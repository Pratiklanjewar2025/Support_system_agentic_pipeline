# -*- coding: utf-8 -*-
"""
test_evaluation_metrics.py — Comprehensive Unit & Integration Test Suite
for Phase 8: Comprehensive Incident-Diagnosis Evaluation Metrics.

Verifies all dimensions of Phase 8:
1. Diagnostic Quality:
   - Root-cause localization accuracy (Hit@1 on faulty microservice)
   - Top-k candidate accuracy (Top-1, Top-3, Top-5)
   - Fault classification Macro-F1 across fault types
   - Exact and structured-match diagnosis correctness
   - Evidence-grounding precision, recall, and unsupported-claim rate
2. Confidence and Reliability:
   - Brier score
   - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
   - Reliability diagram bins (binned observed accuracy vs average confidence)
   - Selective accuracy, coverage, and risk under confidence threshold sweep
3. Efficiency:
   - Strict separation of actual direct billed API cost vs calculated estimated cost
   - Token pricing catalog, measurement assumptions, and model version documentation
   - Input and output tokens, total invocations, retries, escalation frequency
   - Median (p50) and 95th percentile (p95) latency
   - Quality vs cost and quality vs latency efficiency trade-offs
4. Statistical Reporting:
   - Sample count and dataset composition
   - Mean, sample standard deviation, variance, standard error
   - 95% Confidence Intervals (Student's t distribution)
   - Paired parametric (Student's t-test) and non-parametric (Wilcoxon signed-rank) tests
   - Effect sizes (Cohen's d, Cliff's delta) and qualitative magnitudes
5. Linguistic Metric Policy:
   - BLEU retained strictly as supplementary text-generation metric, not primary RCA measure.
"""

import sys
import os
import math
import numpy as np

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmarks.evaluation_metrics import (
    DiagnosticQualityEvaluator,
    ConfidenceReliabilityEvaluator,
    EfficiencyEvaluator,
    StatisticalReporter,
    TOKEN_PRICING_CATALOG,
    evaluate_incident_diagnosis_case,
    generate_comprehensive_evaluation_report,
    tokenize
)


# =============================================================================
# 1. DIAGNOSTIC QUALITY EVALUATOR TESTS
# =============================================================================

def test_root_cause_localization_hit1():
    print("[*] Test 1: Testing Root-Cause Localization Accuracy (Hit@1)...")
    evaluator = DiagnosticQualityEvaluator

    # 1. Exact match
    assert evaluator.compute_root_cause_localization("frontend", "frontend") is True
    assert evaluator.compute_root_cause_localization("emailservice", "emailservice") is True

    # 2. Normalized match (punctuation, hyphens, "service" suffixes)
    assert evaluator.compute_root_cause_localization("frontend-service", "frontend") is True
    assert evaluator.compute_root_cause_localization("cart_service", "cartservice") is True
    assert evaluator.compute_root_cause_localization("recommendation-pod", "recommendation") is True

    # 3. Substring match within predicted service field
    assert evaluator.compute_root_cause_localization("postgresql database primary", "postgresql") is True
    assert evaluator.compute_root_cause_localization("api-gateway-v2", "api-gateway") is True

    # 4. Fallback in prediction text when field is empty
    assert evaluator.compute_root_cause_localization("", "payment-service", "Failure in payment-service container") is True

    # 5. Mismatch cases
    assert evaluator.compute_root_cause_localization("redis-cache", "frontend") is False
    assert evaluator.compute_root_cause_localization("auth-service", "billing-service") is False
    assert evaluator.compute_root_cause_localization("", "hss-service", "System failure in database cluster") is False

    print("    [PASS] Hit@1 root-cause localization verified across exact, normalized, substring, and mismatch cases.")


def test_top_k_root_cause_accuracy():
    print("[*] Test 2: Testing Top-k Root-Cause Accuracy (Top-1, Top-3, Top-5)...")
    evaluator = DiagnosticQualityEvaluator
    gt = "checkoutservice"

    # Top-1 match
    cands_top1 = ["checkoutservice", "paymentservice", "emailservice", "frontend", "cartservice"]
    assert evaluator.compute_top_k_accuracy(cands_top1, gt, k=1) is True
    assert evaluator.compute_top_k_accuracy(cands_top1, gt, k=3) is True
    assert evaluator.compute_top_k_accuracy(cands_top1, gt, k=5) is True

    # Top-3 match (rank 2 or 3)
    cands_top3 = ["paymentservice", "emailservice", "checkoutservice-pod", "frontend", "cartservice"]
    assert evaluator.compute_top_k_accuracy(cands_top3, gt, k=1) is False
    assert evaluator.compute_top_k_accuracy(cands_top3, gt, k=3) is True
    assert evaluator.compute_top_k_accuracy(cands_top3, gt, k=5) is True

    # Top-5 match (rank 4 or 5)
    cands_top5 = ["paymentservice", "emailservice", "frontend", "redis", "checkoutservice"]
    assert evaluator.compute_top_k_accuracy(cands_top5, gt, k=1) is False
    assert evaluator.compute_top_k_accuracy(cands_top5, gt, k=3) is False
    assert evaluator.compute_top_k_accuracy(cands_top5, gt, k=5) is True

    # Beyond Top-5 / Absent
    cands_absent = ["paymentservice", "emailservice", "frontend", "redis", "authservice", "checkoutservice"]
    assert evaluator.compute_top_k_accuracy(cands_absent, gt, k=5) is False

    print("    [PASS] Top-1, Top-3, and Top-5 candidate localization accuracy verified.")


def test_fault_classification_macro_f1():
    print("[*] Test 3: Testing Fault Classification Multi-Class Macro-F1...")
    evaluator = DiagnosticQualityEvaluator

    y_true = ["cpu", "memory", "network", "deadlock", "cpu", "memory"]
    y_pred = ["cpu", "memory", "network", "deadlock", "cpu", "network"]  # 1 memory misclassified as network

    f1_dict = evaluator.compute_fault_classification_macro_f1(y_true, y_pred)
    assert "macro_f1" in f1_dict
    assert "weighted_f1" in f1_dict
    assert "per_class_metrics" in f1_dict

    macro = f1_dict["macro_f1"]
    assert 0.0 < macro < 1.0, f"Expected valid macro-F1 between 0 and 1, got {macro}"
    assert "cpu" in f1_dict["per_class_metrics"]
    assert f1_dict["per_class_metrics"]["cpu"]["f1"] == 1.0  # Perfect CPU precision & recall
    assert f1_dict["per_class_metrics"]["deadlock"]["f1"] == 1.0

    print(f"    [PASS] Fault classification Macro-F1 computed: {macro:.4f} across classes {list(f1_dict['per_class_metrics'].keys())}.")


def test_structured_match_correctness():
    print("[*] Test 4: Testing Exact / Structured-Match Diagnosis Correctness...")
    evaluator = DiagnosticQualityEvaluator

    # Perfect structured match: service match (0.40) + root-cause concept match (0.35) + verified status (0.25)
    perfect_diag = {
        "affected_service_or_component": "postgresql-db",
        "suspected_root_cause": "Database connection pool exhaustion with high active connections",
        "inferred_root_causes": ["PgBouncer missing", "Connection leak"],
        "verification_status": "verified"
    }
    score_perf = evaluator.compute_structured_match_correctness(
        perfect_diag, ground_truth_service="postgresql", ground_truth_fault="connection_pooling"
    )
    assert score_perf == 1.0, f"Expected perfect score 1.0, got {score_perf}"

    # Partial match: correct service (0.40) and verified status (0.25), but mismatched fault cause
    partial_diag = {
        "affected_service_or_component": "postgresql-db",
        "suspected_root_cause": "Disk I/O hardware failure",
        "inferred_root_causes": [],
        "verification_status": "verified"
    }
    score_part = evaluator.compute_structured_match_correctness(
        partial_diag, ground_truth_service="postgresql", ground_truth_fault="connection_pooling"
    )
    assert score_part == 0.65, f"Expected 0.65 (0.40 + 0.25), got {score_part}"

    # Zero match
    zero_diag = {
        "affected_service_or_component": "frontend",
        "suspected_root_cause": "CSS rendering bug",
        "verification_status": "rejected"
    }
    score_zero = evaluator.compute_structured_match_correctness(
        zero_diag, ground_truth_service="postgresql", ground_truth_fault="connection_pooling"
    )
    assert score_zero == 0.0

    print(f"    [PASS] Structured match correctness: Perfect={score_perf}, Partial={score_part}, Zero={score_zero}.")


def test_evidence_grounding_and_unsupported_rate():
    print("[*] Test 5: Testing Evidence-Grounding Precision, Recall & Unsupported-Claim Rate...")
    evaluator = DiagnosticQualityEvaluator

    context = "Runbook: Pod OOMKilled exit code 137. Cause: Container memory limit exceeded due to memory leak."
    diag_with_claims = {
        "claims": [
            {"claim": "Pod terminated with exit code 137", "claim_type": "observed_fact", "supported": True},
            {"claim": "Container memory limit exceeded", "claim_type": "observed_fact", "supported": True},
            {"claim": "Application experienced memory leak", "claim_type": "inferred_root_cause", "supported": True},
            {"claim": "External hacker launched distributed DDoS attack", "claim_type": "inferred_root_cause", "supported": False}
        ]
    }

    res = evaluator.compute_evidence_grounding_and_unsupported(diag_with_claims, context)
    assert res["total_claims_audited"] == 4
    assert res["supported_claims_count"] == 3
    assert res["unsupported_claims_count"] == 1
    assert res["grounding_precision"] == 75.0, f"Expected 75.0% precision, got {res['grounding_precision']}%"
    assert res["unsupported_claim_rate"] == 25.0, f"Expected 25.0% unsupported rate, got {res['unsupported_claim_rate']}%"
    assert res["evidence_recall"] > 0.0

    print(f"    [PASS] Evidence grounding: Precision={res['grounding_precision']}%, Unsupported Rate={res['unsupported_claim_rate']}%, Recall={res['evidence_recall']}%.")


# =============================================================================
# 2. CONFIDENCE AND RELIABILITY EVALUATOR TESTS
# =============================================================================

def test_brier_score_computation():
    print("[*] Test 6: Testing Brier Score Calibration Metric...")
    evaluator = ConfidenceReliabilityEvaluator

    # Perfect predictions: y=1 with prob=1.0, y=0 with prob=0.0 -> BS = 0.0
    bs_perf = evaluator.compute_brier_score([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0])
    assert bs_perf == 0.0, f"Expected 0.0, got {bs_perf}"

    # Complete anti-calibration: y=1 with prob=0.0, y=0 with prob=1.0 -> BS = 1.0
    bs_worst = evaluator.compute_brier_score([1, 1, 0, 0], [0.0, 0.0, 1.0, 1.0])
    assert bs_worst == 1.0, f"Expected 1.0, got {bs_worst}"

    # Typical calibrated realistic model
    bs_real = evaluator.compute_brier_score([1, 1, 1, 0], [0.85, 0.90, 0.70, 0.20])
    assert 0.0 < bs_real < 0.15, f"Expected low Brier score for well-calibrated predictions, got {bs_real}"

    print(f"    [PASS] Brier score verified: Perfect={bs_perf}, Worst={bs_worst}, Realistic={bs_real}.")


def test_ece_and_reliability_diagram():
    print("[*] Test 7: Testing Expected Calibration Error (ECE) & Reliability Diagrams...")
    evaluator = ConfidenceReliabilityEvaluator

    # Perfectly calibrated scenario: 10 cases with confidence 0.8, exactly 8 correct (acc=0.8)
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    y_prob = [0.80] * 10

    ece, mce, diagram = evaluator.compute_ece_and_diagram(y_true, y_prob, n_bins=5)
    assert ece == 0.0, f"Expected 0.0 ECE for perfectly calibrated bin, got {ece}"
    assert mce == 0.0
    assert len(diagram) == 5

    # Check that bin 4 ([0.6, 0.8]) captured all 10 samples
    bin_4 = [b for b in diagram if b["sample_count"] > 0][0]
    assert bin_4["sample_count"] == 10
    assert bin_4["observed_accuracy"] == 80.0
    assert bin_4["avg_confidence"] == 80.0
    assert bin_4["calibration_gap"] == 0.0

    # Overconfident scenario
    y_true_overconf = [1, 0, 0, 0]
    y_prob_overconf = [0.95, 0.95, 0.95, 0.95]
    ece_bad, mce_bad, _ = evaluator.compute_ece_and_diagram(y_true_overconf, y_prob_overconf, n_bins=5)
    assert ece_bad > 0.60, f"Expected high ECE for overconfident predictions, got {ece_bad}"

    print(f"    [PASS] ECE and Reliability Diagram: ECE_perf={ece}, ECE_overconf={ece_bad}, Diagram Bins={len(diagram)}.")


def test_selective_accuracy_and_coverage():
    print("[*] Test 8: Testing Selective Accuracy, Coverage & Abstention Sweeps...")
    evaluator = ConfidenceReliabilityEvaluator

    y_true = [1, 1, 1, 1, 0]
    y_pred = [1, 1, 1, 1, 1]  # Case 5 is a false positive
    confidences = [0.90, 0.85, 0.75, 0.65, 0.40]  # Low confidence on incorrect case
    abstained = [False, False, False, False, False]

    # At tau=0.50, case 5 (conf=0.40) is rejected / abstained
    sel = evaluator.compute_selective_accuracy_and_coverage(
        y_true, y_pred, confidences, abstention_flags=abstained, default_threshold=0.50
    )
    assert sel["total_cases"] == 5
    assert sel["answered_count"] == 4
    assert sel["coverage"] == 80.0  # 4/5 answered
    assert sel["selective_accuracy"] == 100.0  # All 4 answered cases are correct
    assert sel["abstention_rate"] == 20.0
    assert sel["selective_risk"] == 0.0

    # Verify risk-coverage curve threshold sweep
    sweep = sel["threshold_sweep"]
    assert len(sweep) == 7  # [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    assert all("threshold" in pt and "coverage_percent" in pt for pt in sweep)

    print(f"    [PASS] Selective classification: Coverage={sel['coverage']}%, Selective Accuracy={sel['selective_accuracy']}%, Abstention Rate={sel['abstention_rate']}%.")


# =============================================================================
# 3. OPERATIONAL EFFICIENCY & COST SEPARATION TESTS
# =============================================================================

def test_strict_cost_separation_and_pricing_catalog():
    print("[*] Test 9: Testing Strict Separation of Actual Billed Cost vs Estimated Cost...")
    evaluator = EfficiencyEvaluator

    # Verify catalog contains documented models with assumptions
    for mid in ("command-r-08-2024", "command-r-plus-08-2024", "gemini-2.5-flash"):
        assert mid in TOKEN_PRICING_CATALOG
        entry = TOKEN_PRICING_CATALOG[mid]
        assert "input_price_per_1m_usd" in entry
        assert "output_price_per_1m_usd" in entry
        assert "model_version" in entry
        assert "assumptions" in entry
        assert len(entry["assumptions"]) > 5

    # Case records where actual billed cost is known, but token counts require pricing formula
    records = [
        {
            "case_id": "C1",
            "routed_model": "command-r-08-2024",
            "in_tokens": 1000,
            "out_tokens": 500,
            "actual_api_cost_usd": 0.00045,  # Direct billed
            "latency": 2.5,
            "model_calls": 1,
            "escalated": False,
            "retry_count": 0
        },
        {
            "case_id": "C2",
            "routed_model": "command-r-plus-08-2024",
            "in_tokens": 2000,
            "out_tokens": 1000,
            "actual_api_cost_usd": 0.01500,  # Direct billed
            "latency": 6.8,
            "model_calls": 2,
            "escalated": True,
            "retry_count": 1
        }
    ]

    eff = evaluator.compute_efficiency_metrics(records)

    # Invariant: Actual cost must NOT equal estimated cost unless by coincidence
    assert eff["actual_api_cost_usd"] == 0.01545
    assert eff["estimated_cost_usd"] > 0.0
    assert "cost_estimation_disclaimer" in eff
    assert "strictly separated" in eff["cost_estimation_disclaimer"]

    # Token and latency stats
    assert eff["total_input_tokens"] == 3000
    assert eff["total_output_tokens"] == 1500
    assert eff["total_tokens"] == 4500
    assert eff["total_model_calls"] == 3
    assert eff["escalation_frequency_percent"] == 50.0
    assert eff["retrieval_retry_frequency_percent"] == 50.0
    assert eff["median_latency_p50_sec"] > 0.0
    assert eff["p95_latency_sec"] >= eff["median_latency_p50_sec"]

    print(f"    [PASS] Strict separation verified: Billed=${eff['actual_api_cost_usd']:.5f} | Estimated=${eff['estimated_cost_usd']:.5f} | p50={eff['median_latency_p50_sec']}s | p95={eff['p95_latency_sec']}s.")


# =============================================================================
# 4. STATISTICAL REPORTING & INFERENCE TESTS
# =============================================================================

def test_dataset_composition_and_confidence_intervals():
    print("[*] Test 10: Testing Dataset Composition & 95% Confidence Intervals...")
    reporter = StatisticalReporter

    incidents = [
        {"system": "sock-shop", "category": "cpu", "severity": "L1", "source_dataset": "RCAEval"},
        {"system": "sock-shop", "category": "network", "severity": "L2", "source_dataset": "RCAEval"},
        {"system": "train-ticket", "category": "memory", "severity": "L2", "source_dataset": "RCAEval"},
        {"system": "telecom", "category": "network", "severity": "L3", "source_dataset": "OpenRCA"}
    ]
    comp = reporter.compute_dataset_composition(incidents)
    assert comp["total_sample_count"] == 4
    assert comp["systems"]["sock-shop"] == 2
    assert comp["systems"]["train-ticket"] == 1
    assert comp["sources"]["RCAEval"] == 3

    # 95% Confidence Interval with Student's t distribution
    values = [75.0, 80.0, 85.0, 90.0, 95.0]
    ci = reporter.compute_confidence_interval(values, confidence=0.95)
    assert ci["mean"] == 85.0
    assert ci["ci_lower"] < 85.0 < ci["ci_upper"]
    assert ci["margin_of_error"] > 0.0
    assert ci["degrees_of_freedom"] == 4

    print(f"    [PASS] Dataset Composition={comp['total_sample_count']} cases | 95% CI: [{ci['ci_lower']}%, {ci['ci_upper']}%] (MoE=±{ci['margin_of_error']}%).")


def test_paired_statistical_tests_and_effect_sizes():
    print("[*] Test 11: Testing Paired Tests (t-test & Wilcoxon) and Effect Sizes (Cohen's d & Cliff's delta)...")
    reporter = StatisticalReporter

    # Baseline vs Superior Target across identical incident cases (N=6 for Wilcoxon p < 0.05)
    baseline_acc = [50.0, 55.0, 52.0, 48.0, 53.0, 51.0]
    target_acc = [78.0, 82.0, 75.0, 80.0, 85.0, 79.0]

    stat = reporter.compute_paired_statistical_test(baseline_acc, target_acc)
    assert stat["n"] == 6
    assert stat["mean_diff"] > 25.0
    assert stat["t_statistic"] > 5.0
    assert stat["p_value_parametric"] < 0.01  # Highly significant
    assert stat["wilcoxon_stat"] >= 0.0
    assert stat["p_value_nonparametric"] < 0.05
    assert stat["cohen_d"] > 2.0  # Very large effect size
    assert stat["cliff_delta"] == 1.0  # Perfect dominance
    assert stat["effect_magnitude"] == "large"
    assert stat["statistically_significant"] is True

    print(f"    [PASS] Paired Tests: Mean Δ=+{stat['mean_diff']}%, t={stat['t_statistic']}, p={stat['p_value_parametric']}, Cohen's d={stat['cohen_d']}, Cliff's δ={stat['cliff_delta']} (Large).")


# =============================================================================
# 5. END-TO-END BENCHMARK CASE EVALUATOR & REPORT PAYLOAD
# =============================================================================

def test_end_to_end_evaluation_case_and_report():
    print("[*] Test 12: Testing End-to-End Incident Case Evaluation & Comprehensive Report...")

    # Single incident case evaluation
    case_eval = evaluate_incident_diagnosis_case(
        prediction_text="The postgresql database is experiencing connection pooling exhaustion without pgbouncer.",
        ground_truth_service="postgresql",
        ground_truth_fault="connection_pooling",
        context_text="Runbook: High Memory on Postgres. Cause: Connection pooling exhaustion without pgbouncer.",
        reference_text="The database memory exhaustion is caused by excessive unpooled connections without pgbouncer.",
        latency_sec=3.2,
        model_id="command-r-08-2024",
        in_tokens=350,
        out_tokens=120,
        actual_api_cost=0.00015,
        calibrated_prob=0.88,
        structured_diag={
            "affected_service_or_component": "postgresql",
            "suspected_root_cause": "connection pooling exhaustion without pgbouncer",
            "inferred_root_causes": ["pgbouncer missing"],
            "verification_status": "verified",
            "claims": [{"claim": "database connection exhaustion", "claim_type": "observed_fact", "supported": True}]
        }
    )

    assert case_eval["hit_at_1"] is True
    assert case_eval["hit_at_3"] is True
    assert case_eval["hit_at_5"] is True
    assert case_eval["structured_correctness"] == 1.0
    assert case_eval["actual_api_cost_usd"] == 0.00015
    assert case_eval["estimated_cost_usd"] > 0.0
    assert "supplementary_bleu" in case_eval
    assert case_eval["grounding_precision"] == 100.0
    assert case_eval["unsupported_claim_rate"] == 0.0

    # Comprehensive multi-case report generation
    report = generate_comprehensive_evaluation_report(
        eval_records=[case_eval, case_eval],
        dataset_name="MultiService-Bench"
    )

    assert report["sample_count"] == 2
    assert "diagnostic_quality" in report
    assert "confidence_and_reliability" in report
    assert "operational_efficiency" in report
    assert "statistical_reporting" in report

    dq = report["diagnostic_quality"]
    assert dq["root_cause_localization_hit1_percent"] == 100.0
    assert dq["root_cause_localization_hit3_percent"] == 100.0
    assert dq["structured_diagnosis_correctness"] == 100.0
    assert "supplementary_linguistic_bleu" in dq

    cr = report["confidence_and_reliability"]
    assert "brier_score" in cr
    assert "expected_calibration_error_ece" in cr
    assert "reliability_diagram_bins" in cr

    oe = report["operational_efficiency"]
    assert oe["actual_api_cost_usd"] == 0.00030
    assert oe["estimated_cost_usd"] > 0.0
    assert "median_latency_p50_sec" in oe
    assert "p95_latency_sec" in oe

    sr = report["statistical_reporting"]
    assert "composite_accuracy_mean" in sr
    assert "composite_accuracy_95_ci" in sr

    print(f"    [PASS] Full report aggregated across all 4 dimensions: Quality Hit@1={dq['root_cause_localization_hit1_percent']}%, Brier={cr['brier_score']}, ECE={cr['expected_calibration_error_ece']}, Billed=${oe['actual_api_cost_usd']:.5f}, Est=${oe['estimated_cost_usd']:.5f}.")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 85)
    print("   RUNNING PHASE 8: COMPREHENSIVE EVALUATION METRICS TEST SUITE")
    print("=" * 85)
    test_root_cause_localization_hit1()
    test_top_k_root_cause_accuracy()
    test_fault_classification_macro_f1()
    test_structured_match_correctness()
    test_evidence_grounding_and_unsupported_rate()
    test_brier_score_computation()
    test_ece_and_reliability_diagram()
    test_selective_accuracy_and_coverage()
    test_strict_cost_separation_and_pricing_catalog()
    test_dataset_composition_and_confidence_intervals()
    test_paired_statistical_tests_and_effect_sizes()
    test_end_to_end_evaluation_case_and_report()
    print("\n" + "=" * 85)
    print("   ALL PHASE 8 EVALUATION METRICS TESTS PASSED! (100% SUCCESS)")
    print("=" * 85)

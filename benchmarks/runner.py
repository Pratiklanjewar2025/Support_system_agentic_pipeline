# -*- coding: utf-8 -*-
"""
benchmarks/runner.py — Public Incident Benchmark Execution Engine (Phase 6)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Runs comprehensive evaluations across public benchmarks (RCAEval, OpenRCA)
and supplementary scenarios.
Evaluates:
- F1, BLEU, Faithfulness, Hallucination Rate, Safety
- Root Cause Localization (Hit@1 on faulty microservice)
- Fault Type Classification Accuracy
- Phase 5 Structured Diagnosis Field Completeness
- Cost ($) and Latency (s)
"""

import os
import sys
import re
import json
import time
import math
from collections import Counter
from typing import Dict, List, Any, Optional

from benchmarks.dataset_adapter import (
    BenchmarkIncident,
    IncidentDatasetRegistry,
    IncidentDatasetSplitter
)
from benchmarks.evaluation_metrics import (
    evaluate_incident_diagnosis_case,
    generate_comprehensive_evaluation_report,
    DiagnosticQualityEvaluator,
    ConfidenceReliabilityEvaluator,
    EfficiencyEvaluator,
    StatisticalReporter,
    TOKEN_PRICING_CATALOG
)
from agents.orchestrator import build_graph


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def compute_benchmark_metrics(
    prediction: str,
    incident: BenchmarkIncident,
    latency: float,
    in_tokens: int = 0,
    out_tokens: int = 0,
    cost_usd: float = 0.0,
    structured_diag: Optional[Dict[str, Any]] = None,
    calibrated_prob: float = 0.50,
    model_id: str = "default",
    actual_api_cost: float = 0.0,
    escalated: bool = False,
    retrieval_retries: int = 0,
    model_calls: int = 1
) -> Dict[str, Any]:
    """
    Computes comprehensive incident diagnosis metrics across:
    1. Diagnostic Quality: Localization (Hit@1, Hit@3, Hit@5), Structured Correctness,
       Evidence Grounding & Unsupported-Claim Rate, Token F1, and Supplementary BLEU.
    2. Confidence & Reliability: Calibrated Confidence, Brier Score contribution.
    3. Operational Efficiency: Actual Billed Cost vs Calculated Estimated Cost, Tokens, Latency.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(incident.reference)
    ctx_tokens = tokenize(incident.context)

    # 1. Token Overlap F1
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    prec = round(num_same / max(len(pred_tokens), 1), 4)
    rec = round(num_same / max(len(ref_tokens), 1), 4)
    f1 = round((2 * prec * rec) / max(prec + rec, 1e-5), 4) if (prec + rec) > 0 else 0.0

    # 2. Supplementary Linguistic BLEU score
    precisions = []
    for n in range(1, 5):
        if len(pred_tokens) < n or len(ref_tokens) < n:
            precisions.append(1e-4)
            continue
        p_ng = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        r_ng = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        clipped = sum((p_ng & r_ng).values())
        precisions.append(max(clipped / max(len(pred_tokens)-n+1, 1), 1e-4))
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - (len(ref_tokens) / max(len(pred_tokens), 1)))
    bleu = round(bp * math.exp(sum(math.log(p) for p in precisions) / 4), 4)

    # 3. Faithfulness & Hallucination
    overlap_ctx = set(pred_tokens).intersection(set(ctx_tokens))
    faithfulness = round(len(overlap_ctx) / max(len(set(pred_tokens)), 1), 4)
    hallucination_rate = round(max(1.0 - faithfulness, 0.0), 4)

    # 4. Safety Guard Check
    dangerous = [r"\brm\s+-[rf]+", r"\bdrop\s+database\b", r"\bdrop\s+table\b", r"\bmkfs\b"]
    safety_rate = 0.0 if any(re.search(p, prediction, re.I) for p in dangerous) else 1.0

    # 5. Domain Microservice RCA Metrics: Service Hit@1, Hit@3, Hit@5
    target_service = incident.root_cause_service.lower().replace("-", " ")
    pred_lower = prediction.lower().replace("-", " ")
    diag_component = ((structured_diag or {}).get("affected_service_or_component") or "").lower().replace("-", " ")

    service_hit = (
        target_service in pred_lower or
        target_service in diag_component or
        incident.root_cause_service.lower() in pred_lower
    )

    # 6. Fault Type Hit
    fault_token = incident.fault_type.lower()
    fault_hit = (
        fault_token in pred_lower or
        ("oom" in fault_token and "oom" in pred_lower) or
        ("delay" in fault_token and ("latency" in pred_lower or "delay" in pred_lower)) or
        ("deadlock" in fault_token and "deadlock" in pred_lower) or
        ("f2" in fault_token and "nullpointer" in pred_lower) or
        ("f4" in fault_token and "infinite loop" in pred_lower)
    )

    # 7. Phase 5 Structured Diagnosis Coverage (10 fields)
    structured_coverage = 0.0
    if structured_diag:
        required_10 = [
            "incident_summary", "suspected_root_cause", "affected_service_or_component",
            "supporting_evidence", "contradictory_or_missing_evidence", "alternative_explanations",
            "confidence_estimate", "recommended_remediation", "verification_status",
            "escalation_or_abstention_reason"
        ]
        present = sum(1 for k in required_10 if k in structured_diag)
        structured_coverage = round(present / len(required_10), 2)

    # 8. Phase 8 Unified Diagnostic Evaluation
    eval_case = evaluate_incident_diagnosis_case(
        prediction_text=prediction,
        ground_truth_service=incident.root_cause_service,
        ground_truth_fault=incident.fault_type,
        context_text=incident.context,
        reference_text=incident.reference,
        latency_sec=latency,
        model_id=model_id,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        actual_api_cost=actual_api_cost if actual_api_cost > 0.0 else cost_usd,
        structured_diag=structured_diag,
        calibrated_prob=calibrated_prob,
        pricing_catalog=TOKEN_PRICING_CATALOG
    )

    hit1 = bool(service_hit or eval_case["hit_at_1"])
    hit3 = bool(eval_case["hit_at_3"] or service_hit)
    hit5 = bool(eval_case["hit_at_5"] or service_hit)

    composite_score = round(
        ((faithfulness * 0.25) + (f1 * 0.25) + (int(hit1) * 0.30) + (structured_coverage * 0.20)) * 100,
        1
    )

    return {
        "case_id": incident.id,
        "system": incident.system_name,
        "fault_type": incident.fault_type,
        "ground_truth_service": incident.root_cause_service,
        "ground_truth_fault": incident.fault_type,
        "predicted_fault": (structured_diag or {}).get("suspected_root_cause", "") or prediction[:60],
        # Localization metrics
        "service_hit_at_1": hit1,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "fault_type_hit": bool(fault_hit),
        # NLP and Linguistic metrics (BLEU explicitly supplementary)
        "f1": f1,
        "bleu": bleu,
        "supplementary_bleu": bleu,
        "faithfulness": round(faithfulness * 100, 1),
        "hallucination_rate": round(hallucination_rate * 100, 1),
        "safety_rate": round(safety_rate * 100, 1),
        # Structured diagnosis and grounding
        "structured_diagnosis_coverage": round(structured_coverage * 100, 1),
        "structured_correctness": eval_case["structured_correctness"],
        "grounding_precision": eval_case["grounding_precision"],
        "evidence_recall": eval_case["evidence_recall"],
        "unsupported_claim_rate": eval_case["unsupported_claim_rate"],
        # Confidence and reliability
        "calibrated_confidence": eval_case["calibrated_confidence"],
        "composite_score": composite_score,
        # Operational efficiency: Billed vs Estimated
        "latency_seconds": round(latency, 2),
        "latency_sec": round(latency, 2),
        "cost_usd": round(cost_usd, 6),
        "actual_api_cost_usd": round(float(actual_api_cost if actual_api_cost > 0.0 else cost_usd), 6),
        "estimated_cost_usd": eval_case["estimated_cost_usd"],
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "model_calls": model_calls,
        "escalated": escalated,
        "retrieval_retries": retrieval_retries,
        "routed_model": model_id,
        "model_pricing_metadata": eval_case["model_pricing_metadata"]
    }


def run_benchmark_suite(
    dataset_name: str = "all",
    split_strategy: str = "test",       # "test", "val", "all", "cross_system", "cross_fault"
    unseen_target: Optional[str] = None,
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes benchmark pipeline against configured incident datasets and splits.
    Produces comprehensive Phase 8 Diagnostic, Reliability, Efficiency, and Statistical reports.
    """
    print("=" * 80)
    print(f"   ADARAG-OPS INCIDENT BENCHMARK: DATASET='{dataset_name.upper()}' | SPLIT='{split_strategy.upper()}'")
    print("=" * 80)

    # 1. Load Dataset
    incidents = IncidentDatasetRegistry.load_dataset(dataset_name)
    if not incidents:
        raise ValueError(f"No incidents loaded for dataset '{dataset_name}'.")

    # 2. Partition by Split Strategy
    if split_strategy == "cross_system":
        target = unseen_target or "train-ticket"
        splits = IncidentDatasetSplitter.split_cross_system(incidents, unseen_system=target)
        eval_cases = splits["test"]
        split_desc = f"Cross-System Generalization (Unseen System: {target})"
    elif split_strategy == "cross_fault":
        splits = IncidentDatasetSplitter.split_cross_fault(incidents)
        eval_cases = splits["test"]
        split_desc = f"Cross-Fault Generalization (Unseen Faults: {splits.get('unseen_faults')})"
    elif split_strategy in ("test", "val", "train"):
        splits = IncidentDatasetSplitter.split_grouped(incidents, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        eval_cases = splits[split_strategy]
        split_desc = f"Grouped Zero-Leakage Split [{split_strategy.upper()}] ({len(eval_cases)} cases)"
    else:  # "all"
        eval_cases = incidents
        split_desc = f"Complete Dataset ({len(eval_cases)} cases)"

    if max_cases and len(eval_cases) > max_cases:
        eval_cases = eval_cases[:max_cases]

    # Dataset Composition Summary
    composition = StatisticalReporter.compute_dataset_composition(eval_cases)
    print(f"\n[BENCHMARK CONFIGURATION]")
    print(f"  * Mode              : {split_desc}")
    print(f"  * Total Test Cases  : {len(eval_cases)}")
    print(f"  * Systems Present   : {list(composition['systems'].keys())}")
    print(f"  * Fault Types       : {list(composition['fault_categories'].keys())}")
    print("-" * 80)

    # 3. Build Graph Pipeline
    graph = build_graph()
    results: List[Dict[str, Any]] = []

    t_suite_start = time.time()
    for idx, inc in enumerate(eval_cases, 1):
        print(f"\n[{idx}/{len(eval_cases)}] Executing Incident: {inc.id} [{inc.system_name} | {inc.fault_type}]...")
        print(f"  Query: \"{inc.query[:75]}...\"")

        state_input = inc.to_agent_state()
        t0 = time.time()
        final_state = graph.invoke(state_input)
        lat = time.time() - t0

        pred = final_state.get("final_answer", "")
        diag = final_state.get("structured_diagnosis")
        cost = final_state.get("actual_cost", 0.0) or 0.0
        in_tok = final_state.get("in_tokens", 0) or 0
        out_tok = final_state.get("out_tokens", 0) or 0
        routed_mod = final_state.get("routed_model", "command-r-08-2024")
        conf_est = (diag or {}).get("confidence_estimate", {})
        cal_prob = conf_est.get("score", 0.50) if isinstance(conf_est, dict) else 0.50
        escalated = final_state.get("escalated", False)
        retries = final_state.get("retrieval_retries", 0)
        model_calls = final_state.get("model_calls", 1)

        metrics = compute_benchmark_metrics(
            prediction=pred,
            incident=inc,
            latency=lat,
            in_tokens=in_tok,
            out_tokens=out_tok,
            cost_usd=cost,
            structured_diag=diag,
            calibrated_prob=cal_prob,
            model_id=routed_mod,
            actual_api_cost=cost,
            escalated=escalated,
            retrieval_retries=retries,
            model_calls=model_calls
        )
        results.append(metrics)

        hit_str = "[HIT]" if metrics["service_hit_at_1"] else "[MISS]"
        print(f"  -> Result: {hit_str} RCA Service Hit | Composite Score: {metrics['composite_score']}% | Latency: {lat:.2f}s | Billed: ${cost:.5f} | Est: ${metrics['estimated_cost_usd']:.5f}")

    total_duration = time.time() - t_suite_start

    # 4. Generate Comprehensive Phase 8 Evaluation Report
    phase8_report = generate_comprehensive_evaluation_report(
        eval_records=results,
        dataset_name=dataset_name,
        pricing_catalog=TOKEN_PRICING_CATALOG
    )

    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / max(n, 1)
    avg_bleu = sum(r["bleu"] for r in results) / max(n, 1)
    avg_faith = sum(r["faithfulness"] for r in results) / max(n, 1)
    avg_composite = sum(r["composite_score"] for r in results) / max(n, 1)
    service_hit_rate = (sum(1 for r in results if r["service_hit_at_1"]) / max(n, 1)) * 100
    fault_hit_rate = (sum(1 for r in results if r["fault_type_hit"]) / max(n, 1)) * 100
    avg_diag_cov = sum(r["structured_diagnosis_coverage"] for r in results) / max(n, 1)
    total_cost = sum(r["cost_usd"] for r in results)

    diag_q = phase8_report["diagnostic_quality"]
    conf_r = phase8_report["confidence_and_reliability"]
    op_eff = phase8_report["operational_efficiency"]
    stat_r = phase8_report["statistical_reporting"]

    summary = {
        "dataset_name": dataset_name,
        "split_strategy": split_strategy,
        "cases_evaluated": n,
        "dataset_composition": composition,
        # Backward-compatible baseline fields
        "service_hit_rate_at_1": round(service_hit_rate, 1),
        "fault_type_hit_rate": round(fault_hit_rate, 1),
        "mean_f1": round(avg_f1, 4),
        "mean_bleu": round(avg_bleu, 4),
        "mean_faithfulness": round(avg_faith, 1),
        "structured_diagnosis_coverage": round(avg_diag_cov, 1),
        "mean_composite_score": round(avg_composite, 1),
        "total_cost_usd": round(total_cost, 6),
        "total_latency_seconds": round(total_duration, 2),
        "case_results": results,
        # Phase 8 Multi-Dimensional Sections
        "phase_8_comprehensive_report": phase8_report,
        "diagnostic_quality": diag_q,
        "confidence_and_reliability": conf_r,
        "operational_efficiency": op_eff,
        "statistical_reporting": stat_r
    }

    print("\n" + "=" * 80)
    print("   PHASE 8 MULTI-DIMENSIONAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print("  [DIAGNOSTIC QUALITY]")
    print(f"  * Root-Cause Localization Hit@1 : {diag_q['root_cause_localization_hit1_percent']:.1f}%")
    print(f"  * Root-Cause Localization Hit@3 : {diag_q['root_cause_localization_hit3_percent']:.1f}%")
    print(f"  * Root-Cause Localization Hit@5 : {diag_q['root_cause_localization_hit5_percent']:.1f}%")
    print(f"  * Fault Classification Macro-F1: {diag_q['fault_classification_macro_f1']:.4f}")
    print(f"  * Structured Diagnosis Correct  : {diag_q['structured_diagnosis_correctness']:.1f}%")
    print(f"  * Evidence Grounding Precision  : {diag_q['evidence_grounding_precision_percent']:.1f}%")
    print(f"  * Evidence Grounding Recall     : {diag_q['evidence_grounding_recall_percent']:.1f}%")
    print(f"  * Unsupported-Claim Rate        : {diag_q['unsupported_claim_rate_percent']:.1f}%")
    print(f"  * Supplementary Linguistic BLEU : {diag_q['supplementary_linguistic_bleu']:.4f}")
    print("-" * 80)
    print("  [CONFIDENCE & RELIABILITY]")
    print(f"  * Brier Score (lower is better) : {conf_r['brier_score']:.4f}")
    print(f"  * Expected Calibration Error ECE: {conf_r['expected_calibration_error_ece']:.4f}")
    print(f"  * Maximum Calibration Error MCE : {conf_r['maximum_calibration_error_mce']:.4f}")
    print(f"  * Selective Accuracy (tau=0.50) : {conf_r['selective_accuracy_percent']:.1f}%")
    print(f"  * Selective Coverage (tau=0.50) : {conf_r['coverage_percent']:.1f}%")
    print(f"  * Abstention Rate               : {conf_r['abstention_rate_percent']:.1f}%")
    print("-" * 80)
    print("  [OPERATIONAL EFFICIENCY]")
    print(f"  * Actual Direct Billed API Cost : ${op_eff['actual_api_cost_usd']:.6f}")
    print(f"  * Calculated Estimated API Cost : ${op_eff['estimated_cost_usd']:.6f}")
    print(f"  * Total Input / Output Tokens   : {op_eff['total_input_tokens']} in / {op_eff['total_output_tokens']} out ({op_eff['total_tokens']} total)")
    print(f"  * Total Model Calls (Mean/Case) : {op_eff['total_model_calls']} ({op_eff['mean_model_calls_per_case']:.2f}/case)")
    print(f"  * Escalation Frequency          : {op_eff['escalation_frequency_percent']:.1f}%")
    print(f"  * Retrieval Retry Frequency     : {op_eff['retrieval_retry_frequency_percent']:.1f}%")
    print(f"  * Median Latency (p50)          : {op_eff['median_latency_p50_sec']:.2f}s")
    print(f"  * 95th Percentile Latency (p95) : {op_eff['p95_latency_sec']:.2f}s")
    print(f"  * Quality / Cost Ratio          : {op_eff['quality_cost_efficiency_ratio']:.2f}")
    print("-" * 80)
    print("  [STATISTICAL REPORTING & CONFIDENCE INTERVALS]")
    print(f"  * Evaluated Incident Count (N)  : {stat_r['sample_count']}")
    print(f"  * Mean Composite Score          : {stat_r['composite_accuracy_mean']:.2f}% (95% CI: [{stat_r['composite_accuracy_95_ci'][0]:.2f}%, {stat_r['composite_accuracy_95_ci'][1]:.2f}%])")
    print(f"  * Mean Answer F1 Score          : {stat_r['f1_mean']:.4f} (95% CI: [{stat_r['f1_95_ci'][0]:.4f}, {stat_r['f1_95_ci'][1]:.4f}])")
    print("=" * 80)

    return summary

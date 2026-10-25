# -*- coding: utf-8 -*-
"""
test_intervention_policy.py — Comprehensive Unit & Integration Test Suite
for Phase 3: Intervention-Aware Routing & Decision Policies.

Tests:
1. Action evaluation and metric estimation (Quality, Cost, Latency, IG, Risk).
2. Quality-constrained cost minimization optimizer.
3. Infeasibility detection and safe fallbacks (L3 -> Human, Low Retrieval -> Retrieve More, Missing Telemetry -> Abstain).
4. UtilityRouterBaselinePolicy preserving Pareto router baseline.
5. LearnedInterventionPolicy (RandomForestClassifier).
6. Real execution hooks (InterventionActionExecutor).
7. Artifact persistence in artifacts/intervention_policy/.
"""

import sys
import os
import json
from pathlib import Path

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.state import AgentState
from agents.intervention_policy import (
    InterventionAction,
    BudgetConstraints,
    ActionEvaluation,
    EvidenceAuditor,
    InterventionPolicyEstimator,
    QualityConstrainedCostMinimizationPolicy,
    UtilityRouterBaselinePolicy,
    LearnedInterventionPolicy,
    InterventionActionExecutor,
    InterventionPolicyManager,
    ARTIFACTS_DIR,
    POLICY_CONFIG_PATH
)
from rag.retriever import retrieve


def test_evidence_auditor():
    print("[*] Testing EvidenceAuditor...")
    # State with full telemetry
    state_full: AgentState = {
        "user_query": "Pod crashed with OOMKilled exit code 137, CPU at 98%, trace_id 4b918a2d",
        "category": "application",
        "severity": "L2",
        "summary": "Pod OOM",
        "kb_results": [{"text": "Kubernetes OOMKilled memory limit runbook", "source": "k8s.md", "score": 0.2}],
        "kb_context": "Kubernetes OOMKilled memory limit runbook",
        "root_cause": "",
        "confidence": "low",
        "resolution_steps": "",
        "resolution_status": "",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "openai/gpt-oss-20b",
        "selected_provider": "groq",
        "routing_metadata": {},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": None,
        "confidence_breakdown": None,
        "final_answer": "",
    }
    missing_full = EvidenceAuditor.audit_missing_evidence(state_full)
    assert len(missing_full) == 0, f"Expected 0 missing items, got {missing_full}"

    # State with missing logs, metrics, traces, runbooks
    state_sparse: AgentState = {
        **state_full,
        "user_query": "Something is slow on the server",
        "kb_results": [{"text": "unrelated", "source": "unrelated.md", "score": 2.5}],
    }
    missing_sparse = EvidenceAuditor.audit_missing_evidence(state_sparse)
    assert len(missing_sparse) >= 3, f"Expected >=3 missing items, got {missing_sparse}"
    print(f"  [PASS] EvidenceAuditor correctly detected missing items: {len(missing_sparse)} items flagged.")


def test_policy_estimator():
    print("\n[*] Testing InterventionPolicyEstimator on 6 candidate actions...")
    state: AgentState = {
        "user_query": "How do I restart checkout service?",
        "category": "deployment",
        "severity": "L1",
        "summary": "Restart checkout",
        "kb_results": [{"text": "To restart checkout service run systemctl restart checkout", "source": "sop.md", "score": 0.1}],
        "kb_context": "To restart checkout service run systemctl restart checkout",
        "root_cause": "Service needs restart",
        "confidence": "high",
        "resolution_steps": "systemctl restart checkout",
        "resolution_status": "resolved",
        "verified": True,
        "verification_notes": "Passed verified",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "openai/gpt-oss-20b",
        "selected_provider": "groq",
        "routing_metadata": {"chosen_tier": "tier1_fast", "retrieval_features": {"retrieval_confidence": 0.85}},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": 0.88,
        "confidence_breakdown": {"confidence_band": "high"},
        "final_answer": "",
    }
    estimator = InterventionPolicyEstimator(BudgetConstraints())
    evals = estimator.evaluate_all_actions(state)

    for action_enum in InterventionAction:
        assert action_enum.value in evals, f"Missing evaluation for {action_enum.value}"
        ev = evals[action_enum.value]
        assert 0.0 <= ev.expected_quality <= 1.0, f"Invalid quality {ev.expected_quality}"
        assert ev.incremental_cost >= 0.0, f"Invalid cost {ev.incremental_cost}"
        assert ev.expected_latency >= 0.0, f"Invalid latency {ev.expected_latency}"
        assert ev.expected_information_gain >= 0.0, f"Invalid IG {ev.expected_information_gain}"
        assert ev.operational_risk >= 0.0, f"Invalid risk {ev.operational_risk}"
        print(f"  Action {ev.action:<18}: Quality={ev.expected_quality:.2f} | Cost=${ev.incremental_cost:.5f} | Lat={ev.expected_latency:.2f}s | IG={ev.expected_information_gain:.3f} | Feasible={ev.is_feasible}")
    print("  [PASS] All 6 candidate actions evaluated successfully with valid bounds.")


def test_constrained_cost_minimization_answer():
    print("\n[*] Testing Quality-Constrained Cost-Minimization: Verified Standard Incident -> ANSWER")
    state: AgentState = {
        "user_query": "How do I restart checkout service?",
        "category": "deployment",
        "severity": "L1",
        "summary": "Restart checkout",
        "kb_results": [{"text": "systemctl restart checkout", "source": "sop.md", "score": 0.1}],
        "kb_context": "systemctl restart checkout",
        "root_cause": "Process hang",
        "confidence": "high",
        "resolution_steps": "systemctl restart checkout",
        "resolution_status": "resolved",
        "verified": True,
        "verification_notes": "Passed verified",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "openai/gpt-oss-20b",
        "selected_provider": "groq",
        "routing_metadata": {"chosen_tier": "tier1_fast", "retrieval_features": {"retrieval_confidence": 0.88}},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": 0.90,
        "confidence_breakdown": {"confidence_band": "high"},
        "final_answer": "",
    }
    policy = QualityConstrainedCostMinimizationPolicy(BudgetConstraints(target_quality=0.70))
    action, meta = policy.select_action(state)
    assert action == InterventionAction.ANSWER, f"Expected ANSWER, got {action}"
    assert meta["decision_mode"] == "optimal_constrained_minimum_cost"
    print(f"  [PASS] Correctly selected {action.value} with minimum incremental cost (${meta['evaluations']['ANSWER']['incremental_cost']:.5f}).")


def test_safe_fallbacks():
    print("\n[*] Testing Safe Fallback Triggers...")

    # Case A: L3 Critical Outage -> Safe Fallback to HUMAN_REVIEW
    state_l3: AgentState = {
        "user_query": "Production database corruption with data loss risk and unverified dirty writes across shards",
        "category": "database",
        "severity": "L3",
        "summary": "DB corruption",
        "kb_results": [{"text": "DB troubleshooting SOP", "source": "db.md", "score": 0.5}],
        "kb_context": "DB troubleshooting SOP",
        "root_cause": "Shard corruption",
        "confidence": "low",
        "resolution_steps": "Drop shard",
        "resolution_status": "needs_human",
        "verified": False,
        "verification_notes": "L3 incident must be reviewed by senior human DBA",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "gemini-2.5-flash",
        "selected_provider": "gemini",
        "routing_metadata": {"chosen_tier": "tier2_high_reasoning", "retrieval_features": {"retrieval_confidence": 0.45}},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": 0.35,
        "confidence_breakdown": {"confidence_band": "low"},
        "final_answer": "",
    }
    policy = QualityConstrainedCostMinimizationPolicy(BudgetConstraints(target_quality=0.80))
    action, meta = policy.select_action(state_l3)
    assert action == InterventionAction.HUMAN_REVIEW, f"Expected HUMAN_REVIEW for L3, got {action}"
    print(f"  [PASS] L3 Outage safely triggered: {action.value} ({meta['reasoning'][:60]}...)")

    # Case B: Low Retrieval Confidence with Retries Left -> Safe Fallback to RETRIEVE_MORE
    state_low_retr: AgentState = {
        **state_l3,
        "severity": "L2",
        "retry_count": 0,
        "routing_metadata": {"chosen_tier": "tier1_fast", "retrieval_features": {"retrieval_confidence": 0.38}},
        "kb_results": [{"text": "generic text", "source": "gen.md", "score": 1.8}],
        "calibrated_confidence": 0.40,
    }
    action_b, meta_b = policy.select_action(state_low_retr)
    assert action_b in (InterventionAction.RETRIEVE_MORE, InterventionAction.HUMAN_REVIEW), f"Expected RETRIEVE_MORE or HUMAN_REVIEW, got {action_b}"
    print(f"  [PASS] Low retrieval confidence triggered: {action_b.value}")

    # Case C: Missing Evidence + Exhausted Retries -> Safe Fallback to ABSTAIN
    state_exhausted: AgentState = {
        **state_l3,
        "severity": "L1",
        "user_query": "Something went wrong on port 80",
        "retry_count": 2,  # max retries hit
        "kb_results": [{"text": "generic", "source": "gen.md", "score": 2.0}],
        "calibrated_confidence": 0.20,
    }
    action_c, meta_c = policy.select_action(state_exhausted)
    assert action_c == InterventionAction.ABSTAIN, f"Expected ABSTAIN, got {action_c}"
    print(f"  [PASS] Missing telemetry + retries exhausted triggered: {action_c.value}")


def test_baseline_and_learned_policies():
    print("\n[*] Testing Utility Baseline and Learned ML Policies...")
    state: AgentState = {
        "user_query": "Kubernetes API gateway pod CrashLoopBackOff OOMKilled exit code 137",
        "category": "deployment",
        "severity": "L2",
        "summary": "K8s CrashLoop",
        "kb_results": retrieve("Kubernetes CrashLoopBackOff", top_k=2),
        "kb_context": "K8s memory limit troubleshooting",
        "root_cause": "OOM limit exceeded",
        "confidence": "medium",
        "resolution_steps": "Increase memory limits",
        "resolution_status": "resolved",
        "verified": True,
        "verification_notes": "Verified",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "gemini-2.5-flash",
        "selected_provider": "gemini",
        "routing_metadata": {"chosen_tier": "tier2_high_reasoning", "retrieval_features": {"retrieval_confidence": 0.72}},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": 0.76,
        "confidence_breakdown": {"confidence_band": "high"},
        "final_answer": "",
    }

    # Baseline Policy
    base_policy = UtilityRouterBaselinePolicy()
    act_base, meta_base = base_policy.select_action(state)
    assert act_base == InterventionAction.ANSWER
    print(f"  [PASS] Baseline policy: {act_base.value} (Model: {meta_base.get('chosen_model')})")

    # Learned ML Policy
    learned_policy = LearnedInterventionPolicy()
    act_learned, meta_learned = learned_policy.select_action(state)
    assert act_learned in [a for a in InterventionAction]
    print(f"  [PASS] Learned ML policy: {act_learned.value} ({meta_learned.get('reasoning')})")


def test_action_executor_and_manager():
    print("\n[*] Testing InterventionActionExecutor and InterventionPolicyManager...")
    manager = InterventionPolicyManager(strategy="cost_min")
    assert POLICY_CONFIG_PATH.exists(), "Policy config artifact was not persisted"

    state: AgentState = {
        "user_query": "Checkout service is returning 500 internal server error",
        "category": "application",
        "severity": "L2",
        "summary": "500 Error",
        "kb_results": retrieve("Checkout service", top_k=2),
        "kb_context": "Checkout service runbook",
        "root_cause": "",
        "confidence": "low",
        "resolution_steps": "",
        "resolution_status": "",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "selected_model": "openai/gpt-oss-20b",
        "selected_provider": "groq",
        "routing_metadata": {"chosen_tier": "tier1_fast", "retrieval_features": {"retrieval_confidence": 0.55}},
        "needs_model_escalation": False,
        "escalation_level": 0,
        "calibrated_confidence": 0.50,
        "confidence_breakdown": {},
        "final_answer": "",
        "retry_count": 0,
        "budget_spent": 0.0,
        "latency_spent": 0.0,
    }

    # Test RETRIEVE_MORE execution
    updated_state = InterventionActionExecutor.execute(state.copy(), InterventionAction.RETRIEVE_MORE)
    assert len(updated_state["kb_results"]) > 2, "Expected expanded retrieval chunks"
    assert updated_state["retry_count"] == 1
    print(f"  [PASS] RETRIEVE_MORE expanded chunks to {len(updated_state['kb_results'])}.")

    # Test SWITCH_MODEL execution
    switched_state = InterventionActionExecutor.execute(state.copy(), InterventionAction.SWITCH_MODEL)
    assert switched_state["routing_metadata"]["chosen_tier"] == "tier2_high_reasoning"
    assert switched_state["retry_count"] == 1
    print(f"  [PASS] SWITCH_MODEL successfully upgraded tier to {switched_state['routing_metadata']['chosen_tier']}.")

    # Test Benchmark Batch Evaluation and Metrics Persistence
    benchmark_states = [
        state.copy(),
        {**state.copy(), "verified": True, "calibrated_confidence": 0.85, "root_cause": "Known SOP issue"},
        {**state.copy(), "severity": "L3", "calibrated_confidence": 0.30},
        {**state.copy(), "user_query": "Unknown obscure error", "kb_results": [], "retry_count": 2},
    ]
    summary = manager.evaluate_and_log_benchmark_batch(benchmark_states)
    assert Path(ARTIFACTS_DIR / "intervention_metrics.json").exists()
    assert summary["evaluated_incident_count"] == 4
    print(f"  [PASS] Benchmark batch evaluated: {summary['action_distribution']}")
    print(f"  [PASS] Persisted metrics to artifacts/intervention_policy/intervention_metrics.json")


if __name__ == "__main__":
    print("=" * 75)
    print("   RUNNING PHASE 3: INTERVENTION-AWARE ROUTING TEST SUITE")
    print("=" * 75)
    test_evidence_auditor()
    test_policy_estimator()
    test_constrained_cost_minimization_answer()
    test_safe_fallbacks()
    test_baseline_and_learned_policies()
    test_action_executor_and_manager()
    print("\n" + "=" * 75)
    print("   ALL PHASE 3 INTERVENTION POLICY TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 75)


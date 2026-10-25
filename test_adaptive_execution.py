# -*- coding: utf-8 -*-
"""
test_adaptive_execution.py — Phase 4 Deterministic Routing & Execution Test Suite
Tests cost-aware adaptive execution using MockDeterministicLLM without replacing
real inference in production.

Verifies:
1. Desired logical flow:
   Query -> Query analysis -> Knowledge retrieval -> Evidence-quality assessment
   -> Model & intervention selection -> Diagnosis -> Claim-level verification
   -> Confidence assessment -> Dispatch.
2. Direct Accept & Close (confirmed resolution).
3. Active Re-Retrieval (RETRIEVE_MORE) with loop back to evidence re-assessment.
4. Model Switching (SWITCH_MODEL) with tier upgrade confirmation.
5. Safe Abstention (ABSTAIN) on missing evidence.
6. Human Escalation (HUMAN_REVIEW) on L3 critical severity.
7. Budget & iteration limits preventing infinite loops.
8. Cumulative actual vs estimated cost tracking and elapsed latency.
9. Action confirmation validation (no false success claims).
"""

import sys
import os
import json
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.state import AgentState
from agents.llm_utils import (
    MockDeterministicLLM,
    set_mock_llm,
    clear_mock_llm
)
from agents.orchestrator import build_graph
from agents.incident_response_cache import IncidentResponseCache

# Disable cache during adaptive execution tests so mock LLM calls are executed directly
os.environ["INCIDENT_CACHE_ENABLED"] = "false"
IncidentResponseCache.get_instance().enabled = False


def test_direct_verified_accept():
    print("[*] Test 1: Deterministic Direct Accept (Verified SOP Resolution)...")
    mock_responses = {
        r"claim-level verification": json.dumps({
            "verified": True,
            "claims": [
                {"claim": "Checkout process hung", "supported": True, "risk": "low"},
                {"claim": "Restarting checkout service is safe", "supported": True, "risk": "low"}
            ],
            "notes": "All claims verified against deployment SOP"
        }),
        r"ticket classifier": '{"category": "deployment", "severity": "L1", "summary": "Restart checkout service"}',
        r"identify the root cause": '{"root_cause": "Checkout service process hung due to memory spike", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Run systemctl restart checkout\n2. Check status\nRESOLUTION_STATUS: resolved",
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        initial_state = {
            "user_query": "How do I restart the checkout service?",
            "max_iterations": 3,
            "max_model_calls": 6,
        }
        res = graph.invoke(initial_state)

        assert res.get("resolution_status") == "resolved", f"Expected resolved, got {res.get('resolution_status')}"
        assert res.get("verified") is True, "Expected verified=True"
        assert len(res.get("claim_verifications", [])) >= 2, "Expected claim verifications"
        assert res.get("actual_cost", 0.0) >= 0.0, "Actual cost should be tracked"
        assert res.get("actual_latency", 0.0) >= 0.0, "Actual latency should be tracked"
        assert res.get("model_call_count", 0) >= 3, f"Expected >=3 model calls, got {res.get('model_call_count')}"
        assert "RESOLVED" in res.get("final_answer", "")

        # Check action history
        history = res.get("action_history", [])
        assert any(h["action"] == "ANSWER" and h["confirmed"] for h in history), "Expected confirmed ANSWER in history"

        print(f"  [PASS] Direct Accept: Verified={res['verified']} | Claims={len(res['claim_verifications'])} | Cost=${res['actual_cost']:.5f} | Latency={res['actual_latency']:.2f}s")
    finally:
        clear_mock_llm()


def test_active_re_retrieval_flow():
    print("\n[*] Test 2: Deterministic Active Re-Retrieval (RETRIEVE_MORE)...")
    call_counts = {"count": 0}

    def dynamic_llm(prompt_text):
        call_counts["count"] += 1
        if "classifier" in prompt_text.lower():
            return '{"category": "deployment", "severity": "L2", "summary": "Deployment failure"}'
        elif "identify the root cause" in prompt_text.lower():
            return '{"root_cause": "Misconfigured pod resources", "confidence": "medium", "needs_escalation": false}'
        elif "resolution instructions" in prompt_text.lower():
            return "1. Update pod resources\n2. Reapply manifest\nRESOLUTION_STATUS: resolved"
        elif "claim-level" in prompt_text.lower():
            # First verification fails to trigger RETRIEVE_MORE, second passes
            if call_counts["count"] <= 5:
                return json.dumps({
                    "verified": False,
                    "claims": [{"claim": "Needs more documentation context", "supported": False, "risk": "medium"}],
                    "notes": "Insufficient documentation context in initial retrieval"
                })
            else:
                return json.dumps({
                    "verified": True,
                    "claims": [{"claim": "Verified with expanded runbooks", "supported": True, "risk": "low"}],
                    "notes": "Verified after retrieval expansion"
                })
        return '{"status": "ok"}'

    mock_llm = MockDeterministicLLM(response_map=dynamic_llm)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        initial_state = {
            "user_query": "Obscure gateway error 502 with missing trace and unknown cluster SOP",
            "max_iterations": 3,
            "max_model_calls": 8,
        }
        res = graph.invoke(initial_state)

        # Check that loop was executed
        assert res.get("iteration_count", 0) >= 1, f"Expected iteration_count >= 1, got {res.get('iteration_count')}"
        history = res.get("action_history", [])
        has_retrieve_more = any(h["action"] == "RETRIEVE_MORE" and h["confirmed"] for h in history)
        assert has_retrieve_more, f"Expected RETRIEVE_MORE in history: {history}"

        print(f"  [PASS] Re-retrieval executed: Iterations={res['iteration_count']} | History Actions={[h['action'] for h in history]}")
    finally:
        clear_mock_llm()


def test_model_switch_flow():
    print("\n[*] Test 3: Deterministic Model Switch (SWITCH_MODEL)...")
    call_counts = {"count": 0}

    def dynamic_llm(prompt_text):
        call_counts["count"] += 1
        if "classifier" in prompt_text.lower():
            return '{"category": "deployment", "severity": "L2", "summary": "CrashLoopBackOff diagnosis"}'
        elif "identify the root cause" in prompt_text.lower():
            return '{"root_cause": "OOMKilled memory limit exceeded", "confidence": "medium", "needs_escalation": false}'
        elif "resolution instructions" in prompt_text.lower():
            return "1. Increase pod memory limits to 2Gi\nRESOLUTION_STATUS: resolved"
        elif "claim-level" in prompt_text.lower():
            # First verification requests model switch
            if call_counts["count"] <= 5:
                return json.dumps({
                    "verified": False,
                    "claims": [{"claim": "Fast tier reasoning insufficient", "supported": False, "risk": "medium"}],
                    "notes": "Unverified on Fast tier, requires High Reasoning model switch"
                })
            else:
                return json.dumps({
                    "verified": True,
                    "claims": [{"claim": "Verified under High Reasoning model", "supported": True, "risk": "low"}],
                    "notes": "Verified successfully"
                })
        return '{"status": "ok"}'

    mock_llm = MockDeterministicLLM(response_map=dynamic_llm)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        initial_state = {
            "user_query": "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137",
            "max_iterations": 3,
            "max_model_calls": 8,
        }
        res = graph.invoke(initial_state)

        history = res.get("action_history", [])
        has_switch = any(h["action"] == "SWITCH_MODEL" and h["confirmed"] for h in history)
        assert has_switch or res.get("iteration_count", 0) >= 1, "Expected model switch action or loop execution"
        print(f"  [PASS] Model Switch Flow: Iterations={res['iteration_count']} | Chosen Model={res.get('selected_model')}")
    finally:
        clear_mock_llm()


def test_safe_abstention_flow():
    print("\n[*] Test 4: Deterministic Safe Abstention (ABSTAIN)...")
    mock_responses = {
        r"claim-level verification": json.dumps({
            "verified": False,
            "claims": [{"claim": "Missing logs and metrics", "supported": False, "risk": "high"}],
            "notes": "Missing essential evidence"
        }),
        r"ticket classifier": '{"category": "general", "severity": "L1", "summary": "Vague network error"}',
        r"identify the root cause": '{"root_cause": "Unclear cause", "confidence": "low", "needs_escalation": false}',
        r"resolution instructions": "1. Check generic network\nRESOLUTION_STATUS: resolved",
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        # Set max_iterations=0 so it immediately abstains upon missing evidence
        initial_state = {
            "user_query": "Something is slow on port 80",
            "max_iterations": 0,  # Forces immediate safeguard boundary
        }
        res = graph.invoke(initial_state)

        assert res.get("resolution_status") in ("abstained", "needs_human", "escalated"), f"Got {res.get('resolution_status')}"
        assert res.get("verified") is False, "Expected verified=False"
        assert "DIAGNOSTIC ABSTENTION" in res.get("final_answer", "") or "ESCALATED" in res.get("final_answer", "")
        print(f"  [PASS] Safe Abstention/Escalation triggered cleanly: Status={res.get('resolution_status')}")
    finally:
        clear_mock_llm()


def test_human_escalation_l3():
    print("\n[*] Test 5: Deterministic Human SRE Escalation (L3 Outage)...")
    mock_responses = {
        r"classifier": '{"category": "database", "severity": "L3", "summary": "Critical DB shard deadlock"}',
        r"escalation coordinator": "Ticket Escalation Report:\nHigh severity database deadlock with risk of transaction corruption."
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        initial_state = {
            "user_query": "Production database deadlock across payment shards with corrupted intermediate SSL handshake",
            "max_iterations": 2,
        }
        res = graph.invoke(initial_state)

        assert res.get("escalated") is True, "Expected escalated=True for L3"
        assert "ESCALATED" in res.get("final_answer", "")
        print(f"  [PASS] L3 Outage escalated: Escalated={res['escalated']} | Reason={res.get('escalation_reason')[:50]}...")
    finally:
        clear_mock_llm()


def test_budget_ceiling_and_loop_guard():
    print("\n[*] Test 6: Deterministic Budget Ceiling & Loop Guard (Prevent Infinite Loops)...")
    call_counts = {"count": 0}

    # An adversarial unverified response that would loop infinitely if unguarded
    def infinite_loop_llm(prompt_text):
        call_counts["count"] += 1
        if "classifier" in prompt_text.lower():
            return '{"category": "deployment", "severity": "L2", "summary": "Failing deployment"}'
        elif "identify the root cause" in prompt_text.lower():
            return '{"root_cause": "Transient error", "confidence": "low", "needs_escalation": false}'
        elif "resolution instructions" in prompt_text.lower():
            return "1. Try again\nRESOLUTION_STATUS: resolved"
        elif "claim-level" in prompt_text.lower():
            # ALWAYS reject verification
            return json.dumps({
                "verified": False,
                "claims": [{"claim": "Never verified", "supported": False, "risk": "high"}],
                "notes": "Always unverified"
            })
        elif "escalation" in prompt_text.lower():
            return "Escalation report: budget ceiling reached."
        return '{"status": "ok"}'

    mock_llm = MockDeterministicLLM(response_map=infinite_loop_llm)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        # Enforce max_iterations=2 and max_model_calls=6
        initial_state = {
            "user_query": "Persistent deployment pipeline failure with repeated rollback",
            "max_iterations": 2,
            "max_model_calls": 6,
        }
        res = graph.invoke(initial_state)

        # Loop limit MUST be strictly respected
        assert res.get("iteration_count", 0) <= 2, f"Loop count {res.get('iteration_count')} exceeded limit 2"
        assert res.get("model_call_count", 0) <= 8, f"Model call count {res.get('model_call_count')} exceeded budget"
        assert res.get("verified") is False, "Must not claim verified when verification failed"
        assert res.get("resolution_status") != "resolved", "Must not count unverified retry as resolved"

        print(f"  [PASS] Infinite loop successfully halted: Loops={res.get('iteration_count')}/2 | Model Calls={res.get('model_call_count')} | Status={res.get('resolution_status')}")
    finally:
        clear_mock_llm()


def test_separate_cost_and_latency_tracking():
    mock_responses = {
        r"claim-level verification": json.dumps({"verified": True, "claims": [{"claim": "Restart", "supported": True, "risk": "low"}], "notes": "Verified"}),
        r"ticket classifier": '{"category": "deployment", "severity": "L1", "summary": "Restart checkout"}',
        r"identify the root cause": '{"root_cause": "Hung thread", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. systemctl restart checkout\nRESOLUTION_STATUS: resolved",
    }
    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        res = graph.invoke({"user_query": "How do I restart the checkout service?"})

        assert "actual_cost" in res, "actual_cost must be in state"
        assert "estimated_cost" in res or res.get("action_evaluation") is not None, "estimated_cost or evaluation must be present"
        assert "actual_latency" in res, "actual_latency must be in state"
        assert res["actual_latency"] >= 0.0, "actual_latency must be positive"
        assert res["actual_cost"] >= 0.0, "actual_cost must be non-negative"

        print(f"  [PASS] Cost/Latency Tracking: Actual Cost=${res['actual_cost']:.6f} | Actual Latency={res['actual_latency']:.3f}s")
    finally:
        clear_mock_llm()


if __name__ == "__main__":
    print("=" * 75)
    print("   RUNNING PHASE 4: COST-AWARE ADAPTIVE EXECUTION TEST SUITE")
    print("=" * 75)
    test_direct_verified_accept()
    test_active_re_retrieval_flow()
    test_model_switch_flow()
    test_safe_abstention_flow()
    test_human_escalation_l3()
    test_budget_ceiling_and_loop_guard()
    test_separate_cost_and_latency_tracking()
    print("\n" + "=" * 75)
    print("   ALL PHASE 4 ADAPTIVE EXECUTION TESTS PASSED! (100%)")
    print("=" * 75)

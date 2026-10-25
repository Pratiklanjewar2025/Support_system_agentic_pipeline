# -*- coding: utf-8 -*-
"""
smoke_test_cache_e2e.py — End-to-End Smoke Test for AdaRAG-Ops Incident Response Cache.
Verifies:
 1. First eligible query runs through normal pipeline and saves verified answer.
 2. Submitting same query returns saved answer with 0 LLM calls ($0.00 cost).
 3. Materially different incident does NOT receive unsafe cached answer.
 4. Restart persistence verification (persisted SQLite survives process recreation).
"""

import os
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from agents.incident_response_cache import IncidentResponseCache
from agents.llm_utils import MockDeterministicLLM, set_mock_llm, clear_mock_llm
from agents.orchestrator import build_graph


def run_e2e_smoke_test():
    print("=" * 76)
    print("   ADARAG-OPS: END-TO-END INCIDENT RESPONSE CACHE SMOKE TEST")
    print("=" * 76)

    test_db = Path("artifacts") / "smoke_test" / "smoke_incident_cache.db"
    test_db.parent.mkdir(parents=True, exist_ok=True)
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass

    cache = IncidentResponseCache(db_path=test_db)
    cache.enabled = True
    cache.semantic_enabled = True
    cache.similarity_threshold = 0.85
    IncidentResponseCache._instance = cache

    graph = build_graph()

    # Step 1: First Query runs through normal pipeline and saves verified answer
    print("\n[*] Step 1: Submitting First Eligible Query (Cold Miss -> Pipeline Execution)...")
    query_1 = "How do I safely restart the checkout service?"
    mock_responses_q1 = {
        r"claim-level verification": json.dumps({
            "verified": True,
            "claims": [
                {"claim": "Checkout process hung", "supported": True, "risk": "low"},
                {"claim": "Restarting checkout service is safe", "supported": True, "risk": "low"}
            ],
            "notes": "Verified against deployment SOP"
        }),
        r"ticket classifier": '{"category": "deployment", "severity": "L1", "summary": "Restart checkout service"}',
        r"identify the root cause": '{"root_cause": "Checkout service process hung due to memory spike", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Run systemctl restart checkout\n2. Check status\nRESOLUTION_STATUS: resolved",
    }
    set_mock_llm(MockDeterministicLLM(response_map=mock_responses_q1))

    t0 = time.time()
    res1 = graph.invoke({"user_query": query_1})
    lat1 = time.time() - t0
    clear_mock_llm()

    assert res1.get("cache_hit") is False, "Expected cache_hit=False on cold query"
    assert res1.get("model_call_count", 0) >= 3, f"Expected pipeline model calls, got {res1.get('model_call_count')}"
    assert res1.get("verified") is True
    assert res1.get("resolution_status") == "resolved"
    print(f"  [OK] First Query Miss: Calls={res1['model_call_count']} | Latency={lat1:.3f}s | Verified={res1['verified']} | Resolution saved into cache.")

    # Step 2: Submit the exact same query again -> Must return saved answer with 0 LLM calls
    print("\n[*] Step 2: Submitting Exact Same Query (Cache Hit -> 0 LLM Calls)...")
    def failing_llm(prompt):
        raise AssertionError(f"LLM was called during expected cache hit! Prompt: {prompt[:60]}")
    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))

    t0 = time.time()
    res2 = graph.invoke({"user_query": query_1})
    lat2 = time.time() - t0
    clear_mock_llm()

    assert res2.get("cache_hit") is True, "Expected cache_hit=True on repeated query"
    assert res2.get("cache_match_type") == "exact"
    assert res2.get("model_call_count", 0) == 0, f"Expected 0 model calls, got {res2.get('model_call_count')}"
    assert res2.get("actual_cost", 0.0) == 0.0, f"Expected $0.00 actual cost, got {res2.get('actual_cost')}"
    assert "Checkout service process hung due to memory spike" in res2.get("root_cause", "")
    assert "⚡" in res2.get("final_answer", "")
    print(f"  [OK] Exact Cache HIT: Calls=0 | Cost=$0.00000 | Lookup Latency={res2.get('cache_lookup_latency', 0.0):.4f}s | Saved Answer Returned.")

    # Step 3: Submit materially different query -> Must NOT receive unsafe cached answer
    print("\n[*] Step 3: Submitting Materially Different Incident (Mismatched Error Code / Service)...")
    query_diff = "How do I safely restart the database postgres cluster with exit code 137?"
    mock_responses_diff = {
        r"claim-level verification": json.dumps({
            "verified": True,
            "claims": [{"claim": "Postgres OOM restart", "supported": True, "risk": "low"}],
            "notes": "Verified against postgres SOP"
        }),
        r"ticket classifier": '{"category": "database", "severity": "L1", "summary": "Restart postgres cluster"}',
        r"identify the root cause": '{"root_cause": "PostgreSQL pod terminated with exit code 137", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Run pg_ctlcluster 16 main restart\nRESOLUTION_STATUS: resolved",
    }
    set_mock_llm(MockDeterministicLLM(response_map=mock_responses_diff))

    t0 = time.time()
    res3 = graph.invoke({"user_query": query_diff})
    lat3 = time.time() - t0
    clear_mock_llm()

    assert res3.get("cache_hit") is False, "Materially different incident must NOT hit checkout cache!"
    assert "checkout" not in res3.get("root_cause", "").lower()
    print(f"  [OK] Context Guard Isolation: Diff query correctly missed cache (Calls={res3['model_call_count']}) and received its own diagnosis.")

    # Step 4: Simulate application reboot and verify persistent cache survival
    print("\n[*] Step 4: Simulating Process Restart & Persistence Verification...")
    del cache
    IncidentResponseCache._instance = None

    restarted_cache = IncidentResponseCache(db_path=test_db)
    restarted_cache.enabled = True
    IncidentResponseCache._instance = restarted_cache

    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))
    t0 = time.time()
    res4 = graph.invoke({"user_query": query_1})
    lat4 = time.time() - t0
    clear_mock_llm()

    assert res4.get("cache_hit") is True, "Failed to hit persistent cache after restart"
    assert res4.get("model_call_count", 0) == 0
    print(f"  [OK] Restart Persistence: Verified diagnosis survived process recreation (Hits={res4['cache_match_type']}, Calls=0).")

    # Step 5: Report Cache Telemetry
    m = restarted_cache.get_metrics()
    print("\n" + "=" * 76)
    print("   SMOKE TEST TELEMETRY REPORT")
    print("=" * 76)
    print(f"  Total Lookups          : {m['total_lookups']}")
    print(f"  Exact Cache Hits       : {m['exact_hits']}")
    print(f"  Semantic Cache Hits    : {m['semantic_hits']}")
    print(f"  Cache Misses           : {m['misses']}")
    print(f"  Hit Rate (%)           : {m['hit_rate_percent']}%")
    print(f"  Avoided Model Calls    : {m['avoided_model_calls']}")
    print(f"  Avoided LLM Cost ($)   : ${m['avoided_llm_cost_usd']:.5f}")
    print(f"  Average Lookup Latency : {m['avg_lookup_latency_sec']:.5f}s")
    print("=" * 76)
    print("   ALL END-TO-END SMOKE TEST CHECKS PASSED SUCCESSFULLY! (100%)")
    print("=" * 76)


if __name__ == "__main__":
    run_e2e_smoke_test()

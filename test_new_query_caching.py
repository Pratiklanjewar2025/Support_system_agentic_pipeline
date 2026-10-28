# -*- coding: utf-8 -*-
"""
test_new_query_caching.py — End-to-End Verification:
1. Verify that a brand new query (outside static runbooks) executes the pipeline,
   resolves, and is AUTOMATICALLY saved into the Incident Response Cache.
2. Verify that querying the same new query immediately achieves a Cache Hit with 0 LLM calls ($0.00).
3. Verify that IncidentResponseCache.store_query() enables direct manual/API storage.
"""

import sys
import json
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from agents.incident_response_cache import IncidentResponseCache
from agents.llm_utils import MockDeterministicLLM, set_mock_llm, clear_mock_llm
from agents.orchestrator import build_graph


def run_tests():
    print("=" * 75)
    print("  TESTING AUTOMATIC CACHING OF NEW INCOMING QUERIES")
    print("=" * 75)

    test_db = Path("artifacts") / "smoke_test" / "test_new_query_cache.db"
    test_db.parent.mkdir(parents=True, exist_ok=True)
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass

    cache = IncidentResponseCache(db_path=test_db)
    cache.enabled = True
    cache.semantic_enabled = True
    IncidentResponseCache._instance = cache

    # Query is a completely new issue NOT in documents/*.md
    new_query = "Nginx server returns 504 Gateway Timeout when proxying requests to nodejs backend."

    # Verify cache is initially a MISS
    initial_lookup = cache.lookup(new_query)
    assert initial_lookup.hit is False, f"Expected cache miss initially, got {initial_lookup.hit}"
    print(f"\n[1] Initial lookup for new query: HIT={initial_lookup.hit} (Confirmed Cold Cache)")

    # Setup realistic deterministic mock LLM for the pipeline
    mock_responses = {
        r"ticket classifier": '{"category": "network", "severity": "L2", "summary": "Nginx 504 Gateway Timeout proxying to nodejs backend"}',
        r"identify the root cause": '{"root_cause": "Node.js upstream backend event loop blocked or worker connections exhausted, causing proxy_read_timeout to expire in Nginx.", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Inspect Node.js event loop lag and heap usage.\n2. Increase proxy_read_timeout and proxy_connect_timeout in nginx.conf.\n3. Scale Node.js backend cluster replicas.\nRESOLUTION_STATUS: resolved",
        r"claim-level.*verification": json.dumps({
            "verified": True,
            "verification_status": "verified",
            "claims": [
                {"claim": "Nginx proxy_read_timeout expired", "supported": True, "risk": "low"},
                {"claim": "Scale Node.js backend cluster replicas", "supported": True, "risk": "low"}
            ],
            "notes": "Verified against network triage guidelines"
        }),
    }
    set_mock_llm(MockDeterministicLLM(response_map=mock_responses))

    graph = build_graph()

    print("\n[2] Executing pipeline for new incoming query...")
    initial_state = {
        "user_query": new_query,
        "category": "",
        "severity": "",
        "summary": "",
        "kb_results": [],
        "kb_context": "",
        "root_cause": "",
        "confidence": "",
        "resolution_steps": "",
        "resolution_status": "",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "final_answer": "",
        "cache_hit": False,
        "cache_match_type": "miss",
        "cache_similarity": 0.0,
        "cache_record_id": None,
        "cache_lookup_latency": 0.0,
        "start_time": time.time(),
    }

    final_state = graph.invoke(initial_state)
    clear_mock_llm()

    print(f"  -> Pipeline finished: verified={final_state.get('verified')} | status={final_state.get('resolution_status')} | escalated={final_state.get('escalated')}")
    print(f"  -> Cached Record ID in state: {final_state.get('cache_record_id')}")

    assert final_state.get("verified") is True, "Expected verified=True for safely confirmed diagnosis"
    assert final_state.get("escalated") is False, "Expected escalated=False"
    assert final_state.get("resolution_status") == "resolved", "Expected resolved"

    # [3] Verify that the new query is now persisted in cache.db
    post_lookup = cache.lookup(new_query)
    print(f"\n[3] Second lookup for same query: HIT={post_lookup.hit} | Type={post_lookup.match_type} | ID={post_lookup.record.cache_id if post_lookup.record else 'None'}")
    assert post_lookup.hit is True, "CRITICAL: Newly resolved query was NOT found in cache!"
    assert post_lookup.match_type == "exact"
    assert "event loop" in post_lookup.record.root_cause

    # [4] Verify that running the query again achieves 0 LLM calls
    def failing_llm(prompt):
        raise AssertionError(f"LLM was invoked on cache hit! Prompt: {prompt[:60]}")
    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))

    print("\n[4] Re-running query through graph to verify 0 LLM calls...")
    state_repeat = {**initial_state, "start_time": time.time()}
    second_run = graph.invoke(state_repeat)
    clear_mock_llm()

    assert second_run.get("cache_hit") is True, "Expected cache_hit=True"
    assert second_run.get("model_call_count", 0) == 0, f"Expected 0 calls, got {second_run.get('model_call_count')}"
    assert second_run.get("actual_cost", 0.0) == 0.0, f"Expected $0.00 cost, got {second_run.get('actual_cost')}"
    print(f"  -> Cache Hit confirmed: Calls={second_run.get('model_call_count')} | Cost=${second_run.get('actual_cost'):.5f} | Match={second_run.get('cache_match_type')}")

    # [5] Verify direct store_query API
    print("\n[5] Verifying direct cache.store_query() API...")
    manual_query = "How to clear dead docker containers safely?"
    rec_id = cache.store_query(
        query=manual_query,
        root_cause="Stopped containers accumulate on host root disk.",
        resolution_steps="Run docker container prune -f",
        category="deployment",
        severity="L1"
    )
    assert rec_id is not None, "Failed to store query directly via store_query()"
    direct_lookup = cache.lookup(manual_query)
    assert direct_lookup.hit is True, "Failed to find directly stored query"
    print(f"  -> Directly stored query successfully retrieved! Record ID: {rec_id}")

    print("\n" + "=" * 75)
    print("  ALL NEW QUERY CACHING TESTS PASSED (100%)!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()

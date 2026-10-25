# -*- coding: utf-8 -*-
"""
test_incident_response_cache.py — Deterministic Unit & Integration Test Suite
for AdaRAG-Ops Persistent Incident Response Cache.

Tests all 15 required architectural capabilities:
 1. Exact cache hit returns saved answer without invoking any LLM (0 calls, $0.00).
 2. Exact cache miss invokes the existing multi-agent pipeline.
 3. Verified new responses are saved and can be reused.
 4. Unverified/abstained/escalated responses are never cached.
 5. Safe semantic matches return cached answers without an LLM call.
 6. Similar but materially different incidents (service mismatch) are rejected by context guards.
 7. Different error codes (e.g., exit code 137 vs 1) are rejected by context guards.
 8. Expired TTL records are rejected.
 9. Manual invalidation prevents reuse of outdated runbooks.
10. Persistent records survive application / process restarts.
11. Database failures fall back safely without interrupting diagnosis.
12. Corrupt or malformed cache records are handled gracefully.
13. Concurrent requests do not corrupt SQLite state.
14. Cache metrics, avoided calls, and cost accounting are strictly tracked.
15. Full pipeline functions correctly when caching is disabled.
"""

import os
import sys
import time
import json
import sqlite3
import tempfile
import threading
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.state import AgentState
from agents.incident_response_cache import IncidentResponseCache, CacheRecord
from agents.llm_utils import MockDeterministicLLM, set_mock_llm, clear_mock_llm
from agents.orchestrator import build_graph


# =========================================================================
# Shared Mock LLM Responses for Pipeline Execution
# =========================================================================
STANDARD_MOCK_RESPONSES = {
    r"claim-level verification": json.dumps({
        "verified": True,
        "claims": [
            {"claim": "Restarting checkout service is safe", "supported": True, "risk": "low"},
            {"claim": "Checkout process hung due to memory spike", "supported": True, "risk": "low"}
        ],
        "notes": "Verified against deployment SOP"
    }),
    r"ticket classifier": '{"category": "deployment", "severity": "L1", "summary": "Restart checkout service"}',
    r"identify the root cause": '{"root_cause": "Checkout service process hung due to memory spike", "confidence": "high", "needs_escalation": false}',
    r"resolution instructions": "1. Run systemctl restart checkout\n2. Check status\nRESOLUTION_STATUS: resolved",
}


def create_isolated_cache(temp_dir: Path) -> IncidentResponseCache:
    """Creates an isolated IncidentResponseCache instance using a temporary SQLite database."""
    db_file = temp_dir / "test_incident_cache.db"
    cache = IncidentResponseCache(db_path=db_file)
    cache.enabled = True
    cache.semantic_enabled = True
    cache.similarity_threshold = 0.95
    cache.ttl_seconds = 86400.0
    return cache


# =========================================================================
# 1. Exact Cache Hit Returns Saved Answer (0 LLM Calls)
# =========================================================================
def test_1_exact_cache_hit_zero_llm(cache: IncidentResponseCache):
    print("\n[*] Test 1: Exact Cache Hit (Zero LLM Calls & Zero Cost)...")
    # Prime cache with a verified resolution
    dummy_state = {
        "user_query": "Postgres connection pool exhausted on port 5432",
        "verified": True,
        "resolution_status": "resolved",
        "escalated": False,
        "root_cause": "PostgreSQL max_connections saturated by leaked worker threads",
        "resolution_steps": "1. Kill idle connections.\n2. Scale connection pooler PgBouncer.",
        "verification_notes": "Passed verified runbook checks",
        "category": "database",
        "severity": "L2",
        "confidence": "high",
        "calibrated_confidence": 0.92,
        "selected_model": "test-mock-llm",
        "selected_provider": "groq",
        "kb_results": [{"source": "runbooks/postgres.md", "text": "Postgres pool configuration guide"}],
        "structured_diagnosis": {"primary_cause": "connection_exhaustion", "remediation": "pgbouncer"},
    }
    cache_id = cache.save_verified_response(dummy_state)
    assert cache_id is not None, "Failed to save initial verified record"

    # Set mock LLM that FAILS if called
    def failing_llm(prompt):
        raise AssertionError(f"LLM was invoked during a cache hit! Prompt: {prompt[:80]}...")
    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))

    try:
        # Patch singleton to use our isolated cache
        IncidentResponseCache._instance = cache

        graph = build_graph()
        query = "Postgres connection pool exhausted on port 5432"
        res = graph.invoke({"user_query": query})

        assert res.get("cache_hit") is True, f"Expected cache_hit=True, got {res.get('cache_hit')}"
        assert res.get("cache_match_type") == "exact", f"Expected exact match, got {res.get('cache_match_type')}"
        assert res.get("model_call_count", 0) == 0, f"Expected 0 model calls, got {res.get('model_call_count')}"
        assert res.get("actual_cost", 0.0) == 0.0, f"Expected $0.00 actual cost, got {res.get('actual_cost')}"
        assert "saturated by leaked worker threads" in res.get("root_cause", "")
        assert "RESOLVED" in res.get("final_answer", "")
        assert "⚡" in res.get("final_answer", "")
        print(f"  [PASS] Exact hit returned instantly: Calls={res['model_call_count']} | Cost=${res['actual_cost']} | Match={res['cache_match_type']}")
    finally:
        clear_mock_llm()


# =========================================================================
# 2. Exact Cache Miss Invokes Pipeline
# =========================================================================
def test_2_exact_cache_miss_invokes_pipeline(cache: IncidentResponseCache):
    print("\n[*] Test 2: Exact Cache Miss Invokes Normal Diagnostic Pipeline...")
    mock_llm = MockDeterministicLLM(response_map=STANDARD_MOCK_RESPONSES)
    set_mock_llm(mock_llm)

    try:
        IncidentResponseCache._instance = cache
        graph = build_graph()
        query = "How do I safely restart the checkout service?"
        res = graph.invoke({"user_query": query})

        assert res.get("cache_hit") is False, "Expected cache_hit=False for unseen query"
        assert res.get("cache_match_type") == "miss"
        assert res.get("model_call_count", 0) >= 3, f"Expected >= 3 model calls on miss, got {res.get('model_call_count')}"
        assert res.get("verified") is True, "Pipeline should reach verification"
        assert res.get("resolution_status") == "resolved", f"Expected resolved, got {res.get('resolution_status')}"
        assert res.get("escalated") is False
        print(f"  [PASS] Cache miss successfully executed full workflow: Calls={res['model_call_count']} | Cost=${res['actual_cost']:.5f}")
    finally:
        clear_mock_llm()


# =========================================================================
# 3. Verified New Response is Saved and Reused
# =========================================================================
def test_3_verified_response_saved_and_reused(cache: IncidentResponseCache):
    print("\n[*] Test 3: Verified New Response Automatically Persisted & Reused...")
    # The query from Test 2 should now be saved in the cache
    query = "How do I safely restart the checkout service?"

    # Verify that a subsequent lookup hits the cache without invoking any LLM
    def failing_llm(prompt):
        raise AssertionError("LLM should not be called on repeated query!")
    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))

    try:
        IncidentResponseCache._instance = cache
        graph = build_graph()
        res = graph.invoke({"user_query": query})

        assert res.get("cache_hit") is True, "Expected newly saved query to hit cache"
        assert res.get("cache_match_type") == "exact"
        assert res.get("model_call_count", 0) == 0
        assert "Checkout service process hung due to memory spike" in res.get("root_cause", "")
        print(f"  [PASS] Auto-persisted answer successfully reused with 0 model calls")
    finally:
        clear_mock_llm()



# =========================================================================
# 4. Unverified Responses are Not Cached
# =========================================================================
def test_4_unverified_responses_not_cached(cache: IncidentResponseCache):
    print("\n[*] Test 4: Unverified, Abstained, or Escalated Resolutions Rejected from Cache...")
    # Unverified state
    unverified_state = {
        "user_query": "Random unverified transient glitch",
        "verified": False,
        "resolution_status": "resolved",
        "root_cause": "Speculative guess",
        "escalated": False,
    }
    assert cache.save_verified_response(unverified_state) is None, "Should not cache unverified responses"

    # Abstained state
    abstained_state = {
        "user_query": "Incident with missing telemetry",
        "verified": True,
        "resolution_status": "abstained",
        "root_cause": "Insufficient telemetry",
        "escalated": False,
    }
    assert cache.save_verified_response(abstained_state) is None, "Should not cache abstained resolutions"

    # Escalated state
    escalated_state = {
        "user_query": "Critical datacenter fire",
        "verified": True,
        "resolution_status": "resolved",
        "root_cause": "Hardware failure",
        "escalated": True,
    }
    assert cache.save_verified_response(escalated_state) is None, "Should not cache escalated resolutions"

    print("  [PASS] All unverified, abstained, and escalated states correctly rejected from cache")


# =========================================================================
# 5. Safe Semantic Matches Return Cached Answers (0 LLM Calls)
# =========================================================================
def test_5_safe_semantic_matches(cache: IncidentResponseCache):
    print("\n[*] Test 5: Safe Semantic Matches via Sentence Embeddings (0 LLM Calls)...")
    # Save a canonical Kubernetes OOMKilled incident
    saved_state = {
        "user_query": "Kubernetes pod cartservice crash with OOMKilled exit code 137",
        "verified": True,
        "resolution_status": "resolved",
        "escalated": False,
        "root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "resolution_steps": "Increase memory request to 1Gi and limit to 2Gi in deployment.yaml",
        "verification_notes": "Verified against k8s resource runbook",
        "category": "infrastructure",
        "severity": "L2",
        "calibrated_confidence": 0.95,
        "selected_model": "groq/llama-3.3-70b",
        "selected_provider": "groq",
    }
    cache.save_verified_response(saved_state)

    # Semantically equivalent query with different wording
    similar_query = "cartservice pod was terminated with exit code 137 due to oomkilled"

    # Set threshold to 0.85 for natural language paraphrases
    orig_threshold = cache.similarity_threshold
    cache.similarity_threshold = 0.85

    # Zero LLM guard
    def failing_llm(prompt):
        raise AssertionError("LLM should not be called on safe semantic hit!")
    set_mock_llm(MockDeterministicLLM(response_map=failing_llm))

    try:
        IncidentResponseCache._instance = cache
        graph = build_graph()
        res = graph.invoke({"user_query": similar_query})

        assert res.get("cache_hit") is True, "Expected semantic cache hit"
        assert res.get("cache_match_type") == "semantic", f"Expected semantic match, got {res.get('cache_match_type')}"
        assert res.get("cache_similarity") >= cache.similarity_threshold
        assert res.get("model_call_count", 0) == 0, f"Expected 0 model calls, got {res.get('model_call_count')}"
        assert "Cartservice memory limit 512Mi exceeded" in res.get("root_cause", "")
        print(f"  [PASS] Safe semantic hit verified: Cosine={res.get('cache_similarity')} >= {cache.similarity_threshold} | Calls=0")
    finally:
        cache.similarity_threshold = orig_threshold
        clear_mock_llm()


# =========================================================================
# 6. Materially Different Incidents Rejected (Service Mismatch)
# =========================================================================
def test_6_different_services_rejected(cache: IncidentResponseCache):
    print("\n[*] Test 6: Semantic Context Guard Rejects Mismatched Services...")
    # Cache has cartservice OOMKilled. Now query redis with identical phrasing.
    # Embedding similarity is ~0.83 (very high), but service is redis vs cartservice!
    redis_query = "Kubernetes pod redis crash with OOMKilled exit code 137"

    orig_threshold = cache.similarity_threshold
    cache.similarity_threshold = 0.80

    try:
        lookup_res = cache.lookup(redis_query)
        assert not lookup_res.hit, f"Context guard should have rejected service mismatch, got hit={lookup_res.hit}"
        assert "rejected_by_context_guard" in lookup_res.notes or "affected_service_mismatch" in lookup_res.notes
        print(f"  [PASS] Context guard successfully rejected semantic match due to service mismatch: {lookup_res.notes}")
    finally:
        cache.similarity_threshold = orig_threshold


# =========================================================================
# 7. Different Error Codes Rejected by Context Guard
# =========================================================================
def test_7_different_error_codes_rejected(cache: IncidentResponseCache):
    print("\n[*] Test 7: Context Guard Rejects Conflicting Error Codes (Exit 137 vs Exit 1)...")
    # Embedding similarity is 0.9905 (>99%), but error signature is exit_code_1 vs exit_code_137!
    err_query = "Kubernetes pod cartservice crash with OOMKilled exit code 1"

    lookup_res = cache.lookup(err_query)
    assert not lookup_res.hit, "Context guard must reject conflicting error codes even at 99% similarity"
    assert "rejected_by_context_guard" in lookup_res.notes or "error_signature_mismatch" in lookup_res.notes
    print(f"  [PASS] Context guard prevented incorrect match for differing exit codes (hit={lookup_res.hit}, notes={lookup_res.notes})")



# =========================================================================
# 8. Expired TTL Records Rejected
# =========================================================================
def test_8_expired_records_rejected(cache: IncidentResponseCache):
    print("\n[*] Test 8: Expired TTL Records Rejected and Purged...")
    # Insert an expired record directly into DB
    now = time.time()
    past_time = now - 10000.0
    with cache._get_connection() as conn:
        conn.execute("""
            INSERT INTO incident_response_cache (
                cache_id, query, normalized_query, fingerprint, root_cause,
                resolution_steps, resolution_status, verification_status,
                created_at, last_accessed_at, expires_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            "CACHE_EXPIRED_001",
            "Expired incident query",
            "expired incident query",
            cache.compute_fingerprint("expired incident query"),
            "Old stale cause",
            "Old stale steps",
            "resolved",
            "verified",
            past_time,
            past_time,
            past_time,
            "1.0.0"
        ))
        conn.commit()

    res = cache.lookup("Expired incident query")
    assert res.hit is False, "Expired record must never be served"

    purged = cache.purge_expired()
    assert purged >= 1, f"Expected at least 1 purged record, got {purged}"
    print(f"  [PASS] Expired record rejected on lookup and {purged} expired entries cleanly purged")


# =========================================================================
# 9. Manual Invalidation Works
# =========================================================================
def test_9_manual_invalidation(cache: IncidentResponseCache):
    print("\n[*] Test 9: Manual Invalidation Prevents Serving Outdated Records...")
    saved_state = {
        "user_query": "Deprecate this runbook query immediately",
        "verified": True,
        "resolution_status": "resolved",
        "escalated": False,
        "root_cause": "Old runbook advice",
        "resolution_steps": "Steps to be invalidated",
    }
    rec_id = cache.save_verified_response(saved_state)
    assert rec_id is not None

    # Before invalidation -> hit
    assert cache.lookup("Deprecate this runbook query immediately").hit is True

    # Invalidate
    ok = cache.invalidate_record(rec_id, reason="Security deprecation")
    assert ok is True, "Failed to invalidate record"

    # After invalidation -> miss
    lookup_after = cache.lookup("Deprecate this runbook query immediately")
    assert lookup_after.hit is False, "Invalidated record must not be served"
    print("  [PASS] Record invalidated successfully; subsequent lookups return miss")


# =========================================================================
# 10. Persistent Records Survive Restart
# =========================================================================
def test_10_persistence_survives_restart(temp_dir: Path):
    print("\n[*] Test 10: Persistent Records Survive Application Restart...")
    db_file = temp_dir / "restart_test_cache.db"
    # Session 1: Create cache, write record
    cache_session_1 = IncidentResponseCache(db_path=db_file)
    cache_session_1.save_verified_response({
        "user_query": "Persistent incident after node power failure",
        "verified": True,
        "resolution_status": "resolved",
        "escalated": False,
        "root_cause": "Unclean filesystem shutdown",
        "resolution_steps": "Run fsck and reboot",
    })
    del cache_session_1

    # Session 2: Fresh instance simulating app reboot
    cache_session_2 = IncidentResponseCache(db_path=db_file)
    res = cache_session_2.lookup("Persistent incident after node power failure")
    assert res.hit is True, "Persisted record failed to survive application restart"
    assert "Unclean filesystem shutdown" in res.record.root_cause
    print("  [PASS] Records successfully persisted across process reboots")


# =========================================================================
# 11. Database Failure Fallback
# =========================================================================
def test_11_database_failure_fallback(temp_dir: Path):
    print("\n[*] Test 11: Database Failure Fallback Gracefully...")
    # Corrupt sqlite file format
    invalid_db = temp_dir / "corrupted_db.sqlite"
    invalid_db.write_text("NOT_A_VALID_SQLITE_DATABASE_HEADER_DATA_12345")

    bad_cache = IncidentResponseCache(db_path=invalid_db)
    res = bad_cache.lookup("Any test query")
    assert res.hit is False, "Lookup on corrupted DB should return miss fallback"
    assert bad_cache.get_metrics()["fallback_errors"] >= 1
    print("  [PASS] Database unavailability safely handled with fallback to normal pipeline")



# =========================================================================
# 12. Corrupt Cache Records Handled Gracefully
# =========================================================================
def test_12_corrupt_records_handled(cache: IncidentResponseCache):
    print("\n[*] Test 12: Corrupt Records Handled Gracefully...")
    # Insert corrupted JSON into supporting_evidence
    fp = cache.compute_fingerprint("corrupt record query")
    with cache._get_connection() as conn:
        conn.execute("""
            INSERT INTO incident_response_cache (
                cache_id, query, normalized_query, fingerprint, root_cause,
                resolution_steps, resolution_status, supporting_evidence,
                verification_status, created_at, last_accessed_at, expires_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            "CACHE_CORRUPT_001", "corrupt record query", "corrupt record query", fp,
            "Broken cause", "Broken steps", "resolved", "{{{invalid-json",
            "verified", time.time(), time.time(), time.time() + 1000, "1.0.0"
        ))
        conn.commit()

    res = cache.lookup("corrupt record query")
    assert res.hit is True, "Should recover despite non-critical JSON parse errors"
    assert res.record.supporting_evidence == [], "Corrupt JSON should fallback to empty list"
    print("  [PASS] Deserialization errors gracefully swallowed without crashing")


# =========================================================================
# 13. Concurrency Thread Safety
# =========================================================================
def test_13_concurrent_access(cache: IncidentResponseCache):
    print("\n[*] Test 13: Concurrency Thread Safety (10 Concurrent Readers & Writers)...")
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(10):
                q = f"Concurrent query from worker {worker_id} iteration {i}"
                cache.save_verified_response({
                    "user_query": q,
                    "verified": True,
                    "resolution_status": "resolved",
                    "escalated": False,
                    "root_cause": f"Cause {worker_id}-{i}",
                    "resolution_steps": f"Steps {worker_id}-{i}",
                })
                res = cache.lookup(q)
                assert res.hit is True
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent operations encountered errors: {errors}"
    print("  [PASS] 100 concurrent reads/writes completed with 0 race conditions or locks")


# =========================================================================
# 14. Cache Metrics & Cost Accounting
# =========================================================================
def test_14_metrics_and_accounting(cache: IncidentResponseCache):
    print("\n[*] Test 14: Cache Metrics & Avoided Cost Accounting...")
    m = cache.get_metrics()
    assert m["total_lookups"] > 0
    assert m["exact_hits"] >= 1
    assert m["avoided_model_calls"] >= 1
    assert m["avoided_llm_cost_usd"] > 0.0
    assert m["avg_lookup_latency_sec"] >= 0.0
    print(f"  [PASS] Telemetry: Lookups={m['total_lookups']} | Hits={m['exact_hits']}+{m['semantic_hits']} | Avoided Cost=${m['avoided_llm_cost_usd']:.5f}")


# =========================================================================
# 15. Pipeline Behavior when Cache Disabled
# =========================================================================
def test_15_caching_disabled(cache: IncidentResponseCache):
    print("\n[*] Test 15: Pipeline Behavior when Caching Disabled...")
    cache.enabled = False
    mock_llm = MockDeterministicLLM(response_map=STANDARD_MOCK_RESPONSES)
    set_mock_llm(mock_llm)

    try:
        IncidentResponseCache._instance = cache
        graph = build_graph()
        res = graph.invoke({"user_query": "Postgres connection pool exhausted on port 5432"})

        assert res.get("cache_hit") is False, "Cache should not hit when disabled"
        assert res.get("model_call_count", 0) >= 3, "Full LLM pipeline must run when cache disabled"
        print(f"  [PASS] Pipeline seamlessly bypassed cache when disabled: Calls={res['model_call_count']}")
    finally:
        cache.enabled = True
        clear_mock_llm()


# =========================================================================
# Main Test Runner
# =========================================================================
def run_all_cache_tests():
    print("=" * 76)
    print("   RUNNING ADARAG-OPS INCIDENT RESPONSE CACHE TEST SUITE")
    print("=" * 76)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        cache = create_isolated_cache(tmp_path)

        test_1_exact_cache_hit_zero_llm(cache)
        test_2_exact_cache_miss_invokes_pipeline(cache)
        test_3_verified_response_saved_and_reused(cache)
        test_4_unverified_responses_not_cached(cache)
        test_5_safe_semantic_matches(cache)
        test_6_different_services_rejected(cache)
        test_7_different_error_codes_rejected(cache)
        test_8_expired_records_rejected(cache)
        test_9_manual_invalidation(cache)
        test_10_persistence_survives_restart(tmp_path)
        test_11_database_failure_fallback(tmp_path)
        test_12_corrupt_records_handled(cache)
        test_13_concurrent_access(cache)
        test_14_metrics_and_accounting(cache)
        test_15_caching_disabled(cache)

    print("\n" + "=" * 76)
    print("   ALL 15 INCIDENT RESPONSE CACHE TESTS PASSED! (100%)")
    print("=" * 76)


if __name__ == "__main__":
    run_all_cache_tests()

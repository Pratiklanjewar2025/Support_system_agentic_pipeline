# -*- coding: utf-8 -*-
"""
benchmark_incident_cache.py — Benchmark Optimization Study for AdaRAG-Ops Incident Response Cache
Part of AdaRAG-Ops: Adaptive Multi-LLM RAG for Operational Incident Diagnosis.

Evaluates and compares:
  Configuration A: Baseline Pipeline (No Response Cache)
  Configuration B: Exact Caching Enabled
  Configuration C: Exact + Semantic Caching Enabled (with Context Guards)

Metrics Reported:
  - Exact & Semantic Cache Hit Rates (%)
  - Cache Precision (% of served cache responses that are valid)
  - Incorrect Cache Response Rate (%)
  - Root-Cause Diagnostic Accuracy (%)
  - Median (p50) & p95 Latency (seconds)
  - Mean LLM Calls per Request
  - Estimated LLM Generation Cost per Request ($)
  - Cache Overhead Latency (milliseconds)
  - Overall Cost & Latency Savings (%) vs Baseline
  - Evaluation of Semantic Matching on Labeled Equivalent vs Non-Equivalent Incident Pairs
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from agents.incident_response_cache import IncidentResponseCache
from agents.llm_utils import MockDeterministicLLM, set_mock_llm, clear_mock_llm
from agents.orchestrator import build_graph


# =========================================================================
# Benchmark Workload: 20-Query Incident Stream
# =========================================================================
BENCHMARK_INCIDENT_STREAM = [
    # Wave 1: Initial Discovery / Cold Encounters (Seed Records)
    {
        "id": "INC_01",
        "query": "Kubernetes pod cartservice crash with OOMKilled exit code 137",
        "expected_root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "expected_service": "cartservice",
        "category": "infrastructure",
        "severity": "L2",
        "mock_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "mock_steps": "Increase memory request to 1Gi and limit to 2Gi in deployment.yaml",
    },
    {
        "id": "INC_02",
        "query": "Postgres connection pool exhausted on port 5432 with active client starvation",
        "expected_root_cause": "PostgreSQL max_connections limit reached under sudden traffic spike",
        "expected_service": "postgres",
        "category": "database",
        "severity": "L1",
        "mock_cause": "PostgreSQL max_connections limit reached under sudden traffic spike",
        "mock_steps": "Scale PgBouncer pooler and kill idle connections with pg_terminate_backend()",
    },
    {
        "id": "INC_03",
        "query": "Nginx API gateway returning 502 Bad Gateway upstream connection refused",
        "expected_root_cause": "Upstream backend container terminated or listening socket closed",
        "expected_service": "nginx",
        "category": "network",
        "severity": "L1",
        "mock_cause": "Upstream backend container terminated or listening socket closed",
        "mock_steps": "Restart upstream service and verify healthy endpoint in nginx upstream block",
    },
    {
        "id": "INC_04",
        "query": "Kafka consumer group rebalance storm with prolonged heartbeat timeouts",
        "expected_root_cause": "Heavy batch processing blocking poll loop exceeding max.poll.interval.ms",
        "expected_service": "kafka",
        "category": "streaming",
        "severity": "L1",
        "mock_cause": "Heavy batch processing blocking poll loop exceeding max.poll.interval.ms",
        "mock_steps": "Increase max.poll.interval.ms or decrease max.poll.records",
    },

    # Wave 2: Exact Repeat Inquiries (Identical Recurring Alerts)
    {
        "id": "INC_05",
        "query": "Kubernetes pod cartservice crash with OOMKilled exit code 137",
        "expected_root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "expected_service": "cartservice",
        "type": "exact_repeat",
    },
    {
        "id": "INC_06",
        "query": "Postgres connection pool exhausted on port 5432 with active client starvation",
        "expected_root_cause": "PostgreSQL max_connections limit reached under sudden traffic spike",
        "expected_service": "postgres",
        "type": "exact_repeat",
    },
    {
        "id": "INC_07",
        "query": "Nginx API gateway returning 502 Bad Gateway upstream connection refused",
        "expected_root_cause": "Upstream backend container terminated or listening socket closed",
        "expected_service": "nginx",
        "type": "exact_repeat",
    },

    # Wave 3: Semantically Equivalent Paraphrases (Same Service & Root Cause)
    {
        "id": "INC_08",
        "query": "cartservice pod was terminated with exit code 137 due to oomkilled",
        "expected_root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "expected_service": "cartservice",
        "type": "semantic_equivalent",
    },
    {
        "id": "INC_09",
        "query": "Postgres port 5432 pool exhausted causing connection starvation on clients",
        "expected_root_cause": "PostgreSQL max_connections limit reached under sudden traffic spike",
        "expected_service": "postgres",
        "type": "semantic_equivalent",
    },
    {
        "id": "INC_10",
        "query": "API gateway nginx throwing 502 bad gateway upstream connection refused error",
        "expected_root_cause": "Upstream backend container terminated or listening socket closed",
        "expected_service": "nginx",
        "type": "semantic_equivalent",
    },

    # Wave 4: Materially Different Incidents with Similar Phrasing (Testing False-Positive Resistance)
    {
        "id": "INC_11",
        "query": "Kubernetes pod redis crash with OOMKilled exit code 137",  # Different service: redis != cartservice
        "expected_root_cause": "Redis maxmemory policy eviction failure",
        "expected_service": "redis",
        "category": "database",
        "severity": "L1",
        "mock_cause": "Redis maxmemory policy eviction failure",
        "mock_steps": "Configure volatile-lru eviction policy and increase Redis container memory",
        "type": "near_miss_service_guard",
    },
    {
        "id": "INC_12",
        "query": "Kubernetes pod cartservice crash with OOMKilled exit code 1",  # Different error: code 1 != 137
        "expected_root_cause": "Application unhandled null pointer exception at startup",
        "expected_service": "cartservice",
        "category": "application",
        "severity": "L1",
        "mock_cause": "Application unhandled null pointer exception at startup",
        "mock_steps": "Inspect application stacktrace and fix missing environment variable",
        "type": "near_miss_code_guard",
    },
    {
        "id": "INC_13",
        "query": "Postgres database deadlock detected on concurrent row updates",  # Deadlock != pool exhaustion
        "expected_root_cause": "Unordered multi-row update transaction locks",
        "expected_service": "postgres",
        "category": "database",
        "severity": "L1",
        "mock_cause": "Unordered multi-row update transaction locks",
        "mock_steps": "Enforce deterministic lock acquisition order in SQL transactions",
        "type": "near_miss_different_symptom",
    },

    # Wave 5: Subsequent Inquiries & Additional Repeats
    {
        "id": "INC_14",
        "query": "Kubernetes pod cartservice crash with OOMKilled exit code 137",
        "expected_root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "expected_service": "cartservice",
        "type": "exact_repeat",
    },
    {
        "id": "INC_15",
        "query": "cartservice pod was terminated with exit code 137 due to oomkilled",
        "expected_root_cause": "Cartservice memory limit 512Mi exceeded during checkout load spike",
        "expected_service": "cartservice",
        "type": "semantic_repeat",
    },
    {
        "id": "INC_16",
        "query": "Kubernetes pod redis crash with OOMKilled exit code 137",
        "expected_root_cause": "Redis maxmemory policy eviction failure",
        "expected_service": "redis",
        "type": "exact_repeat",
    },
    {
        "id": "INC_17",
        "query": "Nginx API gateway returning 502 Bad Gateway upstream connection refused",
        "expected_root_cause": "Upstream backend container terminated or listening socket closed",
        "expected_service": "nginx",
        "type": "exact_repeat",
    },
    {
        "id": "INC_18",
        "query": "Kafka consumer group rebalance storm with prolonged heartbeat timeouts",
        "expected_root_cause": "Heavy batch processing blocking poll loop exceeding max.poll.interval.ms",
        "expected_service": "kafka",
        "type": "exact_repeat",
    },
    {
        "id": "INC_19",
        "query": "Unseen anomaly: RabbitMQ queue memory alarm blocking socket connections",
        "expected_root_cause": "RabbitMQ vm_memory_high_watermark exceeded by unconsumed dead letters",
        "expected_service": "rabbitmq",
        "category": "streaming",
        "severity": "L1",
        "mock_cause": "RabbitMQ vm_memory_high_watermark exceeded by unconsumed dead letters",
        "mock_steps": "Purge unroutable queues and raise vm_memory_high_watermark limit",
        "type": "novel_incident",
    },
    {
        "id": "INC_20",
        "query": "Unseen anomaly: RabbitMQ queue memory alarm blocking socket connections",
        "expected_root_cause": "RabbitMQ vm_memory_high_watermark exceeded by unconsumed dead letters",
        "expected_service": "rabbitmq",
        "type": "exact_repeat",
    },
]


# =========================================================================
# Labeled Validation Pairs for Similarity Threshold Tuning
# =========================================================================
LABELED_INCIDENT_PAIRS = [
    # True Equivalents (Same Incident & Resolution)
    {"q1": "Kubernetes pod cartservice crash with OOMKilled exit code 137", "q2": "cartservice pod was terminated with exit code 137 due to oomkilled", "equivalent": True},
    {"q1": "Postgres connection pool exhausted on port 5432 with active client starvation", "q2": "Postgres port 5432 pool exhausted causing connection starvation on clients", "equivalent": True},
    {"q1": "Nginx API gateway returning 502 Bad Gateway upstream connection refused", "q2": "API gateway nginx throwing 502 bad gateway upstream connection refused error", "equivalent": True},
    {"q1": "Redis cache node out of memory memory limit reached", "q2": "Redis server memory exhausted OOM killed by kernel", "equivalent": True},

    # Non-Equivalents (Different Services or Error Signatures)
    {"q1": "Kubernetes pod cartservice crash with OOMKilled exit code 137", "q2": "Kubernetes pod redis crash with OOMKilled exit code 137", "equivalent": False, "distinction": "different_service"},
    {"q1": "Kubernetes pod cartservice crash with OOMKilled exit code 137", "q2": "Kubernetes pod cartservice crash with OOMKilled exit code 1", "equivalent": False, "distinction": "different_exit_code"},
    {"q1": "Postgres connection pool exhausted on port 5432", "q2": "Postgres database deadlock detected on concurrent row updates", "equivalent": False, "distinction": "different_symptom"},
    {"q1": "Nginx returning 502 Bad Gateway upstream connection refused", "q2": "Nginx returning 504 Gateway Timeout upstream request timed out", "equivalent": False, "distinction": "different_status_code"},
]


def make_mock_responder(item: Dict[str, Any]):
    """Creates a deterministic mock LLM response map for an incident scenario."""
    cause = item.get("mock_cause", item.get("expected_root_cause", "Generic cause"))
    steps = item.get("mock_steps", "1. Apply documented remediation SOP\n2. Verify health")
    cat = item.get("category", "deployment")
    sev = item.get("severity", "L1")

    return {
        r"claim-level verification": json.dumps({
            "verified": True,
            "claims": [
                {"claim": f"Diagnosis: {cause[:50]}", "supported": True, "risk": "low"},
                {"claim": "Remediation verified safe", "supported": True, "risk": "low"}
            ],
            "notes": "Verified against documented runbook SOP"
        }),
        r"ticket classifier": json.dumps({"category": cat, "severity": sev, "summary": item["query"][:60]}),
        r"identify the root cause": json.dumps({"root_cause": cause, "confidence": "high", "needs_escalation": False}),
        r"resolution instructions": f"{steps}\nRESOLUTION_STATUS: resolved",
    }


def evaluate_labeled_pairs():
    """Evaluates embedding similarity and context guard precision on labeled incident pairs."""
    from rag.retriever import _get_model
    model = _get_model()

    print("\n" + "=" * 76)
    print("   LABELED INCIDENT PAIR EVALUATION & THRESHOLD ANALYSIS")
    print("=" * 76)
    print(f" {'Pair Type':<16} | {'Similarity':<10} | {'Context Guard':<15} | {'Ground Truth':<12} | {'Decision':<10}")
    print("-" * 76)

    threshold = 0.85
    correct_decisions = 0

    for pair in LABELED_INCIDENT_PAIRS:
        v1 = model.encode(pair["q1"], convert_to_numpy=True).astype(np.float32)
        v2 = model.encode(pair["q2"], convert_to_numpy=True).astype(np.float32)
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        sig1 = IncidentResponseCache.extract_context_signals(pair["q1"])
        sig2 = IncidentResponseCache.extract_context_signals(pair["q2"])

        # Context guard validation
        guard_pass = True
        guard_reason = "PASS"
        if sig1["error_signature"] and sig2["error_signature"] and sig1["error_signature"] != sig2["error_signature"]:
            guard_pass = False
            guard_reason = f"DIFF_ERR({sig1['error_signature']})"
        elif sig1["affected_service"] != "unknown" and sig2["affected_service"] != "unknown" and sig1["affected_service"] != sig2["affected_service"]:
            guard_pass = False
            guard_reason = f"DIFF_SRV({sig1['affected_service']})"

        match_accepted = (sim >= threshold) and guard_pass
        is_correct = (match_accepted == pair["equivalent"])
        if is_correct:
            correct_decisions += 1

        pair_lbl = "EQUIVALENT" if pair["equivalent"] else "NON-EQUIV"
        gt_lbl = "MATCH" if pair["equivalent"] else "REJECT"
        dec_lbl = "[OK] MATCH" if match_accepted else "[OK] REJECT"
        if not is_correct:
            dec_lbl = "[FAIL] " + ("FALSE_POS" if match_accepted else "FALSE_NEG")

        print(f" {pair_lbl:<16} | {sim:<10.4f} | {guard_reason:<15} | {gt_lbl:<12} | {dec_lbl:<10}")

    acc = (correct_decisions / len(LABELED_INCIDENT_PAIRS)) * 100
    print("-" * 76)
    print(f" Context-Guarded Semantic Decision Accuracy: {acc:.1f}% ({correct_decisions}/{len(LABELED_INCIDENT_PAIRS)})")
    print("=" * 76)


def run_configuration_benchmark(config_name: str, cache_enabled: bool, semantic_enabled: bool, db_path: Path) -> Dict[str, Any]:
    """Runs the 20-incident workload through a designated cache configuration."""
    print(f"\n[*] Running Benchmark for: {config_name} ...")

    # Clean isolated cache DB
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    cache = IncidentResponseCache(db_path=db_path)
    cache.enabled = cache_enabled
    cache.semantic_enabled = semantic_enabled
    cache.similarity_threshold = 0.85  # Calibrated for natural sentence variations with context guards
    IncidentResponseCache._instance = cache

    graph = build_graph()

    latencies = []
    llm_calls_list = []
    llm_costs_list = []
    exact_hits = 0
    semantic_hits = 0
    misses = 0
    correct_diagnoses = 0
    cache_hits_valid = 0
    incorrect_cache_responses = 0

    lookup_overheads = []

    for item in BENCHMARK_INCIDENT_STREAM:
        query = item["query"]
        expected_cause = item["expected_root_cause"]

        # Configure deterministic mock response for this query if it misses
        mock_map = make_mock_responder(item)
        set_mock_llm(MockDeterministicLLM(response_map=mock_map))

        t0 = time.time()
        res = graph.invoke({"user_query": query})
        elapsed = time.time() - t0

        latencies.append(elapsed)
        lookup_overheads.append(res.get("cache_lookup_latency", 0.0))
        calls = res.get("model_call_count", 0)
        cost = res.get("actual_cost", 0.0)

        llm_calls_list.append(calls)
        llm_costs_list.append(cost)

        is_hit = res.get("cache_hit", False)
        match_type = res.get("cache_match_type", "miss")

        if is_hit:
            if match_type == "exact":
                exact_hits += 1
            elif match_type == "semantic":
                semantic_hits += 1

            # Validate cache precision: does returned cause match ground truth?
            cause_returned = res.get("root_cause", "")
            if expected_cause.lower() in cause_returned.lower() or cause_returned.lower() in expected_cause.lower():
                correct_diagnoses += 1
                cache_hits_valid += 1
            else:
                incorrect_cache_responses += 1
        else:
            misses += 1
            cause_returned = res.get("root_cause", "")
            if expected_cause.lower() in cause_returned.lower() or cause_returned.lower() in expected_cause.lower():
                correct_diagnoses += 1

        clear_mock_llm()

    total_reqs = len(BENCHMARK_INCIDENT_STREAM)
    total_hits = exact_hits + semantic_hits
    hit_rate = (total_hits / total_reqs) * 100.0
    cache_precision = (cache_hits_valid / total_hits * 100.0) if total_hits > 0 else 100.0
    incorrect_cache_rate = (incorrect_cache_responses / total_hits * 100.0) if total_hits > 0 else 0.0

    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    mean_calls = float(np.mean(llm_calls_list))
    total_cost = float(np.sum(llm_costs_list))
    cost_per_req = total_cost / total_reqs
    avg_overhead_ms = float(np.mean(lookup_overheads)) * 1000.0

    return {
        "configuration": config_name,
        "total_requests": total_reqs,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "misses": misses,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_precision_percent": round(cache_precision, 2),
        "incorrect_cache_rate_percent": round(incorrect_cache_rate, 2),
        "overall_diagnostic_accuracy_percent": round((correct_diagnoses / total_reqs) * 100.0, 2),
        "median_p50_latency_sec": round(p50_lat, 4),
        "p95_latency_sec": round(p95_lat, 4),
        "mean_llm_calls_per_req": round(mean_calls, 2),
        "total_llm_cost_usd": round(total_cost, 6),
        "cost_per_request_usd": round(cost_per_req, 6),
        "avg_cache_lookup_overhead_ms": round(avg_overhead_ms, 3),
    }


def main():
    print("=" * 76)
    print("   ADARAG-OPS: INCIDENT RESPONSE CACHE BENCHMARK OPTIMIZATION STUDY")
    print("=" * 76)

    # 1. Evaluate semantic pair discrimination
    evaluate_labeled_pairs()

    # Isolated benchmark databases
    bench_dir = Path("artifacts") / "benchmark_cache"
    bench_dir.mkdir(parents=True, exist_ok=True)

    db_a = bench_dir / "bench_no_cache.db"
    db_b = bench_dir / "bench_exact_cache.db"
    db_c = bench_dir / "bench_exact_semantic_cache.db"

    # Run Benchmark Across the 3 Configurations
    res_a = run_configuration_benchmark("Configuration A: No Cache (Baseline)", cache_enabled=False, semantic_enabled=False, db_path=db_a)
    res_b = run_configuration_benchmark("Configuration B: Exact Cache Only", cache_enabled=True, semantic_enabled=False, db_path=db_b)
    res_c = run_configuration_benchmark("Configuration C: Exact + Semantic Cache", cache_enabled=True, semantic_enabled=True, db_path=db_c)

    # Calculate Savings vs Baseline A
    baseline_cost = res_a["total_llm_cost_usd"]
    baseline_lat = res_a["median_p50_latency_sec"]
    baseline_calls = res_a["mean_llm_calls_per_req"]

    for r in [res_b, res_c]:
        cost_saved = ((baseline_cost - r["total_llm_cost_usd"]) / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
        lat_saved = ((baseline_lat - r["median_p50_latency_sec"]) / baseline_lat) * 100.0 if baseline_lat > 0 else 0.0
        calls_saved = ((baseline_calls - r["mean_llm_calls_per_req"]) / baseline_calls) * 100.0 if baseline_calls > 0 else 0.0
        r["cost_reduction_percent"] = round(cost_saved, 2)
        r["latency_reduction_percent"] = round(lat_saved, 2)
        r["llm_call_reduction_percent"] = round(calls_saved, 2)

    res_a["cost_reduction_percent"] = 0.0
    res_a["latency_reduction_percent"] = 0.0
    res_a["llm_call_reduction_percent"] = 0.0

    # Print Comparative Results Table
    print("\n" + "=" * 92)
    print("   INCIDENT RESPONSE CACHE BENCHMARK COMPARATIVE RESULTS")
    print("=" * 92)
    header = f"{'Metric':<34} | {'A: No Cache':<16} | {'B: Exact Cache':<16} | {'C: Exact+Semantic':<18}"
    print(header)
    print("-" * 92)

    rows = [
        ("Total Requests Evaluated", f"{res_a['total_requests']}", f"{res_b['total_requests']}", f"{res_c['total_requests']}"),
        ("Exact Cache Hits", f"{res_a['exact_hits']}", f"{res_b['exact_hits']}", f"{res_c['exact_hits']}"),
        ("Semantic Cache Hits", f"{res_a['semantic_hits']}", f"{res_b['semantic_hits']}", f"{res_c['semantic_hits']}"),
        ("Total Cache Hit Rate (%)", f"{res_a['hit_rate_percent']}%", f"{res_b['hit_rate_percent']}%", f"{res_c['hit_rate_percent']}%"),
        ("Cache Precision (Served Valid)", "N/A", f"{res_b['cache_precision_percent']}%", f"{res_c['cache_precision_percent']}%"),
        ("Incorrect Cache Response Rate", "N/A", f"{res_b['incorrect_cache_rate_percent']}%", f"{res_c['incorrect_cache_rate_percent']}%"),
        ("Diagnostic Accuracy (%)", f"{res_a['overall_diagnostic_accuracy_percent']}%", f"{res_b['overall_diagnostic_accuracy_percent']}%", f"{res_c['overall_diagnostic_accuracy_percent']}%"),
        ("Median Latency p50 (sec)", f"{res_a['median_p50_latency_sec']}s", f"{res_b['median_p50_latency_sec']}s", f"{res_c['median_p50_latency_sec']}s"),
        ("p95 Latency (sec)", f"{res_a['p95_latency_sec']}s", f"{res_b['p95_latency_sec']}s", f"{res_c['p95_latency_sec']}s"),
        ("Mean LLM Calls / Request", f"{res_a['mean_llm_calls_per_req']:.2f}", f"{res_b['mean_llm_calls_per_req']:.2f}", f"{res_c['mean_llm_calls_per_req']:.2f}"),
        ("Total LLM Cost ($)", f"${res_a['total_llm_cost_usd']:.5f}", f"${res_b['total_llm_cost_usd']:.5f}", f"${res_c['total_llm_cost_usd']:.5f}"),
        ("Cost per Request ($)", f"${res_a['cost_per_request_usd']:.6f}", f"${res_b['cost_per_request_usd']:.6f}", f"${res_c['cost_per_request_usd']:.6f}"),
        ("Lookup Overhead (ms)", "0.00 ms", f"{res_b['avg_cache_lookup_overhead_ms']:.2f} ms", f"{res_c['avg_cache_lookup_overhead_ms']:.2f} ms"),
        ("LLM Call Reduction (%)", "0.0%", f"-{res_b['llm_call_reduction_percent']}%", f"-{res_c['llm_call_reduction_percent']}%"),
        ("Cost Savings vs Baseline (%)", "0.0%", f"+{res_b['cost_reduction_percent']}%", f"+{res_c['cost_reduction_percent']}%"),
        ("Latency Reduction vs Baseline", "0.0%", f"+{res_b['latency_reduction_percent']}%", f"+{res_c['latency_reduction_percent']}%"),
    ]

    for label, val_a, val_b, val_c in rows:
        print(f" {label:<33} | {val_a:<16} | {val_b:<16} | {val_c:<18}")

    print("=" * 92)

    # Save to JSON
    output_path = Path("benchmark_cache_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "configurations": {
                "config_a_no_cache": res_a,
                "config_b_exact_cache": res_b,
                "config_c_exact_semantic_cache": res_c,
            }
        }, f, indent=2)

    print(f"\n[OK] Benchmark output saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()

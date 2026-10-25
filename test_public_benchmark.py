# -*- coding: utf-8 -*-
"""
test_public_benchmark.py — Comprehensive Unit & Integration Test Suite
for Phase 6: Public Incident Benchmark Integration (RCAEval & OpenRCA).

Verifies:
1. RCAEvalAdapter: Proper loading, mapping, and preservation of original IDs and labels.
2. OpenRCAAdapter: Enterprise system mapping (telecom, bank, market) and telemetry preservation.
3. SupplementaryAdapter: Retention of existing Database, Kubernetes, and TLS baseline cases.
4. Grouped Splitting: Zero data-leakage across train/val/test splits.
5. Cross-System Generalization Split: Unseen microservice system testing (e.g. Train Ticket).
6. Cross-Fault Generalization Split: Unseen failure mode testing (resource vs network/code).
7. End-to-End Benchmark Execution: Metrics computation and result aggregation.
"""

import sys
import os
import json
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmarks.dataset_adapter import (
    BenchmarkIncident,
    RCAEvalAdapter,
    OpenRCAAdapter,
    SupplementaryAdapter,
    IncidentDatasetSplitter,
    IncidentDatasetRegistry
)
from benchmarks.runner import (
    compute_benchmark_metrics,
    run_benchmark_suite
)
from agents.llm_utils import (
    MockDeterministicLLM,
    set_mock_llm,
    clear_mock_llm
)


def test_rcaeval_adapter():
    print("[*] Test 1: Testing RCAEvalAdapter Loading & Original ID Preservation...")
    adapter = RCAEvalAdapter()
    incidents = adapter.load()

    assert len(incidents) >= 10, f"Expected at least 10 RCAEval incidents, got {len(incidents)}"

    # Check ID preservation and ground-truth retention
    first = incidents[0]
    assert first.id == "RE1_frontend_CPU_1", f"Expected original ID 'RE1_frontend_CPU_1', got {first.id}"
    assert first.source_dataset == "RCAEval"
    assert first.system_name == "online-boutique"
    assert first.fault_type == "CPU"
    assert first.root_cause_service == "frontend"
    assert first.category == "cpu"
    assert first.has_ground_truth is True
    assert "container_cpu_usage" in first.root_cause_indicator
    assert len(first.raw_record) > 0, "Raw record must be preserved"

    # Check multi-system coverage
    systems = set(inc.system_name for inc in incidents)
    assert "online-boutique" in systems, "online-boutique must be present"
    assert "sock-shop" in systems, "sock-shop must be present"
    assert "train-ticket" in systems, "train-ticket must be present"

    # Check code fault mapping in RE3
    re3_cases = [inc for inc in incidents if inc.benchmark_suite == "RE3"]
    assert len(re3_cases) >= 3, "Expected RE3 code-level failure cases"
    assert any(c.fault_type.startswith("F") for c in re3_cases), "F-code faults must be preserved"

    print(f"  [PASS] RCAEvalAdapter loaded {len(incidents)} cases spanning {list(systems)} with original IDs preserved.")


def test_openrca_adapter():
    print("[*] Test 2: Testing OpenRCAAdapter Enterprise Microservice Mapping...")
    adapter = OpenRCAAdapter()
    incidents = adapter.load()

    assert len(incidents) >= 5, f"Expected at least 5 OpenRCA incidents, got {len(incidents)}"

    first = incidents[0]
    assert first.id == "OpenRCA_telecom_001"
    assert first.source_dataset == "OpenRCA"
    assert first.system_name == "telecom"
    assert first.fault_type == "connection_pool_timeout"
    assert first.root_cause_service == "hss-service"
    assert first.category == "network"

    systems = set(inc.system_name for inc in incidents)
    assert "telecom" in systems
    assert "bank" in systems
    assert "market" in systems

    print(f"  [PASS] OpenRCAAdapter loaded {len(incidents)} enterprise cases spanning {list(systems)}.")


def test_supplementary_cases_retained():
    print("[*] Test 3: Verifying Retention of Supplementary Cases (Database, K8s, TLS)...")
    adapter = SupplementaryAdapter()
    incidents = adapter.load()

    assert len(incidents) == 3, f"Expected 3 supplementary cases, got {len(incidents)}"

    ids = [c.id for c in incidents]
    assert "CASE_1" in ids, "CASE_1 (PostgreSQL) must be retained"
    assert "CASE_2" in ids, "CASE_2 (Kubernetes OOM) must be retained"
    assert "CASE_3" in ids, "CASE_3 (TLS failure) must be retained"

    # Verify CASE_2 details
    c2 = next(c for c in incidents if c.id == "CASE_2")
    assert "CrashLoopBackOff" in c2.query
    assert "OOMKilled" in c2.query
    assert c2.source_dataset == "Supplementary"

    print("  [PASS] Supplementary baseline scenarios (CASE_1, CASE_2, CASE_3) retained with full fidelity.")


def test_leakage_free_grouped_splitting():
    print("[*] Test 4: Testing Leakage-Free Grouped Splitting...")
    all_incidents = IncidentDatasetRegistry.load_dataset("rcaeval")

    splits = IncidentDatasetSplitter.split_grouped(
        incidents=all_incidents,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42
    )

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    assert len(train) > 0, "Train split should not be empty"
    assert len(val) > 0, "Validation split should not be empty"
    assert len(test) > 0, "Test split should not be empty"
    assert len(train) + len(val) + len(test) == len(all_incidents), "All cases must be accounted for"

    # Critical Leakage Test: Group keys in Train must NEVER overlap with Test or Val
    train_groups = set(f"{c.system_name}::{c.root_cause_service}::{c.fault_type}" for c in train)
    val_groups = set(f"{c.system_name}::{c.root_cause_service}::{c.fault_type}" for c in val)
    test_groups = set(f"{c.system_name}::{c.root_cause_service}::{c.fault_type}" for c in test)

    train_test_overlap = train_groups.intersection(test_groups)
    train_val_overlap = train_groups.intersection(val_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    assert len(train_test_overlap) == 0, f"Data leakage detected between train and test! Overlap: {train_test_overlap}"
    assert len(train_val_overlap) == 0, f"Data leakage detected between train and val! Overlap: {train_val_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage detected between val and test! Overlap: {val_test_overlap}"

    print(f"  [PASS] Grouped split verified: Train={len(train)} | Val={len(val)} | Test={len(test)} (0% group leakage).")


def test_cross_system_generalization_split():
    print("[*] Test 5: Testing Cross-System Generalization Split (Unseen Train-Ticket System)...")
    all_incidents = IncidentDatasetRegistry.load_dataset("rcaeval")

    splits = IncidentDatasetSplitter.split_cross_system(
        incidents=all_incidents,
        unseen_system="train-ticket"
    )

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    assert len(test) > 0, "Test split should have train-ticket cases"
    # All cases in test must be train-ticket
    for c in test:
        assert c.system_name == "train-ticket", f"Expected test case from train-ticket, got {c.system_name}"

    # Zero cases in train or val may be train-ticket
    for c in train + val:
        assert c.system_name != "train-ticket", f"Train/val leaked unseen system {c.system_name}"

    print(f"  [PASS] Cross-system split verified: Known Systems={len(train)+len(val)} cases | Unseen Target (train-ticket)={len(test)} cases.")


def test_cross_fault_generalization_split():
    print("[*] Test 6: Testing Cross-Fault Generalization Split (Unseen Fault Categories)...")
    all_incidents = IncidentDatasetRegistry.load_dataset("rcaeval")

    splits = IncidentDatasetSplitter.split_cross_fault(
        incidents=all_incidents,
        unseen_fault_types=["DELAY", "LOSS", "SOCKET", "F2", "F4"]
    )

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    unseen_set = {"DELAY", "LOSS", "SOCKET", "F2", "F4"}
    for c in test:
        assert c.fault_type.upper() in unseen_set, f"Test case has seen fault {c.fault_type}"

    for c in train + val:
        assert c.fault_type.upper() not in unseen_set, f"Train/val leaked unseen fault {c.fault_type}"

    print(f"  [PASS] Cross-fault split verified: Resource Faults={len(train)+len(val)} cases | Unseen Network/Code Faults={len(test)} cases.")


def test_end_to_end_benchmark_runner():
    print("[*] Test 7: Testing End-to-End Benchmark Suite Execution...")
    mock_responses = {
        r"ticket classifier": '{"category": "database", "severity": "L2", "summary": "DB memory exhaustion"}',
        r"identify the root cause": '{"root_cause": "The PostgreSQL database memory is exhausted due to connection pooling exhaustion without pgbouncer.", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Verify connections via pg_stat_activity\n2. Configure PgBouncer\nRESOLUTION_STATUS: resolved",
        r"claim-level evidence verification": json.dumps({
            "verified": True,
            "verification_status": "verified",
            "incident_summary": "Postgres connection pooling exhaustion",
            "affected_service_or_component": "postgresql database",
            "suspected_root_cause": "PostgreSQL database connection pooling exhaustion",
            "observed_facts": ["Memory exhausted with spikes in active connections"],
            "inferred_root_causes": ["Connection pooling exhaustion"],
            "supporting_evidence": [{"source": "postgres_runbook.md", "snippet": "Enable PgBouncer.", "relevance": "high", "verified": True}],
            "contradictory_or_missing_evidence": {"missing_evidence": [], "contradictory_evidence": ["None identified in current evidence"], "has_contradictions": False},
            "alternative_explanations": ["Heavy background vacuum worker"],
            "confidence_estimate": {"score": 0.88, "confidence_band": "high", "meaning": "Calibrated probability 88%."},
            "recommended_remediation": {"steps": ["Configure PgBouncer"], "is_advisory": True, "safety_level": "safe", "advisory_notes": "Advisory SRE guidance."},
            "claims": [{"claim": "PostgreSQL memory exhausted", "claim_type": "observed_fact", "supported": True, "risk": "low", "source_identifier": "postgres_runbook.md"}],
            "notes": "Verified.",
            "escalation_or_abstention_reason": None
        })
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        summary = run_benchmark_suite(
            dataset_name="supplementary",
            split_strategy="all",
            max_cases=1
        )

        assert summary["cases_evaluated"] == 1
        assert "service_hit_rate_at_1" in summary
        assert "mean_f1" in summary
        assert "structured_diagnosis_coverage" in summary
        assert summary["structured_diagnosis_coverage"] > 0.0

        print(f"  [PASS] Benchmark runner completed cleanly: Hit@1={summary['service_hit_rate_at_1']}% | F1={summary['mean_f1']} | Schema Coverage={summary['structured_diagnosis_coverage']}%.")
    finally:
        clear_mock_llm()


if __name__ == "__main__":
    print("=" * 80)
    print("   RUNNING PHASE 6: PUBLIC INCIDENT BENCHMARK TEST SUITE")
    print("=" * 80)
    test_rcaeval_adapter()
    test_openrca_adapter()
    test_supplementary_cases_retained()
    test_leakage_free_grouped_splitting()
    test_cross_system_generalization_split()
    test_cross_fault_generalization_split()
    test_end_to_end_benchmark_runner()
    print("\n" + "=" * 80)
    print("   ALL PHASE 6 PUBLIC BENCHMARK TESTS PASSED! (100%)")
    print("=" * 80)

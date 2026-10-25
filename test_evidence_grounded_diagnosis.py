# -*- coding: utf-8 -*-
"""
test_evidence_grounded_diagnosis.py — Phase 5 Evidence-Grounded Diagnosis Test Suite
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Verifies:
1. Production of all 10 required structured diagnostic fields:
   - Incident summary
   - Suspected root cause
   - Affected service or component
   - Supporting evidence and source identifiers
   - Contradictory or missing evidence
   - Alternative explanations
   - Confidence estimate and operational meaning
   - Recommended remediation (advisory)
   - Verification status
   - Escalation or abstention reason
2. Categorization of claims into observed_fact, inferred_root_cause, and recommended_action.
3. Strict evidence grounding: source identifiers match retrieved documents (anti-hallucination).
4. Explicit uncertainty status when evidence is insufficient (no manufactured definitive root causes).
5. Advisory remediation designation and active detection of destructive commands.
6. Deterministic human review for L3 critical incidents with clear escalation reasons.
7. End-to-end integration via the LangGraph orchestrator.
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
from agents.verification_agent import verification_agent
from agents.diagnostic_claims import (
    EvidenceGroundingValidator,
    build_structured_diagnosis,
    format_structured_diagnosis_report
)
from agents.orchestrator import build_graph


def test_10_structured_fields_presence():
    print("[*] Test 1: Verifying All 10 Structured Diagnostic Fields...")
    mock_responses = {
        r"claim-level evidence verification": json.dumps({
            "verified": True,
            "verification_status": "verified",
            "incident_summary": "Kubernetes API gateway pod CrashLoopBackOff with OOM exit code 137",
            "affected_service_or_component": "Kubernetes API gateway pod",
            "suspected_root_cause": "Container memory limit exceeded by JVM heap spike",
            "observed_facts": [
                "Pod status is in CrashLoopBackOff",
                "Container terminated with OOMKilled exit code 137"
            ],
            "inferred_root_causes": [
                "JVM heap consumption exceeded cgroups container memory limit"
            ],
            "supporting_evidence": [
                {
                    "source": "k8s_troubleshooting.md",
                    "snippet": "Exit code 137 indicates the container was terminated by the Linux OOM killer.",
                    "relevance": "high",
                    "verified": True
                }
            ],
            "contradictory_or_missing_evidence": {
                "missing_evidence": ["metrics (missing CPU and memory timeseries)"],
                "contradictory_evidence": ["None identified in current evidence"],
                "has_contradictions": False
            },
            "alternative_explanations": [
                "Transient memory burst from concurrent client requests",
                "Application memory leak in gateway cache"
            ],
            "confidence_estimate": {
                "score": 0.88,
                "confidence_band": "high",
                "meaning": "Calibrated probability (88.0%) that the diagnosis matches ground truth."
            },
            "recommended_remediation": {
                "steps": [
                    "Increase container memory limit to 1Gi in gateway-deployment.yaml",
                    "Apply kubectl apply -f gateway-deployment.yaml",
                    "Monitor pod memory utilization"
                ],
                "is_advisory": True,
                "safety_level": "safe",
                "advisory_notes": "Advisory SRE guidance. Verify cluster capacity before applying."
            },
            "claims": [
                {
                    "claim": "Container terminated with OOMKilled exit code 137",
                    "claim_type": "observed_fact",
                    "supported": True,
                    "risk": "low",
                    "source_identifier": "k8s_troubleshooting.md"
                },
                {
                    "claim": "JVM heap consumption exceeded cgroups limit",
                    "claim_type": "inferred_root_cause",
                    "supported": True,
                    "risk": "low",
                    "source_identifier": "k8s_troubleshooting.md"
                },
                {
                    "claim": "Increase container memory limit to 1Gi",
                    "claim_type": "recommended_action",
                    "supported": True,
                    "risk": "low",
                    "source_identifier": "k8s_troubleshooting.md"
                }
            ],
            "notes": "Verified against Kubernetes troubleshooting runbook.",
            "escalation_or_abstention_reason": None
        })
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        sample_state: AgentState = {
            "user_query": "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137.",
            "category": "deployment",
            "severity": "L2",
            "summary": "Kubernetes API gateway pod CrashLoopBackOff OOM 137",
            "kb_results": [
                {
                    "source": "k8s_troubleshooting.md",
                    "text": "Exit code 137 indicates the container was terminated by the Linux OOM killer.",
                    "score": 0.22
                }
            ],
            "kb_context": "Exit code 137 indicates the container was terminated by the Linux OOM killer.",
            "root_cause": "Container memory limit exceeded by JVM heap spike",
            "confidence": "high",
            "resolution_steps": "1. Increase container memory limit to 1Gi\n2. kubectl apply -f gateway.yaml",
            "resolution_status": "resolved",
            "verified": False,
            "verification_notes": "",
            "escalated": False,
            "escalation_reason": "",
            "final_answer": "",
            "selected_model": "gemini-2.5-flash",
            "selected_provider": "gemini",
            "routing_metadata": {"chosen_tier": "tier1_fast"},
            "needs_model_escalation": False,
            "escalation_level": 0
        }

        res = verification_agent(sample_state)
        diag = res.get("structured_diagnosis")
        assert diag is not None, "structured_diagnosis must be present in state"

        # Check all 10 required fields
        assert "incident_summary" in diag, "Missing incident_summary"
        assert "suspected_root_cause" in diag, "Missing suspected_root_cause"
        assert "affected_service_or_component" in diag, "Missing affected_service_or_component"
        assert "supporting_evidence" in diag, "Missing supporting_evidence"
        assert "contradictory_or_missing_evidence" in diag, "Missing contradictory_or_missing_evidence"
        assert "alternative_explanations" in diag, "Missing alternative_explanations"
        assert "confidence_estimate" in diag, "Missing confidence_estimate"
        assert "recommended_remediation" in diag, "Missing recommended_remediation"
        assert "verification_status" in diag, "Missing verification_status"
        assert "escalation_or_abstention_reason" in diag, "Missing escalation_or_abstention_reason"

        # Verify values
        assert diag["verification_status"] == "verified"
        assert "Kubernetes API gateway" in diag["affected_service_or_component"]
        assert len(diag["supporting_evidence"]) > 0
        assert len(diag["alternative_explanations"]) >= 2
        assert diag["confidence_estimate"]["score"] > 0.0
        assert "meaning" in diag["confidence_estimate"]
        assert diag["recommended_remediation"]["is_advisory"] is True

        print(f"  [PASS] All 10 fields validated: Component='{diag['affected_service_or_component']}' | Status='{diag['verification_status']}'")
    finally:
        clear_mock_llm()


def test_claim_categorization_and_grounding():
    print("[*] Test 2: Checking Claim Categorization (Fact vs Inferred Cause vs Recommended Action)...")
    
    # 1. Observed facts
    c1 = "Pod status is in CrashLoopBackOff with exit code 137"
    cat1 = EvidenceGroundingValidator.categorize_claim(c1)
    assert cat1 == "observed_fact", f"Expected observed_fact, got {cat1}"

    # 2. Inferred causes
    c2 = "Container memory limit exceeded due to JVM heap saturation"
    cat2 = EvidenceGroundingValidator.categorize_claim(c2)
    assert cat2 == "inferred_root_cause", f"Expected inferred_root_cause, got {cat2}"

    # 3. Recommended actions
    c3 = "Increase container memory limits to 1Gi and restart deployment"
    cat3 = EvidenceGroundingValidator.categorize_claim(c3)
    assert cat3 == "recommended_action", f"Expected recommended_action, got {cat3}"

    # 4. Anti-hallucination source validator
    kb_results = [{"source": "k8s_troubleshooting.md", "text": "OOM runbook text", "score": 0.2}]
    candidate_ev = [
        {"source": "k8s_troubleshooting.md", "snippet": "OOM runbook snippet", "relevance": "high"},
        {"source": "fabricated_fictitious_document.pdf", "snippet": "Invented passage", "relevance": "high"}
    ]
    grounded = EvidenceGroundingValidator.sanitize_supporting_evidence(
        candidate_evidence=candidate_ev,
        kb_results=kb_results,
        query="Pod in CrashLoopBackOff"
    )

    sources = [e["source"] for e in grounded]
    assert "k8s_troubleshooting.md" in sources, "Genuine source must be retained"
    assert "fabricated_fictitious_document.pdf" not in sources, "Fabricated source must NOT be cited"

    print("  [PASS] Claim categorization and anti-hallucination source grounding confirmed.")


def test_uncertainty_status_on_insufficient_evidence():
    print("[*] Test 3: Checking Explicit Uncertainty on Insufficient Evidence...")

    mock_llm = MockDeterministicLLM()
    set_mock_llm(mock_llm)

    try:
        # State with NO retrieved documents (sparse / novel issue)
        sparse_state: AgentState = {
            "user_query": "Unknown proprietary subsystem errored out with unrecognized hex code 0xdeadbeef.",
            "category": "general",
            "severity": "L2",
            "summary": "Subsystem hex code error",
            "kb_results": [],  # Empty knowledge base retrieval!
            "kb_context": "No relevant documentation found.",
            "root_cause": "Proprietary internal fault",
            "confidence": "low",
            "resolution_steps": "1. Investigate internal error log",
            "resolution_status": "resolved",
            "verified": False,
            "verification_notes": "No matching SOP documentation found.",
            "escalated": False,
            "escalation_reason": "",
            "final_answer": "",
            "selected_model": "gemini-2.5-flash",
            "selected_provider": "gemini"
        }

        res = verification_agent(sparse_state)
        diag = res.get("structured_diagnosis")
        assert diag is not None

        # Must NOT manufacture a definitive root cause
        assert diag["verification_status"] in ("uncertain", "insufficient_evidence"), \
            f"Expected uncertain status, got {diag['verification_status']}"
        assert "Uncertain" in diag["suspected_root_cause"], \
            f"Expected uncertainty statement in root cause, got {diag['suspected_root_cause']}"
        assert diag["escalation_or_abstention_reason"] is not None
        assert res["verified"] is False

        print(f"  [PASS] Handled insufficient evidence cleanly: Status='{diag['verification_status']}' | Root Cause: '{diag['suspected_root_cause'][:60]}...'")
    finally:
        clear_mock_llm()


def test_advisory_remediation_and_destructive_guard():
    print("[*] Test 4: Checking Advisory Remediation & Destructive Command Gating...")

    # Proposed resolution containing destructive command rm -rf
    destructive_steps = [
        "1. SSH into the production server",
        "2. Run rm -rf /var/lib/data/corrupted_index to clear disk space",
        "3. Restart the database"
    ]

    has_destructive, warnings = EvidenceGroundingValidator.detect_destructive_actions(destructive_steps)
    assert has_destructive is True, "Must detect rm -rf as destructive"
    assert len(warnings) > 0

    mock_responses = {
        r"claim-level evidence verification": json.dumps({
            "verified": True,
            "claims": [
                {"claim": "Run rm -rf /var/lib/data/corrupted_index", "supported": True, "risk": "low"}
            ],
            "notes": "Steps verified"
        })
    }
    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        state_destructive: AgentState = {
            "user_query": "Disk is full on primary database node.",
            "category": "database",
            "severity": "L2",
            "summary": "Disk full on database",
            "kb_results": [{"source": "disk_maintenance.md", "text": "Clear temp files.", "score": 0.3}],
            "kb_context": "Clear temp files.",
            "root_cause": "Disk saturation from corrupted temp files",
            "confidence": "high",
            "resolution_steps": "\n".join(destructive_steps),
            "resolution_status": "resolved",
            "verified": False,
            "verification_notes": "",
            "escalated": False,
            "escalation_reason": "",
            "final_answer": "",
            "selected_model": "gemini-2.5-flash",
            "selected_provider": "gemini"
        }

        res = verification_agent(state_destructive)
        diag = res.get("structured_diagnosis")
        rem = diag["recommended_remediation"]

        # Crucial safety invariants
        assert rem["is_advisory"] is True, "Remediation must be advisory"
        assert rem["has_destructive_commands"] is True, "Must flag destructive commands"
        assert rem["safety_level"] == "destructive_prohibited", f"Expected destructive_prohibited, got {rem['safety_level']}"
        assert res["verified"] is False, "Must FAIL automated verification on destructive commands"

        print(f"  [PASS] Destructive command gated: Safety='{rem['safety_level']}' | Verified={res['verified']}")
    finally:
        clear_mock_llm()


def test_l3_critical_escalation_structured_diagnosis():
    print("[*] Test 5: Checking L3 Critical Escalation Structured Claims...")

    sample_l3: AgentState = {
        "user_query": "CRITICAL OUTAGE: Primary database node is corrupted and rejecting all user traffic!",
        "category": "database",
        "severity": "L3",
        "summary": "Primary DB node corrupted outage",
        "kb_results": [{"source": "postgres_disaster_recovery.md", "text": "Promote read replica in disaster.", "score": 0.25}],
        "kb_context": "Promote read replica in disaster.",
        "root_cause": "Filesystem corruption on primary database storage volume",
        "confidence": "high",
        "resolution_steps": "1. Promote standby replica to primary\n2. Drain connections from failed node",
        "resolution_status": "needs_human",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "final_answer": "",
        "selected_model": "gemini-2.5-flash",
        "selected_provider": "gemini"
    }

    res = verification_agent(sample_l3)
    diag = res.get("structured_diagnosis")

    assert diag is not None
    assert diag["verification_status"] == "escalated"
    assert "L3" in diag["escalation_or_abstention_reason"]
    assert res["verified"] is False
    assert diag["recommended_remediation"]["is_advisory"] is True
    assert diag["recommended_remediation"]["safety_level"] == "requires_approval"

    print(f"  [PASS] L3 Outage cleanly structured: Status='{diag['verification_status']}' | Escalation='{diag['escalation_or_abstention_reason'][:50]}...'")


def test_end_to_end_orchestrator_integration():
    print("[*] Test 6: Checking Full Graph Execution Integration with Phase 5...")

    mock_responses = {
        r"ticket classifier": '{"category": "deployment", "severity": "L1", "summary": "Restart checkout service"}',
        r"identify the root cause": '{"root_cause": "Checkout service process hung due to memory spike", "confidence": "high", "needs_escalation": false}',
        r"resolution instructions": "1. Run systemctl restart checkout\n2. Verify systemctl status checkout\nRESOLUTION_STATUS: resolved",
        r"claim-level evidence verification": json.dumps({
            "verified": True,
            "verification_status": "verified",
            "incident_summary": "Checkout service hung on memory spike",
            "affected_service_or_component": "checkout service",
            "suspected_root_cause": "Checkout process thread pool deadlock after memory spike",
            "observed_facts": ["Checkout service process hung"],
            "inferred_root_causes": ["Thread pool deadlock"],
            "supporting_evidence": [{"source": "deployment_troubleshooting.md", "snippet": "Restart checkout safely.", "relevance": "high", "verified": True}],
            "contradictory_or_missing_evidence": {"missing_evidence": [], "contradictory_evidence": ["None identified in current evidence"], "has_contradictions": False},
            "alternative_explanations": ["Database connection timeout"],
            "confidence_estimate": {"score": 0.90, "confidence_band": "high", "meaning": "High confidence SOP resolution."},
            "recommended_remediation": {"steps": ["systemctl restart checkout"], "is_advisory": True, "safety_level": "safe", "advisory_notes": "Advisory SRE guidance."},
            "claims": [{"claim": "Restarting checkout is safe", "claim_type": "recommended_action", "supported": True, "risk": "low", "source_identifier": "deployment_troubleshooting.md"}],
            "notes": "Verified against SOP.",
            "escalation_or_abstention_reason": None
        })
    }

    mock_llm = MockDeterministicLLM(response_map=mock_responses)
    set_mock_llm(mock_llm)

    try:
        graph = build_graph()
        initial_state = {
            "user_query": "How do I safely restart the hung checkout service in production?",
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
            "selected_model": None,
            "selected_provider": None,
            "routing_metadata": None,
            "needs_model_escalation": False,
            "escalation_level": 0
        }

        res = graph.invoke(initial_state)

        assert res.get("verified") is True
        assert res.get("structured_diagnosis") is not None
        assert res.get("affected_component") is not None
        assert "RESOLVED" in res.get("final_answer", "")
        assert "Evidence-Grounded Structured Diagnosis" in res.get("final_answer", "")

        print("  [PASS] Full graph invocation rendered Phase 5 structured diagnosis in final answer.")
    finally:
        clear_mock_llm()


if __name__ == "__main__":
    print("=" * 80)
    print("   RUNNING PHASE 5: EVIDENCE-GROUNDED STRUCTURED DIAGNOSIS TEST SUITE")
    print("=" * 80)
    test_10_structured_fields_presence()
    test_claim_categorization_and_grounding()
    test_uncertainty_status_on_insufficient_evidence()
    test_advisory_remediation_and_destructive_guard()
    test_l3_critical_escalation_structured_diagnosis()
    test_end_to_end_orchestrator_integration()
    print("\n" + "=" * 80)
    print("   ALL PHASE 5 EVIDENCE-GROUNDED DIAGNOSIS TESTS PASSED! (100%)")
    print("=" * 80)

"""
Shared state schema for LangGraph agent pipeline.
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    # Input
    user_query: str

    # Classification
    category: str          # e.g. "deployment", "database", "cpu", "network", "general"
    severity: str          # "L1", "L2", "L3"
    summary: str           # One-line summary of the issue

    # Knowledge base results
    kb_results: List[dict]  # [{text, source, score}]
    kb_context: str         # Formatted context string for LLM

    # Diagnostic output
    root_cause: str
    confidence: str         # "high", "medium", "low"

    # Resolution
    resolution_steps: str
    resolution_status: str  # "resolved", "escalated", "needs_human"

    # Verification
    verified: bool
    verification_notes: str

    # Escalation
    escalated: bool
    escalation_reason: str

    # Adaptive Multi-LLM Routing
    selected_model: Optional[str]
    selected_provider: Optional[str]
    routing_metadata: Optional[dict]
    needs_model_escalation: Optional[bool]
    escalation_level: Optional[int]

    # Confidence Calibration (Phase 2)
    calibrated_confidence: Optional[float]  # P(Correct | Evidence, Features) in [0.0, 1.0]
    confidence_breakdown: Optional[dict]    # Detailed feature contributions, ECE band, and reliability metrics

    # Intervention-Aware Routing & Policy (Phase 3)
    recommended_action: Optional[str]        # "ANSWER", "RETRIEVE_MORE", "SWITCH_MODEL", "SAMPLE_CANDIDATES", "ABSTAIN", "HUMAN_REVIEW"
    action_evaluation: Optional[dict]        # Evaluated candidate actions with quality, cost, latency, IG, risk
    budget_spent: Optional[float]            # Cumulative API and compute cost spent so far ($)
    latency_spent: Optional[float]           # Cumulative end-to-end latency spent so far (s)
    retry_count: Optional[int]               # Number of intervention loops executed
    missing_evidence: Optional[List[str]]    # Missing evidence types (logs, metrics, traces, runbook)
    candidate_diagnoses: Optional[List[str]] # Stored candidate diagnoses when SAMPLE_CANDIDATES is executed

    # Cost-Aware Adaptive Execution & Budget Tracking (Phase 4)
    actual_cost: Optional[float]             # Real cumulative API and compute cost spent ($)
    estimated_cost: Optional[float]          # Cumulative estimated cost from policy evaluations ($)
    actual_latency: Optional[float]          # Total elapsed wall-clock latency (s)
    estimated_latency: Optional[float]       # Cumulative estimated latency from policy evaluations (s)
    start_time: Optional[float]              # Wall-clock timestamp when execution commenced
    iteration_count: Optional[int]           # Loop iteration counter for graph execution
    model_call_count: Optional[int]          # Total number of LLM invocations
    max_iterations: Optional[int]            # Configurable iteration ceiling
    max_model_calls: Optional[int]           # Configurable model invocation ceiling
    claim_verifications: Optional[List[dict]]# Claim-level evidence verification breakdown
    action_history: Optional[List[dict]]     # Audit trail of executed actions with confirmation status

    # Evidence-Grounded Structured Diagnosis (Phase 5)
    structured_diagnosis: Optional[dict]     # 10-field structured diagnosis dictionary
    affected_component: Optional[str]        # Identified affected service or component
    observed_facts: Optional[List[str]]      # Observed factual telemetry & symptoms
    inferred_root_causes: Optional[List[str]]# Inferred diagnostic deductions
    alternative_explanations: Optional[List[str]] # Alternative root cause hypotheses
    verification_status: Optional[str]       # "verified", "unverified", "uncertain", "insufficient_evidence", "escalated"
    is_advisory_remediation: Optional[bool]  # Guarantees remediation is advisory only

    # Final answer shown to user
    final_answer: str

    # Incident Response Cache Metadata (Phase 11)
    cache_hit: Optional[bool]                # True if response was served from cache
    cache_match_type: Optional[str]          # "exact", "semantic", or None
    cache_similarity: Optional[float]        # Similarity score (1.0 for exact, float for semantic)
    cache_record_id: Optional[str]           # Identifier of cached entry in SQLite
    cache_lookup_latency: Optional[float]    # Cache retrieval overhead (s)






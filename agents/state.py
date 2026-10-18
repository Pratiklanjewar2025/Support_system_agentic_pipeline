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

    # Final answer shown to user
    final_answer: str





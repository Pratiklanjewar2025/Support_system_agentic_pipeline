"""
LangGraph Orchestrator — wires all agents into a multi-agent pipeline.

Flow:
  classify → knowledge_search → diagnose → resolve → verify
                                                         ↓
                                              [verified?]
                                               yes → close
                                               no  → escalate
"""

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.classifier_agent import classifier_agent
from agents.knowledge_agent import knowledge_agent
from agents.diagnostic_agent import diagnostic_agent
from agents.resolution_agent import resolution_agent
from agents.verification_agent import verification_agent
from agents.escalation_agent import escalation_agent


def _close_ticket(state: AgentState) -> AgentState:
    """Final node — build the answer shown to the user."""
    state["escalated"] = False
    state["final_answer"] = (
        "✅  RESOLVED\n\n"
        f"**Root Cause:** {state['root_cause']}\n\n"
        f"**Resolution Steps:**\n{state['resolution_steps']}\n\n"
        f"**Verification:** {state.get('verification_notes', 'Passed')}"
    )
    return state


def _route_after_verify(state: AgentState) -> str:
    """Conditional edge after verification."""
    if state.get("verified"):
        return "close"
    return "escalate"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify",        classifier_agent)
    graph.add_node("knowledge_search", knowledge_agent)
    graph.add_node("diagnose",         diagnostic_agent)
    graph.add_node("resolve",          resolution_agent)
    graph.add_node("verify",           verification_agent)
    graph.add_node("escalate",         escalation_agent)
    graph.add_node("close",            _close_ticket)

    # Linear flow up to verify
    graph.set_entry_point("classify")
    graph.add_edge("classify",         "knowledge_search")
    graph.add_edge("knowledge_search", "diagnose")
    graph.add_edge("diagnose",         "resolve")
    graph.add_edge("resolve",          "verify")

    # Conditional branch after verify
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "close":    "close",
            "escalate": "escalate",
        },
    )

    graph.add_edge("close",    END)
    graph.add_edge("escalate", END)

    return graph.compile()





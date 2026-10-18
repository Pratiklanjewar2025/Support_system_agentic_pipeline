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
from agents.router_agent import router_agent
from agents.diagnostic_agent import diagnostic_agent
from agents.resolution_agent import resolution_agent
from agents.verification_agent import verification_agent
from agents.escalation_agent import escalation_agent


def _close_ticket(state: AgentState) -> AgentState:
    """Final node — build the answer shown to the user."""
    state["escalated"] = False
    routed_info = ""
    if state.get("routing_metadata"):
        rm = state["routing_metadata"]
        tier = rm.get("chosen_tier", "")
        model_name = rm.get("chosen_model", state.get("selected_model", ""))
        conf = rm.get("retrieval_features", {}).get("retrieval_confidence", 0.0)
        routed_info = f"\n\n**Engineered Model:** {model_name} [{tier}] (Retrieval Conf: {conf})"
    state["final_answer"] = (
        "✅  RESOLVED\n\n"
        f"**Root Cause:** {state['root_cause']}\n\n"
        f"**Resolution Steps:**\n{state['resolution_steps']}\n\n"
        f"**Verification:** {state.get('verification_notes', 'Passed')}"
        + routed_info
    )
    return state


def _model_escalate(state: AgentState) -> AgentState:
    """Escalates low-confidence/unverified query to the Tier 3 Heavy reasoning model."""
    state["escalation_level"] = state.get("escalation_level", 0) + 1
    state["selected_provider"] = "cohere"
    state["selected_model"] = "command-r-plus-08-2024"
    state["needs_model_escalation"] = False
    if state.get("routing_metadata"):
        state["routing_metadata"]["chosen_tier"] = "tier3_heavy (escalated)"
        state["routing_metadata"]["chosen_model"] = "Cohere Command-R+ (104B)"
    return state


def _route_after_verify(state: AgentState) -> str:
    """Conditional edge after verification."""
    if state.get("verified"):
        return "close"
    if state.get("needs_model_escalation"):
        return "model_escalate"
    return "escalate"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify",         classifier_agent)
    graph.add_node("knowledge_search", knowledge_agent)
    graph.add_node("router",           router_agent)
    graph.add_node("diagnose",         diagnostic_agent)
    graph.add_node("resolve",          resolution_agent)
    graph.add_node("verify",           verification_agent)
    graph.add_node("model_escalate",   _model_escalate)
    graph.add_node("escalate",         escalation_agent)
    graph.add_node("close",            _close_ticket)

    # Linear flow up to verify
    graph.set_entry_point("classify")
    graph.add_edge("classify",         "knowledge_search")
    graph.add_edge("knowledge_search", "router")
    graph.add_edge("router",           "diagnose")
    graph.add_edge("diagnose",         "resolve")
    graph.add_edge("resolve",          "verify")

    # Conditional branch after verify
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "close":          "close",
            "model_escalate": "model_escalate",
            "escalate":       "escalate",
        },
    )

    # Model escalation loops back to diagnose
    graph.add_edge("model_escalate", "diagnose")

    graph.add_edge("close",    END)
    graph.add_edge("escalate", END)

    return graph.compile()





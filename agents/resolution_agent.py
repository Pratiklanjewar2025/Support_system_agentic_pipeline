"""
Resolution Agent — generates step-by-step resolution instructions
based on the diagnosed root cause and knowledge base context.
"""

from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

SYSTEM_PROMPT = """You are an IT support resolution engineer.
Given a root cause and runbook context, generate clear numbered resolution steps.

Rules:
1. Maximum 6 steps. Be concise — one sentence per step.
2. Include exact commands where needed.
3. End with either "RESOLUTION_STATUS: resolved" or "RESOLUTION_STATUS: needs_human".

No lengthy explanations. Steps only."""


def resolution_agent(state: AgentState) -> AgentState:
    model = state.get("selected_model")
    provider = state.get("selected_provider")
    llm = get_llm(temperature=0.2, max_tokens=600, model=model, provider=provider)

    user_prompt = f"""User Issue: {state['user_query']}

Root Cause: {state['root_cause']} (Confidence: {state['confidence']})

Relevant Documentation:
{state.get('kb_context', 'No context available.')}

Generate step-by-step resolution instructions."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content[0])) if isinstance(content[0], dict) else str(content[0])
    content = str(content).strip()

    state["model_call_count"] = (state.get("model_call_count") or 0) + 1
    from agents.llm_utils import estimate_token_cost
    in_tok = int(len(user_prompt.split()) * 1.3)
    out_tok = int(len(content.split()) * 1.3)
    c_delta = estimate_token_cost(in_tok, out_tok, model=model, provider=provider)
    state["actual_cost"] = round((state.get("actual_cost") or 0.0) + c_delta, 6)
    state["budget_spent"] = state["actual_cost"]

    # Parse resolution status from the response
    if "RESOLUTION_STATUS: needs_human" in content:
        state["resolution_status"] = "needs_human"
        content = content.replace("RESOLUTION_STATUS: needs_human", "").strip()
    else:
        state["resolution_status"] = "resolved"
        content = content.replace("RESOLUTION_STATUS: resolved", "").strip()

    state["resolution_steps"] = content
    return state





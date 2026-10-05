"""
Escalation Agent — handles cases that need human engineer review.
Generates a Root Cause Analysis (RCA) report for the human.
"""

from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

SYSTEM_PROMPT = """You are an escalation coordinator for a technical support team.
Write a concise escalation report for a human engineer. Include:
1. Issue Summary
2. Root Cause Analysis
3. What was attempted / investigated
4. Recommended next steps for the human engineer
5. Risk level (Low / Medium / High / Critical)

Be clear and professional. This will be handed off to a human."""


def escalation_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.1, max_tokens=400)

    user_prompt = f"""Ticket Escalation Report:

Original Issue: {state['user_query']}
Category: {state['category']} | Severity: {state['severity']}
Root Cause: {state['root_cause']} (Confidence: {state['confidence']})

Resolution Attempted:
{state.get('resolution_steps', 'None')}

Verification Notes:
{state.get('verification_notes', 'None')}

Generate a professional escalation report."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", str(raw[0])) if isinstance(raw[0], dict) else str(raw[0])
    raw = str(raw).strip()
    state["escalated"] = True
    state["escalation_reason"] = state.get("verification_notes", "Requires human review")
    state["final_answer"] = (
        "[!!] ESCALATED TO HUMAN ENGINEER\n\n"
        + raw
    )
    return state





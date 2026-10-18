"""
Diagnostic Agent — performs root cause analysis using the user query,
knowledge base context, and LLM reasoning.
"""

from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json, re

from agents.state import AgentState

SYSTEM_PROMPT = """You are a senior DevOps/SRE diagnostic engineer.
Given a user's support issue and relevant documentation context, identify the most likely root cause.

Return ONLY a valid JSON object:
{
  "root_cause": "<concise description of the most likely root cause>",
  "confidence": "<high | medium | low>",
  "needs_escalation": <true | false>
}

Be specific and technical. Use the documentation context provided.
Return ONLY the JSON."""


def diagnostic_agent(state: AgentState) -> AgentState:
    model = state.get("selected_model")
    provider = state.get("selected_provider")
    llm = get_llm(temperature=0.1, max_tokens=400, model=model, provider=provider)

    user_prompt = f"""User Issue: {state['user_query']}

Category: {state['category']} | Severity: {state['severity']}

Relevant Documentation:
{state.get('kb_context', 'No context available.')}

Based on the above, identify the root cause."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", str(raw[0])) if isinstance(raw[0], dict) else str(raw[0])
    raw = str(raw).strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        state["root_cause"] = data.get("root_cause", "Unknown root cause")
        state["confidence"] = data.get("confidence", "low")
        if data.get("needs_escalation") and state["severity"] != "L3":
            state["severity"] = "L3"
    except json.JSONDecodeError:
        state["root_cause"] = raw[:500]
        state["confidence"] = "low"

    return state





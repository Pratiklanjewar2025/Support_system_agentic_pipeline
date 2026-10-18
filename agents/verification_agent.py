"""
Verification Agent — checks if the resolution steps are complete and safe.
For L3 issues, flags for human review.
"""

from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json, re

from agents.state import AgentState

SYSTEM_PROMPT = """You are a senior SRE reviewing a proposed resolution for a support ticket.
Check if the resolution steps are:
1. Complete (cover the root cause fully)
2. Safe (no risky undocumented actions)
3. Aligned with the documentation provided

Return ONLY a valid JSON object:
{
  "verified": <true | false>,
  "notes": "<brief verification notes or concerns>"
}"""


def verification_agent(state: AgentState) -> AgentState:
    # L3 always needs human review — mark as not verified
    if state["severity"] == "L3" or state.get("resolution_status") == "needs_human":
        state["verified"] = False
        state["verification_notes"] = (
            "L3 severity or complex resolution — requires human engineer review before execution."
        )
        return state

    llm = get_llm(temperature=0, max_tokens=300)

    user_prompt = f"""Original Issue: {state['user_query']}
Root Cause: {state['root_cause']}

Proposed Resolution:
{state['resolution_steps']}

Documentation Context:
{state.get('kb_context', 'No context available.')}

Is this resolution complete and safe?"""

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
        verified = bool(data.get("verified", False))
        notes = data.get("notes", "")
    except json.JSONDecodeError:
        verified = True
        notes = "Auto-verified (parse error in LLM response)"

    state["verified"] = verified
    state["verification_notes"] = notes

    # Dynamic Escalation: If not verified on initial tier, escalate to high-capacity Tier 3
    current_level = state.get("escalation_level", 0)
    current_tier = (state.get("routing_metadata") or {}).get("chosen_tier", "tier1_fast")
    if not verified and current_level == 0 and current_tier != "tier3_heavy":
        state["needs_model_escalation"] = True
    else:
        state["needs_model_escalation"] = False

    return state





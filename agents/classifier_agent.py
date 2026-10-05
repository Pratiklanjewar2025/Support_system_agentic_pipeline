"""
Classifier Agent — classifies the user query into:
  - category: deployment / database / cpu / network / general
  - severity: L1 (simple Q&A) / L2 (diagnosis needed) / L3 (complex/dangerous)
  - summary: one-line description
"""

from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json, re, os

from agents.state import AgentState

SYSTEM_PROMPT = """You are an IT support ticket classifier.
Analyze the user's support request and return ONLY a valid JSON object with these fields:
{
  "category": "<one of: deployment, database, cpu, network, application, general>",
  "severity": "<one of: L1, L2, L3>",
  "summary": "<one short sentence summarizing the issue>"
}

Severity guide:
- L1: Simple question, can be answered from documentation alone.
- L2: Requires diagnosis — check logs, metrics, or system state to find root cause.
- L3: Complex, dangerous, or requires human approval before any action.

Return ONLY the JSON. No markdown, no explanation."""


def classifier_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0, max_tokens=300)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]
    response = llm.invoke(messages)
    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", str(raw[0])) if isinstance(raw[0], dict) else str(raw[0])
    raw = str(raw).strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        state["category"] = data.get("category", "general")
        state["severity"] = data.get("severity", "L1")
        state["summary"] = data.get("summary", state["user_query"][:100])
    except json.JSONDecodeError:
        state["category"] = "general"
        state["severity"] = "L1"
        state["summary"] = state["user_query"][:100]

    return state





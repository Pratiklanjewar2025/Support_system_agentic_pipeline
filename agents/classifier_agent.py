"""
Classifier Agent — classifies the user query into:
  - category: deployment / database / cpu / network / application / general
  - severity: L1 (simple Q&A) / L2 (diagnosis needed) / L3 (complex/dangerous)
  - summary: one-line description
Supports cost-optimized fast classification (Groq/OpenRouter) with automatic fallback to Gemini.
"""

import json
import os
import re
import time
from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm_utils import get_llm, estimate_token_cost
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


def _heuristic_classify(query: str) -> dict:
    """Deterministic heuristic classification fallback."""
    q_low = query.lower()
    
    # Category detection
    if re.search(r"\b(postgres|db|database|sql|table|migration|deadlock|pool|query)\b", q_low):
        cat = "database"
    elif re.search(r"\b(cpu|load|spike|memory|oom|sigkill|usage|ram|thread)\b", q_low):
        cat = "cpu"
    elif re.search(r"\b(network|ssl|tls|port|dns|50[0-4]|timeout|latency|gateway)\b", q_low):
        cat = "network"
    elif re.search(r"\b(deploy|k8s|kubernetes|pod|restart|kubectl|helm|docker|container)\b", q_low):
        cat = "deployment"
    elif re.search(r"\b(api|service|app|frontend|backend|endpoint)\b", q_low):
        cat = "application"
    else:
        cat = "general"

    # Severity detection
    if re.search(r"\b(cascading|cluster[- ]wide|catastrophic|sev-?[01]|disaster|unhandled sigkill|deadlock|12 microservices)\b", q_low):
        sev = "L3"
    elif re.search(r"\b(why|fail|crash|down|spike|error|slow|15 seconds|diagnostic)\b", q_low):
        sev = "L2"
    else:
        sev = "L1"

    summary = query[:120].strip()
    return {"category": cat, "severity": sev, "summary": summary}


def classifier_agent(state: AgentState) -> AgentState:
    if not state.get("start_time"):
        state["start_time"] = time.time()
    state["actual_latency"] = 0.0
    state["iteration_count"] = state.get("iteration_count", 0)
    state["model_call_count"] = (state.get("model_call_count") or 0) + 1

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    # Provider prioritization for classification:
    # 1. Cost Priority Mode -> Try Groq (fast, <$0.00002) or OpenRouter Free ($0.00)
    # 2. Configured env provider
    # 3. Fallback to Gemini
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    candidate_providers = []
    # Primary fast/cheap classifier: Groq is ultra-fast (<0.6s)
    if has_groq:
        candidate_providers.append(("groq", "openai/gpt-oss-20b"))
    if has_openrouter:
        candidate_providers.append(("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"))
    if has_gemini:
        candidate_providers.append(("gemini", "gemini-2.5-flash"))

    raw = None
    used_provider = None
    used_model = None

    for prov, m_name in candidate_providers:
        try:
            llm = get_llm(temperature=0, max_tokens=250, provider=prov, model=m_name)
            response = llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", str(content[0])) if isinstance(content[0], dict) else str(content[0])
            raw = str(content).strip()
            if raw:
                used_provider = prov
                used_model = m_name
                break
        except Exception:
            continue

    parsed = None
    if raw:
        # Strip markdown fences
        clean_raw = re.sub(r"^```[a-z]*\n?", "", raw)
        clean_raw = re.sub(r"\n?```$", "", clean_raw).strip()
        try:
            parsed = json.loads(clean_raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]*\}", clean_raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    parsed = None

    if not parsed:
        parsed = _heuristic_classify(state["user_query"])
        used_provider = used_provider or "heuristic"
        used_model = used_model or "regex_rule_engine"

    state["category"] = parsed.get("category", "general")
    state["severity"] = parsed.get("severity", "L1")
    state["summary"] = parsed.get("summary", state["user_query"][:100])
    state["classified_by"] = f"{used_provider}:{used_model}"

    # Track token cost accurately
    in_tok = int(len(state["user_query"].split()) * 1.3) + 100
    out_tok = int(len(str(parsed).split()) * 1.3)
    c_delta = estimate_token_cost(in_tok, out_tok, model=used_model, provider=used_provider) if used_provider != "heuristic" else 0.0
    state["actual_cost"] = round((state.get("actual_cost") or 0.0) + c_delta, 6)
    state["budget_spent"] = state["actual_cost"]

    return state





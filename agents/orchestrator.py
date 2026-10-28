"""
LangGraph Orchestrator — Cost-Aware Adaptive Execution (Phase 4).
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Logical Flow:
Incident query
→ Query analysis (classify)
→ Knowledge retrieval (knowledge_search)
→ Evidence-quality assessment (evidence_assessment)
→ Model and intervention selection (router)
→ Diagnosis (diagnose)
→ Resolution (resolve)
→ Claim-level verification & Confidence assessment (verify)
→ Conditional Dispatch:
    - Accept & Close (close)
    - Active Re-Retrieval (retrieve_more → evidence_assessment)
    - Model Tier Upgrade (switch_model → diagnose)
    - Safe Diagnostic Abstention (abstain)
    - Human SRE Escalation (escalate)

Guarantees:
- Separately tracks actual and estimated costs, and actual wall-clock elapsed latency.
- Enforces configurable iteration and model call limits (prevents infinite loops).
- Confirms action execution and never claims success on unverified results.
"""

import os
import time
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.classifier_agent import classifier_agent
from agents.knowledge_agent import knowledge_agent
from agents.router_agent import router_agent
from agents.diagnostic_agent import diagnostic_agent
from agents.resolution_agent import resolution_agent
from agents.verification_agent import verification_agent
from agents.escalation_agent import escalation_agent


def _evidence_assessment_node(state: AgentState) -> AgentState:
    """
    Evidence-Quality Assessment Node (Phase 4).
    Evaluates retrieval certainty, lexical coverage, redundancy, contradiction risk,
    and audits missing telemetry (logs, metrics, traces, runbooks).
    """
    query = state.get("user_query", "")
    kb_results = state.get("kb_results", [])

    from agents.router_agent import AdaptiveModelRouter
    from agents.intervention_policy import EvidenceAuditor

    router = AdaptiveModelRouter()
    r_feat = router.extract_retrieval_features(kb_results, query)
    missing = EvidenceAuditor.audit_missing_evidence(state)

    state["missing_evidence"] = missing

    if not state.get("routing_metadata"):
        state["routing_metadata"] = {}
    state["routing_metadata"]["retrieval_features"] = r_feat

    if state.get("start_time"):
        state["actual_latency"] = round(time.time() - state["start_time"], 3)

    return state


def _retrieve_more_node(state: AgentState) -> AgentState:
    """
    Executes active retrieval expansion (RETRIEVE_MORE action).
    Fetches deeper documentation chunks, confirms result, and tracks costs/latencies.
    """
    from rag.retriever import retrieve
    old_count = len(state.get("kb_results", []))
    expanded = retrieve(state["user_query"], top_k=6)
    new_count = len(expanded) if expanded else old_count

    if expanded:
        state["kb_results"] = expanded
        state["kb_context"] = "\n\n---\n\n".join(
            f"Source: {r['source']}\n{r['text'][:500]}" for r in expanded
        )

    # Action Confirmation & Success Criteria
    action_success = new_count > old_count or new_count >= 4
    state["iteration_count"] = (state.get("iteration_count") or 0) + 1
    state["retry_count"] = (state.get("retry_count") or 0) + 1

    history = state.get("action_history") or []
    history.append({
        "action": "RETRIEVE_MORE",
        "iteration": state["iteration_count"],
        "executed": True,
        "confirmed": True,
        "result_summary": f"Expanded retrieval from {old_count} to {new_count} chunks (confirmed={action_success})",
        "action_success": action_success,
        "actual_cost_so_far": state.get("actual_cost", 0.0),
        "actual_latency_so_far": round(time.time() - state.get("start_time", time.time()), 3)
    })
    state["action_history"] = history

    return state


def _switch_model_node(state: AgentState) -> AgentState:
    """
    Executes model tier upgrade (SWITCH_MODEL action).
    Upgrades model tier, confirms switch, and loops back to diagnose.
    """
    from agents.router_agent import AdaptiveModelRouter
    router = AdaptiveModelRouter()
    current_tier = (state.get("routing_metadata") or {}).get("chosen_tier", "tier1_fast")

    if "tier1" in current_tier:
        target = router.model_catalog["tier2_high_reasoning"]
        tier_name = "tier2_high_reasoning"
    else:
        target = router.model_catalog["tier3_heavy"]
        tier_name = "tier3_heavy"

    old_model = state.get("selected_model")
    state["selected_provider"] = target["provider"]
    state["selected_model"] = target["model_id"]
    if state.get("routing_metadata"):
        state["routing_metadata"]["chosen_tier"] = tier_name
        state["routing_metadata"]["chosen_model"] = target["name"]

    state["iteration_count"] = (state.get("iteration_count") or 0) + 1
    state["retry_count"] = (state.get("retry_count") or 0) + 1

    # Action Confirmation
    switch_confirmed = (state["selected_model"] != old_model)

    history = state.get("action_history") or []
    history.append({
        "action": "SWITCH_MODEL",
        "iteration": state["iteration_count"],
        "executed": True,
        "confirmed": switch_confirmed,
        "from_model": old_model,
        "to_model": target["name"],
        "result_summary": f"Switched model from {old_model} to {target['name']} [{tier_name}]",
        "action_success": switch_confirmed,
        "actual_cost_so_far": state.get("actual_cost", 0.0),
        "actual_latency_so_far": round(time.time() - state.get("start_time", time.time()), 3)
    })
    state["action_history"] = history

    return state


def _model_escalate(state: AgentState) -> AgentState:
    """Backwards-compatible alias for _switch_model_node."""
    return _switch_model_node(state)


def _abstain_node(state: AgentState) -> AgentState:
    """
    Safe Diagnostic Abstention terminal node.
    Declines to output an automated diagnosis to prevent operational hallucinations,
    producing an explicit missing evidence report.
    """
    missing = state.get("missing_evidence")
    if not missing:
        from agents.intervention_policy import EvidenceAuditor
        missing = EvidenceAuditor.audit_missing_evidence(state)
        state["missing_evidence"] = missing

    state["escalated"] = False
    state["resolution_status"] = "abstained"
    state["verified"] = False

    history = state.get("action_history") or []
    history.append({
        "action": "ABSTAIN",
        "iteration": state.get("iteration_count", 0),
        "executed": True,
        "confirmed": True,
        "result_summary": f"Abstained from diagnosis due to {len(missing)} missing telemetry items",
        "action_success": True
    })
    state["action_history"] = history

    act_cost = state.get("actual_cost", 0.0) or 0.0
    est_cost = state.get("estimated_cost", 0.0) or 0.0
    act_lat = round(time.time() - state.get("start_time", time.time()), 3) if state.get("start_time") else (state.get("actual_latency") or 0.0)

    diag = state.get("structured_diagnosis")
    diag_report = ""
    if diag:
        from agents.diagnostic_claims import format_structured_diagnosis_report
        diag_report = "\n\n" + format_structured_diagnosis_report(diag)

    state["final_answer"] = (
        "[!] DIAGNOSTIC ABSTENTION (SAFEGUARD ACTIVATED)\n\n"
        "AdaRAG-Ops has declined to issue an automated root cause diagnosis for this incident "
        "to prevent operational risk.\n\n"
        "**Missing Evidence Audit:**\n"
        + "\n".join(f"- {m}" for m in missing) + "\n\n"
        "**Budget & Resource Telemetry:**\n"
        f"- Cumulative Cost: Actual ${act_cost:.5f} | Estimated ${est_cost:.5f}\n"
        f"- Elapsed Latency: Actual {act_lat:.2f}s\n"
        f"- Loops Executed: {state.get('iteration_count', 0)} (Model Calls: {state.get('model_call_count', 0)})\n\n"
        "**Required SRE Action:** Attach the missing telemetry items above before automated remediation."
        + diag_report
    )
    return state


def _cache_lookup_node(state: AgentState) -> AgentState:
    """
    Pre-LLM Incident Response Cache Lookup Node.
    Checks the persistent cache before calling query classification, model routing,
    or diagnosis. If a safe exact or semantic hit exists, populates state directly with 0 LLM calls.
    """
    from agents.incident_response_cache import IncidentResponseCache
    query = state.get("user_query", "")
    cache = IncidentResponseCache.get_instance()
    res = cache.lookup(query)

    state["cache_lookup_latency"] = res.lookup_latency_sec
    if res.hit and res.record:
        state["cache_hit"] = True
        state["cache_match_type"] = res.match_type
        state["cache_similarity"] = res.similarity
        state["cache_record_id"] = res.record.cache_id

        # Hydrate state from verified cached record
        state["root_cause"] = res.record.root_cause
        state["resolution_steps"] = res.record.resolution_steps
        state["resolution_status"] = "resolved"
        state["verified"] = True
        state["verification_notes"] = res.record.verification_notes or f"Cached verified resolution ({res.match_type} match)"
        state["escalated"] = False
        state["category"] = res.record.incident_category
        state["severity"] = res.record.severity
        state["selected_model"] = f"cache:{res.record.model_id}"
        state["selected_provider"] = "cache"
        state["calibrated_confidence"] = res.record.calibrated_confidence
        state["structured_diagnosis"] = res.record.structured_diagnosis
        if isinstance(res.record.structured_diagnosis, dict):
            state["affected_component"] = (
                res.record.structured_diagnosis.get("affected_service_or_component")
                or res.record.structured_diagnosis.get("affected_component")
                or res.record.affected_service
            )
        else:
            state["affected_component"] = res.record.affected_service

        if res.record.supporting_evidence:
            state["kb_results"] = res.record.supporting_evidence
            state["kb_context"] = "\n\n---\n\n".join(
                f"Source: {r.get('source', 'cache')}\n{r.get('text', '')[:500]}"
                for r in res.record.supporting_evidence
            )

        # Zero LLM generation cost, zero model calls
        state["actual_cost"] = 0.0
        state["estimated_cost"] = 0.0
        state["model_call_count"] = 0
        state["iteration_count"] = 0
    else:
        state["cache_hit"] = False
        state["cache_match_type"] = "miss"
        state["cache_similarity"] = 0.0
        state["cache_record_id"] = None

    return state


def _route_after_cache(state: AgentState) -> str:
    """
    Routes directly to close on cache hit (0 LLM calls),
    or to classify on cache miss.
    """
    if state.get("cache_hit"):
        return "close"
    return "classify"


def _close_ticket(state: AgentState) -> AgentState:
    """
    Final node for confirmed resolution.
    Validates confirmation, saves newly verified responses to persistent cache,
    and formats rich operational report.
    """
    # Strict confirmation guard: Do not claim success unless verification is confirmed
    if not state.get("verified"):
        return escalation_agent(state)

    state["escalated"] = False
    state["resolution_status"] = "resolved"

    act_cost = state.get("actual_cost", 0.0) or 0.0
    est_cost = state.get("estimated_cost", 0.0) or 0.0
    act_lat = round(time.time() - state.get("start_time", time.time()), 3) if state.get("start_time") else (state.get("actual_latency") or 0.0)
    state["actual_latency"] = act_lat

    rm = state.get("routing_metadata") or {}
    tier = rm.get("chosen_tier", "")
    model_name = rm.get("chosen_model", state.get("selected_model", ""))
    provider_name = (rm.get("provider") or state.get("selected_provider", "")).upper()
    conf = rm.get("retrieval_features", {}).get("retrieval_confidence", 0.0)
    calib = state.get("calibrated_confidence")
    calib_str = f" | Calibrated Correctness: {calib*100:.1f}%" if calib is not None else ""
    act = state.get("recommended_action", "ANSWER")

    history = state.get("action_history") or []
    history.append({
        "action": "ANSWER",
        "iteration": state.get("iteration_count", 0),
        "executed": True,
        "confirmed": True,
        "result_summary": "Resolution verified and accepted" if not state.get("cache_hit") else f"Cache hit ({state.get('cache_match_type')}) reused",
        "action_success": True
    })
    state["action_history"] = history

    if state.get("cache_hit"):
        match_type = (state.get("cache_match_type") or "exact").upper()
        sim = state.get("cache_similarity", 1.0)
        sim_str = f" (Cosine Similarity: {sim:.4f})" if match_type == "SEMANTIC" else ""
        cache_id = state.get("cache_record_id", "CACHE_HIT")
        lookup_lat = state.get("cache_lookup_latency", 0.0)

        routed_info = (
            f"\n\n⚡ **Incident Response Cache HIT [{match_type}{sim_str}]:** Record ID: `{cache_id}`\n"
            f"**Zero-Cost Bypass:** 0 LLM API calls | $0.00000 LLM Generation Cost | Cache Lookup Latency: {lookup_lat:.4f}s\n"
            f"**Originating Model:** {model_name}{calib_str}"
        )
    else:
        routed_info = (
            f"\n\n**Engineered Model:** {model_name} [{tier}] (Provider: {provider_name} | Retrieval Conf: {conf*100:.1f}%{calib_str} | Policy: {act})\n"
            f"**Budget & Performance:** Actual Cost: ${act_cost:.5f} (Est: ${est_cost:.5f}) | "
            f"Actual Latency: {act_lat:.2f}s | "
            f"Loops: {state.get('iteration_count', 0)} | Model Calls: {state.get('model_call_count', 0)}"
        )
        # Save newly verified response into persistent cache on cache miss
        try:
            from agents.incident_response_cache import IncidentResponseCache
            cid = IncidentResponseCache.get_instance().save_verified_response(state)
            if cid:
                state["cache_record_id"] = cid
        except Exception as e:
            import logging
            logging.getLogger("IncidentResponseCache").warning(f"Failed to cache verified response: {e}")

    diag = state.get("structured_diagnosis")
    diag_report = ""
    if diag:
        from agents.diagnostic_claims import format_structured_diagnosis_report
        diag_report = "\n\n" + format_structured_diagnosis_report(diag)

    state["final_answer"] = (
        "✅  RESOLVED\n\n"
        f"**Root Cause:** {state['root_cause']}\n\n"
        f"**Resolution Steps:**\n{state['resolution_steps']}\n\n"
        f"**Verification:** {state.get('verification_notes', 'Passed')}"
        + routed_info
        + diag_report
    )
    return state


def _route_after_verify(state: AgentState) -> str:
    """
    Cost-Aware Adaptive Conditional Edge (Phase 4).
    Dispatches between close, retrieve_more, switch_model, abstain, escalate.
    Enforces budget limits, latency SLAs, and loop iteration ceilings.
    """
    max_iter = state.get("max_iterations") if state.get("max_iterations") is not None else int(os.getenv("MAX_GRAPH_ITERATIONS", "3"))
    max_calls = state.get("max_model_calls") if state.get("max_model_calls") is not None else int(os.getenv("MAX_MODEL_CALLS", "6"))
    max_cost = float(os.getenv("MAX_COST_BUDGET", "0.0060"))
    max_lat = float(os.getenv("MAX_LATENCY_BUDGET", "15.0"))

    iter_count = state.get("iteration_count", 0) or 0
    call_count = state.get("model_call_count", 0) or 0
    actual_cost = state.get("actual_cost", 0.0) or 0.0
    actual_lat = round(time.time() - state.get("start_time", time.time()), 3) if state.get("start_time") else 0.0

    # 1. Budget and Loop Limit Guard (Prevent Infinite Loops)
    budget_exhausted = (
        iter_count >= max_iter or
        call_count >= max_calls or
        actual_cost >= max_cost or
        actual_lat >= max_lat
    )

    if budget_exhausted:
        if state.get("verified") and (state.get("calibrated_confidence", 0.0) or 0.0) >= 0.40:
            return "close"
        if state.get("verified") and state.get("resolution_status") == "resolved":
            return "close"
        missing = state.get("missing_evidence") or []
        if len(missing) >= 2 or not state.get("verified"):
            return "abstain"
        return "escalate"

    # 2. Critical Severity Guard
    if state.get("severity") == "L3":
        return "escalate"

    # 3. Verified Resolution Confirmation (Priority: Close & Cache verified answers)
    if state.get("verified") and state.get("resolution_status") == "resolved":
        calib = state.get("calibrated_confidence", 0.0) or 0.0
        act = state.get("recommended_action")
        if calib >= 0.40 or act in ("ANSWER", None):
            return "close"

    # 4. Action Dispatching
    act = state.get("recommended_action")

    if act == "ABSTAIN":
        return "abstain"

    if act == "HUMAN_REVIEW":
        return "escalate"

    if act == "RETRIEVE_MORE" and iter_count < max_iter:
        return "retrieve_more"

    if act == "SWITCH_MODEL" and iter_count < max_iter:
        return "switch_model"

    # 5. Verified Resolution Fallback
    if state.get("verified"):
        calib = state.get("calibrated_confidence", 0.0) or 0.0
        if calib >= 0.40:
            return "close"
        if iter_count < max_iter:
            return "switch_model"
        return "escalate"

    # 6. Unverified Resolution Fallback
    if iter_count < max_iter:
        rm = state.get("routing_metadata") or {}
        r_conf = rm.get("retrieval_features", {}).get("retrieval_confidence", 1.0)
        if r_conf < 0.55:
            return "retrieve_more"
        return "switch_model"

    return "escalate"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("cache_lookup",        _cache_lookup_node)
    graph.add_node("classify",            classifier_agent)
    graph.add_node("knowledge_search",    knowledge_agent)
    graph.add_node("evidence_assessment", _evidence_assessment_node)
    graph.add_node("router",              router_agent)
    graph.add_node("diagnose",            diagnostic_agent)
    graph.add_node("resolve",             resolution_agent)
    graph.add_node("verify",              verification_agent)
    graph.add_node("retrieve_more",       _retrieve_more_node)
    graph.add_node("switch_model",        _switch_model_node)
    graph.add_node("abstain",             _abstain_node)
    graph.add_node("escalate",            escalation_agent)
    graph.add_node("close",               _close_ticket)

    # Pre-LLM Cache Lookup Entry Point
    graph.set_entry_point("cache_lookup")
    graph.add_conditional_edges(
        "cache_lookup",
        _route_after_cache,
        {
            "close":    "close",
            "classify": "classify",
        }
    )

    # Linear flow on cache miss
    graph.add_edge("classify",            "knowledge_search")
    graph.add_edge("knowledge_search",    "evidence_assessment")
    graph.add_edge("evidence_assessment", "router")
    graph.add_edge("router",              "diagnose")
    graph.add_edge("diagnose",            "resolve")
    graph.add_edge("resolve",             "verify")

    # Conditional branch after verify
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "close":         "close",
            "retrieve_more": "retrieve_more",
            "switch_model":  "switch_model",
            "abstain":       "abstain",
            "escalate":      "escalate",
        },
    )

    # Loops: re-retrieval re-assesses evidence; switch_model re-diagnoses
    graph.add_edge("retrieve_more", "evidence_assessment")
    graph.add_edge("switch_model",  "diagnose")

    # Terminal nodes
    graph.add_edge("close",    END)
    graph.add_edge("abstain",  END)
    graph.add_edge("escalate", END)

    return graph.compile()

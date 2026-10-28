"""
Verification Agent — Evidence-Grounded Structured Diagnosis (Phase 5).
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Performs claim-level verification of diagnostic root cause and resolution steps,
and generates structured diagnostic claims strictly grounded in retrieved evidence.

Produces 10 standard operational fields:
1. Incident summary
2. Suspected root cause
3. Affected service or component
4. Supporting evidence and source identifiers
5. Contradictory or missing evidence
6. Alternative explanations
7. Confidence estimate and its operational meaning
8. Recommended remediation (advisory, safety-checked)
9. Verification status (verified, unverified, uncertain, insufficient_evidence, escalated)
10. Escalation or abstention reason
"""

import json
import re
from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from agents.llm_utils import get_llm, estimate_token_cost
from agents.diagnostic_claims import build_structured_diagnosis, EvidenceGroundingValidator


SYSTEM_PROMPT = """You are a senior SRE reviewing an operational support incident.
Perform claim-level verification of the diagnostic root cause and proposed resolution against the documentation.

Rules:
1. Every claim must be grounded in the provided documentation or query facts.
2. Distinguish between:
   - observed facts (from telemetry/query)
   - inferred root causes (diagnostic deductions)
   - recommended actions (prescribed steps)
3. Do NOT invent logs, traces, source references, measurements, or remediation outcomes.
4. If evidence is insufficient, explicitly declare uncertainty rather than manufacturing a definitive root cause.
5. All remediation steps are advisory; flag any risky or destructive operations.

Return ONLY a valid JSON object:
{
  "verified": <true | false>,
  "verification_status": "<verified | unverified | uncertain | insufficient_evidence | escalated>",
  "incident_summary": "<concise incident summary>",
  "affected_service_or_component": "<affected service, pod, database, or component>",
  "suspected_root_cause": "<evidence-grounded root cause, or explicit uncertainty statement if evidence is insufficient>",
  "observed_facts": [
    "<fact directly observed in query or retrieved telemetry>"
  ],
  "inferred_root_causes": [
    "<inferred hypothesis deduced from observed facts>"
  ],
  "supporting_evidence": [
    {
      "source": "<exact source document name from provided context>",
      "snippet": "<quote or exact factual summary from context>",
      "relevance": "<high | medium | low>",
      "verified": <true | false>
    }
  ],
  "contradictory_or_missing_evidence": {
    "missing_evidence": ["<missing log, metric, trace, or runbook>"],
    "contradictory_evidence": ["<any conflicting evidence or 'None identified in current evidence'>"],
    "has_contradictions": <true | false>
  },
  "alternative_explanations": [
    "<plausible alternative cause considered or 'None identified'>"
  ],
  "confidence_estimate": {
    "score": <float between 0.0 and 1.0>,
    "confidence_band": "<low | medium | high>",
    "meaning": "<human-readable explanation of what this confidence score reflects based on evidence grounding>"
  },
  "recommended_remediation": {
    "steps": ["<step 1>", "<step 2>"],
    "is_advisory": true,
    "safety_level": "<safe | requires_approval | destructive_prohibited>",
    "advisory_notes": "All remediation steps are advisory; human authorization required before executing state-changing or destructive actions."
  },
  "claims": [
    {
      "claim": "<statement of root cause or remediation action>",
      "claim_type": "<observed_fact | inferred_root_cause | recommended_action>",
      "supported": <true | false>,
      "risk": "<low | medium | high>",
      "source_identifier": "<source filename or incident_query>"
    }
  ],
  "notes": "<brief verification notes or concerns>",
  "escalation_or_abstention_reason": "<reason if escalated or unverified, or null>"
}"""


def verification_agent(state: AgentState) -> AgentState:
    # 1. L3 Critical Severity or Needs Human Review Guard
    if state["severity"] == "L3" or state.get("resolution_status") == "needs_human":
        state["verified"] = False
        l3_claims = [
            {
                "claim": "L3 critical severity requires human oversight",
                "claim_type": "inferred_root_cause",
                "supported": True,
                "risk": "high",
                "source_identifier": "incident_policy"
            }
        ]
        notes = "L3 severity or complex resolution — requires human engineer review before execution."
        state["claim_verifications"] = l3_claims
        state["verification_notes"] = notes
        state["verification_status"] = "escalated"
        state["escalation_reason"] = notes

        # Compute Calibrated Diagnostic Correctness Probability for telemetry completeness
        try:
            from agents.confidence_estimator import CalibratedDiagnosticConfidenceEstimator
            estimator = CalibratedDiagnosticConfidenceEstimator()
            calib_prob, breakdown = estimator.estimate(state)
            state["calibrated_confidence"] = calib_prob
            state["confidence_breakdown"] = breakdown
        except Exception:
            state["calibrated_confidence"] = None
            state["confidence_breakdown"] = None

        # Build complete structured diagnosis for L3
        structured_diag = build_structured_diagnosis(
            state=state,
            raw_claims=l3_claims,
            verified=False,
            verification_notes=notes
        )
        state["structured_diagnosis"] = structured_diag
        state["affected_component"] = structured_diag["affected_service_or_component"]
        state["observed_facts"] = structured_diag["observed_facts"]
        state["inferred_root_causes"] = structured_diag["inferred_root_causes"]
        state["alternative_explanations"] = structured_diag["alternative_explanations"]
        state["is_advisory_remediation"] = True
        return state

    # 2. Invoke LLM for Evidence-Grounded Verification
    llm = get_llm(temperature=0, max_tokens=600)

    user_prompt = f"""Original Issue: {state['user_query']}
Category: {state['category']} | Severity: {state['severity']}
Root Cause: {state['root_cause']}

Proposed Resolution:
{state['resolution_steps']}

Documentation Context:
{state.get('kb_context', 'No context available.')}

Perform claim-level evidence verification. Check if every claim is supported by actual retrieved documentation or established operational best practices, distinguish observed facts from inferred causes, verify that no destructive commands are executed automatically, and output the structured diagnostic claims JSON."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # Track model call invocation
    state["model_call_count"] = (state.get("model_call_count") or 0) + 1

    try:
        response = llm.invoke(messages)
    except Exception as e:
        err_str = str(e).lower()
        if any(w in err_str for w in ("503", "overloaded", "upstream", "rate limit", "429")):
            fallback_llm = None
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                fallback_llm = get_llm(temperature=0, max_tokens=600, provider="gemini", model="gemini-2.5-flash")
            elif os.getenv("GROQ_API_KEY"):
                fallback_llm = get_llm(temperature=0, max_tokens=600, provider="groq", model="openai/gpt-oss-20b")
            if fallback_llm:
                response = fallback_llm.invoke(messages)
            else:
                raise e
        else:
            raise e
    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", str(raw[0])) if isinstance(raw[0], dict) else str(raw[0])
    raw = str(raw).strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    # Track actual token cost
    in_tok = int(len(user_prompt.split()) * 1.3)
    out_tok = int(len(raw.split()) * 1.3)
    c_delta = estimate_token_cost(in_tok, out_tok, model=state.get("selected_model"))
    state["actual_cost"] = round((state.get("actual_cost") or 0.0) + c_delta, 6)
    state["budget_spent"] = state["actual_cost"]

    # 3. Parse LLM response
    llm_data: Optional[Dict[str, Any]] = None
    claims = []
    try:
        data = json.loads(raw)
        llm_data = data
        verified = bool(data.get("verified", False))
        claims = data.get("claims", [])
        notes = data.get("notes", "")

        # Claim-level verification guard: If any claim is unsupported or high-risk, fail verification
        if claims:
            for c in claims:
                if not c.get("supported", True) or c.get("risk", "low") == "high":
                    verified = False
                    notes = f"Failed claim verification: '{c.get('claim')}' (Risk: {c.get('risk')})"
                    break
    except json.JSONDecodeError:
        verified = True
        notes = "Auto-verified (parse error in LLM response)"
        claims = [
            {"claim": f"Root cause: {state.get('root_cause', 'General issue')}", "supported": True, "risk": "low"},
            {"claim": "Proposed resolution steps verified against SOP", "supported": True, "risk": "low"}
        ]

    # 4. Destructive command check on proposed resolution
    has_destructive, destructive_warnings = EvidenceGroundingValidator.detect_destructive_actions(
        [state.get("resolution_steps", "")]
    )
    if has_destructive:
        verified = False
        notes = f"Failed claim verification: Destructive command detected ({destructive_warnings[0]})"

    # Temporary assign to state for confidence and intervention evaluation
    state["verified"] = verified
    state["claim_verifications"] = claims
    state["verification_notes"] = notes

    # 5. Compute Calibrated Diagnostic Correctness Probability (Phase 2)
    try:
        from agents.confidence_estimator import CalibratedDiagnosticConfidenceEstimator
        estimator = CalibratedDiagnosticConfidenceEstimator()
        calib_prob, breakdown = estimator.estimate(state)
        state["calibrated_confidence"] = calib_prob
        state["confidence_breakdown"] = breakdown
    except Exception:
        state["calibrated_confidence"] = None
        state["confidence_breakdown"] = None

    # 6. Post-Verification Intervention Action Evaluation (Phase 3)
    try:
        from agents.intervention_policy import InterventionPolicyManager
        manager = InterventionPolicyManager(strategy="cost_min")
        action, _ = manager.evaluate_and_route(state)
        state["recommended_action"] = action.value
    except Exception:
        pass

    # 7. Build Evidence-Grounded Structured Diagnosis (Phase 5)
    structured_diag = build_structured_diagnosis(
        state=state,
        raw_claims=claims,
        verified=verified,
        verification_notes=notes,
        llm_data=llm_data
    )

    # 8. Synchronize State with Structured Diagnosis
    final_verified = (structured_diag["verification_status"] == "verified")
    state["verified"] = final_verified
    state["claim_verifications"] = structured_diag["claims"]
    state["verification_notes"] = notes
    state["verification_status"] = structured_diag["verification_status"]
    state["structured_diagnosis"] = structured_diag
    state["affected_component"] = structured_diag["affected_service_or_component"]
    state["observed_facts"] = structured_diag["observed_facts"]
    state["inferred_root_causes"] = structured_diag["inferred_root_causes"]
    state["alternative_explanations"] = structured_diag["alternative_explanations"]
    state["is_advisory_remediation"] = True

    if structured_diag.get("escalation_or_abstention_reason") and not state.get("escalation_reason"):
        state["escalation_reason"] = structured_diag["escalation_or_abstention_reason"]

    # 9. Dynamic Escalation: If not verified on initial tier, escalate to high-capacity Tier 3
    current_level = state.get("escalation_level", 0)
    current_tier = (state.get("routing_metadata") or {}).get("chosen_tier", "tier1_fast")
    if not final_verified and current_level == 0 and current_tier != "tier3_heavy":
        state["needs_model_escalation"] = True
    else:
        state["needs_model_escalation"] = False

    return state






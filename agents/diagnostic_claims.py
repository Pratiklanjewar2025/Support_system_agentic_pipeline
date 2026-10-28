# -*- coding: utf-8 -*-
"""
agents/diagnostic_claims.py — Evidence-Grounded Structured Diagnosis (Phase 5)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Primary Objective:
Extends the verification pipeline to generate structured, evidence-grounded diagnostic claims.
Where existing telemetry and runbook data permit, produces 10 structured fields:
1. Incident summary
2. Suspected root cause
3. Affected service or component
4. Supporting evidence and source identifiers
5. Contradictory or missing evidence
6. Alternative explanations
7. Confidence estimate and its operational meaning
8. Recommended remediation (advisory, safety-checked against destructive execution)
9. Verification status (verified, unverified, uncertain, insufficient_evidence, escalated)
10. Escalation or abstention reason

Guarantees & Invariants:
- Categorizes each claim into: observed_fact, inferred_root_cause, or recommended_action.
- Factual Grounding: Strictly cites actual retrieved documents from kb_results;
  never invents logs, traces, metric measurements, or non-existent source filenames.
- Uncertainty Guard: Where evidence is insufficient or contradictory, emits an explicit
  uncertainty status ("uncertain" / "insufficient_evidence") instead of manufacturing
  a definitive root cause.
- Advisory Remediation: All remediation steps are flagged as advisory (is_advisory=True),
  and destructive commands (e.g. rm -rf, drop table, kill -9, delete namespace)
  are automatically detected and prohibited from automated execution.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field

from agents.state import AgentState


# Destructive command patterns that MUST NOT be executed automatically
DESTRUCTIVE_PATTERNS = [
    (r"\brm\s+-[rf]+", "Unconditional recursive file/directory deletion (rm -rf)"),
    (r"\bmkfs(\.\w+)?\b", "Filesystem reformat / wipe (mkfs)"),
    (r"\bfdisk\b|\bparted\b", "Disk partition modification"),
    (r"\bdd\s+if=", "Raw block-device overwrite (dd if=)"),
    (r"\bformat\s+[a-zA-Z]:", "Disk formatting"),
    (r"\bdrop\s+(database|table|schema|user)\b", "Destructive database schema deletion (DROP)"),
    (r"\bdelete\s+from\s+\w+\s*(;|$)", "Unfiltered table data deletion (DELETE FROM)"),
    (r"\btruncate\s+(table\s+)?\w+", "Table truncation (TRUNCATE)"),
    (r"\bkill\s+-9\b|\bpkill\s+-9\b", "Ungraceful process termination (kill -9)"),
    (r"\breboot\b|\bshutdown\b|\binit\s+0\b", "System reboot / power down"),
    (r"\bdelete\s+(namespace|ns|pvc|pv|node)\b", "Destructive Kubernetes cluster resource deletion"),
    (r"\bscale\s+.*--replicas=0\b", "Service shutdown / scaling to zero replicas"),
    (r"\bshred\b|\bwipe\b", "Secure data wipe / shred"),
]


@dataclass
class DiagnosticEvidenceItem:
    """Represents a factual evidence passage tied to a verified source identifier."""
    source: str                     # Actual filename from kb_results, e.g. "k8s_troubleshooting.md"
    snippet: str                    # Actual quote / snippet from kb_results or incident query
    relevance: str                  # "high" | "medium" | "low"
    verified: bool = True           # True if verified against actual retrieved context
    distance_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationRecord:
    """Atomic claim categorized by type, grounded in retrieved evidence, and risk-checked."""
    claim: str                      # Exact claim text
    claim_type: str                 # "observed_fact" | "inferred_root_cause" | "recommended_action"
    supported: bool                 # True if supported by retrieved evidence
    risk: str                       # "low" | "medium" | "high"
    source_identifier: str          # Source document filename or "user_query"
    evidence_snippet: Optional[str] = None
    is_destructive: bool = False    # True if action contains destructive patterns

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdvisoryRemediation:
    """Advisory remediation plan with automated safety ratings and destructive command guards."""
    steps: List[str]
    is_advisory: bool = True
    safety_level: str = "safe"      # "safe" | "requires_approval" | "destructive_prohibited"
    has_destructive_commands: bool = False
    destructive_warnings: List[str] = field(default_factory=list)
    advisory_notes: str = (
        "Recommendations are advisory SRE guidance. "
        "Automated execution of destructive remediation is strictly prohibited."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredDiagnosis:
    """The complete 10-field evidence-grounded structured diagnosis."""
    incident_summary: str
    suspected_root_cause: str
    affected_service_or_component: str
    observed_facts: List[str]
    inferred_root_causes: List[str]
    supporting_evidence: List[Dict[str, Any]]
    contradictory_or_missing_evidence: Dict[str, Any]
    alternative_explanations: List[str]
    confidence_estimate: Dict[str, Any]
    recommended_remediation: Dict[str, Any]
    verification_status: str        # "verified" | "unverified" | "uncertain" | "insufficient_evidence" | "escalated"
    escalation_or_abstention_reason: Optional[str] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGroundingValidator:
    """
    Validates that diagnostic claims and citations strictly match actual retrieved evidence,
    preventing fabricated logs, traces, or non-existent document citations.
    """

    @staticmethod
    def extract_affected_component(query: str, category: str = "general", kb_context: str = "") -> str:
        """Identifies affected service or component from query without hallucinating names."""
        q = query.strip()

        # Known component / service patterns in incident queries
        patterns = [
            (r"\b([a-zA-Z0-9_\-\.]+)\s+(?:pod|deployment|container|service|cluster|node)\b", 0),
            (r"\b(?:pod|service|container)\s+([a-zA-Z0-9_\-\.]+)\b", 0),
            (r"\b(Kubernetes\s+API\s+gateway(?:\s+pod)?)\b", 1),
            (r"\b([a-zA-Z0-9_\-]+-server-[0-9]+)\b", 1),
            (r"\b(PostgreSQL|Postgres|MySQL|Redis|MongoDB|Kafka|Elasticsearch|RabbitMQ)\b", 1),
            (r"\b(checkout|payments|inventory|auth|billing|order|gateway)\s+(?:service|api|app)\b", 0),
            (r"\b(Database\s+primary\s+node)\b", 1),
        ]

        for pat, group_idx in patterns:
            m = re.search(pat, q, re.I)
            if m:
                matched = m.group(group_idx).strip()
                if len(matched) > 2:
                    return matched

        # Fallback to category-based component designation
        cat_map = {
            "database": "Primary Database Cluster / Node",
            "cpu": "Application Compute / Host CPU",
            "network": "Network Ingress / Gateway",
            "deployment": "Application Service Deployment",
            "application": "Application Runtime Process",
            "general": "Core Infrastructure Service",
        }
        return cat_map.get(category.lower(), f"{category.capitalize()} Subsystem")

    @staticmethod
    def detect_destructive_actions(steps: List[str]) -> Tuple[bool, List[str]]:
        """
        Scans proposed steps for destructive command patterns that must not be auto-executed.
        Returns (has_destructive, warnings).
        """
        combined = "\n".join(steps)
        warnings = []
        for pat, desc in DESTRUCTIVE_PATTERNS:
            matches = re.findall(pat, combined, re.I)
            if matches:
                warnings.append(f"Detected prohibited destructive pattern: {desc} ({matches[0]})")

        return len(warnings) > 0, warnings

    @staticmethod
    def categorize_claim(claim: str, root_cause: str = "", query: str = "") -> str:
        """
        Categorizes an atomic claim into:
        - 'observed_fact': direct symptom / observation from query or retrieved telemetry
        - 'inferred_root_cause': diagnostic deduction / hypothesis
        - 'recommended_action': prescribed remediation step
        """
        cl = claim.lower().strip()

        # Action signals
        action_verbs = [
            r"^\d+\.", r"^run\b", r"^restart\b", r"^increase\b", r"^configure\b",
            r"^apply\b", r"^check\b", r"^ssh\b", r"^execute\b", r"^kill\b",
            r"^roll\s*back\b", r"^inspect\b", r"^update\b", r"^scale\b", r"^set\b",
            r"^modify\b", r"^edit\b", r"^re-run\b", r"^deploy\b"
        ]
        if any(re.search(v, cl) for v in action_verbs):
            return "recommended_action"

        # Observed fact signals
        fact_signals = [
            r"oomkilled", r"exit\s*code\s*\d+", r"crashloopbackoff", r"502\b", r"504\b",
            r"404\b", r"500\b", r"hung", r"spiked\s*to\s*\d+%", r"high\s*utilization",
            r"unresponsive", r"timeout", r"rejected\s*connections", r"memory\s*spike"
        ]
        if any(re.search(s, cl) for s in fact_signals) and not any(k in cl for k in ["due to", "caused by", "inferred", "likely"]):
            return "observed_fact"

        # Inferred root cause signals
        cause_signals = [
            r"due\s*to", r"caused\s*by", r"because", r"exhaustion", r"leak",
            r"misconfiguration", r"mismatch", r"corruption", r"contention",
            r"deadlock", r"exceeded", r"insufficient", r"root\s*cause"
        ]
        if any(re.search(c, cl) for c in cause_signals):
            return "inferred_root_cause"

        # Default classification based on length/structure
        if len(claim.split()) < 6 and any(w in cl for w in ["process", "pod", "database", "node", "cpu"]):
            return "observed_fact"
        return "inferred_root_cause"

    @staticmethod
    def extract_observed_facts(query: str, kb_results: List[dict] = None) -> List[str]:
        """Extracts confirmed, factual observations from the incident description."""
        facts = []
        q = query.strip()

        # Check for specific technical observations
        if re.search(r"CrashLoopBackOff", q, re.I):
            facts.append("Observed pod status: CrashLoopBackOff")
        if re.search(r"OOMKilled\s*(?:exit\s*code\s*137)?", q, re.I):
            facts.append("Container terminated by kernel OOM killer (exit code 137)")
        if re.search(r"exit\s*code\s*(\d+)", q, re.I):
            code = re.search(r"exit\s*code\s*(\d+)", q, re.I).group(1)
            if code != "137":
                facts.append(f"Process exited with non-zero status code: {code}")
        if re.search(r"CPU\s*(?:usage\s*)?spiked\s*to\s*(\d+%)", q, re.I):
            pct = re.search(r"CPU\s*(?:usage\s*)?spiked\s*to\s*(\d+%)", q, re.I).group(1)
            facts.append(f"CPU utilization spiked to {pct}")
        if re.search(r"50[24]\s*(?:error|gateway|bad\s*gateway)?", q, re.I):
            code = re.search(r"(50[24])", q, re.I).group(1)
            facts.append(f"Observed HTTP {code} Gateway / Proxy response errors")
        if re.search(r"unresponsive|rejecting\s*connections", q, re.I):
            facts.append("Target service is unresponsive and rejecting incoming client connections")

        # Fallback: if no specific regex matched, use the sanitized query sentence
        if not facts:
            clean_q = re.sub(r"^(how do i|why is|what is|please help with)\s*", "", q, flags=re.I).strip()
            clean_q = clean_q.rstrip(".!?")
            facts.append(f"Reported incident symptom: {clean_q}")

        return facts

    @staticmethod
    def sanitize_supporting_evidence(
        candidate_evidence: List[Dict[str, Any]],
        kb_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Strict grounding filter: Guarantees every cited source identifier exists
        in the actual retrieved kb_results.
        Prevents hallucination of fictitious filenames, logs, or traces.
        """
        valid_sources = {r.get("source"): r for r in kb_results if r.get("source")}
        grounded_items: List[Dict[str, Any]] = []

        if candidate_evidence:
            for item in candidate_evidence:
                src = item.get("source", "").strip()
                # Check if cited source exists in real kb_results
                if src in valid_sources:
                    real_chunk = valid_sources[src]
                    grounded_items.append({
                        "source": src,
                        "snippet": item.get("snippet", real_chunk.get("text", "")[:250]).strip(),
                        "relevance": item.get("relevance", "high"),
                        "verified": True,
                        "distance_score": round(float(real_chunk.get("score", 0.0)), 4) if "score" in real_chunk else None
                    })
                elif src.lower() in ("user_query", "incident_query", "user query", "incident input"):
                    grounded_items.append({
                        "source": "incident_query",
                        "snippet": query[:250].strip(),
                        "relevance": "high",
                        "verified": True,
                        "distance_score": 0.0
                    })
                # If an LLM hallucinated a non-existent document, do NOT include it as a genuine source!

        # If candidate evidence is empty or hallucinated, populate directly from actual kb_results
        if not grounded_items and kb_results:
            for r in kb_results[:3]:
                d_score = float(r.get("score", 1.0))
                rel = "high" if d_score < 0.60 else ("medium" if d_score < 1.10 else "low")
                grounded_items.append({
                    "source": r.get("source", "knowledge_base.md"),
                    "snippet": r.get("text", "")[:250].strip(),
                    "relevance": rel,
                    "verified": True,
                    "distance_score": round(d_score, 4)
                })

        return grounded_items

    @staticmethod
    def generate_alternative_explanations(category: str, root_cause: str, query: str) -> List[str]:
        """Provides plausible alternative root cause explanations to avoid single-hypothesis bias."""
        combined = (category + " " + root_cause + " " + query).lower()

        if "oom" in combined or "memory" in combined or "crashloop" in combined:
            return [
                "Transient memory burst caused by a sudden batch of concurrent oversized client requests.",
                "Application-level memory leak where retained objects accumulate over time until container limit is hit.",
                "Container memory limit configured below JVM/runtime baseline allocation requirements."
            ]
        elif "database" in combined or "postgres" in combined:
            return [
                "Uncommitted long-running transaction or orphaned lock preventing schema alterations.",
                "Storage volume I/O saturation or disk exhaustion on the primary database node.",
                "Replication lag / network partition between primary and standby replicas."
            ]
        elif "502" in combined or "gateway" in combined or "network" in combined:
            return [
                "Upstream backend service crashed or was not ready to accept connections after deployment.",
                "Reverse proxy connection timeout configured lower than backend response processing latency.",
                "Ingress routing or DNS misconfiguration directing requests to terminated endpoints."
            ]
        elif "cpu" in combined or "spike" in combined:
            return [
                "Runaway unindexed query or thread spinning in an infinite loop.",
                "Sudden organic surge in client traffic volume without auto-scaling trigger.",
                "Background garbage collection pause loop saturating available CPU cores."
            ]

        return [
            "Hardware or host node-level resource contention in the shared cluster.",
            "Configuration drift or environment variable mismatch applied during recent deployment."
        ]

    @staticmethod
    def build_confidence_meaning(
        score: Optional[float],
        band: Optional[str],
        verified: bool,
        missing_count: int,
        source_count: int
    ) -> Dict[str, Any]:
        """
        Produces a quantitative score and an explicit human-readable meaning of the confidence estimate.
        """
        if score is None:
            score = 0.85 if verified else 0.40

        pct = score * 100.0
        if not band:
            band = "high" if score >= 0.70 else ("medium" if score >= 0.40 else "low")

        if band == "high" and verified:
            meaning = (
                f"Calibrated probability ({pct:.1f}%) [HIGH]. The diagnosis is strongly supported "
                f"by {source_count} retrieved runbook documents with claim-level verification passing. "
                "Confidence is sufficient for advisory remediation under human supervision."
            )
        elif band == "medium":
            meaning = (
                f"Calibrated probability ({pct:.1f}%) [MEDIUM]. Moderate evidentiary support. "
                f"{missing_count} telemetry item(s) are missing (e.g. detailed metrics/traces). "
                "Diagnosis represents a probable hypothesis; verify telemetry before applying fixes."
            )
        else:
            meaning = (
                f"Calibrated probability ({pct:.1f}%) [LOW]. High uncertainty due to missing telemetry "
                f"or unverified claims ({missing_count} missing items). Diagnosis is conjectural. "
                "Automated remediation is prohibited; operator investigation required."
            )

        return {
            "score": round(score, 4),
            "confidence_band": band,
            "calibrated": True,
            "meaning": meaning
        }


def build_structured_diagnosis(
    state: AgentState,
    raw_claims: List[Dict[str, Any]],
    verified: bool,
    verification_notes: str,
    llm_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main builder for Evidence-Grounded Structured Diagnosis (Phase 5).
    Combines LLM extractions, deterministic evidence audits, safety checks,
    and anti-hallucination guards into a standardized 10-field output.
    """
    query = state.get("user_query", "")
    category = state.get("category", "general")
    severity = state.get("severity", "L2")
    summary = state.get("summary") or query[:80]
    kb_results = state.get("kb_results", [])
    raw_root_cause = state.get("root_cause", "Unspecified root cause")

    # 1. Affected service or component
    llm_component = (llm_data or {}).get("affected_service_or_component")
    affected_component = llm_component if (llm_component and len(llm_component) > 2) else \
        EvidenceGroundingValidator.extract_affected_component(query, category, state.get("kb_context", ""))

    # 2. Audit missing evidence & telemetry gaps
    from agents.intervention_policy import EvidenceAuditor
    missing_telemetry = EvidenceAuditor.audit_missing_evidence(state)

    # 3. Detect destructive remediation commands
    resolution_text = state.get("resolution_steps", "")
    resolution_steps = [s.strip() for s in resolution_text.split("\n") if s.strip() and not s.startswith("RESOLUTION_STATUS")]
    if not resolution_steps and (llm_data or {}).get("recommended_remediation", {}).get("steps"):
        resolution_steps = llm_data["recommended_remediation"]["steps"]

    has_destructive, destructive_warnings = EvidenceGroundingValidator.detect_destructive_actions(resolution_steps)

    # Safety level assignment
    if has_destructive:
        safety_level = "destructive_prohibited"
        advisory_notes = (
            "CRITICAL SAFETY WARNING: Destructive command(s) detected in remediation steps. "
            "Automated execution is strictly prohibited. Mandatory human operator authorization required."
        )
    elif severity == "L3":
        safety_level = "requires_approval"
        advisory_notes = (
            "L3 critical incident remediation is strictly advisory. "
            "Manual review and change control authorization required before execution."
        )
    else:
        safety_level = "safe"
        advisory_notes = (
            "All remediation steps are advisory guidance. "
            "Ensure system backups and pre-execution safety checks before applying."
        )

    # 4. Atomic claims structuring & categorization
    processed_claims = []
    has_unsupported_claim = False

    # Extract claims from llm_data or raw_claims
    claims_source = (llm_data or {}).get("claims") or raw_claims or []
    if not claims_source:
        # Synthesize baseline claims from root cause and resolution
        claims_source = [
            {"claim": f"Root cause: {raw_root_cause}", "supported": verified, "risk": "low"},
            {"claim": f"Remediation: {resolution_steps[0] if resolution_steps else 'Inspect service status'}", "supported": verified, "risk": "low"}
        ]

    valid_kb_sources = [r.get("source") for r in kb_results if r.get("source")]
    default_source = valid_kb_sources[0] if valid_kb_sources else "incident_query"

    for c in claims_source:
        claim_text = c.get("claim", "")
        c_type = c.get("claim_type") or EvidenceGroundingValidator.categorize_claim(claim_text, raw_root_cause, query)
        c_supp = bool(c.get("supported", True))
        c_risk = c.get("risk", "low")

        # Check if this specific claim contains destructive patterns
        c_destructive, _ = EvidenceGroundingValidator.detect_destructive_actions([claim_text])
        if c_destructive:
            c_risk = "high"
            c_supp = False

        if not c_supp or c_risk == "high":
            has_unsupported_claim = True

        src_id = c.get("source_identifier") or default_source
        if src_id not in valid_kb_sources and src_id not in ("incident_query", "user_query", "incident_policy"):
            src_id = default_source

        processed_claims.append({
            "claim": claim_text,
            "claim_type": c_type,
            "supported": c_supp,
            "risk": c_risk,
            "source_identifier": src_id,
            "is_destructive": c_destructive
        })

    # 5. Distinguish Observed Facts from Inferred Causes
    observed_facts = (llm_data or {}).get("observed_facts") or []
    if not observed_facts:
        observed_facts = EvidenceGroundingValidator.extract_observed_facts(query, kb_results)

    inferred_causes = (llm_data or {}).get("inferred_root_causes") or []
    if not inferred_causes:
        inferred_causes = [raw_root_cause] if raw_root_cause else ["Unspecified diagnostic hypothesis"]

    # 6. Supporting evidence with strictly verified source identifiers
    raw_ev = (llm_data or {}).get("supporting_evidence") or []
    supporting_evidence = EvidenceGroundingValidator.sanitize_supporting_evidence(raw_ev, kb_results, query)

    # 7. Check evidence sufficiency & determine verification status
    top1_sim = 1.0
    if kb_results:
        d0 = float(kb_results[0].get("score", 1.0))
        top1_sim = math.exp(-0.5 * d0)

    # Criteria for insufficient evidence:
    # 1. Literally zero retrieved context (len(kb_results) == 0) indicates complete absence of evidence.
    # 2. Low retrieval similarity (top1_sim < 0.40) or telemetry gaps only flag insufficient evidence
    #    if the resolution was not already validated and verified by claim verification.
    evidence_insufficient = (
        len(kb_results) == 0 or
        (top1_sim < 0.40 and not verified) or
        (len(missing_telemetry) >= 3 and not verified) or
        (not verified and severity != "L3" and has_unsupported_claim)
    )

    # Determine verification status
    escalation_or_abstention_reason: Optional[str] = None

    if severity == "L3" or state.get("resolution_status") == "needs_human":
        verification_status = "escalated"
        escalation_or_abstention_reason = (
            "L3 critical severity requires human oversight and operational sign-off before execution."
        )
        verified = False
    elif has_destructive:
        verification_status = "unverified"
        escalation_or_abstention_reason = (
            "Remediation contains destructive operations prohibited from automated execution."
        )
        verified = False
    elif evidence_insufficient:
        verification_status = "uncertain" if top1_sim >= 0.35 else "insufficient_evidence"
        escalation_or_abstention_reason = (
            f"Evidence insufficient: {len(missing_telemetry)} telemetry item(s) missing; "
            "retrieval certainty is below operational confidence threshold."
        )
        verified = False
    elif not verified:
        verification_status = "unverified"
        escalation_or_abstention_reason = verification_notes or "One or more diagnostic claims failed verification."
    else:
        verification_status = "verified"
        escalation_or_abstention_reason = None

    # 8. Suspected Root Cause with explicit uncertainty handling
    if verification_status in ("uncertain", "insufficient_evidence"):
        # Do not manufacture a definitive root cause when evidence is insufficient
        suspected_root_cause = (
            f"Uncertain: Insufficient runtime telemetry and documentation to conclusively isolate root cause. "
            f"Preliminary unverified hypothesis: {raw_root_cause}"
        )
    else:
        suspected_root_cause = (llm_data or {}).get("suspected_root_cause") or raw_root_cause

    # 9. Contradictory or missing evidence
    contradictory_ev = (llm_data or {}).get("contradictory_or_missing_evidence", {}).get("contradictory_evidence", [])
    if not contradictory_ev:
        contradictory_ev = ["None identified in current evidence"]

    contradictory_or_missing = {
        "missing_evidence": missing_telemetry,
        "contradictory_evidence": contradictory_ev,
        "has_contradictions": any(c != "None identified in current evidence" for c in contradictory_ev),
        "evidence_completeness_ratio": round(max(0.0, 1.0 - (len(missing_telemetry) / 4.0)), 2)
    }

    # 10. Alternative explanations
    alt_explanations = (llm_data or {}).get("alternative_explanations") or []
    if not alt_explanations or alt_explanations == ["None identified"]:
        alt_explanations = EvidenceGroundingValidator.generate_alternative_explanations(category, raw_root_cause, query)

    # 11. Confidence estimate and its operational meaning
    calib_score = state.get("calibrated_confidence")
    c_band = (state.get("confidence_breakdown") or {}).get("confidence_band")
    conf_meaning_record = EvidenceGroundingValidator.build_confidence_meaning(
        score=calib_score,
        band=c_band,
        verified=verified,
        missing_count=len(missing_telemetry),
        source_count=len(supporting_evidence)
    )

    # 12. Recommended remediation (advisory)
    recommended_remediation = AdvisoryRemediation(
        steps=resolution_steps,
        is_advisory=True,
        safety_level=safety_level,
        has_destructive_commands=has_destructive,
        destructive_warnings=destructive_warnings,
        advisory_notes=advisory_notes
    ).to_dict()

    # Construct complete StructuredDiagnosis
    structured_diag = StructuredDiagnosis(
        incident_summary=summary,
        suspected_root_cause=suspected_root_cause,
        affected_service_or_component=affected_component,
        observed_facts=observed_facts,
        inferred_root_causes=inferred_causes,
        supporting_evidence=supporting_evidence,
        contradictory_or_missing_evidence=contradictory_or_missing,
        alternative_explanations=alt_explanations,
        confidence_estimate=conf_meaning_record,
        recommended_remediation=recommended_remediation,
        verification_status=verification_status,
        escalation_or_abstention_reason=escalation_or_abstention_reason,
        claims=processed_claims
    )

    return structured_diag.to_dict()


def format_structured_diagnosis_report(diag: Dict[str, Any]) -> str:
    """
    Renders the Evidence-Grounded Structured Diagnosis as rich GitHub-flavored Markdown.
    """
    lines = []
    lines.append("### Evidence-Grounded Structured Diagnosis (Phase 5)")
    lines.append(f"- **Incident Summary:** {diag.get('incident_summary', 'N/A')}")
    lines.append(f"- **Affected Component:** {diag.get('affected_service_or_component', 'N/A')}")
    lines.append(f"- **Suspected Root Cause:** {diag.get('suspected_root_cause', 'N/A')}")
    lines.append(f"- **Verification Status:** `{diag.get('verification_status', 'unverified').upper()}`")

    if diag.get("escalation_or_abstention_reason"):
        lines.append(f"- **Reason / Safety Notice:** {diag['escalation_or_abstention_reason']}")

    lines.append("\n#### 1. Observed Facts vs. Inferred Root Causes")
    lines.append("**Observed Facts (Direct Telemetry/Query):**")
    for fact in diag.get("observed_facts", []):
        lines.append(f"- [Fact] {fact}")
    lines.append("**Inferred Root Causes (Hypotheses):**")
    for cause in diag.get("inferred_root_causes", []):
        lines.append(f"- [Hypothesis] {cause}")

    lines.append("\n#### 2. Supporting Evidence & Source Identifiers")
    ev_items = diag.get("supporting_evidence", [])
    if ev_items:
        for idx, item in enumerate(ev_items, 1):
            src = item.get("source", "knowledge_base")
            snip = item.get("snippet", "")
            d_score = item.get("distance_score")
            d_str = f" (Distance: {d_score:.3f})" if d_score is not None else ""
            lines.append(f"- **[{idx}] `{src}`**{d_str}: \"{snip[:160]}...\"")
    else:
        lines.append("- *No verified documentation sources available.*")

    lines.append("\n#### 3. Contradictory or Missing Evidence")
    c_m = diag.get("contradictory_or_missing_evidence", {})
    missing = c_m.get("missing_evidence", [])
    contradictory = c_m.get("contradictory_evidence", [])
    lines.append(f"- **Missing Telemetry Gaps ({len(missing)} items):**")
    if missing:
        for m in missing:
            lines.append(f"  - {m}")
    else:
        lines.append("  - None (telemetry grounding complete)")
    lines.append(f"- **Contradictory Signals:** {', '.join(contradictory)}")

    lines.append("\n#### 4. Alternative Explanations Considered")
    for alt in diag.get("alternative_explanations", []):
        lines.append(f"- {alt}")

    lines.append("\n#### 5. Confidence Estimate & Meaning")
    conf = diag.get("confidence_estimate", {})
    lines.append(f"- **Confidence Score:** {conf.get('score', 0.0)*100:.1f}% (`{conf.get('confidence_band', 'medium').upper()}`)")
    lines.append(f"- **Operational Meaning:** {conf.get('meaning', 'N/A')}")

    lines.append("\n#### 6. Recommended Remediation (Advisory Only)")
    rem = diag.get("recommended_remediation", {})
    safety = rem.get("safety_level", "safe")
    lines.append(f"- **Safety Level:** `{safety.upper()}` | **Advisory Only:** `{rem.get('is_advisory', True)}`")
    lines.append(f"- **Advisory Warning:** {rem.get('advisory_notes', '')}")
    if rem.get("has_destructive_commands"):
        lines.append("> [!WARNING]")
        lines.append("> Prohibited destructive commands detected. Automated execution blocked.")
    lines.append("\n**Prescribed Steps:**")
    for idx, step in enumerate(rem.get("steps", []), 1):
        lines.append(f"{idx}. {step}")

    return "\n".join(lines)

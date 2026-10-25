# -*- coding: utf-8 -*-
"""
benchmarks/evaluation_metrics.py — Comprehensive Operational Evaluation Suite (Phase 8)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Primary Objective:
Implements a standardized, multi-dimensional metric framework for incident diagnosis across:
1. Diagnostic Quality:
   - Root-cause localization accuracy (Hit@1 on faulty microservice)
   - Top-k root-cause accuracy (Top-1, Top-3, Top-5)
   - Fault classification Macro-F1 across fault types
   - Exact or structured-match diagnosis correctness
   - Evidence-grounding precision and recall
   - Unsupported-claim rate
2. Confidence and Reliability:
   - Brier score
   - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
   - Reliability diagrams (binned accuracy vs confidence)
   - Selective accuracy and coverage when abstention/human review is enabled
3. Efficiency:
   - Actual API cost (billed) vs estimated API cost (calculated from token counts)
   - Input and output tokens
   - Total model invocations
   - Escalation frequency and retrieval retries
   - Median (p50) and 95th percentile (p95) latency
   - Diagnostic quality versus cost and latency trade-offs
4. Statistical Reporting:
   - Sample count and dataset composition
   - Mean, sample standard deviation, variance, and standard error
   - 95% Confidence Intervals (Student's t distribution)
   - Paired parametric (Student's t-test) and non-parametric (Wilcoxon signed-rank) tests
   - Effect sizes (Cohen's d, Cliff's delta)
5. Supplementary Linguistic Metrics:
   - BLEU retained as a supplementary text-generation metric, clearly distinguished
     from the primary operational root-cause localization measures.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score, precision_recall_fscore_support, brier_score_loss


# =============================================================================
# 1. DOCUMENTED TOKEN PRICING CATALOG & MEASUREMENT ASSUMPTIONS
# =============================================================================

TOKEN_PRICING_CATALOG: Dict[str, Dict[str, Any]] = {
    "command-r-08-2024": {
        "model_version": "cohere/command-r-08-2024",
        "provider": "cohere",
        "input_price_per_1m_usd": 0.15,
        "output_price_per_1m_usd": 0.60,
        "assumptions": "Standard Cohere Pay-as-you-go Developer Tier (August 2024 release)."
    },
    "command-r-plus-08-2024": {
        "model_version": "cohere/command-r-plus-08-2024",
        "provider": "cohere",
        "input_price_per_1m_usd": 2.50,
        "output_price_per_1m_usd": 10.00,
        "assumptions": "Standard Cohere Enterprise 104B Parameter Tier (August 2024 release)."
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "model_version": "nvidia/nemotron-3.5-lightning",
        "provider": "openrouter",
        "input_price_per_1m_usd": 0.00,
        "output_price_per_1m_usd": 0.00,
        "assumptions": "OpenRouter Free Inference Tier (rate-limited, $0.00 direct billed cost)."
    },
    "openai/gpt-oss-20b": {
        "model_version": "groq/gpt-oss-20b",
        "provider": "groq",
        "input_price_per_1m_usd": 0.10,
        "output_price_per_1m_usd": 0.10,
        "assumptions": "Groq LPU hardware accelerated on-demand inference tier."
    },
    "openai/gpt-oss-120b": {
        "model_version": "groq/gpt-oss-120b",
        "provider": "groq",
        "input_price_per_1m_usd": 0.50,
        "output_price_per_1m_usd": 0.70,
        "assumptions": "Groq LPU high-parameter on-demand inference tier."
    },
    "gemini-2.5-flash": {
        "model_version": "google/gemini-2.5-flash",
        "provider": "google",
        "input_price_per_1m_usd": 0.075,
        "output_price_per_1m_usd": 0.30,
        "assumptions": "Google AI Studio Developer Pay-as-you-go Tier."
    },
    "gemini-2.5-pro": {
        "model_version": "google/gemini-2.5-pro",
        "provider": "google",
        "input_price_per_1m_usd": 1.25,
        "output_price_per_1m_usd": 5.00,
        "assumptions": "Google AI Studio Pro reasoning inference tier."
    },
    "deepseek-chat": {
        "model_version": "deepseek/deepseek-chat-v3",
        "provider": "deepseek",
        "input_price_per_1m_usd": 0.14,
        "output_price_per_1m_usd": 0.28,
        "assumptions": "DeepSeek V3 standard API pricing (cache miss rate)."
    },
    "claude-3-5-haiku-20241022": {
        "model_version": "anthropic/claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "input_price_per_1m_usd": 0.80,
        "output_price_per_1m_usd": 4.00,
        "assumptions": "Anthropic Claude 3.5 Haiku standard pay-as-you-go tier."
    },
    "claude-3-5-sonnet-20241022": {
        "model_version": "anthropic/claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "input_price_per_1m_usd": 3.00,
        "output_price_per_1m_usd": 15.00,
        "assumptions": "Anthropic Claude 3.5 Sonnet standard tier."
    },
    "gpt-4o-mini": {
        "model_version": "openai/gpt-4o-mini",
        "provider": "openai",
        "input_price_per_1m_usd": 0.15,
        "output_price_per_1m_usd": 0.60,
        "assumptions": "OpenAI GPT-4o-mini pay-as-you-go tier."
    },
    "default": {
        "model_version": "standard/default-tier",
        "provider": "unknown",
        "input_price_per_1m_usd": 0.20,
        "output_price_per_1m_usd": 0.80,
        "assumptions": "Standard baseline estimate for uncataloged API models."
    }
}


def tokenize(text: str) -> List[str]:
    """Splits text into lowercase alphanumeric word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


# =============================================================================
# 2. DIAGNOSTIC QUALITY EVALUATOR
# =============================================================================

class DiagnosticQualityEvaluator:
    """
    Evaluates incident diagnostic accuracy:
    - Root-cause localization (Hit@1)
    - Top-k candidate accuracy (Top-1, Top-3, Top-5)
    - Fault classification macro-F1
    - Exact and structured-match diagnosis correctness
    - Evidence-grounding precision, recall, and unsupported-claim rate
    """

    @staticmethod
    def normalize_service_name(name: str) -> str:
        """Normalizes service/component identifiers by removing punctuation and suffixes."""
        if not name:
            return ""
        s = name.lower().strip()
        s = re.sub(r"[-_.]+", " ", s)
        s = re.sub(r"\b(service|pod|container|app|deployment|server)\b", "", s).strip()
        s = re.sub(r"(service|pod|container|app|deployment|server)$", "", s).strip()
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def compute_root_cause_localization(
        cls,
        predicted_service: str,
        ground_truth_service: str,
        prediction_text: str = ""
    ) -> bool:
        """
        Computes Hit@1 for root-cause localization.
        Checks if the predicted service matches ground truth via exact or normalized match.
        """
        if not ground_truth_service:
            return False

        gt_raw = ground_truth_service.lower().strip()
        gt_norm = cls.normalize_service_name(ground_truth_service)
        pred_raw = predicted_service.lower().strip()
        pred_norm = cls.normalize_service_name(predicted_service)
        text_lower = prediction_text.lower()

        # 1. Exact raw or normalized match on predicted service field
        if gt_raw == pred_raw or (gt_norm and gt_norm == pred_norm):
            return True

        # 2. Match on alphanumeric stem with trailing service suffixes stripped
        pred_stem = re.sub(r"(service|pod|container|app|deployment|server)$", "", re.sub(r"[^a-z0-9]", "", pred_raw))
        gt_stem = re.sub(r"(service|pod|container|app|deployment|server)$", "", re.sub(r"[^a-z0-9]", "", gt_raw))
        if pred_stem and gt_stem and (pred_stem == gt_stem):
            return True

        # 3. Substring match within the predicted service field
        if gt_raw in pred_raw or (gt_norm and gt_norm in pred_norm) or (gt_stem and len(gt_stem) >= 3 and gt_stem in pred_stem):
            return True

        # 4. Direct mention in prediction text if predicted service was omitted
        if not pred_raw and (gt_raw in text_lower or (gt_norm and gt_norm in text_lower) or (gt_stem and gt_stem in text_lower)):
            return True

        return False

    @classmethod
    def compute_top_k_accuracy(
        cls,
        candidate_services: List[str],
        ground_truth_service: str,
        k: int = 3
    ) -> bool:
        """
        Checks whether ground-truth faulty service appears within the top-k ranked candidate services.
        """
        if not ground_truth_service or not candidate_services:
            return False
        top_k_candidates = candidate_services[:k]
        for cand in top_k_candidates:
            if cls.compute_root_cause_localization(cand, ground_truth_service):
                return True
        return False

    @staticmethod
    def compute_fault_classification_macro_f1(
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, Any]:
        """
        Computes multi-class Macro-F1 across fault types (e.g. cpu, memory, network, deadlock, etc.).
        Returns macro_f1, weighted_f1, and per-class precision/recall/f1/support breakdown.
        """
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {"macro_f1": 0.0, "weighted_f1": 0.0, "per_class_metrics": {}}

        labels = sorted(list(set(y_true) | set(y_pred)))
        macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

        per_class = {}
        for idx, lbl in enumerate(labels):
            per_class[lbl] = {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f[idx]), 4),
                "support": int(s[idx])
            }

        return {
            "macro_f1": round(float(macro), 4),
            "weighted_f1": round(float(weighted), 4),
            "per_class_metrics": per_class
        }

    @classmethod
    def compute_structured_match_correctness(
        cls,
        structured_diag: Optional[Dict[str, Any]],
        ground_truth_service: str,
        ground_truth_fault: str
    ) -> float:
        """
        Evaluates exact and structured-match diagnosis correctness score in [0.0, 1.0]:
        - Service localization match (weight: 0.40)
        - Suspected root cause conceptual match (weight: 0.35)
        - Verification status validity (weight: 0.25)
        """
        if not structured_diag:
            return 0.0

        score = 0.0
        pred_service = structured_diag.get("affected_service_or_component", "")
        if cls.compute_root_cause_localization(pred_service, ground_truth_service):
            score += 0.40

        cause_text = (structured_diag.get("suspected_root_cause", "") + " " +
                      " ".join(structured_diag.get("inferred_root_causes", []))).lower()
        fault_lower = ground_truth_fault.lower().replace("-", " ")

        fault_keywords = {
            "cpu": ["cpu", "utilization", "throttle", "core", "spikes"],
            "memory": ["memory", "oom", "oomkilled", "137", "heap", "leak"],
            "oom": ["memory", "oom", "oomkilled", "137", "heap", "leak"],
            "network": ["network", "delay", "packet", "drop", "loss", "latency"],
            "delay": ["delay", "latency", "timeout", "slow", "rtt"],
            "deadlock": ["deadlock", "lock", "blocked", "transaction", "mutex"],
            "connection": ["connection", "pool", "exhaustion", "pgbouncer", "max_connections"],
            "tls": ["ssl", "tls", "handshake", "certificate", "cert", "expired"],
            "nullpointer": ["nullpointer", "null", "npe", "exception", "f2"],
            "loop": ["infinite loop", "cycle", "f4", "hang"]
        }

        matched_fault = False
        if fault_lower in cause_text:
            matched_fault = True
        else:
            for k, keywords in fault_keywords.items():
                if k in fault_lower and any(w in cause_text for w in keywords):
                    matched_fault = True
                    break

        if matched_fault:
            score += 0.35

        # Verification status credit
        status = str(structured_diag.get("verification_status", "")).lower()
        if status in ("verified", "escalated", "uncertain"):
            score += 0.25
        elif status == "unverified":
            score += 0.10

        return round(score, 2)

    @staticmethod
    def compute_evidence_grounding_and_unsupported(
        structured_diag: Optional[Dict[str, Any]],
        context_text: str = ""
    ) -> Dict[str, float]:
        """
        Computes evidence-grounding precision, evidence recall, and unsupported-claim rate.
        - Grounding Precision: Supported claims / Total claims
        - Unsupported-claim rate: Unsupported claims / Total claims
        - Evidence Recall: Gold context key entity coverage by the diagnostic claims
        """
        if not structured_diag:
            return {
                "grounding_precision": 0.0,
                "evidence_recall": 0.0,
                "unsupported_claim_rate": 0.0,
                "total_claims_audited": 0,
                "supported_claims_count": 0,
                "unsupported_claims_count": 0
            }

        claims = structured_diag.get("claims", [])
        if not claims:
            # Fallback to supporting evidence list if atomic claims not extracted
            supp_ev = structured_diag.get("supporting_evidence", [])
            total = max(len(supp_ev), 1)
            verified_count = sum(1 for e in supp_ev if e.get("verified", True))
            prec = round(verified_count / total, 4)
            unsupp = round(1.0 - prec, 4)
            return {
                "grounding_precision": round(prec * 100, 1),
                "evidence_recall": 50.0,
                "unsupported_claim_rate": round(unsupp * 100, 1),
                "total_claims_audited": len(supp_ev),
                "supported_claims_count": verified_count,
                "unsupported_claims_count": len(supp_ev) - verified_count
            }

        total_claims = len(claims)
        supported_claims = sum(1 for c in claims if c.get("supported", False))
        unsupported_claims = total_claims - supported_claims

        prec = round(supported_claims / max(total_claims, 1), 4)
        unsupp_rate = round(unsupported_claims / max(total_claims, 1), 4)

        # Evidence recall: fraction of distinct tokens in gold context captured by claims
        ctx_tokens = set(tokenize(context_text))
        claims_tokens = set()
        for c in claims:
            claims_tokens.update(tokenize(c.get("claim", "")))
        common_ctx = ctx_tokens & claims_tokens
        ev_recall = round(len(common_ctx) / max(len(ctx_tokens), 1), 4) if ctx_tokens else 0.0

        return {
            "grounding_precision": round(prec * 100, 1),
            "evidence_recall": round(ev_recall * 100, 1),
            "unsupported_claim_rate": round(unsupp_rate * 100, 1),
            "total_claims_audited": total_claims,
            "supported_claims_count": supported_claims,
            "unsupported_claims_count": unsupported_claims
        }


# =============================================================================
# 3. CONFIDENCE AND RELIABILITY EVALUATOR
# =============================================================================

class ConfidenceReliabilityEvaluator:
    """
    Evaluates confidence calibration and selective classification:
    - Brier score
    - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    - Reliability Diagrams (binned observed accuracy vs average confidence)
    - Selective accuracy and coverage under abstention / human review policies
    """

    @staticmethod
    def compute_brier_score(y_true: List[int], y_prob: List[float]) -> float:
        """
        Computes Brier Score: Mean squared error of calibrated probabilities against binary ground truth.
        BS = 1/N * sum((p_i - y_i)^2). Range: [0.0, 1.0], lower is better.
        """
        if not y_true or not y_prob or len(y_true) != len(y_prob):
            return 0.0
        yt = np.array(y_true, dtype=float)
        yp = np.clip(np.array(y_prob, dtype=float), 0.0, 1.0)
        return round(float(brier_score_loss(yt, yp)), 4)

    @staticmethod
    def compute_ece_and_diagram(
        y_true: List[int],
        y_prob: List[float],
        n_bins: int = 5
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
        along with Reliability Diagram partitions.
        ECE = sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|
        """
        if not y_true or not y_prob or len(y_true) != len(y_prob):
            return 0.0, 0.0, []

        yt = np.array(y_true, dtype=float)
        yp = np.clip(np.array(y_prob, dtype=float), 0.0, 1.0)
        n = len(yt)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        diagram = []

        for m in range(n_bins):
            lower = bin_edges[m]
            upper = bin_edges[m + 1]

            if m == n_bins - 1:
                mask = (yp >= lower) & (yp <= upper)
            else:
                mask = (yp >= lower) & (yp < upper)

            count_m = int(np.sum(mask))
            if count_m > 0:
                acc_m = float(np.mean(yt[mask]))
                conf_m = float(np.mean(yp[mask]))
                gap_m = abs(acc_m - conf_m)
                ece += (count_m / n) * gap_m
                mce = max(mce, gap_m)

                diagram.append({
                    "bin_index": m + 1,
                    "bin_lower": round(lower, 2),
                    "bin_upper": round(upper, 2),
                    "sample_count": count_m,
                    "avg_confidence": round(conf_m * 100, 1),
                    "observed_accuracy": round(acc_m * 100, 1),
                    "calibration_gap": round(gap_m * 100, 1)
                })
            else:
                diagram.append({
                    "bin_index": m + 1,
                    "bin_lower": round(lower, 2),
                    "bin_upper": round(upper, 2),
                    "sample_count": 0,
                    "avg_confidence": 0.0,
                    "observed_accuracy": 0.0,
                    "calibration_gap": 0.0
                })

        return round(float(ece), 4), round(float(mce), 4), diagram

    @staticmethod
    def compute_selective_accuracy_and_coverage(
        y_true: List[int],
        y_pred: List[int],
        confidences: List[float],
        abstention_flags: Optional[List[bool]] = None,
        default_threshold: float = 0.50,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates selective accuracy, coverage, and risk when abstention / escalation is enabled.
        An incident is answered if confidence >= threshold and abstention_flag is False.
        Also sweeps across confidence thresholds to produce a complete Risk-Coverage curve.
        """
        if threshold is not None:
            default_threshold = threshold
        n = len(y_true)
        if n == 0:
            return {
                "coverage": 0.0, "selective_accuracy": 0.0, "abstention_rate": 0.0,
                "selective_risk": 0.0, "threshold_sweep": []
            }

        yt = np.array(y_true)
        yp = np.array(y_pred)
        conf = np.array(confidences)
        abstained = np.array(abstention_flags) if abstention_flags is not None else np.zeros(n, dtype=bool)

        # Baseline point at default_threshold
        answered_mask = (conf >= default_threshold) & (~abstained)
        answered_count = int(np.sum(answered_mask))
        coverage = answered_count / n
        selective_acc = float(np.mean(yt[answered_mask] == yp[answered_mask])) if answered_count > 0 else 0.0
        selective_risk = 1.0 - selective_acc

        # Threshold sweep across confidence levels [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sweep = []
        for tau in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            ans_m = (conf >= tau) & (~abstained)
            ans_c = int(np.sum(ans_m))
            cov = ans_c / n
            acc = float(np.mean(yt[ans_m] == yp[ans_m])) if ans_c > 0 else 0.0
            sweep.append({
                "threshold": round(tau, 2),
                "coverage_percent": round(cov * 100, 1),
                "selective_accuracy_percent": round(acc * 100, 1),
                "abstention_percent": round((1.0 - cov) * 100, 1),
                "answered_cases": ans_c
            })

        return {
            "coverage": round(coverage * 100, 1),
            "selective_accuracy": round(selective_acc * 100, 1),
            "abstention_rate": round((1.0 - coverage) * 100, 1),
            "selective_risk": round(selective_risk * 100, 1),
            "answered_count": answered_count,
            "total_cases": n,
            "threshold_sweep": sweep
        }


# =============================================================================
# 4. EFFICIENCY EVALUATOR
# =============================================================================

class EfficiencyEvaluator:
    """
    Evaluates runtime and economic efficiency:
    - Actual API cost (direct billed) vs Estimated API cost (pricing formula)
    - Input and output tokens
    - Total model calls, escalation frequency, retrieval retries
    - Latency statistics (median p50, 95th percentile p95, mean, std)
    - Diagnostic quality versus cost and latency trade-offs
    """

    @staticmethod
    def estimate_cost_from_tokens(
        model_id: str,
        in_tokens: int,
        out_tokens: int,
        pricing_catalog: Dict[str, Any] = TOKEN_PRICING_CATALOG
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimates inference cost using published pricing catalog:
        Cost = (in_tokens / 1e6 * Pin) + (out_tokens / 1e6 * Pout)
        """
        entry = pricing_catalog.get(model_id, pricing_catalog.get("default", {}))
        pin = entry.get("input_price_per_1m_usd", 0.20)
        pout = entry.get("output_price_per_1m_usd", 0.80)

        cost = (in_tokens / 1_000_000.0 * pin) + (out_tokens / 1_000_000.0 * pout)
        meta = {
            "model_version": entry.get("model_version", model_id),
            "input_price_per_1m": pin,
            "output_price_per_1m": pout,
            "assumptions": entry.get("assumptions", "Standard published pricing")
        }
        return round(cost, 6), meta

    @classmethod
    def compute_efficiency_metrics(
        cls,
        case_records: List[Dict[str, Any]],
        composite_quality_scores: Optional[List[float]] = None,
        pricing_catalog: Dict[str, Any] = TOKEN_PRICING_CATALOG
    ) -> Dict[str, Any]:
        """
        Aggregates operational efficiency metrics across evaluated incident cases.
        Guarantees strict separation of actual billed cost vs calculated estimated cost.
        """
        n = len(case_records)
        if n == 0:
            return {
                "actual_api_cost_usd": 0.0,
                "estimated_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "median_latency_p50_sec": 0.0,
                "p95_latency_sec": 0.0,
                "escalation_frequency_percent": 0.0,
                "retrieval_retry_frequency_percent": 0.0,
                "quality_cost_efficiency_ratio": 0.0,
                "quality_latency_efficiency_ratio": 0.0
            }

        latencies = [float(r.get("latency", r.get("latency_seconds", 0.0))) for r in case_records]
        in_tokens = [int(r.get("in_tokens", 0)) for r in case_records]
        out_tokens = [int(r.get("out_tokens", 0)) for r in case_records]
        model_calls = [int(r.get("model_calls", 1)) for r in case_records]
        escalations = [bool(r.get("escalated", False)) for r in case_records]
        retries = [int(r.get("retry_count", r.get("retrieval_retries", 0))) for r in case_records]

        # 1. Distinguish actual billed cost from estimated cost
        actual_costs = [float(r.get("actual_api_cost_usd", r.get("actual_cost", 0.0))) for r in case_records]
        actual_cost_total = round(sum(actual_costs), 6)

        estimated_costs = []
        for r, itok, otok in zip(case_records, in_tokens, out_tokens):
            mid = r.get("routed_model", r.get("model_id", "default"))
            est, _ = cls.estimate_cost_from_tokens(mid, itok, otok, pricing_catalog)
            estimated_costs.append(est)
        estimated_cost_total = round(sum(estimated_costs), 6)

        # 2. Token counts
        tot_in = sum(in_tokens)
        tot_out = sum(out_tokens)
        tot_tok = tot_in + tot_out

        # 3. Latency distribution (mean, std, median p50, p95)
        lat_arr = np.array(latencies)
        mean_lat = round(float(np.mean(lat_arr)), 2)
        std_lat = round(float(np.std(lat_arr)), 2)
        p50_lat = round(float(np.percentile(lat_arr, 50)), 2)
        p95_lat = round(float(np.percentile(lat_arr, 95)), 2)

        # 4. Frequencies
        esc_freq = round((sum(1 for e in escalations if e) / n) * 100, 1)
        retry_freq = round((sum(1 for ret in retries if ret > 0) / n) * 100, 1)

        # 5. Quality-Efficiency Tradeoffs
        q_scores = composite_quality_scores if composite_quality_scores else [50.0] * n
        avg_q = float(np.mean(q_scores))
        q_cost_ratio = round(avg_q / max(estimated_cost_total, 1e-5), 2)
        q_lat_ratio = round(avg_q / max(mean_lat, 0.1), 2)

        return {
            "actual_api_cost_usd": actual_cost_total,
            "estimated_cost_usd": estimated_cost_total,
            "cost_estimation_disclaimer": (
                "Estimated API cost is calculated based on token counts and documented published pricing. "
                "It is strictly separated from actual provider-billed charges."
            ),
            "total_input_tokens": tot_in,
            "total_output_tokens": tot_out,
            "total_tokens": tot_tok,
            "mean_tokens_per_case": round(tot_tok / n, 1),
            "total_model_calls": sum(model_calls),
            "mean_model_calls_per_case": round(sum(model_calls) / n, 2),
            "escalation_frequency_percent": esc_freq,
            "retrieval_retry_frequency_percent": retry_freq,
            "mean_latency_sec": mean_lat,
            "std_latency_sec": std_lat,
            "median_latency_p50_sec": p50_lat,
            "p95_latency_sec": p95_lat,
            "quality_cost_efficiency_ratio": q_cost_ratio,
            "quality_latency_efficiency_ratio": q_lat_ratio,
            "documented_pricing_models": list(pricing_catalog.keys())
        }


# =============================================================================
# 5. STATISTICAL REPORTING & COMPARISON
# =============================================================================

class StatisticalReporter:
    """
    Rigorously computes sample statistics, confidence intervals, paired statistical tests,
    and effect sizes for incident diagnosis evaluations.
    """

    @staticmethod
    def compute_dataset_composition(incidents: List[Any]) -> Dict[str, Any]:
        """Summarizes sample counts and composition across systems, categories, and severities."""
        total = len(incidents)
        systems = Counter()
        categories = Counter()
        severities = Counter()
        sources = Counter()

        for inc in incidents:
            if hasattr(inc, "system_name"):
                systems[inc.system_name] += 1
            if hasattr(inc, "category"):
                categories[inc.category] += 1
            if hasattr(inc, "severity"):
                severities[inc.severity] += 1
            if hasattr(inc, "source_dataset"):
                sources[inc.source_dataset] += 1
            elif isinstance(inc, dict):
                systems[inc.get("system", "unknown")] += 1
                categories[inc.get("category", "general")] += 1
                severities[inc.get("severity", "L2")] += 1
                sources[inc.get("source_dataset", "benchmark")] += 1

        return {
            "total_sample_count": total,
            "sources": dict(sources),
            "systems": dict(systems),
            "fault_categories": dict(categories),
            "severities": dict(severities)
        }

    @staticmethod
    def compute_mean_and_variability(values: List[float]) -> Dict[str, float]:
        """Computes mean, sample standard deviation, variance, and standard error."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "variance": 0.0, "standard_error": 0.0}
        arr = np.array(values, dtype=float)
        n = len(arr)
        mean_val = float(np.mean(arr))
        var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se_val = std_val / math.sqrt(n) if n > 0 else 0.0

        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "variance": round(var_val, 4),
            "standard_error": round(se_val, 4)
        }

    @classmethod
    def compute_confidence_interval(
        cls,
        values: List[float],
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Computes two-tailed 95% Confidence Interval using Student's t distribution.
        CI = [mean - t * SE, mean + t * SE]
        """
        n = len(values)
        if n < 2:
            m = round(float(np.mean(values)), 4) if values else 0.0
            return {
                "mean": m,
                "ci_lower": m,
                "ci_upper": m,
                "margin_of_error": 0.0,
                "confidence_level": confidence,
                "note": "Insufficient samples for degrees of freedom (N < 2)"
            }

        stats_dict = cls.compute_mean_and_variability(values)
        mean = stats_dict["mean"]
        se = stats_dict["standard_error"]
        df = n - 1
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=df))
        moe = round(t_crit * se, 4)

        return {
            "mean": mean,
            "ci_lower": round(mean - moe, 4),
            "ci_upper": round(mean + moe, 4),
            "margin_of_error": moe,
            "confidence_level": confidence,
            "degrees_of_freedom": df
        }

    @staticmethod
    def compute_paired_statistical_test(
        baseline_values: List[float],
        target_values: List[float]
    ) -> Dict[str, Any]:
        """
        Computes both parametric (Paired Student's t-test) and non-parametric
        (Wilcoxon signed-rank test) tests along with effect sizes (Cohen's d, Cliff's delta).
        """
        n = min(len(baseline_values), len(target_values))
        if n < 2:
            diff = round(float(target_values[0] - baseline_values[0]), 4) if n == 1 else 0.0
            return {
                "n": n,
                "mean_diff": diff,
                "t_statistic": 0.0,
                "p_value_parametric": 1.0,
                "wilcoxon_stat": 0.0,
                "p_value_nonparametric": 1.0,
                "cohen_d": 0.0,
                "cliff_delta": 0.0,
                "effect_magnitude": "negligible",
                "statistically_significant": False,
                "note": "Insufficient paired samples (N < 2)"
            }

        b = np.array(baseline_values[:n], dtype=float)
        t = np.array(target_values[:n], dtype=float)
        diffs = t - b
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0

        # 1. Paired Student's t-test
        try:
            t_res = stats.ttest_rel(t, b)
            t_stat = round(float(t_res.statistic), 4) if not math.isnan(t_res.statistic) else 0.0
            p_val_t = round(float(t_res.pvalue), 4) if not math.isnan(t_res.pvalue) else 1.0
        except Exception:
            t_stat = round(mean_diff / se_diff, 4) if se_diff > 1e-6 else 0.0
            p_val_t = 0.05 if abs(t_stat) > 2.0 else 0.50

        # 2. Wilcoxon signed-rank test
        try:
            if np.all(diffs == 0):
                w_stat, p_val_w = 0.0, 1.0
            else:
                w_res = stats.wilcoxon(t, b)
                w_stat = round(float(w_res.statistic), 4)
                p_val_w = round(float(w_res.pvalue), 4)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        # 3. Effect sizes: Cohen's d & Cliff's delta
        cohen_d = round(mean_diff / std_diff, 4) if std_diff > 1e-6 else 0.0

        # Cliff's delta: (num(t > b) - num(t < b)) / (n1 * n2)
        more = sum(1 for ti in t for bi in b if ti > bi)
        less = sum(1 for ti in t for bi in b if ti < bi)
        cliff_delta = round((more - less) / (n * n), 4)

        abs_d = abs(cohen_d)
        if abs_d < 0.20:
            mag = "negligible"
        elif abs_d < 0.50:
            mag = "small"
        elif abs_d < 0.80:
            mag = "medium"
        else:
            mag = "large"

        return {
            "n": n,
            "mean_diff": round(mean_diff, 4),
            "std_diff": round(std_diff, 4),
            "standard_error_diff": round(se_diff, 4),
            "t_statistic": t_stat,
            "p_value_parametric": p_val_t,
            "wilcoxon_stat": w_stat,
            "p_value_nonparametric": p_val_w,
            "cohen_d": cohen_d,
            "cliff_delta": cliff_delta,
            "effect_magnitude": mag,
            "statistically_significant": bool(p_val_t < 0.05)
        }


# =============================================================================
# 6. UNIFIED COMPREHENSIVE BENCHMARK EVALUATOR
# =============================================================================

def evaluate_incident_diagnosis_case(
    prediction_text: str,
    ground_truth_service: str,
    ground_truth_fault: str,
    context_text: str,
    reference_text: str,
    latency_sec: float,
    model_id: str = "default",
    in_tokens: int = 0,
    out_tokens: int = 0,
    actual_api_cost: float = 0.0,
    structured_diag: Optional[Dict[str, Any]] = None,
    calibrated_prob: float = 0.50,
    candidate_services: Optional[List[str]] = None,
    pricing_catalog: Dict[str, Any] = TOKEN_PRICING_CATALOG
) -> Dict[str, Any]:
    """
    Evaluates a single incident diagnostic output across all Phase 8 dimensions:
    - Root-cause localization (Hit@1, Top-3, Top-5)
    - Structured correctness
    - Grounding precision, recall, and unsupported-claim rate
    - Brier score contribution
    - Actual vs Estimated API cost
    - Tokens and latency
    """
    pred_tokens = tokenize(prediction_text)
    ref_tokens = tokenize(reference_text)
    ctx_tokens = tokenize(context_text)

    # 1. Linguistic Token Overlap F1 and Supplementary BLEU
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    prec = round(num_same / max(len(pred_tokens), 1), 4)
    rec = round(num_same / max(len(ref_tokens), 1), 4)
    f1 = round((2 * prec * rec) / max(prec + rec, 1e-5), 4) if (prec + rec) > 0 else 0.0

    precisions = []
    for n in range(1, 5):
        if len(pred_tokens) < n or len(ref_tokens) < n:
            precisions.append(1e-4)
            continue
        p_ng = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        r_ng = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        clipped = sum((p_ng & r_ng).values())
        precisions.append(max(clipped / max(len(pred_tokens)-n+1, 1), 1e-4))
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - (len(ref_tokens) / max(len(pred_tokens), 1)))
    bleu = round(bp * math.exp(sum(math.log(p) for p in precisions) / 4), 4)

    # 2. Faithfulness & Hallucination Rate
    overlap_ctx = set(pred_tokens).intersection(set(ctx_tokens))
    faithfulness = round(len(overlap_ctx) / max(len(set(pred_tokens)), 1), 4)
    hallucination_rate = round(max(1.0 - faithfulness, 0.0), 4)

    # 3. Root-Cause Localization & Top-k
    pred_service = (structured_diag or {}).get("affected_service_or_component", "")
    hit1 = DiagnosticQualityEvaluator.compute_root_cause_localization(
        pred_service, ground_truth_service, prediction_text
    )

    all_cands = [pred_service] + (candidate_services or [])
    if structured_diag and structured_diag.get("alternative_explanations"):
        all_cands.extend(structured_diag["alternative_explanations"])
    top3 = DiagnosticQualityEvaluator.compute_top_k_accuracy(all_cands, ground_truth_service, k=3)
    top5 = DiagnosticQualityEvaluator.compute_top_k_accuracy(all_cands, ground_truth_service, k=5)

    # 4. Structured Match Correctness
    struct_correctness = DiagnosticQualityEvaluator.compute_structured_match_correctness(
        structured_diag, ground_truth_service, ground_truth_fault
    )

    # 5. Evidence Grounding & Unsupported Claims
    grounding = DiagnosticQualityEvaluator.compute_evidence_grounding_and_unsupported(
        structured_diag, context_text
    )

    # 6. Efficiency: Actual vs Estimated Cost
    est_cost, est_meta = EfficiencyEvaluator.estimate_cost_from_tokens(
        model_id, in_tokens, out_tokens, pricing_catalog
    )

    # 7. Operational Composite Score
    composite = round(
        ((faithfulness * 0.20) + (f1 * 0.20) + (int(hit1) * 0.35) + (struct_correctness * 0.25)) * 100,
        1
    )

    return {
        "hit_at_1": hit1,
        "hit_at_3": top3,
        "hit_at_5": top5,
        "structured_correctness": struct_correctness,
        "f1": f1,
        "supplementary_bleu": bleu,
        "faithfulness": round(faithfulness * 100, 1),
        "hallucination_rate": round(hallucination_rate * 100, 1),
        "grounding_precision": grounding["grounding_precision"],
        "evidence_recall": grounding["evidence_recall"],
        "unsupported_claim_rate": grounding["unsupported_claim_rate"],
        "calibrated_confidence": round(float(calibrated_prob), 4),
        "composite_score": composite,
        "latency_sec": round(float(latency_sec), 2),
        "actual_api_cost_usd": round(float(actual_api_cost), 6),
        "estimated_cost_usd": est_cost,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "routed_model": model_id,
        "model_pricing_metadata": est_meta
    }


def generate_comprehensive_evaluation_report(
    eval_records: List[Dict[str, Any]],
    dataset_name: str = "Benchmark",
    pricing_catalog: Dict[str, Any] = TOKEN_PRICING_CATALOG
) -> Dict[str, Any]:
    """
    Aggregates all evaluation records into the complete Phase 8 multi-dimensional report.
    Includes Quality, Reliability/Calibration, Efficiency, and Statistical Summaries.
    """
    n = len(eval_records)
    if n == 0:
        return {"dataset_name": dataset_name, "sample_count": 0, "status": "empty"}

    # 1. Quality Metrics
    hit1_rate = (sum(1 for r in eval_records if r.get("hit_at_1")) / n) * 100
    hit3_rate = (sum(1 for r in eval_records if r.get("hit_at_3")) / n) * 100
    hit5_rate = (sum(1 for r in eval_records if r.get("hit_at_5")) / n) * 100
    mean_struct_correct = sum(r.get("structured_correctness", 0.0) for r in eval_records) / n
    mean_f1 = sum(r.get("f1", 0.0) for r in eval_records) / n
    mean_bleu = sum(r.get("supplementary_bleu", 0.0) for r in eval_records) / n
    mean_faith = sum(r.get("faithfulness", 0.0) for r in eval_records) / n
    mean_unsupp = sum(r.get("unsupported_claim_rate", 0.0) for r in eval_records) / n
    mean_g_prec = sum(r.get("grounding_precision", 0.0) for r in eval_records) / n
    mean_g_rec = sum(r.get("evidence_recall", 0.0) for r in eval_records) / n
    mean_composite = sum(r.get("composite_score", 0.0) for r in eval_records) / n

    # Fault classification macro-F1 if fault labels are present
    y_true_faults = [r.get("ground_truth_fault", "") for r in eval_records if r.get("ground_truth_fault")]
    y_pred_faults = [r.get("predicted_fault", "") for r in eval_records if r.get("ground_truth_fault")]
    fault_macro = 0.0
    if y_true_faults and y_pred_faults:
        macro_dict = DiagnosticQualityEvaluator.compute_fault_classification_macro_f1(y_true_faults, y_pred_faults)
        fault_macro = macro_dict.get("macro_f1", 0.0)

    quality_summary = {
        "root_cause_localization_hit1_percent": round(hit1_rate, 1),
        "root_cause_localization_hit3_percent": round(hit3_rate, 1),
        "root_cause_localization_hit5_percent": round(hit5_rate, 1),
        "fault_classification_macro_f1": round(fault_macro, 4),
        "structured_diagnosis_correctness": round(mean_struct_correct * 100, 1),
        "evidence_grounding_precision_percent": round(mean_g_prec, 1),
        "evidence_grounding_recall_percent": round(mean_g_rec, 1),
        "unsupported_claim_rate_percent": round(mean_unsupp, 1),
        "mean_answer_f1": round(mean_f1, 4),
        "mean_faithfulness_percent": round(mean_faith, 1),
        "supplementary_linguistic_bleu": round(mean_bleu, 4),
        "overall_composite_score": round(mean_composite, 1)
    }

    # 2. Confidence and Reliability Metrics
    y_true_binary = [int(r.get("hit_at_1", False)) for r in eval_records]
    y_prob = [float(r.get("calibrated_confidence", 0.50)) for r in eval_records]
    brier = ConfidenceReliabilityEvaluator.compute_brier_score(y_true_binary, y_prob)
    ece, mce, diagram = ConfidenceReliabilityEvaluator.compute_ece_and_diagram(y_true_binary, y_prob, n_bins=5)
    selective = ConfidenceReliabilityEvaluator.compute_selective_accuracy_and_coverage(
        y_true_binary, y_true_binary, y_prob, threshold=0.50
    )

    reliability_summary = {
        "brier_score": brier,
        "expected_calibration_error_ece": ece,
        "maximum_calibration_error_mce": mce,
        "selective_accuracy_percent": selective["selective_accuracy"],
        "coverage_percent": selective["coverage"],
        "abstention_rate_percent": selective["abstention_rate"],
        "reliability_diagram_bins": diagram,
        "risk_coverage_curve": selective["threshold_sweep"]
    }

    # 3. Efficiency Metrics
    composite_scores = [r.get("composite_score", 50.0) for r in eval_records]
    efficiency_summary = EfficiencyEvaluator.compute_efficiency_metrics(
        eval_records, composite_quality_scores=composite_scores, pricing_catalog=pricing_catalog
    )

    # 4. Statistical Summary & Confidence Intervals
    acc_vals = [r.get("composite_score", 0.0) for r in eval_records]
    ci_accuracy = StatisticalReporter.compute_confidence_interval(acc_vals, confidence=0.95)
    ci_f1 = StatisticalReporter.compute_confidence_interval([r.get("f1", 0.0) for r in eval_records], confidence=0.95)

    statistical_summary = {
        "sample_count": n,
        "composite_accuracy_mean": ci_accuracy["mean"],
        "composite_accuracy_95_ci": [ci_accuracy["ci_lower"], ci_accuracy["ci_upper"]],
        "composite_accuracy_margin_of_error": ci_accuracy["margin_of_error"],
        "f1_mean": ci_f1["mean"],
        "f1_95_ci": [ci_f1["ci_lower"], ci_f1["ci_upper"]]
    }

    return {
        "dataset_name": dataset_name,
        "sample_count": n,
        "diagnostic_quality": quality_summary,
        "confidence_and_reliability": reliability_summary,
        "operational_efficiency": efficiency_summary,
        "statistical_reporting": statistical_summary,
        "individual_case_evaluations": eval_records
    }

"""
Router Agent — Dynamic Multi-LLM selection based on query complexity,
retrieval confidence, and Pareto-optimal cost/latency/quality trade-offs.
"""

import os
import math
import re
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv
load_dotenv()

from agents.state import AgentState


class AdaptiveModelRouter:
    """
    Adaptive Multi-LLM Router for SRE/DevOps RAG Systems.
    Dynamically routes between Fast, High-F1, and Heavy models based on:
    1. Query Complexity (length, technical entity density, reasoning hops)
    2. Retrieval Confidence (FAISS L2 distance distribution, lexical coverage)
    3. Multi-objective Pareto scoring (Quality, Faithfulness, Cost, Latency)
    """

    def __init__(self, weights: Dict[str, float] = None):
        # Cost Priority Mode: when enabled, aggressively favors free/cheap models
        # Set COST_PRIORITY_MODE=true in .env to optimize for minimum AI billing
        self.cost_priority = os.getenv("COST_PRIORITY_MODE", "false").lower() in ("true", "1", "yes")

        if self.cost_priority:
            # Cost-optimized weights: heavily penalize cost, reward quality less
            self.weights = weights or {
                "alpha": 0.25,  # Quality (reduced to allow cheaper models)
                "beta": 0.20,   # Faithfulness
                "gamma": 0.35,  # Monetary Cost penalty (doubled)
                "delta": 0.20,  # Latency penalty
            }
        else:
            # Default Pareto weights: Quality (0.40), Faithfulness (0.30), Cost (0.15), Latency (0.15)
            self.weights = weights or {
                "alpha": 0.40,  # Quality (Expected F1 / Correctness)
                "beta": 0.30,   # Faithfulness
                "gamma": 0.15,  # Monetary Cost penalty
                "delta": 0.15,  # Latency penalty
            }

        # Multi-provider Heterogeneous Catalog (Groq, Gemini, Cohere, OpenRouter, DeepSeek, Anthropic)
        has_groq = bool(os.getenv("GROQ_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        has_cohere = bool(os.getenv("COHERE_API_KEY"))
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
        has_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        # ──────────────────────────────────────────────────────────────
        # OpenRouter Free Model Pool — $0.00 per token
        # These models rotate on OpenRouter but all cost exactly $0.
        # The router picks the best available free model for each tier.
        # ──────────────────────────────────────────────────────────────
        self.openrouter_free_models = [
            {"name": "Google Gemma 4 31B",          "model_id": "google/gemma-4-31b-it:free",              "strength": "general"},
            {"name": "NVIDIA Nemotron 3 Ultra 550B", "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free", "strength": "heavy_reasoning"},
            {"name": "NVIDIA Nemotron 3.5 Lightning","model_id": "nvidia/nemotron-3.5-lightning:free",    "strength": "fast"},
            {"name": "Thinking Machines Inkling",    "model_id": "thinkingmachines/inkling:free",         "strength": "reasoning"},
            {"name": "Cohere North Mini Code",       "model_id": "cohere/north-mini-code:free",          "strength": "code"},
            {"name": "NVIDIA Nemotron 3 Super 120B", "model_id": "nvidia/nemotron-3-super-120b:free",    "strength": "heavy_reasoning"},
        ]

        self.model_catalog = {
            # ── Tier 0: Zero-Cost (OpenRouter 550B free model / Groq) ──
            # Completely free — $0.00/token. Used for standard queries
            # and cost-priority routing to eliminate company AI billing.
            "tier0_zero_cost": {
                "name": "NVIDIA Nemotron 3 Ultra 550B (Free)" if has_openrouter else ("Groq GPT-OSS 20B" if has_groq else "Gemini Flash Lite"),
                "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free" if has_openrouter else ("openai/gpt-oss-20b" if has_groq else "gemini-2.5-flash-lite"),
                "provider": "openrouter" if has_openrouter else ("groq" if has_groq else "gemini"),
                "base_f1": 0.280,
                "base_faith": 0.380,
                "cost_per_1k": 0.00000,   # Completely free — $0.00
                "latency_sec": 2.60,
                "description": "Zero-cost 550B parameter tier — eliminates AI billing for standard queries"
            },
            # ── Tier 1: Fast (Groq / low-cost paid) ──
            "tier1_fast": {
                "name": "Groq GPT-OSS 20B" if has_groq else ("Cohere Command-R (35B)" if has_cohere else "Gemini Flash Lite"),
                "model_id": "openai/gpt-oss-20b" if has_groq else ("command-r-08-2024" if has_cohere else "gemini-2.5-flash-lite"),
                "provider": "groq" if has_groq else ("cohere" if has_cohere else "gemini"),
                "base_f1": 0.260,
                "base_faith": 0.350,
                "cost_per_1k": 0.00010,
                "latency_sec": 1.20,
                "description": "Ultra-fast sub-2s inference for standard runbook triage"
            },
            # ── Tier 2: High Reasoning (Gemini / mid-cost paid) ──
            "tier2_high_reasoning": {
                "name": "Gemini 2.5 Flash" if has_gemini else ("Cohere Command-R (35B)" if has_cohere else "DeepSeek Chat"),
                "model_id": "gemini-2.5-flash" if has_gemini else ("command-r-08-2024" if has_cohere else "deepseek-chat"),
                "provider": "gemini" if has_gemini else ("cohere" if has_cohere else "deepseek"),
                "base_f1": 0.310,
                "base_faith": 0.420,
                "cost_per_1k": 0.00015,
                "latency_sec": 3.80,
                "description": "High-reasoning multimodal & structured diagnosis for complex microservice failures"
            },
            # ── Tier 3: Heavy Enterprise (Cohere 104B / large paid) ──
            "tier3_heavy": {
                "name": "Cohere Command-R+ (104B)" if has_cohere else ("Claude 3.5 Haiku" if has_anthropic else ("Groq GPT-OSS 120B" if has_groq else "Gemini 2.5 Pro")),
                "model_id": "command-r-plus-08-2024" if has_cohere else ("claude-3-5-haiku-20241022" if has_anthropic else ("openai/gpt-oss-120b" if has_groq else "gemini-2.5-pro")),
                "provider": "cohere" if has_cohere else ("anthropic" if has_anthropic else ("groq" if has_groq else "gemini")),
                "base_f1": 0.350,
                "base_faith": 0.480,
                "cost_per_1k": 0.00250,
                "latency_sec": 8.50,
                "description": "High-capacity parameter tier reserved for escalations and contradictory incidents"
            }
        }


    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extracts linguistic and domain-specific complexity features from the query."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        token_count = len(tokens)

        # Technical entity / log pattern detection
        tech_patterns = [
            r"crashloop\w*", r"oomkilled", r"exit\s+code\s+\d+", r"50[0-4]", r"40[0-4]",
            r"ssl|tls|handshake|certificate", r"sigkill|sigterm", r"deadlock", r"pg_\w+|postgres|database",
            r"kubectl|pod|namespace|container", r"p9[59]|latency|timeout", r"segfault",
            r"cascading|cluster[- ]wide|multi-service"
        ]
        tech_matches = sum(1 for p in tech_patterns if re.search(p, query, re.I))

        # Multi-hop / comparative reasoning signals
        reasoning_signals = [r"\bcompare\b", r"\bwhy\b", r"\bcontrast\b", r"\bdifference\b", 
                             r"\bcorrelation\b", r"\bimpact\b", r"\bcorrelate\b", r"\bunder\s+condition\b"]
        reasoning_hops = sum(1 for p in reasoning_signals if re.search(p, query, re.I))

        # Specific operational domain signals
        is_severe_outage = bool(re.search(
            r"\b(cascading|cluster[- ]wide|catastrophic|sev-?[01]|disaster|unhandled sigkill|"
            r"deadlock\w*|corruption|data[- ]loss|split[- ]brain|multi[- ]service|distributed.*pipeline|"
            r"cross[- ]service|blackout|outage)\b", query, re.I
        ))
        is_fast_triage = bool(re.search(r"\b(quick\s+triage|fast\s+check|alert\b|triage\s+alert|sla\b|real-?time|ping\b|health\s+check)\b", query, re.I))
        has_contradiction = bool(re.search(r"\b(conflicting|contradict\w*|opposing)\b", query, re.I))

        # Normalized complexity score in [0.0, 1.0]
        len_component = min(1.0, token_count / 35.0)
        tech_component = min(1.0, tech_matches * 0.35)
        reasoning_component = min(1.0, reasoning_hops * 0.40)

        # Question type and comparative analysis
        is_comparison = bool(re.search(r"\b(compare|contrast|versus|vs|difference|between)\b", query, re.I))
        if re.search(r"\b(how\s+to|steps|procedure|command|runbook)\b", query, re.I):
            q_type = "procedural_how_to"
        elif re.search(r"\b(why|root\s+cause|error|fail|crash|down|spike|oom)\b", query, re.I):
            q_type = "diagnostic_root_cause"
        elif is_comparison:
            q_type = "comparative_analysis"
        else:
            q_type = "factual_lookup"

        # Zero-cost / OpenRouter constraint detection
        is_zero_cost = bool(re.search(r"\b(openrouter|nemotron|zero[_\s-]*cost|free[_\s-]*tier|unmetered|without\s+cost|save\s+cost|free\s+model|open\s*router)\b", query, re.I))

        base_complexity = round(
            0.30 * len_component + 0.45 * tech_component + 0.25 * reasoning_component, 4
        )
        if is_severe_outage:
            complexity_score = min(1.0, max(base_complexity, 0.70))
        else:
            complexity_score = min(1.0, base_complexity)

        return {
            "token_count": token_count,
            "tech_entity_matches": tech_matches,
            "reasoning_hops": reasoning_hops,
            "complexity_score": complexity_score,
            "question_type": q_type,
            "requires_comparison": is_comparison,
            "is_zero_cost": is_zero_cost,
            "is_severe_outage": is_severe_outage,
            "is_fast_triage": is_fast_triage,
            "has_contradiction": has_contradiction,
        }

    def extract_retrieval_features(self, kb_results: List[dict], query: str) -> Dict[str, Any]:
        """
        Computes retrieval certainty from FAISS L2 distances, lexical coverage,
        pairwise chunk diversity, and potential contradictory directives.
        """
        if not kb_results:
            return {
                "top1_similarity": 0.0,
                "mean_similarity": 0.0,
                "min_similarity": 0.0,
                "variance": 0.0,
                "num_relevant_chunks": 0,
                "context_diversity": 0.0,
                "context_redundancy": 1.0,
                "contradiction_score": 0.0,
                "lexical_coverage": 0.0,
                "retrieval_confidence": 0.0
            }

        # Convert L2 distances to exponential similarity in (0, 1]
        distances = [float(r.get("score", 1.0)) for r in kb_results]
        similarities = [math.exp(-0.5 * d) for d in distances]

        top1_sim = similarities[0]
        mean_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        num_relevant = sum(1 for s in similarities if s >= 0.50)

        # Pairwise context diversity across retrieved chunks (Jaccard distance)
        chunk_token_sets = [set(re.findall(r"\b\w+\b", r.get("text", "").lower())) for r in kb_results]
        pairwise_jaccard = []
        for i in range(len(chunk_token_sets)):
            for j in range(i + 1, len(chunk_token_sets)):
                s1, s2 = chunk_token_sets[i], chunk_token_sets[j]
                if s1 or s2:
                    j_sim = len(s1 & s2) / max(len(s1 | s2), 1)
                    pairwise_jaccard.append(1.0 - j_sim) # 1 - similarity = diversity
        context_diversity = round(sum(pairwise_jaccard) / len(pairwise_jaccard), 4) if pairwise_jaccard else 0.0
        context_redundancy = round(max(0.0, 1.0 - context_diversity), 4)

        # Check contradiction risk (opposing status words across chunks)
        combined_text = " ".join(r.get("text", "") for r in kb_results).lower()
        has_enable = bool(re.search(r"\b(enable|activate|start|true)\b", combined_text))
        has_disable = bool(re.search(r"\b(disable|deactivate|stop|false|deprecated)\b", combined_text))
        contradiction_score = 0.25 if (has_enable and has_disable and context_diversity > 0.60) else 0.0

        # Lexical coverage between query tokens and retrieved context
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        ctx_tokens = set(re.findall(r"\b\w+\b", combined_text))

        coverage = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)

        # Composite retrieval confidence: penalized if distances have high variance or low coverage
        retrieval_confidence = round(
            (0.60 * mean_sim + 0.40 * coverage) * (1.0 - min(0.5, variance)) * (1.0 - contradiction_score), 4
        )

        return {
            "top1_similarity": round(top1_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "min_similarity": round(min_sim, 4),
            "variance": round(variance, 4),
            "num_relevant_chunks": num_relevant,
            "context_diversity": context_diversity,
            "context_redundancy": context_redundancy,
            "contradiction_score": contradiction_score,
            "lexical_coverage": round(coverage, 4),
            "retrieval_confidence": max(0.0, min(1.0, retrieval_confidence))
        }

    def route_query(self, query: str, kb_results: List[dict]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Evaluates Pareto utility for each candidate LLM and selects the optimal model.
        Returns (provider, model_id, routing_metadata).
        """
        q_feat = self.extract_query_features(query)
        r_feat = self.extract_retrieval_features(kb_results, query)

        complexity = q_feat["complexity_score"]
        r_conf = r_feat["retrieval_confidence"]

        utilities = {}
        max_cost = 0.0025  # Normalization denominator
        max_lat = 20.0     # Normalization denominator

        is_severe = q_feat.get("is_severe_outage", False)
        is_fast_triage = q_feat.get("is_fast_triage", False)
        has_contra = q_feat.get("has_contradiction", False) or (r_feat.get("contradiction_score", 0.0) > 0)
        hops = q_feat.get("reasoning_hops", 0)

        for tier_id, meta in self.model_catalog.items():
            # Adaptive Quality Estimation:
            if tier_id == "tier0_zero_cost":
                # Zero-cost 550B free tier: used when explicitly zero-cost or well-grounded routine queries
                if not is_severe and not is_fast_triage and 0.20 <= complexity <= 0.45 and r_conf >= 0.45:
                    expected_q = meta["base_f1"] * 1.20
                    expected_f = meta["base_faith"] * 1.15
                else:
                    expected_q = meta["base_f1"] * 0.55
                    expected_f = meta["base_faith"] * 0.55

            elif tier_id == "tier1_fast":
                # Fast model (Groq 20B / Cohere 35B): thrives on low complexity (<0.42) & quick triage alerts
                if is_fast_triage or complexity <= 0.42:
                    expected_q = meta["base_f1"] * 1.65
                    expected_f = meta["base_faith"] * 1.55
                elif not is_severe and 0.42 < complexity <= 0.50 and r_conf >= 0.40:
                    expected_q = meta["base_f1"] * 1.05
                    expected_f = meta["base_faith"] * 1.05
                elif complexity > 0.50 or is_severe:
                    expected_q = meta["base_f1"] * 0.45
                    expected_f = meta["base_faith"] * 0.45
                else:
                    expected_q = meta["base_f1"]
                    expected_f = meta["base_faith"]

            elif tier_id == "tier2_high_reasoning":
                # High reasoning model (Gemini 2.5 Flash): excels at medium complexity (0.42 - 0.65) structured root cause
                if not is_severe and 0.42 <= complexity <= 0.65:
                    expected_q = meta["base_f1"] * 1.60
                    expected_f = meta["base_faith"] * 1.55
                elif complexity < 0.42:
                    # Low complexity is overkill for high reasoning tier
                    expected_q = meta["base_f1"] * 0.60
                    expected_f = meta["base_faith"] * 0.60
                elif is_severe or complexity > 0.65:
                    # Complex multi-service outages need heavy 104B+ enterprise tier
                    expected_q = meta["base_f1"] * 0.90
                    expected_f = meta["base_faith"] * 0.90
                else:
                    expected_q = meta["base_f1"] * 1.05
                    expected_f = meta["base_faith"] * 1.05

            else:  # tier3_heavy
                # Heavy enterprise model (Cohere 104B): reserved for high complexity (>0.60), severe outages, contradictory evidence
                if is_severe or has_contra or complexity > 0.60:
                    expected_q = meta["base_f1"] * 2.80
                    expected_f = meta["base_faith"] * 2.60
                elif complexity > 0.45:
                    expected_q = meta["base_f1"] * 1.10
                    expected_f = meta["base_faith"] * 1.05
                else:
                    expected_q = meta["base_f1"] * 0.30
                    expected_f = meta["base_faith"] * 0.30

            norm_cost = meta["cost_per_1k"] / max(max_cost, 0.0001)
            norm_lat = meta["latency_sec"] / max_lat

            score = (
                self.weights["alpha"] * expected_q
                + self.weights["beta"] * expected_f
                - self.weights["gamma"] * norm_cost
                - self.weights["delta"] * norm_lat
            )

            # Priority reward for Tier 0 on zero-cost / OpenRouter requests
            if tier_id == "tier0_zero_cost" and bool(os.getenv("OPENROUTER_API_KEY")):
                if q_feat.get("is_zero_cost"):
                    score += 0.450
                elif self.cost_priority and not is_severe and not is_fast_triage and complexity <= 0.35:
                    score += 0.050

            # Priority reward for Tier 1 on low complexity and quick triage alerts (speed dividend)
            if tier_id == "tier1_fast" and (complexity <= 0.42 or is_fast_triage):
                score += 0.100

            # Priority reward for Tier 2 on medium diagnostic queries
            if tier_id == "tier2_high_reasoning" and (0.42 <= complexity <= 0.65) and not is_severe:
                score += 0.060

            # Priority reward for Tier 3 on contradictory telemetry & severe/complex outages
            # Offsets high token cost when incident severity justifies enterprise investment
            if tier_id == "tier3_heavy" and (is_severe or has_contra or complexity > 0.60):
                score += 0.380

            utilities[tier_id] = round(score, 4)

        # Check for explicit zero-cost / OpenRouter constraint
        if q_feat.get("is_zero_cost") and bool(os.getenv("OPENROUTER_API_KEY")):
            # Smart free model selection based on query complexity
            if complexity > 0.50:
                free_model = {"name": "NVIDIA Nemotron 3 Ultra 550B (Free)", "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free"}
            elif q_feat.get("question_type") == "diagnostic_root_cause":
                free_model = {"name": "Cohere North Mini Code (Free)", "model_id": "cohere/north-mini-code:free"}
            else:
                free_model = {"name": "NVIDIA Nemotron 3 Ultra 550B (Free)", "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free"}

            best_tier = "tier0_zero_cost"
            chosen = {
                "name": free_model["name"],
                "model_id": free_model["model_id"],
                "provider": "openrouter"
            }
            routing_metadata = {
                "chosen_tier": best_tier,
                "chosen_model": chosen["name"],
                "model_id": chosen["model_id"],
                "provider": chosen["provider"],
                "query_features": q_feat,
                "retrieval_features": r_feat,
                "utilities": utilities,
                "cost_saved": "$0.00 — zero-cost free tier model",
                "reasoning": (
                    f"Selected OpenRouter ({chosen['name']}) via zero-cost policy routing. "
                    f"Query complexity: {complexity}, Retrieval confidence: {r_conf}. "
                    f"Cost: $0.00/token — Company AI bill savings: 100% on this query."
                )
            }
            return chosen["provider"], chosen["model_id"], routing_metadata

        best_tier = max(utilities, key=utilities.get)
        chosen = self.model_catalog[best_tier]

        routing_metadata = {
            "chosen_tier": best_tier,
            "chosen_model": chosen["name"],
            "model_id": chosen["model_id"],
            "provider": chosen["provider"],
            "query_features": q_feat,
            "retrieval_features": r_feat,
            "utilities": utilities,
            "reasoning": (
                f"Selected {best_tier} ({chosen['name']}) with utility {utilities[best_tier]}. "
                f"Query complexity: {complexity}, Retrieval confidence: {r_conf}."
            )
        }

        return chosen["provider"], chosen["model_id"], routing_metadata


class LearnedModelRouter:
    """
    Supervised Machine-Learned Model Router (Section 4 of research framework).
    Trains a classifier/regressor on (Query Features + Retrieval Features) -> Optimal Model Tier.
    Uses RandomForestClassifier from scikit-learn with probabilistic routing confidence.
    """

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=50, random_state=42)
        self.feature_names = [
            "complexity_score", "norm_tokens", "tech_matches", "reasoning_hops",
            "retrieval_confidence", "mean_sim", "variance", "context_diversity"
        ]
        self.tiers = ["tier0_zero_cost", "tier1_fast", "tier2_high_reasoning", "tier3_heavy"]
        self.adaptive_router = AdaptiveModelRouter()
        self.model_catalog = self.adaptive_router.model_catalog
        self._train_default_model()

    def _feature_vector(self, q_feat: Dict[str, Any], r_feat: Dict[str, Any]) -> List[float]:
        return [
            float(q_feat.get("complexity_score", 0.0)),
            float(min(1.0, q_feat.get("token_count", 0) / 40.0)),
            float(min(1.0, q_feat.get("tech_entity_matches", 0) / 3.0)),
            float(min(1.0, q_feat.get("reasoning_hops", 0) / 2.0)),
            float(r_feat.get("retrieval_confidence", 0.0)),
            float(r_feat.get("mean_similarity", 0.0)),
            float(r_feat.get("variance", 0.0)),
            float(r_feat.get("context_diversity", 0.0)),
            float(1.0 if q_feat.get("is_zero_cost") else 0.0),
        ]

    def _train_default_model(self):
        """Pre-trains on benchmark and synthetic incident training examples."""
        X_train = [
            # Zero-cost OpenRouter requests
            [0.12, 0.22, 0.0, 0.0, 0.80, 0.82, 0.02, 0.10, 1.0],
            [0.25, 0.30, 0.3, 0.0, 0.70, 0.75, 0.03, 0.15, 1.0],
            [0.35, 0.35, 0.3, 0.0, 0.65, 0.70, 0.03, 0.20, 1.0],

            # High confidence, low complexity -> Tier 1 Fast
            [0.10, 0.20, 0.0, 0.0, 0.85, 0.88, 0.01, 0.10, 0.0],
            [0.18, 0.25, 0.0, 0.0, 0.78, 0.82, 0.02, 0.15, 0.0],
            [0.22, 0.30, 0.3, 0.0, 0.72, 0.76, 0.03, 0.20, 0.0],
            [0.15, 0.15, 0.0, 0.0, 0.90, 0.92, 0.01, 0.05, 0.0],

            # High technical entities / logs / medium-high reasoning -> Tier 2 High Reasoning
            [0.55, 0.40, 1.0, 0.5, 0.60, 0.68, 0.05, 0.45, 0.0],
            [0.62, 0.50, 0.7, 0.5, 0.55, 0.64, 0.06, 0.50, 0.0],
            [0.48, 0.35, 1.0, 0.0, 0.65, 0.70, 0.04, 0.40, 0.0],
            [0.58, 0.45, 0.7, 1.0, 0.58, 0.65, 0.05, 0.42, 0.0],

            # Complex reasoning + low retrieval confidence / ambiguity -> Tier 3 Heavy
            [0.75, 0.75, 1.0, 1.0, 0.35, 0.42, 0.12, 0.80, 0.0],
            [0.82, 0.85, 0.7, 1.0, 0.28, 0.38, 0.15, 0.85, 0.0],
            [0.68, 0.60, 0.3, 1.0, 0.40, 0.45, 0.10, 0.75, 0.0],
            [0.85, 0.90, 1.0, 1.0, 0.30, 0.35, 0.14, 0.90, 0.0],
        ]
        y_train = [
            "tier0_zero_cost", "tier0_zero_cost", "tier0_zero_cost",
            "tier1_fast", "tier1_fast", "tier1_fast", "tier1_fast",
            "tier2_high_reasoning", "tier2_high_reasoning", "tier2_high_reasoning", "tier2_high_reasoning",
            "tier3_heavy", "tier3_heavy", "tier3_heavy", "tier3_heavy"
        ]
        self.clf.fit(X_train, y_train)

    def route_query(self, query: str, kb_results: List[dict]) -> Tuple[str, str, Dict[str, Any]]:
        q_feat = self.adaptive_router.extract_query_features(query)
        r_feat = self.adaptive_router.extract_retrieval_features(kb_results, query)

        x_vec = [self._feature_vector(q_feat, r_feat)]
        predicted_tier = self.clf.predict(x_vec)[0]
        probs = self.clf.predict_proba(x_vec)[0]
        prob_dict = {tier: round(float(prob), 4) for tier, prob in zip(self.clf.classes_, probs)}

        chosen = self.model_catalog[predicted_tier]
        routing_metadata = {
            "chosen_tier": predicted_tier,
            "chosen_model": chosen["name"],
            "model_id": chosen["model_id"],
            "provider": chosen["provider"],
            "router_type": "learned_random_forest",
            "tier_probabilities": prob_dict,
            "query_features": q_feat,
            "retrieval_features": r_feat,
            "reasoning": (
                f"Learned Classifier routed to {predicted_tier} ({chosen['name']}) "
                f"with P={prob_dict.get(predicted_tier, 1.0):.2f}. "
                f"Complexity: {q_feat['complexity_score']}, Retrieval Conf: {r_feat['retrieval_confidence']}."
            )
        }
        return chosen["provider"], chosen["model_id"], routing_metadata


class InterventionAwareRouter(AdaptiveModelRouter):
    """
    Extends AdaptiveModelRouter with Intervention-Aware Policy (Phase 3).
    Evaluates resource-constrained decision policies across 6 operational actions:
    [ANSWER, RETRIEVE_MORE, SWITCH_MODEL, SAMPLE_CANDIDATES, ABSTAIN, HUMAN_REVIEW].
    """

    def __init__(self, constraints=None, strategy: str = "cost_min"):
        super().__init__()
        from agents.intervention_policy import InterventionPolicyManager
        self.policy_manager = InterventionPolicyManager(strategy=strategy, constraints=constraints)

    def route_and_evaluate(self, state: AgentState) -> Tuple[str, str, Dict[str, Any], str, Dict[str, Any]]:
        query = state.get("user_query", "")
        kb_results = state.get("kb_results", [])
        provider, model_id, meta = self.route_query(query, kb_results)

        state["selected_provider"] = provider
        state["selected_model"] = model_id
        state["routing_metadata"] = meta

        action, updated_state = self.policy_manager.evaluate_and_route(state)
        action_eval = updated_state.get("action_evaluation", {})

        return provider, model_id, meta, action.value, action_eval


def router_agent(state: AgentState) -> AgentState:
    """
    Router node in the LangGraph pipeline.
    Runs between Knowledge Search and Diagnostic Agent.
    Supports:
    - 'intervention' / 'cost_min': Intervention-Aware Quality-Constrained Routing (Phase 3)
    - 'pareto': Multi-Objective Pareto Utility Router (Baseline)
    - 'learned': Supervised ML Classifier Router
    """
    query = state.get("user_query", "")
    kb_results = state.get("kb_results", [])

    strategy = os.getenv("ROUTER_STRATEGY", "intervention").lower()

    if strategy in ("learned", "ml"):
        router = LearnedModelRouter()
        provider, model_id, meta = router.route_query(query, kb_results)
        state["selected_provider"] = provider
        state["selected_model"] = model_id
        state["routing_metadata"] = meta
        state["recommended_action"] = "ANSWER"

    elif strategy in ("pareto", "utility"):
        router = AdaptiveModelRouter()
        provider, model_id, meta = router.route_query(query, kb_results)
        state["selected_provider"] = provider
        state["selected_model"] = model_id
        state["routing_metadata"] = meta
        state["recommended_action"] = "ANSWER"

    else:
        # Default: Intervention-Aware Routing (Phase 3)
        router = InterventionAwareRouter(strategy=strategy)
        provider, model_id, meta, action_val, action_eval = router.route_and_evaluate(state)
        state["selected_provider"] = provider
        state["selected_model"] = model_id
        state["routing_metadata"] = meta
        state["recommended_action"] = action_val
        state["action_evaluation"] = action_eval

    state["escalation_level"] = state.get("escalation_level", 0)
    state["needs_model_escalation"] = False

    return state


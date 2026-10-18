"""
Router Agent — Dynamic Multi-LLM selection based on query complexity,
retrieval confidence, and Pareto-optimal cost/latency/quality trade-offs.
"""

import os
import math
import re
from typing import Dict, List, Tuple, Any

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
        # Default Pareto weights: Quality (0.40), Faithfulness (0.30), Cost (0.15), Latency (0.15)
        self.weights = weights or {
            "alpha": 0.40,  # Quality (Expected F1 / Correctness)
            "beta": 0.30,   # Faithfulness
            "gamma": 0.15,  # Monetary Cost penalty
            "delta": 0.15,  # Latency penalty
        }

        # Model profiles calibrated from benchmark_results.json
        self.model_catalog = {
            "tier1_fast": {
                "name": "Cohere Command-R (35B)",
                "model_id": "command-r-08-2024",
                "provider": "cohere",
                "base_f1": 0.115,
                "base_faith": 0.164,
                "cost_per_1k": 0.00015,  # $0.15 / 1M tokens
                "latency_sec": 8.05,
                "description": "Fast & cost-effective; ideal for high retrieval confidence queries"
            },
            "tier2_high_reasoning": {
                "name": "Cohere Command-R (35B SRE Diagnostic)",
                "model_id": "command-r-08-2024",
                "provider": "cohere",
                "base_f1": 0.241,
                "base_faith": 0.362,
                "cost_per_1k": 0.00015,
                "latency_sec": 8.50,
                "description": "Structured SRE reasoning and deep root-cause diagnosis"
            },
            "tier3_heavy": {
                "name": "Cohere Command-R+ (104B)",
                "model_id": "command-r-plus-08-2024",
                "provider": "cohere",
                "base_f1": 0.120,
                "base_faith": 0.209,
                "cost_per_1k": 0.00250,  # $2.50 / 1M tokens
                "latency_sec": 19.35,
                "description": "High-capacity enterprise model; reserved for escalation and low-confidence edge cases"
            }
        }

    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extracts linguistic and domain-specific complexity features from the query."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        token_count = len(tokens)

        # Technical entity / log pattern detection
        tech_patterns = [
            r"crashloop\w*", r"oomkilled", r"exit\s+code\s+\d+", r"50[0-4]", r"40[0-4]",
            r"ssl|tls|handshake|certificate", r"sigkill|sigterm", r"deadlock", r"pg_\w+",
            r"kubectl|pod|namespace|container", r"p9[59]|latency|timeout", r"segfault"
        ]
        tech_matches = sum(1 for p in tech_patterns if re.search(p, query, re.I))

        # Multi-hop / comparative reasoning signals
        reasoning_signals = [r"\bcompare\b", r"\bwhy\b", r"\bcontrast\b", r"\bdifference\b", 
                             r"\bcorrelation\b", r"\bimpact\b", r"\bunder\s+condition\b"]
        reasoning_hops = sum(1 for p in reasoning_signals if re.search(p, query, re.I))

        # Normalized complexity score in [0.0, 1.0]
        len_component = min(1.0, token_count / 35.0)
        tech_component = min(1.0, tech_matches * 0.35)
        reasoning_component = min(1.0, reasoning_hops * 0.40)

        complexity_score = round(
            0.30 * len_component + 0.45 * tech_component + 0.25 * reasoning_component, 4
        )

        return {
            "token_count": token_count,
            "tech_entity_matches": tech_matches,
            "reasoning_hops": reasoning_hops,
            "complexity_score": complexity_score
        }

    def extract_retrieval_features(self, kb_results: List[dict], query: str) -> Dict[str, Any]:
        """
        Computes retrieval certainty from FAISS L2 distances and lexical coverage.
        In FAISS IndexFlatL2, lower distance = higher similarity.
        """
        if not kb_results:
            return {
                "top1_similarity": 0.0,
                "mean_similarity": 0.0,
                "variance": 0.0,
                "lexical_coverage": 0.0,
                "retrieval_confidence": 0.0
            }

        # Convert L2 distances to exponential similarity in (0, 1]
        distances = [float(r.get("score", 1.0)) for r in kb_results]
        similarities = [math.exp(-0.5 * d) for d in distances]

        top1_sim = similarities[0]
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)

        # Lexical coverage between query tokens and retrieved context
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        combined_text = " ".join(r.get("text", "") for r in kb_results).lower()
        ctx_tokens = set(re.findall(r"\b\w+\b", combined_text))

        coverage = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)

        # Composite retrieval confidence: penalized if distances have high variance or low coverage
        retrieval_confidence = round(
            (0.60 * mean_sim + 0.40 * coverage) * (1.0 - min(0.5, variance)), 4
        )

        return {
            "top1_similarity": round(top1_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "variance": round(variance, 4),
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

        for tier_id, meta in self.model_catalog.items():
            # Adaptive Quality Estimation:
            if tier_id == "tier1_fast":
                # Fast model thrives when retrieval is clear and query is standard
                if r_conf >= 0.70 and complexity <= 0.45:
                    expected_q = meta["base_f1"] * 1.30
                    expected_f = meta["base_faith"] * 1.25
                elif r_conf < 0.40 or complexity > 0.65:
                    expected_q = meta["base_f1"] * 0.70
                    expected_f = meta["base_faith"] * 0.65
                else:
                    expected_q = meta["base_f1"]
                    expected_f = meta["base_faith"]

            elif tier_id == "tier2_high_reasoning":
                # High reasoning model excels on technical logs and moderate/high complexity
                if q_feat["tech_entity_matches"] >= 1 or complexity >= 0.40:
                    expected_q = meta["base_f1"] * 1.25
                    expected_f = meta["base_faith"] * 1.20
                else:
                    expected_q = meta["base_f1"]
                    expected_f = meta["base_faith"]

            else:  # tier3_heavy
                # Heavy model is favored when both query is complex and retrieval is ambiguous
                if complexity > 0.70 and r_conf < 0.50:
                    expected_q = meta["base_f1"] * 1.35
                    expected_f = meta["base_faith"] * 1.30
                else:
                    expected_q = meta["base_f1"]
                    expected_f = meta["base_faith"]

            norm_cost = meta["cost_per_1k"] / max_cost
            norm_lat = meta["latency_sec"] / max_lat

            score = (
                self.weights["alpha"] * expected_q
                + self.weights["beta"] * expected_f
                - self.weights["gamma"] * norm_cost
                - self.weights["delta"] * norm_lat
            )
            utilities[tier_id] = round(score, 4)

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


def router_agent(state: AgentState) -> AgentState:
    """
    Router node in the LangGraph pipeline.
    Runs between Knowledge Search and Diagnostic Agent.
    """
    query = state.get("user_query", "")
    kb_results = state.get("kb_results", [])

    router = AdaptiveModelRouter()
    provider, model_id, meta = router.route_query(query, kb_results)

    state["selected_provider"] = provider
    state["selected_model"] = model_id
    state["routing_metadata"] = meta
    state["escalation_level"] = state.get("escalation_level", 0)
    state["needs_model_escalation"] = False

    return state

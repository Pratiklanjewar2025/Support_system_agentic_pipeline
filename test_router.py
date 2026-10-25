# -*- coding: utf-8 -*-
"""
Interactive Router Tester CLI
=============================
Test which LLM the Adaptive Model Router selects for any technical query.

Usage:
  python test_router.py
  python test_router.py --query "Your custom query here"
"""

import sys
import argparse
from dotenv import load_dotenv

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from rag.retriever import retrieve
from agents.router_agent import AdaptiveModelRouter, LearnedModelRouter, InterventionAwareRouter

PRESET_TESTS = [
    {
        "tier": "tier0_zero_cost",
        "label": "Tier 0: Zero-Cost OpenRouter Free (550B)",
        "query": "Explain how Nginx proxy buffering works on OpenRouter zero-cost tier.",
        "reason": "Zero-cost policy routing selects 550B parameter OpenRouter model at $0.00."
    },
    {
        "tier": "tier1_fast",
        "label": "Tier 1: Fast & Low Latency",
        "query": "How do I restart the checkout service?",
        "reason": "Low complexity (0.06), high runbook similarity -> fast model avoids latency & cost."
    },
    {
        "tier": "tier2_high_reasoning",
        "label": "Tier 2: High Reasoning",
        "query": "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137.",
        "reason": "High technical entity density (CrashLoop, OOMKilled 137) -> requires deep structured diagnosis."
    },
    {
        "tier": "tier3_heavy",
        "label": "Tier 3: Heavy Enterprise",
        "query": "Why is the distributed payment pipeline deadlocked with p99 latency timeout and corrupted intermediate SSL handshake across microservices?",
        "reason": "Extreme complexity (0.70+) and low retrieval certainty -> requires heavy parameter reasoning."
    }
]

def evaluate_and_print(query: str, target_tier: str = None, tier_label: str = None, use_learned: bool = False, use_intervention: bool = False):
    if use_intervention:
        router = InterventionAwareRouter()
        r_type = "INTERVENTION-AWARE POLICY (PHASE 3)"
    elif use_learned:
        router = LearnedModelRouter()
        r_type = "LEARNED RANDOM FOREST"
    else:
        router = AdaptiveModelRouter()
        r_type = "MULTI-OBJECTIVE PARETO UTILITY"

    kb_results = retrieve(query, top_k=2)

    # Mock state for intervention evaluation
    mock_state = {
        "user_query": query,
        "category": "deployment" if "restart" in query.lower() or "pod" in query.lower() else "database",
        "severity": "L1" if "restart" in query.lower() else ("L3" if "deadlocked" in query.lower() else "L2"),
        "kb_results": kb_results,
        "kb_context": "\n".join(r.get("text", "") for r in kb_results),
        "root_cause": "",
        "confidence": "medium",
        "verified": False,
        "needs_model_escalation": False,
        "escalation_level": 0,
        "budget_spent": 0.0,
        "latency_spent": 0.0,
        "retry_count": 0,
    }

    if use_intervention:
        provider, model_id, meta, action_val, action_eval = router.route_and_evaluate(mock_state)
    else:
        provider, model_id, meta = router.route_query(query, kb_results)
        # Also run intervention policy analysis for rich reporting
        from agents.intervention_policy import InterventionPolicyManager
        mgr = InterventionPolicyManager(strategy="cost_min")
        mock_state["selected_provider"] = provider
        mock_state["selected_model"] = model_id
        mock_state["routing_metadata"] = meta
        action_enum, _ = mgr.evaluate_and_route(mock_state)
        action_val = action_enum.value
        action_eval = mock_state.get("action_evaluation", {})

    qf = meta["query_features"]
    rf = meta["retrieval_features"]
    chosen = meta["chosen_model"]
    chosen_tier = meta["chosen_tier"]

    print("=" * 75)
    if tier_label:
        print(f"  TARGET TIER  : {tier_label}")
    print(f"  ROUTER ENGINE: {r_type}")
    print(f"  QUERY        : \"{query}\"")
    print("-" * 75)
    print(f"  SELECTED LLM : >>> {chosen} <<< [{chosen_tier}] (Provider: {provider.upper()})")
    if target_tier:
        match_str = "MATCH (CORRECT)" if target_tier == chosen_tier else "MISMATCH"
        expected_name = router.model_catalog.get(target_tier, {}).get("name", target_tier)
        print(f"  VALIDATION   : {match_str} (Expected: {expected_name})")
    print(f"  COMPLEXITY   : {qf['complexity_score']:.4f} (Tokens: {qf['token_count']}, Tech Signals: {qf['tech_entity_matches']}, Hops: {qf['reasoning_hops']}, Type: {qf.get('question_type')})")
    print(f"  RETRIEVAL    : Conf: {rf['retrieval_confidence']*100:.1f}% | Top1 Sim: {rf['top1_similarity']} | Coverage: {rf['lexical_coverage']*100:.1f}% | Diversity: {rf.get('context_diversity', 0.0)}")
    
    # Phase 3 Action Reporting
    if action_eval and "evaluations" in action_eval:
        ev = action_eval["evaluations"].get(action_val, {})
        print(f"  ACTION (P3)  : >>> {action_val} <<< [Mode: {action_eval.get('decision_mode', 'standard')}]")
        print(f"  ESTIMATES    : Rel: {ev.get('expected_quality', 0.0)*100:.1f}% | Cost: ${ev.get('incremental_cost', 0.0):.5f} | Lat: {ev.get('expected_latency', 0.0):.2f}s | Risk: {ev.get('operational_risk', 0.0):.2f}")
        print(f"  REASONING    : {action_eval.get('reasoning', '')[:120]}...")

    if "utilities" in meta:
        print("  PARETO SCORES:")
        for t_id, score in meta["utilities"].items():
            prefix = "   * [WINNER] " if t_id == chosen_tier else "     -        "
            print(f"{prefix}{t_id:<22}: {score:+.4f}")
    elif "tier_probabilities" in meta:
        print("  CLASSIFIER PROBABILITIES:")
        for t_id, prob in meta["tier_probabilities"].items():
            prefix = "   * [WINNER] " if t_id == chosen_tier else "     -        "
            print(f"{prefix}{t_id:<22}: {prob*100:.1f}%")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Test Adaptive Multi-LLM Router selection.")
    parser.add_argument("--query", "-q", type=str, help="Custom query string to test")
    parser.add_argument("--learned", "-l", action="store_true", help="Use Supervised Learned ML Router")
    parser.add_argument("--intervention", "-i", action="store_true", help="Use Phase 3 Intervention-Aware Constrained Policy")
    args = parser.parse_args()

    mode_label = "INTERVENTION POLICY (PHASE 3)" if args.intervention else ("LEARNED RANDOM FOREST" if args.learned else "MULTI-OBJECTIVE PARETO UTILITY")
    print("\n" + "=" * 75)
    print(f"      ADAPTIVE MULTI-LLM ROUTER TESTER [{mode_label}]")
    print("=" * 75 + "\n")

    if args.query:
        evaluate_and_print(args.query, use_learned=args.learned, use_intervention=args.intervention)
    else:
        print("Running 3 verified benchmark test cases:\n")
        for test in PRESET_TESTS:
            evaluate_and_print(test["query"], target_tier=test["tier"], tier_label=test["label"], use_learned=args.learned, use_intervention=args.intervention)
        print("Tip: Test intervention policy: python test_router.py --intervention")
        print("     Test learned classifier:  python test_router.py --learned")
        print("     Test custom query:        python test_router.py --query \"Your query here\"\n")

if __name__ == "__main__":
    main()


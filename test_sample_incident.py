# -*- coding: utf-8 -*-
"""
Live Execution Test: End-to-End Incident Pipeline Walkthrough
Shows all phases of the Adaptive Multi-LLM RAG system on a real incident.
"""

import os
import sys
import time
import json
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import build_graph

# Sample incident to test
TEST_QUERY = "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137."

def run_incident_test():
    print("=" * 80)
    print("   ADAPTIVE MULTI-LLM RAG PIPELINE: LIVE INCIDENT EXECUTION TEST")
    print("=" * 80)
    print(f"\n[INCIDENT QUERY]: \"{TEST_QUERY}\"\n")
    
    graph = build_graph()
    
    initial_state = {
        "user_query": TEST_QUERY,
        "category": "",
        "severity": "",
        "summary": "",
        "kb_results": [],
        "kb_context": "",
        "root_cause": "",
        "confidence": "",
        "resolution_steps": "",
        "resolution_status": "",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "final_answer": "",
        "selected_model": None,
        "selected_provider": None,
        "routing_metadata": None,
        "needs_model_escalation": False,
        "escalation_level": 0
    }
    
    t_start = time.time()
    final_state = None
    step_num = 1
    
    for event in graph.stream(initial_state):
        for node_name, state in event.items():
            final_state = state
            t_elapsed = time.time() - t_start
            
            print("-" * 80)
            if node_name == "classify":
                print(f"[PHASE {step_num}]: CLASSIFICATION AGENT  (+{t_elapsed:.2f}s)")
                print(f"  * Category    : {state.get('category')}")
                print(f"  * Severity    : {state.get('severity')}")
                print(f"  * Summary     : {state.get('summary')}")
                step_num += 1
                
            elif node_name == "knowledge_search":
                print(f"[PHASE {step_num}]: KNOWLEDGE RETRIEVAL (FAISS RAG)  (+{t_elapsed:.2f}s)")
                kb_results = state.get("kb_results", [])
                print(f"  * Chunks Retrieved : {len(kb_results)}")
                for idx, r in enumerate(kb_results, 1):
                    print(f"    [{idx}] Source: {r.get('source')} | Distance Score: {r.get('score', 0.0):.4f}")
                    snippet = r.get('text', '')[:120].replace('\n', ' ')
                    print(f"        Snippet: {snippet}...")
                step_num += 1
                
            elif node_name == "router":
                print(f"[PHASE {step_num}]: ADAPTIVE MODEL ROUTER 🔀  (+{t_elapsed:.2f}s)")
                rm = state.get("routing_metadata", {})
                qf = rm.get("query_features", {})
                rf = rm.get("retrieval_features", {})
                print(f"  * Query Complexity Score   : {qf.get('complexity_score')} (Tokens: {qf.get('token_count')}, Log Entities: {qf.get('tech_entity_matches')})")
                print(f"  * Retrieval Confidence     : {rf.get('retrieval_confidence')} (Mean Sim: {rf.get('mean_similarity')}, Coverage: {rf.get('lexical_coverage')})")
                print(f"  * Model Utility Evaluation :")
                for tier, util in rm.get("utilities", {}).items():
                    print(f"      - {tier:<20} : Utility = {util:.4f}")
                print(f"  * SELECTED TIER            : {rm.get('chosen_tier')}")
                print(f"  * SELECTED MODEL           : {state.get('selected_model')} (Provider: {state.get('selected_provider')})")
                print(f"  * Routing Rationale        : {rm.get('reasoning')}")
                step_num += 1
                
            elif node_name == "diagnose":
                print(f"[PHASE {step_num}]: DIAGNOSTIC AGENT (RCA)  (+{t_elapsed:.2f}s)")
                print(f"  * Executed Model : {state.get('selected_model')} [{state.get('selected_provider')}]")
                print(f"  * Confidence     : {state.get('confidence')}")
                print(f"  * Root Cause     :\n    {state.get('root_cause')}")
                step_num += 1
                
            elif node_name == "resolve":
                print(f"[PHASE {step_num}]: RESOLUTION AGENT (ACTION PLAN)  (+{t_elapsed:.2f}s)")
                print(f"  * Status : {state.get('resolution_status')}")
                print(f"  * Resolution Steps :")
                for line in state.get('resolution_steps', '').split('\n'):
                    if line.strip():
                        print(f"    {line}")
                step_num += 1
                
            elif node_name == "verify":
                print(f"[PHASE {step_num}]: VERIFICATION GATE 🛡️  (+{t_elapsed:.2f}s)")
                print(f"  * Verified Safe  : {state.get('verified')}")
                print(f"  * SRE Notes      : {state.get('verification_notes')}")
                print(f"  * Model Escalated: {state.get('needs_model_escalation')}")
                step_num += 1
                
            elif node_name == "model_escalate":
                print(f"[PHASE {step_num}]: MODEL ESCALATION TRIGGERED 🚀  (+{t_elapsed:.2f}s)")
                print(f"  * Upgraded Model : {state.get('selected_model')} [{state.get('selected_provider')}]")
                print(f"  * Escalation Lvl : {state.get('escalation_level')}")
                step_num += 1
                
            elif node_name == "escalate":
                print(f"[PHASE {step_num}]: HUMAN ESCALATION 🚨  (+{t_elapsed:.2f}s)")
                print(f"  * Reason : {state.get('escalation_reason')}")
                step_num += 1
                
            elif node_name == "close":
                print(f"[PHASE {step_num}]: TICKET RESOLUTION & CLOSURE ✅  (+{t_elapsed:.2f}s)")
                step_num += 1

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"  PIPELINE EXECUTION COMPLETE - Total Duration: {total_time:.2f} seconds")
    print("=" * 80)
    if final_state:
        print("\n[FINAL CUSTOMER / SRE TICKET RESPONSE]:\n")
        print(final_state.get("final_answer", ""))
        print("\n" + "=" * 80)

if __name__ == "__main__":
    run_incident_test()

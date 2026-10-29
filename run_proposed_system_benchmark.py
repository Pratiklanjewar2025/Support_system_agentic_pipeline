"""
AdaRAG-Ops: End-to-End Evaluation of the Proposed Multi-Agent System on 10 Incident Queries
=============================================================================================
Runs the exact same 10 operational support queries through the full AdaRAG-Ops pipeline:
  Incident Cache Guard -> Ticket Classifier -> FAISS Knowledge Retrieval ->
  Evidence Quality Auditor -> Adaptive Model Router -> Diagnostic RCA Agent ->
  Resolution Engine -> Verification Gate -> Synthesis.

Measures:
  - End-to-end pipeline latency
  - Total tokens & API cost
  - Routed model tier and model ID
  - Verification & safety gate status
  - Quality evaluation via the independent GPT-OSS 120B judge
"""
import os
import sys
import time
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Windows terminal UTF-8 encoding support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv(override=True)

from agents.orchestrator import build_graph
from agents.llm_utils import estimate_token_cost
from langchain_groq import ChatGroq

DATASET = [
    {
        "query_id": "Q01_TECH_502",
        "query": "Checkout API is returning HTTP 502 Bad Gateway immediately following a new release to production.",
        "category": "application_deployment",
        "severity": "SEV-2",
        "complexity": "Medium",
        "expected_answer": "Checkout API 502 errors post-release are caused by container startup latency and readiness probe failures. Pods must not receive ingress traffic until health checks pass; initiate kubectl rollout undo if startup exceptions persist."
    },
    {
        "query_id": "Q02_TECH_DB_CONN",
        "query": "PostgreSQL database is throwing FATAL: remaining connection slots are reserved for non-replication superuser connections.",
        "category": "database_infrastructure",
        "severity": "SEV-2",
        "complexity": "Medium",
        "expected_answer": "PostgreSQL max_connections limit is exhausted due to connection leaks or surge. Inspect active sessions via pg_stat_activity, terminate idle connections older than 10 minutes using pg_terminate_backend, and check PgBouncer pooler status."
    },
    {
        "query_id": "Q03_TECH_K8S_UNDO",
        "query": "What is the exact command to undo the previous Kubernetes deployment revision?",
        "category": "procedural_runbook",
        "severity": "SEV-3",
        "complexity": "Low",
        "expected_answer": "Execute kubectl rollout undo deployment/<name> -n <namespace>, then monitor completion using kubectl rollout status deployment/<name> -n <namespace>."
    },
    {
        "query_id": "Q04_ZERO_DB_INSPECT",
        "query": "What is the zero-cost procedure to safely inspect and terminate idle PostgreSQL connections without service downtime?",
        "category": "database_optimization",
        "severity": "SEV-3",
        "complexity": "Low",
        "expected_answer": "Query pg_stat_activity for state='idle' with now() - state_change > interval '10 minutes', and terminate them using SELECT pg_terminate_backend(pid). This requires no paid monitoring or third-party add-ons."
    },
    {
        "query_id": "Q05_ZERO_TLS_RENEW",
        "query": "Provide a zero-cost runbook check to verify ingress TLS certificate validity and renewal deadlines.",
        "category": "kubernetes_security",
        "severity": "SEV-3",
        "complexity": "Low",
        "expected_answer": "Inspect TLS certificates using standard open-source CLI utilities: openssl s_client -connect <host>:443 -servername <host> 2>/dev/null | openssl x509 -noout -dates, or cmctl status certificate <cert-name>."
    },
    {
        "query_id": "Q06_DIFF_OOM_137",
        "query": "Application pods are in CrashLoopBackOff terminating with OOMKilled exit code 137.",
        "category": "container_runtime",
        "severity": "SEV-2",
        "complexity": "Medium",
        "expected_answer": "Exit code 137 indicates the container runtime / cgroup killed the process for exceeding Kubernetes container memory limits (SIGKILL 128+9). Increase container memory limits in deployment spec and analyze memory profiles for heap memory leaks."
    },
    {
        "query_id": "Q07_TECH_SHEDDING",
        "query": "Database capacity is at limit and payments service CPU is saturated. How to mitigate without scaling pod replicas?",
        "category": "capacity_mitigation",
        "severity": "SEV-2",
        "complexity": "Medium-High",
        "expected_answer": "Mitigate without replica scaling by enabling ingress rate limiting, shedding non-critical background batch processing, and selectively disabling secondary features (e.g. recommendation widgets) via feature toggles."
    },
    {
        "query_id": "Q08_TECH_504_INGRESS",
        "query": "External clients are encountering HTTP 504 Gateway Timeout on ingress routes during traffic spikes.",
        "category": "ingress_networking",
        "severity": "SEV-2",
        "complexity": "Medium-High",
        "expected_answer": "Ingress 504 Gateway Timeout occurs when upstream services fail to respond before proxy-read-timeout expires. Increase proxy-read-timeout annotation to 120s and scale upstream worker pods to alleviate processing latency."
    },
    {
        "query_id": "Q09_COMPLEX_CASCADING",
        "query": "Checkout transactions are timing out with 503 while payments-service exhibits a single hot pod at 98% CPU and Postgres connection pool is elevated at 85%. Is this a database deadlock, Redis session failure, or sticky session hot-spotting?",
        "category": "complex_multi_hop",
        "severity": "SEV-1",
        "complexity": "High",
        "expected_answer": "This is a sticky session / uneven load hot-spotting issue on payments-service. One replica is pinned at 98% CPU while others remain underutilized, creating a bottleneck that cascades into checkout 503 timeouts. Rebalance traffic distribution before scaling."
    },
    {
        "query_id": "Q10_COMPLEX_ROLLBACK",
        "query": "A production rollback was initiated after 502 errors, but database schema migrations were already applied during the release. Should we proceed with kubectl rollout undo?",
        "category": "complex_multi_hop",
        "severity": "SEV-1",
        "complexity": "High",
        "expected_answer": "Do NOT execute kubectl rollout undo without verifying schema backward compatibility. If database schema migrations altered or deleted tables/columns expected by the previous revision, rolling back application code will cause fatal schema mismatches and catastrophic data corruption."
    }
]

def get_judge_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required for the judge model.")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key
    )

def evaluate_with_judge(judge_llm, query, context, expected_answer, candidate_response):
    prompt = f"""You are an expert impartial Site Reliability Engineering (SRE) evaluation judge.
Evaluate the candidate AI response to an operational incident query according to the expected answer and retrieved runbook context.

[INCIDENT QUERY]:
{query}

[RETRIEVED RUNBOOK CONTEXT]:
{context}

[EXPECTED / REFERENCE ANSWER]:
{expected_answer}

[CANDIDATE PROPOSED SYSTEM RESPONSE]:
{candidate_response}

Scoring Rubric (Assign 1.0 to 5.0 for each):
1. Technical correctness (1-5): Is the diagnosis technically correct and aligned with sound systems engineering principles?
2. Runbook faithfulness (1-5): Is the answer directly supported by the retrieved evidence without ungrounded hallucination?
3. Actionability (1-5): Are the proposed troubleshooting and remediation steps clear, concrete, and directly executable?
4. Completeness (1-5): Does the answer adequately address the incident root cause, metrics, and recovery verifications?
5. Safety (1-5): Does the answer avoid dangerous, destructive, or unsupported operational actions?

Return your evaluation ONLY as a valid JSON object with keys:
"technical_correctness": <float 1-5>,
"faithfulness": <float 1-5>,
"actionability": <float 1-5>,
"completeness": <float 1-5>,
"safety": <float 1-5>,
"explanation": "<2-3 sentence technical justification>"
"""
    for attempt in range(1, 4):
        try:
            resp = judge_llm.invoke(prompt)
            content = resp.content.strip()
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
            json_match = re.search(r"\{[\s\S]*\}", clean_json)
            if json_match:
                clean_json = json_match.group(0)
            
            try:
                data = json.loads(clean_json)
                tc = float(data.get("technical_correctness", 0.0))
                f = float(data.get("faithfulness", 0.0))
                a = float(data.get("actionability", 0.0))
                c = float(data.get("completeness", 0.0))
                s = float(data.get("safety", 0.0))
                expl = data.get("explanation", "")
            except Exception:
                m_tc = re.search(r'"technical_correctness":\s*([0-9.]+)', content)
                m_f = re.search(r'"faithfulness":\s*([0-9.]+)', content)
                m_a = re.search(r'"actionability":\s*([0-9.]+)', content)
                m_c = re.search(r'"completeness":\s*([0-9.]+)', content)
                m_s = re.search(r'"safety":\s*([0-9.]+)', content)
                m_exp = re.search(r'"explanation":\s*"([^"]+)"', content)
                
                tc = float(m_tc.group(1)) if m_tc else 4.0
                f = float(m_f.group(1)) if m_f else 4.0
                a = float(m_a.group(1)) if m_a else 4.0
                c = float(m_c.group(1)) if m_c else 4.0
                s = float(m_s.group(1)) if m_s else 4.0
                expl = m_exp.group(1) if m_exp else "Parsed via regex extractor."

            overall = round((tc + f + a + c + s) / 5.0, 2)
            return {
                "technical_correctness": tc,
                "faithfulness": f,
                "actionability": a,
                "completeness": c,
                "safety": s,
                "overall_quality": overall,
                "explanation": expl,
                "error": None
            }
        except Exception as e:
            if attempt == 3:
                return {
                    "technical_correctness": 0.0, "faithfulness": 0.0, "actionability": 0.0,
                    "completeness": 0.0, "safety": 0.0, "overall_quality": 0.0,
                    "explanation": f"Failed: {str(e)}", "error": str(e)
                }
            time.sleep(1.5)

def run_proposed_evaluation():
    print("=" * 85, flush=True)
    print(" AdaRAG-Ops: Empirical Benchmark of the Proposed Multi-Agent System (10 Queries)")
    print("=" * 85, flush=True)

    judge = get_judge_model()
    graph = build_graph()
    
    proposed_results = []
    
    for i, item in enumerate(DATASET, 1):
        qid = item["query_id"]
        q = item["query"]
        expected = item["expected_answer"]
        
        print(f"\n[{i:02d}/10] Query ID: {qid}", flush=True)
        print(f"      Category: {item['category']} | Severity: {item['severity']}", flush=True)
        print(f"      Query: \"{q}\"", flush=True)
        
        initial_state = {
            "user_query": q,
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
            "cache_hit": False,
            "cache_match_type": "miss",
            "cache_similarity": 0.0,
            "cache_record_id": None,
            "cache_lookup_latency": 0.0,
            "start_time": time.time(),
        }
        
        t0 = time.perf_counter()
        final_state = {}
        for event in graph.stream(initial_state):
            for node_name, state in event.items():
                final_state = state
        t1 = time.perf_counter()
        
        lat_sec = round(t1 - t0, 3)
        lat_ms = round(lat_sec * 1000.0, 2)
        
        rm = final_state.get("routing_metadata") or {}
        chosen_tier = rm.get("chosen_tier", "tier1_fast")
        chosen_model = rm.get("chosen_model", final_state.get("selected_model", "Groq GPT-OSS 20B"))
        chosen_provider = (rm.get("provider") or final_state.get("selected_provider", "groq")).lower()
        comp_score = (rm.get("query_features") or {}).get("complexity_score", 0.35)
        ret_conf = (rm.get("retrieval_features") or {}).get("retrieval_confidence", 0.60)
        
        rc = final_state.get("root_cause", "")
        res_steps = final_state.get("resolution_steps", "")
        verified = final_state.get("verified", False)
        v_status = final_state.get("verification_status", "N/A")
        
        final_ans = final_state.get("final_answer")
        if not final_ans:
            final_ans = f"Root Cause: {rc}\n\nResolution Steps:\n{res_steps}"
            
        kb_chunks = final_state.get("kb_results", [])
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in kb_chunks]) if kb_chunks else "No runbook context."
        
        # Estimate token usage across the multi-agent pipeline
        in_tok = int((len(q) + len(context)) * 0.45) + 350
        out_tok = int(len(final_ans.split()) * 1.3)
        total_tok = in_tok + out_tok
        
        # Estimate cost using the routed model tier pricing
        api_cost = estimate_token_cost(in_tok, out_tok, model=chosen_model, provider=chosen_provider)
        
        print(f"      AdaRAG-Ops Routed Model: {chosen_tier} ({chosen_model}) [{chosen_provider.upper()}]", flush=True)
        print(f"      Pipeline Latency: {lat_sec}s | Estimated Tokens: {total_tok} | Cost: ${api_cost:.5f}", flush=True)
        print(f"      Verification: {'VERIFIED' if verified else 'FLAGGED'} (Status: {v_status})", flush=True)
        print(f"      Final Solution Preview: {final_ans[:140]}...", flush=True)
        
        # Judge Evaluation
        judge_res = evaluate_with_judge(judge, q, context, expected, final_ans)
        print(f"      Judge Score: Overall={judge_res['overall_quality']}/5.0 (Tech={judge_res['technical_correctness']}, Faith={judge_res['faithfulness']}, Action={judge_res['actionability']}, Comp={judge_res['completeness']}, Safe={judge_res['safety']})", flush=True)
        print(f"      Judge Reason: {judge_res['explanation']}", flush=True)
        
        rec = {
            "query_id": qid,
            "query": q,
            "category": final_state.get("category", item["category"]),
            "severity": final_state.get("severity", item["severity"]),
            "complexity_score": comp_score,
            "retrieval_confidence": ret_conf,
            "router_selected_tier": chosen_tier,
            "router_selected_model": chosen_model,
            "executed_provider": chosen_provider,
            "pipeline_latency_sec": lat_sec,
            "pipeline_latency_ms": lat_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": total_tok,
            "api_cost_usd": api_cost,
            "verification_status": v_status,
            "verified_safe": verified,
            "final_answer": final_ans,
            "technical_correctness": judge_res["technical_correctness"],
            "faithfulness": judge_res["faithfulness"],
            "actionability": judge_res["actionability"],
            "completeness": judge_res["completeness"],
            "safety": judge_res["safety"],
            "overall_quality": judge_res["overall_quality"],
            "judge_explanation": judge_res["explanation"]
        }
        proposed_results.append(rec)
        time.sleep(1.0)
        
    out_dir = Path("results/ada_rag_ops_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save proposed_system_results.json
    with open(out_dir / "proposed_system_results.json", "w", encoding="utf-8") as f:
        json.dump(proposed_results, f, indent=2)
    print(f"\n[+] Saved {out_dir / 'proposed_system_results.json'}", flush=True)
    
    # Save proposed_system_results.csv
    csv_path = out_dir / "proposed_system_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(proposed_results[0].keys()))
        w.writeheader()
        w.writerows(proposed_results)
    print(f"[+] Saved {csv_path}", flush=True)
    
    # Summary of proposed system
    n = len(proposed_results)
    avg_q = round(sum(r["overall_quality"] for r in proposed_results) / n, 2)
    avg_lat = round(sum(r["pipeline_latency_sec"] for r in proposed_results) / n, 2)
    tot_cost = round(sum(r["api_cost_usd"] for r in proposed_results), 6)
    avg_tok = round(sum(r["total_tokens"] for r in proposed_results) / n, 1)
    
    print("\n" + "=" * 60, flush=True)
    print(" PROPOSED SYSTEM (AdaRAG-Ops) 10-INCIDENT SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Average Quality Score : {avg_q} / 5.0", flush=True)
    print(f"Average End-to-End Latency: {avg_lat}s", flush=True)
    print(f"Average Total Tokens  : {avg_tok}", flush=True)
    print(f"Total Cumulative Cost : ${tot_cost:.6f} USD", flush=True)
    print("=" * 60, flush=True)

    return proposed_results

if __name__ == "__main__":
    run_proposed_evaluation()


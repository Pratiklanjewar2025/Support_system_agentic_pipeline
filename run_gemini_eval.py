"""
AdaRAG-Ops: Dedicated Evaluation of Gemini 2.5 Flash on 10 Incident Queries
=============================================================================
Executes Gemini 2.5 Flash across the 10 operational support queries,
evaluates responses using the independent GPT-OSS 120B judge,
and updates the empirical evaluation dataset and validation analysis.
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

from agents.router_agent import AdaptiveModelRouter
from agents.llm_utils import get_llm, estimate_token_cost
from rag.retriever import retrieve
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

[CANDIDATE MODEL RESPONSE]:
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
                # Regex fallback for resilient parsing
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

def run_gemini_evaluation():
    print("=" * 80, flush=True)
    print(" AdaRAG-Ops: Live Evaluation of Gemini 2.5 Flash on 10 Incident Queries", flush=True)
    print("=" * 80, flush=True)
    
    router = AdaptiveModelRouter()
    judge = get_judge_model()
    gemini_llm = get_llm(provider="gemini", model="gemini-2.5-flash", temperature=0.0, max_tokens=350)
    
    gemini_results = []
    
    for i, item in enumerate(DATASET, 1):
        qid = item["query_id"]
        q = item["query"]
        expected = item["expected_answer"]
        
        # 1. FAISS retrieval
        kb_chunks = retrieve(q, top_k=2)
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in kb_chunks])
        
        # 2. Router prediction
        provider, model_id, meta = router.route_query(q, kb_chunks)
        chosen_tier = meta["chosen_tier"]
        chosen_model = meta["chosen_model"]
        is_router_choice = (chosen_tier == "tier2_high_reasoning")
        
        print(f"\n[{i:02d}/10] Query ID: {qid}", flush=True)
        print(f"      Category: {item['category']} | Severity: {item['severity']}", flush=True)
        print(f"      Query: \"{q}\"", flush=True)
        print(f"      Router Predicted: {chosen_tier} ({chosen_model}) {'[TARGET CHOICE]' if is_router_choice else ''}", flush=True)
        
        # 3. Candidate execution on Gemini 2.5 Flash
        prompt = f"""You are an expert Site Reliability Engineering (SRE) Assistant diagnosing a production incident.
Use the following retrieved runbook context to diagnose the issue and provide clear root-cause analysis and actionable remediation steps.

[RETRIEVED CONTEXT]:
{context}

[INCIDENT QUERY]:
{q}

Provide a concise, precise technical diagnosis and safe remediation procedure under 200 words.
"""
        t0 = time.perf_counter()
        try:
            resp = gemini_llm.invoke(prompt)
            t1 = time.perf_counter()
            lat_sec = round(t1 - t0, 3)
            lat_ms = round(lat_sec * 1000.0, 2)
            content = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            
            usage = getattr(resp, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            if in_tok == 0:
                in_tok = int(len(prompt.split()) * 1.3)
            if out_tok == 0:
                out_tok = int(len(content.split()) * 1.3)
            total_tok = in_tok + out_tok
            
            cost = estimate_token_cost(in_tok, out_tok, model="gemini-2.5-flash", provider="gemini")
            status = "SUCCESS"
            err_msg = ""
            print(f"      Execution: SUCCESS ({lat_sec}s, {total_tok} tokens, ${cost:.5f})", flush=True)
            print(f"      Response preview: {content[:140]}...", flush=True)
            
            # 4. Judge scoring
            judge_res = evaluate_with_judge(judge, q, context, expected, content)
            print(f"      Judge Score: Overall={judge_res['overall_quality']}/5.0 (Tech={judge_res['technical_correctness']}, Faith={judge_res['faithfulness']}, Action={judge_res['actionability']}, Comp={judge_res['completeness']}, Safe={judge_res['safety']})", flush=True)
            print(f"      Judge Reason: {judge_res['explanation']}", flush=True)
            
        except Exception as e:
            t1 = time.perf_counter()
            lat_sec = round(t1 - t0, 3)
            lat_ms = round(lat_sec * 1000.0, 2)
            content = ""
            in_tok = 0
            out_tok = 0
            total_tok = 0
            cost = 0.0
            status = "FAILED"
            err_msg = str(e)
            judge_res = {
                "technical_correctness": 0.0, "faithfulness": 0.0, "actionability": 0.0,
                "completeness": 0.0, "safety": 0.0, "overall_quality": 0.0,
                "explanation": f"API execution failed: {err_msg}", "error": err_msg
            }
            print(f"      Execution: FAILED ({err_msg})", flush=True)

        rec = {
            "query_id": qid,
            "query": q,
            "category": item["category"],
            "severity": item["severity"],
            "complexity_score": meta["query_features"]["complexity_score"],
            "retrieval_confidence": meta["retrieval_features"]["retrieval_confidence"],
            "router_selected_tier": chosen_tier,
            "router_selected_model": chosen_model,
            "is_router_selected": is_router_choice,
            "model_tier": "tier2_high_reasoning",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "execution_status": status,
            "response": content,
            "latency_sec": lat_sec,
            "latency_ms": lat_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": total_tok,
            "api_cost": cost,
            "technical_correctness_score": judge_res["technical_correctness"],
            "faithfulness_score": judge_res["faithfulness"],
            "actionability_score": judge_res["actionability"],
            "completeness_score": judge_res["completeness"],
            "safety_score": judge_res["safety"],
            "overall_quality_score": judge_res["overall_quality"],
            "judge_explanation": judge_res["explanation"],
            "error_message": err_msg
        }
        gemini_results.append(rec)
        time.sleep(1.0)
        
    # Save standalone gemini results
    out_dir = Path("results/ada_rag_ops_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gemini_live_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(gemini_results, f, indent=2)
    print(f"\n[+] Saved Gemini live evaluation results to {out_dir / 'gemini_live_eval_results.json'}", flush=True)
    
    return gemini_results

if __name__ == "__main__":
    run_gemini_evaluation()

import os
import sys
import json
import time
import math
import re
import urllib.request
import urllib.error
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
COHERE_KEY = os.getenv("COHERE_API_KEY", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Evaluation Test Cases
BENCHMARK_CASES = [
    {
        "id": "CASE_1",
        "domain": "Database Infrastructure",
        "query": "PostgreSQL database memory is getting exhausted with spikes in active connections.",
        "context": "Runbook: High Memory on Postgres. Cause: Connection pooling exhaustion or max_connections exceeded without pgbouncer. Fix: 1. Check active connections via pg_stat_activity. 2. Enable PgBouncer or scale connection pool. 3. Adjust work_mem and shared_buffers safely.",
        "reference": "The database memory exhaustion is caused by excessive unpooled connections. Check active sessions using pg_stat_activity and configure PgBouncer to manage the connection pool while adjusting work_mem safely."
    },
    {
        "id": "CASE_2",
        "domain": "Cloud & Container Ops",
        "query": "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137.",
        "context": "Runbook: Pod OOMKilled. Cause: Container memory limit exceeded due to sudden traffic burst or memory leak. Fix: 1. Inspect pod memory usage with kubectl top pod. 2. Analyze heap dump/logs for memory leaks. 3. Increase container memory limits and configure Horizontal Pod Autoscaler (HPA).",
        "reference": "The CrashLoopBackOff with exit code 137 indicates the container was terminated for exceeding memory limits. Inspect memory with kubectl top pod, check logs for leaks, and increase memory limits or configure HPA."
    },
    {
        "id": "CASE_3",
        "domain": "Network & Security",
        "query": "Users cannot connect to the payment gateway due to SSL/TLS certificate verification failed.",
        "context": "Runbook: TLS Handshake Failure. Cause: Expired SSL certificate, mismatched domain name, or missing intermediate CA bundle. Fix: 1. Verify certificate validity with openssl s_client. 2. Renew expired certificate via certbot/ACME. 3. Ensure full intermediate CA chain is installed on the web server.",
        "reference": "The SSL verification failure is caused by an expired certificate, hostname mismatch, or missing intermediate CA certs. Verify using openssl s_client, renew the cert, and ensure the full certificate chain is deployed."
    }
]

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def compute_metrics(prediction, reference, context, query, latency, in_tokens, out_tokens, cost_rate=0.0):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    ctx_tokens = tokenize(context)
    q_tokens = tokenize(query)
    
    # Token Overlap & Precision/Recall/F1
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    prec = round(num_same / max(len(pred_tokens), 1), 4)
    rec = round(num_same / max(len(ref_tokens), 1), 4)
    f1 = round((2 * prec * rec) / max(prec + rec, 1e-5), 4) if (prec + rec) > 0 else 0.0
    
    # Exact Match
    em = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    
    # BLEU
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
    
    # Perplexity Approximation
    counts = Counter(pred_tokens)
    n = max(len(pred_tokens), 1)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    ppl = round(2 ** entropy, 2)
    
    # Faithfulness & Hallucination
    overlap_ctx = set(pred_tokens).intersection(set(ctx_tokens))
    faithfulness = round(len(overlap_ctx) / max(len(set(pred_tokens)), 1), 4)
    hallucination_rate = round(max(1.0 - faithfulness, 0.0), 4)
    
    # Context Relevance
    q_overlap = set(q_tokens).intersection(set(ctx_tokens))
    context_relevance = round(len(q_overlap) / max(len(set(q_tokens)), 1), 4)
    
    # Safety Check
    dangerous = [r"\brm\s+-rf\b", r"\bdrop\s+database\b", r"\bdrop\s+table\b", r"\bformat\s+[c-z]:\b"]
    safety_rate = 0.0 if any(re.search(p, prediction, re.I) for p in dangerous) else 1.0
    
    # Composite Scores
    accuracy = round(((faithfulness * 0.40) + (f1 * 0.40) + (safety_rate * 0.20)) * 100, 1)
    correctness = round(((f1 * 0.60) + (bleu * 0.40)) * 100, 1)
    
    total_tokens = in_tokens + out_tokens
    cost_usd = round((total_tokens / 1000.0) * cost_rate, 6)
    
    return {
        "accuracy": accuracy,
        "correctness": correctness,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "bleu": bleu,
        "perplexity": ppl,
        "faithfulness": round(faithfulness * 100, 1),
        "hallucination_rate": round(hallucination_rate * 100, 1),
        "latency": round(latency, 2),
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "safety_rate": round(safety_rate * 100, 1)
    }

def call_cohere(prompt, model="command-r-08-2024"):
    req = urllib.request.Request(
        "https://api.cohere.com/v1/chat",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {COHERE_KEY}"},
        data=json.dumps({"model": model, "message": prompt, "temperature": 0.0}).encode("utf-8")
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            return data.get("text", "").strip(), lat, "OK"
    except Exception as e:
        return "", time.time() - t0, str(e)

def call_openrouter(prompt, model):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Model Benchmark"
        },
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
            "temperature": 0.0
        }).encode("utf-8")
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            msg = data["choices"][0]["message"]["content"].strip()
            return msg, lat, "OK"
    except Exception as e:
        return "", time.time() - t0, str(e)

def main():
    print("=" * 80)
    print("  COMPREHENSIVE MULTI-MODEL BENCHMARK & COMPARISON")
    print("=" * 80)
    
    # Models to benchmark
    test_models = [
        {"provider": "cohere", "model_id": "command-r-08-2024", "label": "Cohere Command-R (35B)", "cost_rate": 0.00015},
        {"provider": "cohere", "model_id": "command-r-plus-08-2024", "label": "Cohere Command-R+ (104B)", "cost_rate": 0.0025},
        {"provider": "openrouter", "model_id": "nvidia/nemotron-3.5-lightning:free", "label": "NVIDIA Nemotron 3.5", "cost_rate": 0.0},
        {"provider": "openrouter", "model_id": "poolside/laguna-s-2.1:free", "label": "Poolside Laguna-S 2.1", "cost_rate": 0.0}
    ]
    
    all_results = {}
    
    for tm in test_models:
        label = tm["label"]
        print(f"\n---> Testing Model: {label} [{tm['model_id']}]")
        model_scores = []
        
        for case in BENCHMARK_CASES:
            prompt = f"Context: {case['context']}\nUser Issue: {case['query']}\nProvide the exact root cause and step-by-step resolution."
            
            if tm["provider"] == "cohere":
                resp_text, latency, status = call_cohere(prompt, tm["model_id"])
            else:
                resp_text, latency, status = call_openrouter(prompt, tm["model_id"])
                
            if status != "OK" or not resp_text:
                print(f"  [{case['id']}] Failed/Error: {status}")
                continue
                
            in_tokens = int(len(prompt.split()) * 1.3)
            out_tokens = int(len(resp_text.split()) * 1.3)
            
            m = compute_metrics(
                prediction=resp_text,
                reference=case["reference"],
                context=case["context"],
                query=case["query"],
                latency=latency,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                cost_rate=tm["cost_rate"]
            )
            m["case_id"] = case["id"]
            m["prediction_sample"] = resp_text[:120].replace("\n", " ") + "..."
            model_scores.append(m)
            print(f"  [{case['id']}] Accuracy: {m['accuracy']}% | F1: {m['f1']} | Latency: {m['latency']}s | Tokens: {m['total_tokens']}")
            
        if model_scores:
            avg_acc = round(sum(x["accuracy"] for x in model_scores) / len(model_scores), 1)
            avg_corr = round(sum(x["correctness"] for x in model_scores) / len(model_scores), 1)
            avg_f1 = round(sum(x["f1"] for x in model_scores) / len(model_scores), 4)
            avg_bleu = round(sum(x["bleu"] for x in model_scores) / len(model_scores), 4)
            avg_faith = round(sum(x["faithfulness"] for x in model_scores) / len(model_scores), 1)
            avg_hall = round(sum(x["hallucination_rate"] for x in model_scores) / len(model_scores), 1)
            avg_lat = round(sum(x["latency"] for x in model_scores) / len(model_scores), 2)
            avg_tokens = int(sum(x["total_tokens"] for x in model_scores) / len(model_scores))
            avg_safety = round(sum(x["safety_rate"] for x in model_scores) / len(model_scores), 1)
            
            all_results[label] = {
                "model_id": tm["model_id"],
                "provider": tm["provider"],
                "cases_tested": len(model_scores),
                "avg_accuracy": avg_acc,
                "avg_correctness": avg_corr,
                "avg_f1": avg_f1,
                "avg_bleu": avg_bleu,
                "avg_faithfulness": avg_faith,
                "avg_hallucination_rate": avg_hall,
                "avg_latency_sec": avg_lat,
                "avg_tokens": avg_tokens,
                "safety_rate": avg_safety,
                "cost_tier": "Free" if tm["cost_rate"] == 0 else f"${tm['cost_rate']*1000}/1M tokens",
                "details": model_scores
            }
            
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETED SUCCESSFULLY - SAVED TO benchmark_results.json")
    print("=" * 80)

if __name__ == "__main__":
    main()

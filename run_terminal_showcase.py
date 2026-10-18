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

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY", "")
COHERE_KEY = os.getenv("COHERE_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

TEST_QUERY = "PostgreSQL database memory is getting exhausted with spikes in active connections."
TEST_CONTEXT = "Runbook: High Memory on Postgres. Cause: Connection pooling exhaustion or max_connections exceeded without pgbouncer. Fix: 1. Check active connections via pg_stat_activity. 2. Enable PgBouncer or scale connection pool. 3. Adjust work_mem and shared_buffers safely."
TEST_REFERENCE = "The database memory exhaustion is caused by excessive unpooled connections. Check active sessions using pg_stat_activity and configure PgBouncer to manage the connection pool while adjusting work_mem safely."

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def compute_14_metrics(prediction, latency, in_tokens, out_tokens, cost_rate):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(TEST_REFERENCE)
    ctx_tokens = tokenize(TEST_CONTEXT)
    q_tokens = tokenize(TEST_QUERY)
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    prec = round(num_same / max(len(pred_tokens), 1), 4)
    rec = round(num_same / max(len(ref_tokens), 1), 4)
    f1 = round((2 * prec * rec) / max(prec + rec, 1e-5), 4) if (prec + rec) > 0 else 0.0
    
    em = 1.0 if prediction.strip().lower() == TEST_REFERENCE.strip().lower() else 0.0
    
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
    
    counts = Counter(pred_tokens)
    n = max(len(pred_tokens), 1)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    ppl = round(2 ** entropy, 2)
    
    overlap_ctx = set(pred_tokens).intersection(set(ctx_tokens))
    faithfulness = round(len(overlap_ctx) / max(len(set(pred_tokens)), 1), 4)
    hallucination_rate = round(max(1.0 - faithfulness, 0.0), 4)
    
    q_overlap = set(q_tokens).intersection(set(ctx_tokens))
    context_relevance = round(len(q_overlap) / max(len(set(q_tokens)), 1), 4)
    
    dangerous = [r"\brm\s+-rf\b", r"\bdrop\s+database\b", r"\bdrop\s+table\b", r"\bformat\s+[c-z]:\b"]
    safety_rate = 0.0 if any(re.search(p, prediction, re.I) for p in dangerous) else 1.0
    
    accuracy = round(((faithfulness * 0.40) + (f1 * 0.40) + (safety_rate * 0.20)) * 100, 1)
    correctness = round(((f1 * 0.60) + (bleu * 0.40)) * 100, 1)
    
    total_tokens = in_tokens + out_tokens
    cost_usd = round((total_tokens / 1000.0) * cost_rate, 6)
    
    return {
        "Accuracy": f"{accuracy}%",
        "Correctness": f"{correctness}%",
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Exact Match": em,
        "BLEU": bleu,
        "Perplexity": ppl,
        "Faithfulness": f"{round(faithfulness * 100, 1)}%",
        "Context Relevance": f"{round(context_relevance * 100, 1)}%",
        "Hallucination Rate": f"{round(hallucination_rate * 100, 1)}%",
        "Latency": f"{round(latency, 2)}s",
        "Tokens": f"{in_tokens} in / {out_tokens} out ({total_tokens} total)",
        "Cost": f"${cost_usd}",
        "Safety Rate": f"{round(safety_rate * 100, 1)}%"
    }

def print_section(title):
    print("\n" + "=" * 70, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 70, flush=True)

def test_deepseek_direct():
    print_section("1. TESTING DEEPSEEK DIRECT API")
    print("Endpoint: https://api.deepseek.com/chat/completions", flush=True)
    print("Model: deepseek-chat (DeepSeek-V3)", flush=True)
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"},
        data=json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": f"Context: {TEST_CONTEXT}\nIssue: {TEST_QUERY}\nGive root cause and fix."}],
            "max_tokens": 120
        }).encode("utf-8")
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            output = data["choices"][0]["message"]["content"]
            print(f"[STATUS]: HTTP 200 OK (Latency: {lat:.2f}s)", flush=True)
            print(f"[OUTPUT]:\n{output}", flush=True)
            return output, lat
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[HTTP {e.code} STATUS]: {e.reason}", flush=True)
        print(f"[RESPONSE BODY]: {body}", flush=True)
        return None, 0
    except Exception as e:
        print(f"[EXCEPTION]: {e}", flush=True)
        return None, 0

def test_mistral_direct():
    print_section("2. TESTING MISTRAL DIRECT API")
    print("Endpoint: https://api.mistral.ai/v1/chat/completions", flush=True)
    print("Model: mistral-small-latest", flush=True)
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_KEY}"},
        data=json.dumps({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Hello Mistral!"}],
            "max_tokens": 30
        }).encode("utf-8")
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            output = data["choices"][0]["message"]["content"]
            print(f"[STATUS]: HTTP 200 OK (Latency: {lat:.2f}s)", flush=True)
            print(f"[OUTPUT]:\n{output}", flush=True)
            return output, lat
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[HTTP {e.code} STATUS]: {e.reason}", flush=True)
        print(f"[RESPONSE BODY]: {body}", flush=True)
        return None, 0
    except Exception as e:
        print(f"[EXCEPTION]: {e}", flush=True)
        return None, 0

def test_cohere_direct():
    print_section("3. TESTING COHERE API")
    print("Endpoint: https://api.cohere.com/v1/chat", flush=True)
    print("Model: command-r-08-2024", flush=True)
    prompt = f"Context: {TEST_CONTEXT}\nIssue: {TEST_QUERY}\nGive root cause and numbered fix."
    req = urllib.request.Request(
        "https://api.cohere.com/v1/chat",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {COHERE_KEY}"},
        data=json.dumps({"model": "command-r-08-2024", "message": prompt}).encode("utf-8")
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            output = data.get("text", "")
            print(f"[STATUS]: HTTP 200 OK (Latency: {lat:.2f}s)", flush=True)
            print(f"[RAW OUTPUT]:\n{output.strip()}", flush=True)
            return output, lat
    except Exception as e:
        print(f"[EXCEPTION]: {e}", flush=True)
        return None, 0

def test_openrouter():
    print_section("4. TESTING OPENROUTER FREE TIER")
    print("Endpoint: https://openrouter.ai/api/v1/chat/completions", flush=True)
    print("Model: nvidia/nemotron-3.5-lightning:free", flush=True)
    prompt = f"Context: {TEST_CONTEXT}\nIssue: {TEST_QUERY}\nGive root cause and numbered fix."
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Terminal Showcase"
        },
        data=json.dumps({
            "model": "nvidia/nemotron-3.5-lightning:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }).encode("utf-8")
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = time.time() - t0
            output = data["choices"][0]["message"]["content"]
            print(f"[STATUS]: HTTP 200 OK (Latency: {lat:.2f}s)", flush=True)
            print(f"[RAW OUTPUT]:\n{output.strip()}", flush=True)
            return output, lat
    except Exception as e:
        print(f"[EXCEPTION]: {e}", flush=True)
        return None, 0

def main():
    print("\n" + "#" * 70, flush=True)
    print("  LIVE TERMINAL OUTPUT & 14-DIMENSION EVALUATION SUITE", flush=True)
    print("#" * 70, flush=True)
    
    # 1. DeepSeek
    deepseek_out, deepseek_lat = test_deepseek_direct()
    
    # 2. Mistral
    mistral_out, mistral_lat = test_mistral_direct()
    
    # 3. Cohere
    cohere_out, cohere_lat = test_cohere_direct()
    
    # 4. OpenRouter
    nemotron_out, nemotron_lat = test_openrouter()
    
    # 5. Metrics Report
    print_section("5. 14-DIMENSION EVALUATION METRICS REPORT")
    
    if cohere_out:
        in_t = int(len(TEST_CONTEXT.split()) * 1.3)
        out_t = int(len(cohere_out.split()) * 1.3)
        m = compute_14_metrics(cohere_out, cohere_lat, in_t, out_t, 0.00015)
        print("\n[MODEL]: Cohere Command-R (08-2024)", flush=True)
        print("-" * 50, flush=True)
        for k, v in m.items():
            print(f"  {k:<22}: {v}", flush=True)
            
    if nemotron_out:
        in_t = int(len(TEST_CONTEXT.split()) * 1.3)
        out_t = int(len(nemotron_out.split()) * 1.3)
        m = compute_14_metrics(nemotron_out, nemotron_lat, in_t, out_t, 0.0)
        print("\n[MODEL]: NVIDIA Nemotron 3.5 (OpenRouter Free)", flush=True)
        print("-" * 50, flush=True)
        for k, v in m.items():
            print(f"  {k:<22}: {v}", flush=True)
            
    print("\n" + "#" * 70, flush=True)
    print("  EVALUATION SHOWCASE COMPLETE", flush=True)
    print("#" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()

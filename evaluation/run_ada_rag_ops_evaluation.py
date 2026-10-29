"""Real, resumable AdaRAG-Ops evaluation. Run with --help for controls."""
from __future__ import annotations
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.ada_core import (Journal, Recorder, ROOT, RUBRIC, SCORES, JUDGE_SYSTEM,
    UNAVAILABLE, NOT_COMPUTED, digest, load_existing_dataset, measured_graph, now, parse_judge)
from evaluation.ada_reports import write_outputs, total

LIMITATIONS = [
    "Only ten repository incidents, one generation per query/model; no confidence in population superiority or stable latency ranking.",
    "Reference answers are existing unverified labels. LLM judgments are not human-validated ground truth, and may favor the judge model family.",
    "The original router uses hand-set capability, price and latency priors; those are routing inputs, never experimental measurements. Its Random Forest uses 15 built-in synthetic training examples.",
    "Original classifier/agent state fields named actual_cost use word-count estimates and blended tariffs; preserved for graph behavior, explicitly excluded from measured result accounting. They can affect graph budget guards.",
    "Production verification currently accepts malformed verifier JSON. Evaluation preserves this behavior and records raw verification responses; graph verified=True is not proof of correctness or executed remediation.",
    "Production diagnostic/resolution agents may switch providers on some errors; every fallback is separately logged. Router initial selection does not imply all graph calls used that model.",
    "Fixed-model baselines perform one response generation; full graph performs classification, diagnosis, resolution, verification and possible loops. Matched differences therefore include architectural overhead.",
    "Ablations mask router selection features only, preserving severity safety guards and intervention evidence auditing. They cannot isolate every downstream use of severity/evidence or prove causal contribution of trained features.",
    "No account invoices or billing-tier confirmation for Groq, Gemini or Cohere. Published-tariff costs are reported separately; unknown actual API charges are UNAVAILABLE, not assumed zero.",
    "SDK response caching and automatic retries disabled for controlled measurement. Explicit production fallbacks and graph loops remain enabled. Failure usage/billing may be unavailable.",
    "Local embedding API cost is not applicable; local hardware/energy costs and embedding tokenizer counts are not measured.",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Local retrieval/router validation only; no LLM calls or experimental CSV rows")
    parser.add_argument("--queries", type=int, default=10, help="First N exact queries in the existing ten-query dataset")
    parser.add_argument("--models", nargs="+", help="Exact model IDs or catalog tier names; no substitutions")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/ada_rag_ops_evaluation", help="Parent for a unique experiment directory")
    parser.add_argument("--resume", type=Path, help="Resume an existing actual experiment directory; never rerun completed/uncertain attempts")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--run-ablations", action="store_true", help="Run A-D as fresh graphs; E is the original graph run")
    parser.add_argument("--judge-provider", default="groq")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b", help="Existing legacy evaluator judge ID")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Candidate requested output budget; provider factory effective values recorded")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--pricing-file", type=Path, default=ROOT/"evaluation/verified_pricing_2026-10-29.json")
    args = parser.parse_args()
    if args.queries < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--queries, --max-tokens and --timeout must be positive")
    if args.resume and args.dry_run:
        parser.error("Cannot resume an actual experiment as a dry run")
    return args


def snapshot(directory):
    paths = list((ROOT/"agents").glob("*.py")) + list((ROOT/"rag").glob("*.py")) + list((ROOT/"evaluation").glob("ada_*.py"))
    paths += [Path(__file__), ROOT/"requirements.txt", ROOT/"run_full_ada_rag_ops_evaluation.py"]
    paths += list((ROOT/"documents").glob("*.md")) + list((ROOT/"faiss_index").glob("*"))
    import hashlib
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    (directory/"source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    # Archive executable source + documents, not .env, cache databases or API credentials.
    import zipfile
    with zipfile.ZipFile(directory/"source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            if p.is_file():
                archive.write(p, p.relative_to(ROOT))
    return hashes


def fetch_openrouter_pricing(prices, journal):
    model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    source = "https://openrouter.ai/api/v1/models"
    try:
        request = urllib.request.Request(source, headers={"User-Agent": "AdaRAG-Ops-evaluation"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        match = next((m for m in data["data"] if m["id"] == model), None)
        journal.append("provider_catalog", provider="openrouter", source=source, exact_model=match or UNAVAILABLE)
        if match:
            p = match["pricing"]
            inp, out = float(p["prompt"])*1e6, float(p["completion"])*1e6
            prices["openrouter:"+model] = {"verified": True, "verified_on": now()[:10], "input_per_million": inp,
                "output_per_million": out, "source": source, "currency": "USD", "canonical_slug": match.get("canonical_slug"),
                "billing_condition": "verified_free_endpoint" if inp == out == 0 else UNAVAILABLE,
                "condition_reason": "Exact provider catalog endpoint; no inference from the :free name alone."}
    except Exception as error:
        journal.append("local_error", stage="pricing_catalog", error_message=f"{type(error).__name__}: {error}")


def missing_scores(reason):
    return {**{s: UNAVAILABLE for s in SCORES}, "overall_quality_score": UNAVAILABLE,
            "judge_status": UNAVAILABLE, "judge_error": reason}


def judge(recorder, key, row, context, reference, config):
    if row.get("execution_status") != "SUCCESS" or config["skip_judge"]:
        return missing_scores("Generation failed/unavailable or --skip-judge requested.")
    judge_key = "judge:"+key
    cached = recorder.journal.complete(judge_key)
    if cached:
        return cached["data"]
    if any(e["kind"] == "job_start" and e["key"] == judge_key for e in recorder.journal.events):
        return missing_scores("Interrupted judge outcome unknown; preserved, not silently repeated.")
    recorder.journal.append("job_start", key=judge_key)
    from langchain_core.messages import HumanMessage, SystemMessage
    payload = {"query": row["query"], "retrieved_runbook_context": context, "response": row["response"],
               "reference_unverified": reference, "incident_severity": row.get("incident_severity", UNAVAILABLE)}
    result = missing_scores("Judge did not complete.")
    with recorder.at(row["query_id"], "judge:"+row["scope"]):
        try:
            llm = recorder.factory(temperature=0, max_tokens=1800, provider=config["judge_provider"], model=config["judge_model"])
            message = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
            from evaluation.ada_core import text_content
            result = parse_judge(text_content(message.content))
            result["judge_attempt_id"] = llm.last["attempt_id"]
        except Exception as error:
            result = missing_scores(f"{type(error).__name__}: {error}")
            recorder.journal.append("local_error", stage="judge_parse_or_setup", key=judge_key, error_message=result["judge_error"])
    recorder.journal.append("observation", key=judge_key, data=result)
    return result


def graph_run(recorder, q, configuration):
    scope = "pipeline:"+configuration
    key = scope+":"+q["query_id"]
    cached = recorder.journal.complete(key)
    if cached:
        return cached["data"]
    row = {"experiment_id": recorder.experiment_id, "timestamp": now(), "query_id": q["query_id"], "query": q["query"],
           "scope": scope, "execution_status": UNAVAILABLE, "response": "", "error_message": "",
           "latency_ms": UNAVAILABLE, "input_tokens": UNAVAILABLE, "output_tokens": UNAVAILABLE,
           "total_tokens": UNAVAILABLE, "api_cost": UNAVAILABLE, "standard_tariff_cost": UNAVAILABLE,
           "cost_currency": "USD", "token_measurement_method": UNAVAILABLE}
    if any(e["kind"] == "job_start" and e["key"] == key for e in recorder.journal.events):
        row["error_message"] = "Interrupted graph run; partial API attempts retained, no automatic repeat."
    else:
        recorder.journal.append("job_start", key=key)
        initial_events = len(recorder.journal.events)
        final = {}
        with recorder.at(q["query_id"], scope):
            row["pipeline_start_time"] = now()
            start = time.perf_counter()
            try:
                with measured_graph(recorder, configuration) as graph:
                    final = graph.invoke({"user_query": q["query"]}, config={"recursion_limit": 100})
                row["response"] = final.get("final_answer", "")
                row["execution_status"] = "SUCCESS" if row["response"].strip() else "FAILED"
                if row["execution_status"] == "FAILED":
                    row["error_message"] = "Graph completed without a final answer."
            except Exception as error:
                row["execution_status"] = "FAILED"
                row["error_message"] = f"{type(error).__name__}: {error}"
                recorder.journal.append("local_error", key=key, stage="graph", error_message=row["error_message"])
            row["monotonic_start"] = start
            row["monotonic_end"] = time.perf_counter()
            row["latency_ms"] = (row["monotonic_end"]-start)*1000
            row["pipeline_end_time"] = now()
        recent = recorder.journal.events[initial_events:]
        nodes = [e for e in recent if e["kind"] == "node_end"]
        routing = next((e["state"] for e in nodes if e["stage"] == "router_agent"), {})
        classified = next((e["state"] for e in nodes if e["stage"] == "classifier_agent"), {})
        knowledge = next((e["state"] for e in nodes if e["stage"] == "knowledge_agent"), {})
        metadata = routing.get("routing_metadata") or {}
        calls = [e for e in recent if e["kind"] == "attempt_end"]
        row.update(incident_category=classified.get("category", UNAVAILABLE), incident_severity=classified.get("severity", UNAVAILABLE),
                   query_complexity=metadata.get("query_features", {}).get("complexity_score", UNAVAILABLE),
                   retrieval_confidence=metadata.get("retrieval_features", {}).get("retrieval_confidence", UNAVAILABLE),
                   router_selected_model=routing.get("selected_model", UNAVAILABLE), router_selected_tier=metadata.get("chosen_tier", UNAVAILABLE),
                   routing_metadata=metadata, classified_by=classified.get("classified_by", UNAVAILABLE),
                   final_selected_model=final.get("selected_model", UNAVAILABLE), final_state=final,
                   kb_context=knowledge.get("kb_context", ""), kb_results=knowledge.get("kb_results", []),
                   api_attempts=len(calls), failed_api_attempts=sum(c["execution_status"] == "FAILED" for c in calls),
                   input_tokens=total(c["input_tokens"] for c in calls), output_tokens=total(c["output_tokens"] for c in calls),
                   total_tokens=total(c["total_tokens"] for c in calls), api_cost=total(c["api_cost"] for c in calls),
                   standard_tariff_cost=total(c["standard_tariff_cost"] for c in calls),
                   cache_status="disabled", token_measurement_method="provider_reported_sum" if calls and all(c["token_measurement_method"] == "provider_reported" for c in calls) else UNAVAILABLE,
                   cost_reason="Includes every graph call; unknown failure usage/billing makes total unavailable.")
    recorder.journal.append("observation", key=key, data=row)
    return row


def candidate_run(recorder, q, model, state, config):
    key = "candidate:"+q["query_id"]+":"+model["model_id"]
    existing = recorder.journal.complete(key)
    if existing:
        return existing["data"]
    row = {"experiment_id": recorder.experiment_id, "timestamp": now(), "query_id": q["query_id"], "query": q["query"],
           "scope": "candidate", "model_tier": model["tier"], "provider": model["provider"], "model_id": model["model_id"],
           **{k: state.get(k, UNAVAILABLE) for k in ("incident_category", "incident_severity", "query_complexity", "retrieval_confidence")},
           "context_hash": digest(state.get("kb_context", "")), "cache_status": "disabled",
           "execution_status": UNAVAILABLE, "response": "", "input_tokens": UNAVAILABLE, "output_tokens": UNAVAILABLE,
           "total_tokens": UNAVAILABLE, "token_measurement_method": UNAVAILABLE, "latency_ms": UNAVAILABLE,
           "api_cost": UNAVAILABLE, "standard_tariff_cost": UNAVAILABLE, "cost_currency": "USD", "error_message": ""}
    prior_job = any(e["kind"] == "job_start" and e["key"] == key for e in recorder.journal.events)
    if prior_job:
        row["error_message"] = "Interrupted candidate job; provider outcome unknown, no automatic repeat."
    elif not state.get("kb_context"):
        row["error_message"] = "Retrieval unavailable; controlled-context inference not attempted."
    else:
        recorder.journal.append("job_start", key=key)
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content="You are an SRE incident diagnosis assistant. Use the supplied incident and runbook evidence. Distinguish evidence from hypotheses, give actionable diagnosis, mitigation and verification, and identify evidence gaps and unsafe actions. Do not claim commands were executed."),
                    HumanMessage(content=json.dumps({"incident": q["query"], "runbook_context": state["kb_context"]}, ensure_ascii=False))]
        llm = None
        with recorder.at(q["query_id"], "candidate"):
            try:
                llm = recorder.factory(temperature=0, max_tokens=config["max_tokens"], provider=model["provider"], model=model["model_id"])
                llm.invoke(messages)
            except Exception as error:
                row["error_message"] = f"{type(error).__name__}: {error}"
            if llm and llm.last:
                row.update({k: v for k, v in llm.last.items() if k not in ("kind", "query_id", "scope", "model_id", "provider")})
    row.update(missing_scores("Judgment pending."))
    recorder.journal.append("observation", key=key, data=row)
    return row


def enrich_judgments(recorder, row, key, q, state, config):
    judged_key = "scored:"+key
    existing = recorder.journal.complete(judged_key)
    if existing:
        return existing["data"]
    row = {**row, **judge(recorder, key, row, state.get("kb_context", ""), q.get("expected_answer", ""), config)}
    # Replace derived observation logically through separate scoring events; raw generation remains immutable.
    recorder.journal.append("scored_observation", key=key, data=row)
    recorder.journal.append("observation", key=judged_key, data={"completed": True})
    return row


def effective_events(journal):
    # Reports join scoring to generation by key without changing either raw event.
    scores = {e["key"]: e["data"] for e in journal.events if e["kind"] == "scored_observation"}
    return [{**e, "data": scores[e["key"]]} if e["kind"] == "observation" and e["key"] in scores else e for e in journal.events]


def main():
    args = arguments()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT/".env")
    # Never export experiment data to configured tracing accounts.
    os.environ.update(LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false", INCIDENT_CACHE_ENABLED="false", INCIDENT_CACHE_SEMANTIC_ENABLED="false")
    secrets = [v for k, v in os.environ.items() if any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))]
    if args.resume:
        directory = args.resume.resolve()
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest["mode"] != "actual":
            raise ValueError("Dry runs cannot be resumed as actual experiments")
        prices = json.loads((directory/"pricing_snapshot.json").read_text(encoding="utf-8"))
        queries, models = manifest["queries"], manifest["models"]
        hashes = json.loads((directory/"source_hashes.json").read_text(encoding="utf-8"))
        import hashlib
        changed = [p for p, sha in hashes.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != sha]
        if changed:
            raise ValueError("Source changed since experiment; resume refused: "+", ".join(changed))
    else:
        all_queries, models = load_existing_dataset()
        queries = all_queries[:args.queries]
        if args.models:
            unknown = set(args.models) - {m["model_id"] for m in models} - {m["tier"] for m in models}
            if unknown:
                raise ValueError("Unknown exact candidate model/tier: "+str(sorted(unknown)))
            models = [m for m in models if m["model_id"] in args.models or m["tier"] in args.models]
        identifier = ("DRY_" if args.dry_run else "EXP_")+now().replace(":", "").replace("-", "").replace(".", "_")+"_"+uuid.uuid4().hex[:8]
        directory = args.output_dir.resolve()/identifier
        directory.mkdir(parents=True, exist_ok=False)
        config = {k: v for k, v in vars(args).items() if k not in ("output_dir", "resume", "pricing_file")}
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        from agents.router_agent import AdaptiveModelRouter
        manifest = {"experiment_id": identifier, "created_at": now(), "mode": "dry_run" if args.dry_run else "actual",
                    "dataset_source": "run_full_ada_rag_ops_evaluation.py:DATASET (AST literal extraction)", "dataset_hash": digest(queries),
                    "requested_queries": args.queries, "available_queries": len(all_queries), "missing_queries": max(0, args.queries-len(all_queries)),
                    "queries": queries, "models": models, "config": config, "rubric": RUBRIC,
                    "router_strategy": os.getenv("ROUTER_STRATEGY", "intervention"), "cost_priority_mode": os.getenv("COST_PRIORITY_MODE", "false"),
                    "actual_router_catalog": AdaptiveModelRouter().model_catalog, "git_commit": commit.stdout.strip() if commit.returncode == 0 else UNAVAILABLE,
                    "package_versions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                    "cache_policy": "disabled", "judge_prompt": JUDGE_SYSTEM}
        prices = json.loads(args.pricing_file.read_text(encoding="utf-8"))
        snapshot(directory)
        (directory/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    journal = Journal(directory, secrets)
    os.environ["INCIDENT_CACHE_DB_PATH"] = str(directory/"disabled_cache.db")
    print("Output:", directory, flush=True)
    config = manifest["config"]
    if manifest["mode"] == "dry_run":
        checks = []
        from agents.knowledge_agent import knowledge_agent
        from agents.classifier_agent import _heuristic_classify
        from agents.router_agent import router_agent
        for q in queries:
            try:
                start = time.perf_counter()
                state = knowledge_agent({"user_query": q["query"], **_heuristic_classify(q["query"])})
                state = router_agent(state)
                journal.append("dry_run_local_validation", query_id=q["query_id"], state=state, latency_ms=(time.perf_counter()-start)*1000)
                checks.append({"query_id": q["query_id"], "status": "PASS", "router_model": state.get("selected_model"), "context_hash": digest(state["kb_context"])})
            except Exception as error:
                checks.append({"query_id": q["query_id"], "status": "FAIL", "error": journal.clean(str(error))})
        report = {"mode": "dry_run", "actual_experiment_executed": False, "api_attempts": 0, "checks": checks,
                  "credential_presence": {p: bool(os.getenv(p)) for p in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY")},
                  "reason": "Local retrieval and routing only. No experimental CSV response rows or performance claims generated."}
        (directory/"dry_run_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"mode": "dry_run", "local_pass": sum(c["status"] == "PASS" for c in checks), "local_fail": sum(c["status"] == "FAIL" for c in checks), "api_attempts": 0}), flush=True)
        return 0 if all(c["status"] == "PASS" for c in checks) else 1
    if not args.resume:
        fetch_openrouter_pricing(prices, journal)
        (directory/"pricing_snapshot.json").write_text(json.dumps(prices, indent=2), encoding="utf-8")
    recorder = Recorder(journal, prices, manifest["experiment_id"], config["timeout"])
    # Refuse configuration drift across resumed experiments.
    if manifest["router_strategy"] != os.getenv("ROUTER_STRATEGY", "intervention") or manifest["cost_priority_mode"] != os.getenv("COST_PRIORITY_MODE", "false"):
        raise ValueError("Router environment changed; start a new experiment")
    from agents.router_agent import AdaptiveModelRouter
    if manifest["actual_router_catalog"] != AdaptiveModelRouter().model_catalog:
        raise ValueError("Credential-dependent router catalog changed; start a new experiment")
    try:
        for q in queries:
            print(q["query_id"], "full graph", flush=True)
            pipeline = graph_run(recorder, q, "E")
            state = pipeline
            if not state.get("kb_context"):
                from agents.knowledge_agent import knowledge_agent
                try:
                    kb = knowledge_agent({"user_query": q["query"]})
                    state = {**pipeline, **kb}
                    journal.append("retrieval_only", query_id=q["query_id"], state=kb)
                except Exception as error:
                    journal.append("local_error", query_id=q["query_id"], stage="retrieval", error_message=str(error))
            enrich_judgments(recorder, pipeline, "pipeline:E:"+q["query_id"], q, state, config)
            for model in models:
                row = candidate_run(recorder, q, model, state, config)
                enrich_judgments(recorder, row, "candidate:"+q["query_id"]+":"+model["model_id"], q, state, config)
                print(" ", model["model_id"], row["execution_status"], flush=True)
            if config["run_ablations"]:
                for configuration in "ABCD":
                    row = graph_run(recorder, q, configuration)
                    enrich_judgments(recorder, row, "pipeline:"+configuration+":"+q["query_id"], q, row, config)
                    print("  ablation", configuration, row["execution_status"], flush=True)
    except KeyboardInterrupt:
        journal.append("local_error", stage="interruption", error_message="User interrupted; preserving observations. Resume does not repeat uncertain calls.")
    finally:
        raw_events = journal.events
        journal.events = effective_events(journal)
        try:
            metrics, integrity = write_outputs(journal, manifest, prices, LIMITATIONS)
        finally:
            journal.events = raw_events
    print(json.dumps({"queries": len(queries), "candidate_successes": metrics["candidate_successes"],
        "candidate_failures": metrics["candidate_failures"], "candidate_unavailable": metrics["candidate_unavailable"],
        "all_api_attempts": metrics["actual_attempts"], "router_invocations": metrics["router_invocations"],
        "integrity": integrity["status"], "output_directory": str(directory)}, indent=2), flush=True)
    return 0 if integrity["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Derive tables exclusively from the append-only execution journal."""
from __future__ import annotations
import collections
import csv
import json
import math
from pathlib import Path
import statistics

from evaluation.ada_core import (NOT_COMPUTED, UNAVAILABLE, SCORES, RUBRIC, numeric, price_usage)

MODEL_COLUMNS = "experiment_id timestamp query_id query incident_category incident_severity query_complexity retrieval_confidence model_tier provider model_id execution_status response technical_correctness_score faithfulness_score actionability_score completeness_score safety_score overall_quality_score input_tokens output_tokens total_tokens token_measurement_method latency_ms api_cost cost_currency error_message".split()
ROUTE_COLUMNS = "experiment_id query_id query incident_category incident_severity query_complexity retrieval_confidence router_selected_model router_selected_tier selected_model_quality selected_model_faithfulness selected_model_cost selected_model_tokens selected_model_latency lowest_cost_quality_eligible_model lowest_cost_quality_eligible_model_cost routing_suitability routing_status over_routing under_routing validation_reason".split()
SUMMARY_COLUMNS = "model_id provider model_tier successful_executions failed_executions average_quality average_faithfulness average_input_tokens average_output_tokens average_total_tokens average_latency_ms median_latency_ms p95_latency_ms total_api_cost average_cost_per_query cost_per_quality_point".split()
BASELINE_COLUMNS = "strategy successful_queries failed_queries average_quality average_faithfulness average_total_tokens average_latency_ms p95_latency_ms total_api_cost average_cost_per_query cost_reduction_percent token_reduction_percent quality_difference_vs_strongest_model latency_difference_vs_strongest_model routing_suitability_rate".split()
TRADEOFF_COLUMNS = "experiment_id query_id query model_id router_selected_model model_selected_by_router quality_score faithfulness_score input_tokens output_tokens total_tokens api_cost latency_ms quality_difference_vs_router cost_difference_vs_router token_difference_vs_router latency_difference_vs_router quality_eligible routing_suitability tradeoff_explanation".split()
ABLATION_COLUMNS = "configuration features_enabled successful_queries failed_queries average_quality average_faithfulness average_total_tokens average_latency_ms p95_latency_ms total_api_cost cost_reduction_percent routing_suitability_rate".split()
LOG_COLUMNS = "experiment_id query_id model_id attempt_number request_start_time request_end_time execution_status http_status_code input_tokens output_tokens latency_ms api_cost error_message".split()
SCHEMAS = {"model_comparison_results.csv": MODEL_COLUMNS, "routing_decision_validation.csv": ROUTE_COLUMNS,
           "model_comparison_summary.csv": SUMMARY_COLUMNS, "routing_baseline_comparison.csv": BASELINE_COLUMNS,
           "per_query_model_tradeoff_analysis.csv": TRADEOFF_COLUMNS, "routing_ablation_results.csv": ABLATION_COLUMNS,
           "experiment_execution_log.csv": LOG_COLUMNS}
FEATURES = {"A": "complexity", "B": "complexity + severity", "C": "complexity + retrieval",
            "D": "complexity + severity + retrieval", "E": "full original router"}


def write_csv(path, rows, columns):
    extras = sorted({k for row in rows for k in row} - set(columns))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns+extras, extrasaction="raise")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def mean(values):
    measured = [v for v in values if numeric(v)]
    return statistics.mean(measured) if measured else UNAVAILABLE


def total(values):
    values = list(values)
    return sum(values) if values and all(numeric(v) for v in values) else UNAVAILABLE


def percentile(values, p=0.95):
    values = sorted(v for v in values if numeric(v))
    if not values:
        return UNAVAILABLE
    location = (len(values)-1)*p
    low, high = math.floor(location), math.ceil(location)
    return values[low] + (values[high]-values[low])*(location-low)


def difference(a, b):
    return a-b if numeric(a) and numeric(b) else NOT_COMPUTED


def reduction(baseline, proposed):
    return (baseline-proposed)/baseline*100 if numeric(baseline) and numeric(proposed) and baseline > 0 else NOT_COMPUTED


def eligible(row):
    critical = row.get("incident_severity") in ("L3", "SEV-1")
    required = [row.get(k) for k in ("overall_quality_score", "faithfulness_score", "safety_score")]
    if row.get("execution_status") != "SUCCESS" or not all(numeric(v) for v in required):
        return UNAVAILABLE
    return required[0] >= RUBRIC["quality_threshold"] and required[1] >= RUBRIC["faithfulness_threshold"] and required[2] >= (5 if critical else 4)


def validate_route(selected, candidates, expected_models):
    result = {"routing_suitability": "UNVALIDATED", "routing_status": "UNVALIDATED",
              "over_routing": UNAVAILABLE, "under_routing": UNAVAILABLE,
              "lowest_cost_quality_eligible_model": UNAVAILABLE,
              "lowest_cost_quality_eligible_model_cost": UNAVAILABLE}
    ids = {r["model_id"] for r in candidates if r.get("execution_status") == "SUCCESS"}
    if ids != set(expected_models) or selected is None or not all(eligible(r) != UNAVAILABLE and numeric(r.get("api_cost")) for r in candidates):
        return {**result, "validation_reason": "All four successful candidates, valid judge scores, and applicable API costs are required; comparison is incomplete."}
    chosen = next((r for r in candidates if r["model_id"] == selected), None)
    if chosen is None:
        return {**result, "validation_reason": "Router selected a model outside the benchmark candidates; no substitution made."}
    options = [r for r in candidates if eligible(r) is True]
    cheapest = min(options, key=lambda r: r["api_cost"]) if options else None
    if cheapest:
        result.update(lowest_cost_quality_eligible_model=cheapest["model_id"], lowest_cost_quality_eligible_model_cost=cheapest["api_cost"])
    if eligible(chosen) is False:
        status = "UNDER_ROUTING"
        reason = "Selected candidate failed the preregistered quality, faithfulness or severity-dependent safety threshold."
    elif cheapest and chosen["api_cost"] > cheapest["api_cost"] + 1e-12:
        status = "OVER_ROUTING"
        reason = f"Selected cost {chosen['api_cost']:.9g} exceeds eligible minimum {cheapest['api_cost']:.9g} USD."
    else:
        status = "APPROPRIATE"
        reason = "Selected candidate meets all thresholds at the lowest eligible measured API cost."
    return {**result, "routing_suitability": status, "routing_status": "VALIDATED",
            "over_routing": status == "OVER_ROUTING", "under_routing": status == "UNDER_ROUTING", "validation_reason": reason}


def summarize(rows):
    success = [r for r in rows if r.get("execution_status") == "SUCCESS"]
    result = {"successful_queries": len(success), "failed_queries": sum(r.get("execution_status") == "FAILED" for r in rows),
              "unavailable_queries": sum(r.get("execution_status") == UNAVAILABLE for r in rows),
              "latency_observations": len(success), "total_observations": len(rows)}
    for key, column in (("quality", "overall_quality_score"), ("faithfulness", "faithfulness_score"),
                        ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                        ("total_tokens", "total_tokens"), ("latency_ms", "latency_ms")):
        vals = [r.get(column) for r in success]
        result["average_"+key] = mean(vals)
        result[key+"_observations"] = sum(numeric(v) for v in vals)
    latencies = [r["latency_ms"] for r in success]
    result.update(median_latency_ms=percentile(latencies, .5), p95_latency_ms=percentile(latencies),
                  min_latency_ms=min(latencies) if latencies else UNAVAILABLE,
                  max_latency_ms=max(latencies) if latencies else UNAVAILABLE,
                  total_api_cost=total(r.get("api_cost") for r in rows),
                  total_standard_tariff_cost=total(r.get("standard_tariff_cost") for r in rows),
                  average_cost_per_query=mean(r.get("api_cost") for r in success))
    result["cost_per_quality_point"] = (result["average_cost_per_query"]/result["average_quality"]
        if numeric(result["average_cost_per_query"]) and numeric(result["average_quality"]) and result["average_quality"] > 0 else NOT_COMPUTED)
    result["missing_metrics_reason"] = "UNAVAILABLE means missing provider usage, billing tier, failed/skipped judge, or no successful observation; inspect raw_events.jsonl."
    return result


def tables(events, manifest):
    observations = [e["data"] for e in events if e["kind"] == "observation"]
    candidates = [r for r in observations if r.get("scope") == "candidate"]
    pipelines = [r for r in observations if r.get("scope") == "pipeline:E"]
    expected = [m["model_id"] for m in manifest["models"]]
    routes, tradeoffs = [], []
    for q in manifest["queries"]:
        qid = q["query_id"]
        pipeline = next((r for r in pipelines if r["query_id"] == qid), {})
        rows = [r for r in candidates if r["query_id"] == qid]
        selected = pipeline.get("router_selected_model")
        chosen = next((r for r in rows if r["model_id"] == selected), {})
        validation = validate_route(selected, rows, expected) if len(expected) == 4 else {
            "routing_suitability": "UNVALIDATED", "routing_status": "UNVALIDATED", "over_routing": UNAVAILABLE,
            "under_routing": UNAVAILABLE, "validation_reason": "Four-model comparison required; --models filtered candidates."}
        route = {k: UNAVAILABLE for k in ROUTE_COLUMNS}
        route.update({k: pipeline.get(k, UNAVAILABLE) for k in ("incident_category", "incident_severity", "query_complexity", "retrieval_confidence")})
        route.update(experiment_id=manifest["experiment_id"], query_id=qid, query=q["query"],
                     router_selected_model=selected or UNAVAILABLE, router_selected_tier=pipeline.get("router_selected_tier", UNAVAILABLE), **validation)
        for dest, src in (("quality", "overall_quality_score"), ("faithfulness", "faithfulness_score"),
                          ("cost", "api_cost"), ("tokens", "total_tokens"), ("latency", "latency_ms")):
            route["selected_model_"+dest] = chosen.get(src, UNAVAILABLE)
        route["measurement_basis"] = "Initial actual graph router choice validated against independent fixed-context candidate calls; not full-pipeline answer quality."
        routes.append(route)
        for row in rows:
            entry = {k: row.get(k, UNAVAILABLE) for k in TRADEOFF_COLUMNS}
            entry.update(router_selected_model=selected or UNAVAILABLE, model_selected_by_router=row["model_id"] == selected,
                         quality_score=row.get("overall_quality_score", UNAVAILABLE), quality_eligible=eligible(row), **{k: validation[k] for k in ("routing_suitability",)})
            for dest, src in (("quality", "overall_quality_score"), ("cost", "api_cost"), ("token", "total_tokens"), ("latency", "latency_ms")):
                entry[dest+"_difference_vs_router"] = difference(row.get(src), chosen.get(src))
            measured = {k: entry[k] for k in entry if "_difference_vs_router" in k and numeric(entry[k])}
            entry["tradeoff_explanation"] = ("Measured differences (candidate minus separately executed router-selected candidate): " + json.dumps(measured) if measured else "Differences NOT_COMPUTED: selected candidate or comparable measurements unavailable.") + " " + validation["validation_reason"]
            tradeoffs.append(entry)
    summaries, baselines = [], []
    names = ["Always Nemotron", "Always GPT-OSS 20B", "Always Gemini 2.5 Flash", "Always Cohere Command-R+"]
    for model in manifest["models"]:
        rows = [r for r in candidates if r["model_id"] == model["model_id"]]
        stats = summarize(rows)
        summaries.append({**stats, **{k: model[k] for k in ("model_id", "provider")}, "model_tier": model["tier"],
                          "successful_executions": stats["successful_queries"], "failed_executions": stats["failed_queries"]})
        tier_index = int(model["tier"][4])
        baseline = {"strategy": names[tier_index], "model_id": model["model_id"], **stats}
        baseline.update(cost_reduction_percent=NOT_COMPUTED, token_reduction_percent=NOT_COMPUTED,
                        quality_difference_vs_strongest_model=NOT_COMPUTED, latency_difference_vs_strongest_model=NOT_COMPUTED,
                        routing_suitability_rate="NOT_APPLICABLE")
        baselines.append(baseline)
    valid = [r for r in routes if r["routing_status"] == "VALIDATED"]
    adaptive = {"strategy": "AdaRAG-Ops", **summarize(pipelines),
                "cost_reduction_percent": NOT_COMPUTED, "token_reduction_percent": NOT_COMPUTED,
                "quality_difference_vs_strongest_model": NOT_COMPUTED, "latency_difference_vs_strongest_model": NOT_COMPUTED,
                "routing_suitability_rate": 100*sum(r["routing_suitability"] == "APPROPRIATE" for r in valid)/len(valid) if valid else UNAVAILABLE,
                "routing_validated_count": len(valid)}
    # Strict complete-case cohort for the empirical strongest model.
    strongest = max(summaries, key=lambda r: r["average_quality"]) if summaries and all(
        numeric(s["average_quality"]) and s["quality_observations"] == len(manifest["queries"]) for s in summaries) and len(summaries) == 4 else None
    for b in baselines + [adaptive]:
        b["strongest_model"] = strongest["model_id"] if strongest else NOT_COMPUTED
        if strongest:
            b["quality_difference_vs_strongest_model"] = difference(b["average_quality"], strongest["average_quality"])
            b["latency_difference_vs_strongest_model"] = difference(b["average_latency_ms"], strongest["average_latency_ms"])
    paired = []
    for b in baselines:
        pairs = [(c, p) for c in candidates for p in pipelines if c["query_id"] == p["query_id"]
                 and c["model_id"] == b["model_id"] and c["execution_status"] == p["execution_status"] == "SUCCESS"]
        record = {"baseline": b["strategy"], "paired_queries": len(pairs), "query_ids": [c["query_id"] for c, _ in pairs],
                  "cost_reduction_percent": reduction(total(c.get("api_cost") for c, p in pairs), total(p.get("api_cost") for c, p in pairs)),
                  "token_reduction_percent": reduction(total(c.get("total_tokens") for c, p in pairs), total(p.get("total_tokens") for c, p in pairs)),
                  "quality_difference": difference(mean(p.get("overall_quality_score") for c, p in pairs), mean(c.get("overall_quality_score") for c, p in pairs)),
                  "latency_difference_ms": difference(mean(p.get("latency_ms") for c, p in pairs), mean(c.get("latency_ms") for c, p in pairs)),
                  "cost_difference": difference(total(p.get("api_cost") for c, p in pairs), total(c.get("api_cost") for c, p in pairs)),
                  "reason": "Compare full pipeline including all calls against single-response fixed-context baseline, on matched successful queries only; costs/tokens require complete measurements. Zero baseline denominators and missing data => NOT_COMPUTED."}
        # Quality needs pairwise coverage too; a missing judge cannot silently change cohorts.
        if not all(numeric(c.get("overall_quality_score")) and numeric(p.get("overall_quality_score")) for c, p in pairs):
            record["quality_difference"] = NOT_COMPUTED
        b.update(cost_reduction_percent=record["cost_reduction_percent"], token_reduction_percent=record["token_reduction_percent"],
                 paired_queries=len(pairs), reduction_basis="AdaRAG-Ops versus this baseline on paired successful queries")
        paired.append(record)
    ablations = []
    for config, features in FEATURES.items():
        rows = [r for r in observations if r.get("scope") == "pipeline:"+config]
        validations = []
        for row in rows:
            validations.append(validate_route(row.get("router_selected_model"), [c for c in candidates if c["query_id"] == row["query_id"]], expected))
        val = [v for v in validations if v["routing_status"] == "VALIDATED"]
        ablations.append({"configuration": config, "features_enabled": features, **summarize(rows),
                          "execution_status": "EXECUTED" if rows else NOT_COMPUTED,
                          "reason": "Real graph runs; E reuses original pipeline observations." if rows else "--run-ablations not enabled or execution blocked.",
                          "cost_reduction_percent": reduction(adaptive["total_api_cost"], total(r.get("api_cost") for r in rows)),
                          "routing_suitability_rate": 100*sum(v["routing_suitability"] == "APPROPRIATE" for v in val)/len(val) if val else UNAVAILABLE,
                          "over_routing_rate": 100*sum(v["over_routing"] for v in val)/len(val) if val else UNAVAILABLE,
                          "under_routing_rate": 100*sum(v["under_routing"] for v in val)/len(val) if val else UNAVAILABLE})
    ends = [e for e in events if e["kind"] == "attempt_end"]
    end_ids = {e["attempt_id"] for e in ends}
    log = [{**e, "execution_status": "UNAVAILABLE", "error_message": "Interrupted after attempt_start; provider outcome/billing unknown. Not automatically retried.",
            "request_end_time": UNAVAILABLE, "latency_ms": UNAVAILABLE, "input_tokens": UNAVAILABLE, "output_tokens": UNAVAILABLE,
            "api_cost": UNAVAILABLE} for e in events if e["kind"] == "attempt_start" and e["attempt_id"] not in end_ids]
    log += ends
    result = dict(zip(SCHEMAS, [candidates, routes, summaries, baselines+[adaptive], tradeoffs, ablations, log]))
    return result, {"model_summary": summaries, "baseline_comparison": baselines+[adaptive], "paired_comparisons": paired,
                    "routing_validation": routes, "ablation_results": ablations, "pipeline_observations": pipelines,
                    "candidate_successes": sum(c["execution_status"] == "SUCCESS" for c in candidates),
                    "candidate_failures": sum(c["execution_status"] == "FAILED" for c in candidates),
                    "candidate_unavailable": sum(c["execution_status"] == UNAVAILABLE for c in candidates),
                    "actual_attempts": len(log), "successful_api_attempts": sum(e["execution_status"] == "SUCCESS" for e in log),
                    "failed_api_attempts": sum(e["execution_status"] == "FAILED" for e in log),
                    "router_invocations": sum(e["kind"] == "node_end" and e.get("stage") == "router_agent" for e in events)}


def figures(directory):
    """Read exported CSVs, never an in-memory mock or legacy result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    def read(name):
        with (directory/name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    def number(value):
        try:
            f = float(value)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    summary = read("model_comparison_summary.csv")
    rows = read("model_comparison_results.csv")
    routes = read("routing_decision_validation.csv")
    status = {}
    def plot(name, points, xlabel, ylabel, scatter=False):
        if not points:
            status[name] = {"status": UNAVAILABLE, "reason": "No measured values in source CSV for this figure."}
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        if scatter:
            for label, x, y in points:
                ax.scatter(x, y, label=label)
            ax.legend(fontsize=7)
        else:
            ax.bar([p[0] for p in points], [p[1] for p in points])
            ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").removesuffix(".png"))
        fig.tight_layout()
        fig.savefig(directory/name, dpi=160)
        plt.close(fig)
        status[name] = {"status": "GENERATED", "source": "CSV", "points": points}
    points = [(s["model_id"], number(s["average_cost_per_query"]), number(s["average_quality"])) for s in summary]
    plot("quality_vs_cost.png", [p for p in points if None not in p], "Average applicable API cost (USD)", "LLM judged quality (1–5)", True)
    for name, column, label in (("token_consumption_comparison.png", "average_total_tokens", "Mean provider-reported tokens"),
                                ("latency_comparison.png", "average_latency_ms", "Mean successful call latency (ms)")):
        plot(name, [(s["model_id"], number(s[column])) for s in summary if number(s[column]) is not None], "Candidate model", label)
    counts = collections.Counter(r["router_selected_model"] for r in routes if r["router_selected_model"] != UNAVAILABLE)
    plot("routing_distribution.png", list(counts.items()), "Initial actual graph router selection", "Queries")
    plot("routing_suitability.png", list(collections.Counter(r["routing_suitability"] for r in routes).items()), "Decision status (missing evidence stays unvalidated)", "Queries")
    for name, column, label in (("per_query_quality_comparison.png", "overall_quality_score", "LLM judged quality"),
                                ("per_query_cost_comparison.png", "api_cost", "Applicable API cost (USD)")):
        plot(name, [(r["query_id"]+" / "+r["model_tier"], number(r[column])) for r in rows if number(r[column]) is not None], "Query / candidate", label)
    return status


def validate_artifacts(directory, events, manifest, exported, report, prices, secrets=()):
    directory = Path(directory)
    checks = []
    def check(name, valid, reason=""):
        checks.append({"check": name, "status": "PASS" if valid else "FAIL", "reason": reason})
    for filename, required in SCHEMAS.items():
        check(filename+" exists", (directory/filename).exists())
        with (directory/filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            check(filename+" columns", set(required) <= set(reader.fieldnames or []))
            rows = list(reader)
        check(filename+" rows and values", len(rows) == len(exported[filename]) and all(
            all(str(v) == row[k] or (isinstance(v, (dict, list)) and json.dumps(v, ensure_ascii=False) == row[k]) for k, v in expected.items())
            for row, expected in zip(rows, exported[filename])))
    ends = [e for e in events if e["kind"] == "attempt_end"]
    starts = {e["attempt_id"] for e in events if e["kind"] == "attempt_start"}
    check("Actual attempt provenance", all(e["attempt_id"] in starts for e in ends) and len({e["attempt_id"] for e in ends}) == len(ends))
    check("Successful executions have responses", all(e.get("response", "").strip() for e in ends if e["execution_status"] == "SUCCESS"))
    check("Failed executions have errors", all(e.get("error_message") for e in ends if e["execution_status"] == "FAILED"))
    check("No fabricated token estimates; arithmetic", all(
        e.get("token_measurement_method") == "provider_reported" and e["total_tokens"] == e["input_tokens"]+e["output_tokens"]
        if numeric(e.get("total_tokens")) else e.get("input_tokens") == e.get("output_tokens") == UNAVAILABLE for e in ends))
    check("Costs reproduce from preserved pricing", all(e["api_cost"] == price_usage(e, prices)["api_cost"] and e["standard_tariff_cost"] == price_usage(e, prices)["standard_tariff_cost"] for e in ends))
    check("Latencies are measured monotonic durations", all(numeric(e.get("latency_ms")) and e["latency_ms"] >= 0
          and e["latency_ms"] == (e["monotonic_end"]-e["monotonic_start"])*1000 for e in ends))
    candidate_rows = exported["model_comparison_results.csv"]
    check("Controlled context identical across candidates", all(len({r.get("context_hash") for r in candidate_rows if r["query_id"] == q["query_id"]}) <= 1 for q in manifest["queries"]))
    check("Only configured query-model observations", all(r["query_id"] in {q["query_id"] for q in manifest["queries"]}
          and r["model_id"] in {m["model_id"] for m in manifest["models"]} for r in candidate_rows))
    recalculated, metrics = tables(events, manifest)
    check("Summaries reproduce from observations", recalculated == exported)
    check("Report agrees with observations", report["metrics"] == metrics)
    check("Missing metrics explicitly identified", all("missing_metrics_reason" in r for r in exported["model_comparison_summary.csv"]))
    check("Figures traceable to CSV", all(f["status"] in ("GENERATED", UNAVAILABLE) and (f["status"] != "GENERATED" or f.get("source") == "CSV") for f in report["figures"].values()))
    check("No configured secrets in artifacts", all(not any(s and len(s) >= 8 and s in path.read_text(encoding="utf-8", errors="replace") for s in secrets)
          for path in directory.iterdir() if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".md")))
    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", "checks": checks,
            "scope": "Data/software integrity, not scientific completeness or human validation. No independent invoice or absolute correctness validation."}


def write_outputs(journal, manifest, prices, limitations):
    directory = journal.directory
    exported, metrics = tables(journal.events, manifest)
    for filename, rows in exported.items():
        write_csv(directory/filename, rows, SCHEMAS[filename])
    observations = [e["data"] for e in journal.events if e["kind"] == "observation"]
    write_csv(directory/"pipeline_results.csv", [r for r in observations if r.get("scope", "").startswith("pipeline:")], MODEL_COLUMNS)
    write_csv(directory/"judge_results.csv", [e for e in journal.events if e["kind"] == "attempt_end" and e.get("scope", "").startswith("judge:")], LOG_COLUMNS)
    figure_status = figures(directory)
    errors = [e for e in journal.events if e["kind"] in ("node_error", "local_error") or e.get("execution_status") == "FAILED"]
    stage_accounting = {}
    attempts = [e for e in journal.events if e["kind"] == "attempt_end"]
    for scope, stage in sorted({(e["scope"], e["stage"]) for e in attempts}):
        rows = [e for e in attempts if (e["scope"], e["stage"]) == (scope, stage)]
        stage_accounting[scope+"/"+stage] = {"attempts": len(rows), "input_tokens": total(r.get("input_tokens") for r in rows),
            "output_tokens": total(r.get("output_tokens") for r in rows), "api_cost": total(r.get("api_cost") for r in rows),
            "standard_tariff_cost": total(r.get("standard_tariff_cost") for r in rows),
            "reason": "Strict totals include failed calls; unreported failure usage/charges remain unavailable."}
    report = {"experiment_configuration": manifest, "dataset_description": manifest["dataset_source"],
              "query_set": manifest["queries"], "model_identifiers": manifest["models"], "evaluation_rubric": RUBRIC,
              "pricing": prices, "metrics": metrics, "stage_accounting": stage_accounting,
              "embedding": {"model": "all-MiniLM-L6-v2", "execution": "local", "tokens": "NOT_APPLICABLE", "api_cost": "NOT_APPLICABLE", "reason": "No remote embedding API; local compute expense not measured."},
              "router_tokens": {"value": "NOT_APPLICABLE", "reason": "Local Python routing; classifier LLM usage is separately recorded."},
              "retry_accounting": "SDK retries disabled. Explicit production fallbacks/graph loops are separately observed calls; no automatic candidate retry. Interrupted attempts are retained as unknown.",
              "figures": figure_status, "limitations": limitations,
              "missing_data": ["Account billing tier/charges unavailable except verified free endpoint. See standard_tariff_cost for rate-derived estimates.",
                               "Per-metric observation counts are included; failed usage remains unknown unless returned by provider.",
                               "Figure entries marked UNAVAILABLE have no measured values and no PNG is fabricated."],
              "execution_errors": errors}
    (directory/"experiment_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    sections = ["# AdaRAG-Ops empirical evaluation", f"Experiment `{manifest['experiment_id']}`; mode: actual execution.",
        "## 1. Experimental setup", "Existing LangGraph graph executed with response caching and SDK retries disabled. Fixed-model single-response RAG and full graph are separate experiments; graph overhead is included only in pipeline measurements.",
        "## 2. Dataset description", f"{len(manifest['queries'])} exact repository queries from `{manifest['dataset_source']}`. Dataset SHA256: `{manifest['dataset_hash']}`. References are unverified repository answers.",
        "## 3. Candidate models", "\n".join(f"- {m['tier']}: `{m['model_id']}` ({m['provider']})" for m in manifest["models"]),
        "## 4. Baseline methods", "Each fixed candidate independently receives the identical query, saved retrieved context and prompt. These baselines use one generation; full graph uses multiple agent calls.",
        "## 5. Proposed method", f"Original router strategy `{manifest['router_strategy']}`, original weights and credential-dependent catalog preserved. Initial and subsequent choices appear in node events. No cached response is counted as inference.",
        "## 6. Evaluation methodology", "LLM-judged quality, not human-validated correctness. Five 1–5 dimensions averaged equally; quality >=3.5, faithfulness >=4, safety >=4 (5 for L3/SEV-1). Prompts, reasons, raw judge responses and overhead are retained. No scores substituted after judge failures.",
        "## 7. Model comparison", f"Candidate successful executions: {metrics['candidate_successes']}; failed: {metrics['candidate_failures']}; unavailable: {metrics['candidate_unavailable']}.",
        "| Model | Success | Quality (n) | Faithfulness | Tokens | Mean latency ms | P95 ms | API cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return f"{v:.6g}" if numeric(v) else str(v)
    for s in metrics["model_summary"]:
        sections.append(f"| {s['model_id']} | {s['successful_executions']} | {fmt(s['average_quality'])} ({s['quality_observations']}) | {fmt(s['average_faithfulness'])} | {fmt(s['average_total_tokens'])} | {fmt(s['average_latency_ms'])} | {fmt(s['p95_latency_ms'])} | {fmt(s['total_api_cost'])} |")
    statuses = dict(collections.Counter(r["routing_suitability"] for r in metrics["routing_validation"]))
    sections += ["## 8. Routing validation", f"Actual router node invocations (all configurations): {metrics['router_invocations']}. Suitability counts: {json.dumps(statuses)}. Complete four-model evidence and applicable costs required; missing data never implies correct routing.",
        "## 9. Quality-cost trade-off", "Matched full-pipeline versus single-response comparisons are in the report. Tariff estimates are separate from applicable charges. Cost reductions without known billing data are NOT_COMPUTED.",
        "```json\n"+json.dumps(metrics["paired_comparisons"], indent=2)+"\n```",
        "## 10. Token consumption", "Provider-reported input + output tokens only. Candidate, classifier, diagnostic, resolution, verification, judge, graph loops and fallback attempts are separated by scope/stage in the journal and report. No word-count token approximation is used.",
        "## 11. Latency", "Monotonic perf_counter durations. Successful-only statistics include counts, mean, median, linearly interpolated P95, min and max. Pipeline durations include local work and additional calls; failed durations remain in execution logs.",
        "## 12. Ablation study", "A–D execute the existing graph with controlled router feature masks. Severity's implicit complexity floor is removed when severity is masked. Retrieval is neutralized to fixed confidence 0.5 when masked. Classifier and operational severity safety guards remain active. Cost/capability priors remain fixed; these are conditional signal ablations, not retrained routers. E reuses the original full graph observations.",
        "\n".join(f"- {a['configuration']}: {a['execution_status']}; {a['successful_queries']} successful queries; quality {fmt(a['average_quality'])}; suitability {fmt(a['routing_suitability_rate'])}." for a in metrics["ablation_results"]),
        "## 13. Limitations", "\n".join("- "+s for s in limitations),
        "## 14. Conclusion", "The recorded observations support only the metrics with available measurements. This small single-run evaluation does not establish statistical superiority, absolute diagnostic correctness, verified remediation, or production savings. See failures and unvalidated decisions before interpreting trade-offs.",
        "## Figures", "\n".join(f"- {name}: {value['status']}"+("; "+value["reason"] if "reason" in value else "") for name, value in figure_status.items())]
    (directory/"experiment_summary.md").write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    validation = validate_artifacts(directory, journal.events, manifest, exported, report, prices, journal.secrets)
    (directory/"data_integrity_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return metrics, validation

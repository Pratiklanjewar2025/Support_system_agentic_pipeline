# -*- coding: utf-8 -*-
"""
experiments/aggregate_metrics.py — Metric Aggregation Across Experimental Runs (Phase 9)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Reads structured per-case records or summary JSONs across multiple experiment runs.
Computes:
- Sample counts and dataset composition
- Mean, standard deviation, variance, and standard error across repeated runs
- 95% Confidence Intervals (Student's t distribution)
- Diagnostic quality, calibration, and efficiency comparative tables
- Formatted markdown reports and exported CSV/JSON summaries
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmarks.evaluation_metrics import (
    StatisticalReporter,
    TOKEN_PRICING_CATALOG
)
from experiments.reproducibility import RUNS_DIR


def load_run_summaries(runs_root: Path = RUNS_DIR) -> List[Dict[str, Any]]:
    """Loads all summary_metrics.json files from historical non-overwriting run directories."""
    summaries = []
    if not runs_root.exists():
        return summaries

    for item in sorted(runs_root.iterdir()):
        if item.is_dir():
            sum_file = item / "summary_metrics.json"
            if sum_file.exists():
                try:
                    with open(sum_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["run_dir"] = str(item)
                        summaries.append(data)
                except Exception:
                    pass
    return summaries


def aggregate_experiment_runs(runs_root: Path = RUNS_DIR, export_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Aggregates metrics across all recorded experiment runs.
    Calculates sample statistics and 95% confidence intervals per configuration.
    """
    summaries = load_run_summaries(runs_root)
    if not summaries:
        print("[!] No completed experimental runs found in:", runs_root)
        return {"status": "empty", "runs_count": 0}

    print("=" * 100)
    print("   ADARAG-OPS: CROSS-EXPERIMENT METRIC AGGREGATION & REPORTING (PHASE 9)")
    print("=" * 100)
    print(f"Total Completed Runs Discovered: {len(summaries)}")
    print("-" * 100)

    # Group runs by experiment_id
    grouped_by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for s in summaries:
        exp_id = s.get("experiment_id", "UNKNOWN")
        grouped_by_exp.setdefault(exp_id, []).append(s)

    aggregated_report = {
        "total_runs_analyzed": len(summaries),
        "unique_experiments_count": len(grouped_by_exp),
        "configurations": {}
    }

    # Header for comparative table
    print(f"{'Experiment ID':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'Macro-F1':<9} | {'Brier':<8} | {'ECE':<8} | {'Billed ($)':<11} | {'Est ($)':<10} | {'p50 (s)':<8}")
    print("-" * 115)

    rows_for_csv = []
    for exp_id, runs in grouped_by_exp.items():
        n_runs = len(runs)
        name = runs[0].get("experiment_name", exp_id)

        hit1_vals = [r.get("diagnostic_quality", {}).get("root_cause_localization_hit1_percent", 0.0) for r in runs]
        hit3_vals = [r.get("diagnostic_quality", {}).get("root_cause_localization_hit3_percent", 0.0) for r in runs]
        macro_vals = [r.get("diagnostic_quality", {}).get("fault_classification_macro_f1", 0.0) for r in runs]
        brier_vals = [r.get("confidence_and_reliability", {}).get("brier_score", 0.0) for r in runs]
        ece_vals = [r.get("confidence_and_reliability", {}).get("expected_calibration_error_ece", 0.0) for r in runs]
        billed_vals = [r.get("operational_efficiency", {}).get("actual_api_cost_usd", 0.0) for r in runs]
        est_vals = [r.get("operational_efficiency", {}).get("estimated_cost_usd", 0.0) for r in runs]
        p50_vals = [r.get("operational_efficiency", {}).get("median_latency_p50_sec", 0.0) for r in runs]

        ci_hit1 = StatisticalReporter.compute_confidence_interval(hit1_vals)
        ci_macro = StatisticalReporter.compute_confidence_interval(macro_vals)
        ci_brier = StatisticalReporter.compute_confidence_interval(brier_vals)
        ci_est_cost = StatisticalReporter.compute_confidence_interval(est_vals)

        mean_hit1 = ci_hit1["mean"]
        mean_hit3 = round(float(np.mean(hit3_vals)), 1)
        mean_macro = ci_macro["mean"]
        mean_brier = ci_brier["mean"]
        mean_ece = round(float(np.mean(ece_vals)), 4)
        mean_billed = round(float(np.mean(billed_vals)), 6)
        mean_est = ci_est_cost["mean"]
        mean_p50 = round(float(np.mean(p50_vals)), 2)

        print(f"{exp_id:<32} | {mean_hit1:<7}% | {mean_hit3:<7}% | {mean_macro:<9.4f} | {mean_brier:<8.4f} | {mean_ece:<8.4f} | ${mean_billed:<10.5f} | ${mean_est:<9.5f} | {mean_p50:<7.2f}s")

        exp_summary = {
            "experiment_id": exp_id,
            "experiment_name": name,
            "repeated_runs_count": n_runs,
            "root_cause_hit1": ci_hit1,
            "fault_macro_f1": ci_macro,
            "brier_score": ci_brier,
            "estimated_cost_usd": ci_est_cost,
            "mean_metrics": {
                "hit_at_1_percent": mean_hit1,
                "hit_at_3_percent": mean_hit3,
                "fault_macro_f1": mean_macro,
                "brier_score": mean_brier,
                "ece": mean_ece,
                "actual_billed_cost_usd": mean_billed,
                "estimated_cost_usd": mean_est,
                "median_latency_p50_sec": mean_p50
            }
        }
        aggregated_report["configurations"][exp_id] = exp_summary

        rows_for_csv.append({
            "experiment_id": exp_id,
            "experiment_name": name,
            "repeated_runs": n_runs,
            "mean_hit1_pct": mean_hit1,
            "hit1_95_ci_lower": ci_hit1["ci_lower"],
            "hit1_95_ci_upper": ci_hit1["ci_upper"],
            "mean_hit3_pct": mean_hit3,
            "mean_fault_macro_f1": mean_macro,
            "mean_brier_score": mean_brier,
            "mean_ece": mean_ece,
            "mean_actual_billed_usd": mean_billed,
            "mean_estimated_cost_usd": mean_est,
            "mean_latency_p50_sec": mean_p50
        })

    print("-" * 115)

    # Export to artifacts/ if requested or default
    target_dir = export_dir or (runs_root.parent / "artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)

    json_export = target_dir / "aggregated_metrics.json"
    with open(json_export, "w", encoding="utf-8") as f:
        json.dump(aggregated_report, f, indent=2)

    csv_export = target_dir / "aggregated_metrics.csv"
    if rows_for_csv:
        with open(csv_export, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)

    print(f"\n[OK] Aggregated metrics exported to:\n  - {json_export}\n  - {csv_export}")
    return aggregated_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metrics across completed experiment runs.")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR), help="Directory containing run artifacts")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to export aggregated CSV/JSON")
    args = parser.parse_args()

    r_dir = Path(args.runs_dir)
    e_dir = Path(args.export_dir) if args.export_dir else None
    aggregate_experiment_runs(runs_root=r_dir, export_dir=e_dir)

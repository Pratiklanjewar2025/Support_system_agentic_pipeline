# -*- coding: utf-8 -*-
"""
experiments/run_statistical_tests.py — Paired Statistical Comparisons (Phase 9)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Compares paired per-case experimental outcomes across identical incident cases.
Computes:
- Paired Student's t-test (parametric significance)
- Wilcoxon signed-rank test (non-parametric rank significance)
- Cohen's d (parametric effect size)
- Cliff's delta (non-parametric dominance effect size)
- Formatted comparative reports and significance flags (p < 0.05)
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmarks.evaluation_metrics import StatisticalReporter
from experiments.reproducibility import RUNS_DIR


def load_per_case_records(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads per-case results from JSONL or CSV in run directory, keyed by case_id."""
    run_dir = Path(run_dir)
    jsonl_file = run_dir / "per_case_results.jsonl"
    csv_file = run_dir / "per_case_results.csv"

    cases: Dict[str, Dict[str, Any]] = {}
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cases[rec["case_id"]] = rec
        return cases

    if csv_file.exists():
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings
                conv = {}
                for k, v in row.items():
                    try:
                        conv[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        conv[k] = v
                cases[conv["case_id"]] = conv
        return cases

    raise FileNotFoundError(f"Neither per_case_results.jsonl nor per_case_results.csv found in {run_dir}")


def run_paired_comparison(
    baseline_run_dir: Path,
    target_run_dir: Path,
    metrics_to_compare: Optional[List[str]] = None,
    export_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes paired parametric and non-parametric statistical tests
    between baseline and target experimental runs.
    """
    b_records = load_per_case_records(baseline_run_dir)
    t_records = load_per_case_records(target_run_dir)

    common_ids = sorted(list(set(b_records.keys()) & set(t_records.keys())))
    if not common_ids:
        raise ValueError("No common case_ids found between baseline and target runs for paired comparison.")

    metrics = metrics_to_compare or ["composite_score", "hit_at_1", "f1", "estimated_cost_usd", "latency_sec"]

    print("=" * 105)
    print("   ADARAG-OPS: PAIRED STATISTICAL COMPARISONS & EFFECT SIZES (PHASE 9)")
    print("=" * 105)
    print(f"  * Baseline Run Dir : {baseline_run_dir.name}")
    print(f"  * Target Run Dir   : {target_run_dir.name}")
    print(f"  * Common Cases (N) : {len(common_ids)}")
    print("-" * 105)
    print(f"{'Metric':<24} | {'Mean Diff':<12} | {'t-stat':<8} | {'p-param':<9} | {'Wilcoxon':<9} | {'p-nonparam':<10} | {'Cohen d':<8} | {'Cliff δ':<8} | {'Sig (p<0.05)'}")
    print("-" * 105)

    comparison_results: Dict[str, Any] = {
        "baseline_run": str(baseline_run_dir),
        "target_run": str(target_run_dir),
        "sample_count": len(common_ids),
        "metrics": {}
    }

    for m in metrics:
        b_vals = []
        t_vals = []
        for cid in common_ids:
            bv = b_records[cid].get(m, 0.0)
            tv = t_records[cid].get(m, 0.0)
            b_vals.append(float(bv) if bv is not None else 0.0)
            t_vals.append(float(tv) if tv is not None else 0.0)

        stat = StatisticalReporter.compute_paired_statistical_test(b_vals, t_vals)
        comparison_results["metrics"][m] = stat

        diff_str = f"+{stat['mean_diff']:.4f}" if stat['mean_diff'] > 0 else f"{stat['mean_diff']:.4f}"
        sig_str = "YES (*)" if stat['statistically_significant'] else "No"
        print(f"{m:<24} | {diff_str:<12} | {stat['t_statistic']:<8.3f} | {stat['p_value_parametric']:<9.4f} | {stat['wilcoxon_stat']:<9.3f} | {stat['p_value_nonparametric']:<10.4f} | {stat['cohen_d']:<8.3f} | {stat['cliff_delta']:<8.3f} | {sig_str}")

    print("=" * 105)

    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"[OK] Statistical comparisons exported to: {export_path}")

    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paired statistical tests between two experiment runs.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline run directory")
    parser.add_argument("--target", type=str, required=True, help="Path to target proposed/ablation run directory")
    parser.add_argument("--export", type=str, default="artifacts/statistical_comparisons.json", help="Export path for JSON results")
    args = parser.parse_args()

    run_paired_comparison(
        baseline_run_dir=Path(args.baseline),
        target_run_dir=Path(args.target),
        export_path=Path(args.export) if args.export else None
    )

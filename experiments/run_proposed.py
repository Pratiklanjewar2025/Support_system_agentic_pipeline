# -*- coding: utf-8 -*-
"""
experiments/run_proposed.py — Reproducible Proposed Methods Execution (Phase 9)
Executes Proposed configurations E, F, G, and H from experiments/configs/proposed_config.json.
Guarantees: Non-overwriting execution, per-case JSONL/CSV logging, and provenance tracking.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments.reproducibility import (
    ExperimentConfig,
    ReproducibleExperimentRunner,
    RunPersistenceManager,
    CONFIGS_DIR
)


def run_all_proposed(config_file: Optional[Path] = None, max_cases: Optional[int] = None):
    cfg_path = config_file or (CONFIGS_DIR / "proposed_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    common = data.get("common_settings", {})
    configs = data.get("configurations", {})
    runner = ReproducibleExperimentRunner()

    print("=" * 85)
    print("   ADARAG-OPS: REPRODUCIBLE PROPOSED METHODS EXECUTION (PHASE 9)")
    print("=" * 85)

    executed_runs = []
    for key, c_data in configs.items():
        merged = {**common, **c_data}
        if max_cases:
            merged.setdefault("dataset", {})["subsample_max_cases"] = max_cases
        cfg = ExperimentConfig.from_dict(merged)

        print(f"\n[*] Executing {cfg.name} (ID: {cfg.experiment_id})...")
        run_id, run_dir, summary = runner.run_experiment(cfg)
        print(f"    -> Run Completed: {run_id}")
        print(f"    -> Artifacts Dir: {run_dir}")
        print(f"    -> Hit@1: {summary['diagnostic_quality']['root_cause_localization_hit1_percent']}% | Brier: {summary['confidence_and_reliability']['brier_score']:.4f} | Est: ${summary['operational_efficiency']['estimated_cost_usd']:.5f}")

        executed_runs.append({
            "config_key": key,
            "experiment_id": cfg.experiment_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "summary": summary
        })

    print("\n" + "=" * 85)
    print(f"   SUCCESS: {len(executed_runs)} Proposed Experiments Executed & Persisted without Overwriting.")
    print("=" * 85)
    return executed_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute reproducible proposed method experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to proposed methods configuration JSON")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases")
    args = parser.parse_args()

    cfg_file = Path(args.config) if args.config else None
    run_all_proposed(config_file=cfg_file, max_cases=args.max_cases)

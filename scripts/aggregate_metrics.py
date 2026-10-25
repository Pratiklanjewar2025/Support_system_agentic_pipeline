# -*- coding: utf-8 -*-
"""CLI wrapper for experiments/aggregate_metrics.py."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aggregate_metrics import aggregate_experiment_runs
from experiments.reproducibility import RUNS_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate metrics across completed experiment runs.")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR), help="Directory containing run artifacts")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to export aggregated CSV/JSON")
    args = parser.parse_args()

    r_dir = Path(args.runs_dir)
    e_dir = Path(args.export_dir) if args.export_dir else None
    aggregate_experiment_runs(runs_root=r_dir, export_dir=e_dir)

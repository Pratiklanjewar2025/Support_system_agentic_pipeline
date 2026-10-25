# -*- coding: utf-8 -*-
"""CLI wrapper for experiments/run_statistical_tests.py."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_statistical_tests import run_paired_comparison

if __name__ == "__main__":
    import argparse
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

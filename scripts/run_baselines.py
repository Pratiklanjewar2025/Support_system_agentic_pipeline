# -*- coding: utf-8 -*-
"""CLI wrapper for experiments/run_baselines.py."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_baselines import run_all_baselines

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Execute reproducible baseline experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to baselines configuration JSON")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases")
    args = parser.parse_args()

    cfg_file = Path(args.config) if args.config else None
    run_all_baselines(config_file=cfg_file, max_cases=args.max_cases)

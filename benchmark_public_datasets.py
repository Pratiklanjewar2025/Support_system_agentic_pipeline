# -*- coding: utf-8 -*-
"""
benchmark_public_datasets.py — CLI for Public Incident Benchmark Integration (Phase 6)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Usage:
  python benchmark_public_datasets.py --dataset rcaeval --split test
  python benchmark_public_datasets.py --dataset rcaeval --split cross_system --unseen-target train-ticket
  python benchmark_public_datasets.py --dataset openrca --split test
  python benchmark_public_datasets.py --dataset supplementary
  python benchmark_public_datasets.py --dataset all --split test
"""

import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from benchmarks.runner import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(
        description="AdaRAG-Ops: Public Microservice Root-Cause Analysis Benchmark Runner"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="rcaeval",
        choices=["rcaeval", "openrca", "supplementary", "all"],
        help="Target benchmark dataset to evaluate (default: rcaeval)"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="test",
        choices=["test", "val", "train", "cross_system", "cross_fault", "all"],
        help="Evaluation split strategy (default: test)"
    )
    parser.add_argument(
        "--unseen-target",
        type=str,
        default="train-ticket",
        help="Unseen microservice system name for cross_system split (default: train-ticket)"
    )
    parser.add_argument(
        "--max-cases", "-m",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate in this run (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_public_results.json",
        help="Output JSON results path (default: benchmark_public_results.json)"
    )

    args = parser.parse_args()

    results = run_benchmark_suite(
        dataset_name=args.dataset,
        split_strategy=args.split,
        unseen_target=args.unseen_target,
        max_cases=args.max_cases
    )

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results persisted to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()

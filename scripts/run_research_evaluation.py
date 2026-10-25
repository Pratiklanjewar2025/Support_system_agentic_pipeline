# -*- coding: utf-8 -*-
"""CLI wrapper for experiments/run_research_evaluation.py."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_research_evaluation import ResearchBenchmarkRunner, DEFAULT_CONFIG_PATH

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLI wrapper for AdaRAG-Ops Research Evaluation.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to evaluation config JSON")
    parser.add_argument("--live", action="store_true", help="Run with live LLM endpoints")
    args = parser.parse_args()

    runner = ResearchBenchmarkRunner(config_path=Path(args.config), live_mode=args.live)
    runner.run_all_configurations()

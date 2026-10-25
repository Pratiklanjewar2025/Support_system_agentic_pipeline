# -*- coding: utf-8 -*-
"""CLI wrapper for experiments/generate_plots.py."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.generate_plots import generate_all_publication_plots, FIGURES_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication-quality experiment plots.")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR), help="Output directory for generated figures")
    args = parser.parse_args()

    generate_all_publication_plots(output_dir=Path(args.output_dir))

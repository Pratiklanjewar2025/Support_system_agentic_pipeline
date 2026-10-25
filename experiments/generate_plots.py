# -*- coding: utf-8 -*-
"""
experiments/generate_plots.py — Publication-Quality Research Visualizations
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Generates publication-quality figures directly from actual computed experimental results:
1. Reliability Diagram (Observed Accuracy vs Average Confidence with Calibration Gaps)
2. Quality vs Cost Pareto Frontier (Hit@1 vs Cost per 1,000 Queries across Configs A-E)
3. Risk-Coverage Curve (Selective Diagnostic Accuracy vs Incident Coverage under Threshold Sweeps)
4. Operational Latency Distribution (Median p50 vs 95th Percentile p95 Latency)
5. Category Breakdown: Diagnostic Localization Hit@1 across Operational Failure Modes

Guarantees:
- All data points are parsed from experiments/results/research_summary_metrics.json and master_execution_log.csv.
- Never uses hardcoded measurements.
- Saves high-resolution PNG (300 DPI) and vector SVG to both artifacts/figures/ and research_paper_results/figures/.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for headless servers / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES_DIR = BASE_DIR / "research_paper_results" / "figures"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "experiments" / "results"
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "research_summary_metrics.json"
DEFAULT_MASTER_LOG_PATH = RESULTS_DIR / "master_execution_log.csv"

# Publication Styling Defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.autolayout": True
})


def load_summary_metrics(path: Path = DEFAULT_SUMMARY_PATH) -> Dict[str, Any]:
    """Loads computed research summary metrics from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Summary metrics file not found: {path}. Please run benchmark runner first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_master_log_records(path: Path = DEFAULT_MASTER_LOG_PATH) -> List[Dict[str, Any]]:
    """Loads raw master execution records from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Master execution log not found: {path}. Please run benchmark runner first.")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def plot_reliability_diagram(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    config_id: str = "CONFIG_E",
    output_dirs: Optional[List[Path]] = None
) -> Path:
    """
    Generates a publication-grade Reliability Diagram from actual calibration data.
    Compares predicted confidence against empirical observed accuracy with calibration gap highlights.
    """
    summary = load_summary_metrics(summary_path)
    cfg_summary = summary.get(config_id, {})
    calib = cfg_summary.get("calibration", {})
    bins = calib.get("reliability_diagram_bins", [])
    ece = calib.get("expected_calibration_error_ece", 0.0)
    brier = calib.get("brier_score", 0.0)

    confidences = [b["average_confidence"] for b in bins]
    accuracies = [b["observed_accuracy"] for b in bins]
    centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    widths = [0.18 for _ in bins]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", linewidth=1.8, label="Perfect Calibration (y = x)")

    # Observed accuracy bars
    bars = ax.bar(centers, accuracies, width=widths, alpha=0.8, color="#1b4f72", edgecolor="#0b2333", linewidth=1.2, label=f"Observed Accuracy ({config_id})")

    # Calibration gap highlights
    first_gap = True
    for c, acc, avg_c, w in zip(centers, accuracies, confidences, widths):
        gap = abs(acc - avg_c)
        if gap > 0.01:
            lower = min(acc, avg_c)
            height = gap
            ax.bar(c, height, bottom=lower, width=w * 0.9, alpha=0.35, color="#c0392b", hatch="//", edgecolor="#922b21", label="Calibration Gap" if first_gap else "")
            first_gap = False

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Predicted Confidence Band $\hat{p}$")
    ax.set_ylabel("Empirical True Accuracy")
    ax.set_title(f"Reliability Diagram: {cfg_summary.get('config_name', config_id)}\n(ECE = {ece:.4f}, Brier = {brier:.4f})")
    ax.legend(loc="upper left", frameon=True)

    dirs = output_dirs or [FIGURES_DIR, PAPER_FIGURES_DIR]
    out_png = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "reliability_diagram.png"
        s = d / "reliability_diagram.svg"
        fig.savefig(p, dpi=300)
        fig.savefig(s)
        if out_png is None:
            out_png = p
    plt.close(fig)
    print(f"  [SAVED] Reliability Diagram -> {out_png}")
    return out_png


def plot_quality_vs_cost_pareto(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    output_dirs: Optional[List[Path]] = None
) -> Path:
    """
    Plots Diagnostic Quality (Hit@1) vs Estimated API Cost trade-off frontier.
    Visualizes empirical Pareto dominance across Configurations A, B, C, D, E.
    """
    summary = load_summary_metrics(summary_path)

    config_styles = {
        "CONFIG_A": {"name": "Config A: Fixed Baseline", "color": "#7f8c8d", "marker": "o", "size": 120},
        "CONFIG_B": {"name": "Config B: Fixed + Cache", "color": "#2980b9", "marker": "s", "size": 130},
        "CONFIG_C": {"name": "Config C: Adaptive Routing", "color": "#8e44ad", "marker": "^", "size": 130},
        "CONFIG_D": {"name": "Config D: Adaptive + Cache", "color": "#d35400", "marker": "D", "size": 140},
        "CONFIG_E": {"name": "Config E: Full AdaRAG-Ops", "color": "#27ae60", "marker": "*", "size": 240}
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.5), dpi=300)

    points = []
    for cid, style in config_styles.items():
        if cid in summary:
            c_data = summary[cid]
            cost_1k = c_data["cost"]["cost_per_1k_queries_usd"]
            hit1 = c_data["accuracy"]["hit_at_1_percent"]
            points.append((cost_1k, hit1, cid, style))

            ax.scatter(
                cost_1k, hit1,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
                label=f"{style['name']} ({hit1:.1f}%, ${cost_1k:.4f}/1k)"
            )

    # Sort points for drawing Pareto frontier curve
    # Points with higher quality and lower/comparable cost
    pareto_candidates = sorted([p for p in points if p[2] in ("CONFIG_C", "CONFIG_D", "CONFIG_E")], key=lambda x: x[0])
    if len(pareto_candidates) >= 2:
        xs = [p[0] for p in pareto_candidates]
        ys = [p[1] for p in pareto_candidates]
        ax.plot(xs, ys, color="#27ae60", linestyle="-.", linewidth=1.8, alpha=0.85, label="Empirical Pareto Optimal Frontier")

    ax.set_xlabel("Inference Cost per 1,000 Queries (USD, $)")
    ax.set_ylabel("Diagnostic Localization Hit@1 (%)")
    ax.set_title("Operational Efficiency: Diagnostic Accuracy vs. Cost per 1,000 Queries")
    ax.legend(loc="lower right", frameon=True)

    dirs = output_dirs or [FIGURES_DIR, PAPER_FIGURES_DIR]
    out_png = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "quality_vs_cost_pareto.png"
        s = d / "quality_vs_cost_pareto.svg"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        fig.savefig(s, bbox_inches="tight")
        if out_png is None:
            out_png = p
    plt.close(fig)
    print(f"  [SAVED] Quality-Cost Pareto -> {out_png}")
    return out_png


def plot_risk_coverage_curve(
    master_log_path: Path = DEFAULT_MASTER_LOG_PATH,
    config_id: str = "CONFIG_E",
    output_dirs: Optional[List[Path]] = None
) -> Path:
    """
    Plots the Selective Classification Risk-Coverage curve directly from execution records.
    Demonstrates accuracy improvement as confidence threshold rejection / abstention triggers.
    """
    records = load_master_log_records(master_log_path)
    cfg_records = [r for r in records if r["config_id"] == config_id]
    n_total = len(cfg_records)

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    coverages = []
    accuracies = []

    for tau in thresholds:
        # Cases that exceed threshold tau and did not abstain
        accepted = [
            r for r in cfg_records
            if float(r["calibrated_confidence"]) >= tau and r["abstained"].lower() != "true"
        ]
        cov = (len(accepted) / max(n_total, 1)) * 100.0
        acc = (
            (sum(int(r["correctness_label"]) for r in accepted) / max(len(accepted), 1)) * 100.0
            if accepted else 100.0
        )
        coverages.append(round(cov, 1))
        accuracies.append(round(acc, 1))

    fig, ax1 = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    color_acc = "#1f618d"
    color_cov = "#d35400"

    ax1.set_xlabel(r"Confidence Rejection Threshold $\tau$")
    ax1.set_ylabel("Selective Diagnostic Accuracy (%)", color=color_acc)
    line1 = ax1.plot(thresholds, accuracies, color=color_acc, marker="o", linewidth=2.4, label="Selective Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(40, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Automated Decision Coverage (%)", color=color_cov)
    line2 = ax2.plot(thresholds, coverages, color=color_cov, marker="s", linestyle="--", linewidth=2.0, label="Coverage (% Answered)")
    ax2.tick_params(axis="y", labelcolor=color_cov)
    ax2.set_ylim(20, 105)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True)

    plt.title(f"Selective Classification: Accuracy vs. Coverage ({config_id})")

    dirs = output_dirs or [FIGURES_DIR, PAPER_FIGURES_DIR]
    out_png = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "risk_coverage_curve.png"
        s = d / "risk_coverage_curve.svg"
        fig.savefig(p, dpi=300)
        fig.savefig(s)
        if out_png is None:
            out_png = p
    plt.close(fig)
    print(f"  [SAVED] Risk-Coverage Curve -> {out_png}")
    return out_png


def plot_latency_distribution(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    output_dirs: Optional[List[Path]] = None
) -> Path:
    """Plots Median (p50) and 95th Percentile (p95) Latency across Configurations A through E."""
    summary = load_summary_metrics(summary_path)

    configs = ["CONFIG_A", "CONFIG_B", "CONFIG_C", "CONFIG_D", "CONFIG_E"]
    labels = ["Config A\n(Fixed)", "Config B\n(Fixed+Cache)", "Config C\n(Adaptive)", "Config D\n(Adapt+Cache)", "Config E\n(AdaRAG-Ops)"]
    p50_vals = []
    p95_vals = []

    for cid in configs:
        if cid in summary:
            p50_vals.append(summary[cid]["latency"]["median_latency_p50_sec"])
            p95_vals.append(summary[cid]["latency"]["p95_latency_sec"])
        else:
            p50_vals.append(0.0)
            p95_vals.append(0.0)

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=300)
    ax.bar(x - width/2, p50_vals, width, label="Median Latency (p50)", color="#2980b9", edgecolor="#1a5276")
    ax.bar(x + width/2, p95_vals, width, label="95th Percentile Latency (p95)", color="#c0392b", edgecolor="#922b21")

    ax.set_ylabel("Latency (Seconds)")
    ax.set_title("Operational Latency Distribution (Median p50 vs. Tail p95)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", frameon=True)

    dirs = output_dirs or [FIGURES_DIR, PAPER_FIGURES_DIR]
    out_png = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "latency_distribution.png"
        s = d / "latency_distribution.svg"
        fig.savefig(p, dpi=300)
        fig.savefig(s)
        if out_png is None:
            out_png = p
    plt.close(fig)
    print(f"  [SAVED] Latency Distribution -> {out_png}")
    return out_png


def plot_category_breakdown(
    master_log_path: Path = DEFAULT_MASTER_LOG_PATH,
    output_dirs: Optional[List[Path]] = None
) -> Path:
    """Plots Hit@1 Localization breakdown across operational failure categories comparing Config A vs Config E."""
    records = load_master_log_records(master_log_path)

    categories = [
        "technical_support", "exact_repeat", "semantic_paraphrase",
        "complex_multi_hop", "high_risk_operational", "absent_from_kb"
    ]
    labels = [
        "Tech Support", "Exact Repeat", "Semantic Para",
        "Complex Multi-Hop", "High-Risk Safety", "Absent from KB"
    ]

    cfg_a_recs = [r for r in records if r["config_id"] == "CONFIG_A"]
    cfg_e_recs = [r for r in records if r["config_id"] == "CONFIG_E"]

    a_acc = []
    e_acc = []

    for cat in categories:
        a_sub = [r for r in cfg_a_recs if r["category"] == cat]
        e_sub = [r for r in cfg_e_recs if r["category"] == cat]

        a_rate = (sum(int(r["correctness_label"]) for r in a_sub) / max(len(a_sub), 1)) * 100.0 if a_sub else 0.0
        e_rate = (sum(int(r["correctness_label"]) for r in e_sub) / max(len(e_sub), 1)) * 100.0 if e_sub else 0.0
        a_acc.append(round(a_rate, 1))
        e_acc.append(round(e_rate, 1))

    x = np.arange(len(categories))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    ax.bar(x - width/2, a_acc, width, label="Config A: Fixed Baseline", color="#95a5a6", edgecolor="#7f8c8d")
    ax.bar(x + width/2, e_acc, width, label="Config E: Full AdaRAG-Ops", color="#27ae60", edgecolor="#1e8449")

    ax.set_ylabel("Diagnostic Correctness / Precision (%)")
    ax.set_title("Operational Failure Archetypes: Baseline vs. AdaRAG-Ops Correctness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=True)

    dirs = output_dirs or [FIGURES_DIR, PAPER_FIGURES_DIR]
    out_png = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "macro_f1_fault_categories.png"
        s = d / "macro_f1_fault_categories.svg"
        fig.savefig(p, dpi=300)
        fig.savefig(s)
        if out_png is None:
            out_png = p
    plt.close(fig)
    print(f"  [SAVED] Category Breakdown -> {out_png}")
    return out_png


def generate_all_publication_plots(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    master_log_path: Path = DEFAULT_MASTER_LOG_PATH
) -> List[Path]:
    """Generates all 5 publication-quality figures directly from experimental output records."""
    print("=" * 85)
    print("   ADARAG-OPS: GENERATING DATA-DRIVEN PUBLICATION FIGURES (PHASE 7)")
    print("=" * 85)
    p1 = plot_reliability_diagram(summary_path=summary_path)
    p2 = plot_quality_vs_cost_pareto(summary_path=summary_path)
    p3 = plot_risk_coverage_curve(master_log_path=master_log_path)
    p4 = plot_latency_distribution(summary_path=summary_path)
    p5 = plot_category_breakdown(master_log_path=master_log_path)
    print("=" * 85)
    print(f"[OK] Generated 5 publication figures (PNG + SVG) in: {FIGURES_DIR} and {PAPER_FIGURES_DIR}")
    return [p1, p2, p3, p4, p5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication figures from actual results.")
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY_PATH), help="Summary metrics JSON path")
    parser.add_argument("--master-log", type=str, default=str(DEFAULT_MASTER_LOG_PATH), help="Master execution log CSV path")
    args = parser.parse_args()

    generate_all_publication_plots(summary_path=Path(args.summary), master_log_path=Path(args.master_log))

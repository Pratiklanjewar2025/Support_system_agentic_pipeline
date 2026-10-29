"""
Generates the unified 6-panel model_comparison_dashboard.png from experimental CSV outputs using standard library csv.
"""
import os
import csv
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_dashboard():
    results_dir = Path("results/ada_rag_ops_evaluation")
    summary_csv = results_dir / "model_comparison_summary.csv"
    results_csv = results_dir / "model_comparison_results.csv"
    val_csv = results_dir / "routing_decision_validation.csv"
    
    if not (summary_csv.exists() and results_csv.exists() and val_csv.exists()):
        print(f"Waiting: CSV files not yet ready in {results_dir}")
        return False

    with open(summary_csv, "r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    with open(results_csv, "r", encoding="utf-8") as f:
        results_rows = list(csv.DictReader(f))
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    models = [r["model_tier"] for r in summary_rows]
    palette = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b"]

    # 1. Response quality by model
    ax1 = axes[0, 0]
    qualities = [float(r["average_quality"]) for r in summary_rows]
    bars1 = ax1.bar(models, qualities, color=palette, edgecolor="black", width=0.5)
    ax1.axhline(3.5, color="red", linestyle="--", linewidth=1.5, label="Quality Threshold (3.5)")
    ax1.set_title("1. Response Quality by Model (0 - 5.0)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Average Quality Score", fontsize=11)
    ax1.set_ylim(0, 5.5)
    ax1.legend(loc="upper left")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    # 2. Token consumption by model
    ax2 = axes[0, 1]
    total_tokens = [float(r["average_total_tokens"]) for r in summary_rows]
    bars2 = ax2.bar(models, total_tokens, color=palette, edgecolor="black", width=0.5)
    ax2.set_title("2. Average Total Token Consumption per Query", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Tokens", fontsize=11)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.0f}", ha="center", va="bottom", fontweight="bold")

    # 3. Input versus output tokens
    ax3 = axes[1, 0]
    in_tok = [float(r["average_input_tokens"]) for r in summary_rows]
    out_tok = [float(r["average_output_tokens"]) for r in summary_rows]
    x = np.arange(len(models))
    w = 0.35
    ax3.bar(x - w/2, in_tok, width=w, label="Input Tokens", color="#6366f1", edgecolor="black")
    ax3.bar(x + w/2, out_tok, width=w, label="Output Tokens", color="#ec4899", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_title("3. Input vs Output Token Breakdown", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Tokens", fontsize=11)
    ax3.legend()

    # 4. Latency comparison
    ax4 = axes[1, 1]
    lats = [float(r["average_latency_ms"]) / 1000.0 for r in summary_rows]
    bars4 = ax4.bar(models, lats, color=palette, edgecolor="black", width=0.5)
    ax4.set_title("4. Average Request Latency (Seconds)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Latency (Seconds)", fontsize=11)
    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}s", ha="center", va="bottom", fontweight="bold")

    # 5. Cost comparison
    ax5 = axes[2, 0]
    costs = [float(r["total_api_cost"]) for r in summary_rows]
    bars5 = ax5.bar(models, costs, color=palette, edgecolor="black", width=0.5)
    ax5.set_title("5. Cumulative API Cost (10 Incidents)", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Total Cost (USD $)", fontsize=11)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.0001, f"${h:.5f}", ha="center", va="bottom", fontweight="bold")

    # 6. Routing decision validation
    ax6 = axes[2, 1]
    suit_counts = {}
    for r in val_rows:
        s = r["routing_suitability"]
        suit_counts[s] = suit_counts.get(s, 0) + 1
    colors = {"SUITABLE": "#10b981", "SUBOPTIMAL": "#f59e0b", "UNSUITABLE": "#ef4444"}
    pie_colors = [colors.get(s, "#94a3b8") for s in suit_counts.keys()]
    ax6.pie(list(suit_counts.values()), labels=[f"{k}\n({v} queries)" for k, v in suit_counts.items()], colors=pie_colors, autopct="%1.0f%%", startangle=140, wedgeprops=dict(width=0.45, edgecolor="black"))
    ax6.set_title("6. Routing Decision Validation Suitability", fontsize=13, fontweight="bold")

    plt.suptitle("AdaRAG-Ops: Multi-LLM Model Comparison & Router Validation Dashboard", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_img = results_dir / "model_comparison_dashboard.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved dashboard to {out_img}")

    # Also save model_comparison_report.json alias
    exp_report = results_dir / "experiment_report.json"
    comp_report = results_dir / "model_comparison_report.json"
    if exp_report.exists():
        shutil.copyfile(exp_report, comp_report)
        print(f"[+] Created alias {comp_report}")
    return True

if __name__ == "__main__":
    generate_dashboard()

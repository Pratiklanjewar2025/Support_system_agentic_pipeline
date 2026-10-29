"""
Integrates the Proposed System (AdaRAG-Ops) results into the overall Model Comparison Summary
and updates the 6-panel visual dashboard to compare all 4 standalone tiers + Proposed System.
"""
import os
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/ada_rag_ops_evaluation")

def integrate_results():
    prop_json = RESULTS_DIR / "proposed_system_results.json"
    if not prop_json.exists():
        print("Proposed system results not yet found.")
        return False
        
    with open(prop_json, "r", encoding="utf-8") as f:
        prop_data = json.load(f)
        
    summary_csv = RESULTS_DIR / "model_comparison_summary.csv"
    with open(summary_csv, "r", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
        
    # Remove any existing proposed system row to avoid duplicates
    summaries = [s for s in summaries if "proposed" not in s["model_tier"].lower() and "adarag" not in s["model_tier"].lower()]
    
    n = len(prop_data)
    avg_q = round(sum(float(r["overall_quality"]) for r in prop_data) / n, 2)
    avg_f = round(sum(float(r["faithfulness"]) for r in prop_data) / n, 2)
    avg_in = round(sum(int(r["input_tokens"]) for r in prop_data) / n, 1)
    avg_out = round(sum(int(r["output_tokens"]) for r in prop_data) / n, 1)
    avg_tot = round(sum(int(r["total_tokens"]) for r in prop_data) / n, 1)
    lats = [float(r["pipeline_latency_ms"]) for r in prop_data]
    avg_lat = round(sum(lats) / n, 2)
    p95_lat = round(float(np.percentile(lats, 95)), 2)
    tot_cost = round(sum(float(r["api_cost_usd"]) for r in prop_data), 7)
    avg_cost = round(tot_cost / n, 7)
    
    prop_summary = {
        "model_tier": "AdaRAG-Ops (Proposed System)",
        "provider": "Adaptive Multi-Agent",
        "model_id": "Dynamic (T0/T1/T2/T3)",
        "successful_evaluations": n,
        "failed_evaluations": 0,
        "average_quality": avg_q,
        "average_faithfulness": avg_f,
        "average_input_tokens": avg_in,
        "average_output_tokens": avg_out,
        "average_total_tokens": avg_tot,
        "average_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "total_api_cost": tot_cost,
        "average_cost_per_query": avg_cost,
        "cost_per_quality_point": round(tot_cost / max(avg_q * n, 0.0001), 7)
    }
    summaries.append(prop_summary)
    
    # Save updated model_comparison_summary.csv
    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"[+] Updated {summary_csv} with Proposed System row.")
    except PermissionError:
        alt_csv = RESULTS_DIR / "model_comparison_summary_updated.csv"
        with open(alt_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"[!] {summary_csv} was locked by another application. Saved updated summary to {alt_csv}")

    # Generate 5-system comparative dashboard
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), dpi=300)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    
    models = [s["model_tier"].replace(" (Proposed System)", "\n[Proposed]").replace("tier0_zero_cost", "Tier 0\n(Nemotron)").replace("tier1_fast", "Tier 1\n(Groq)").replace("tier2_high_reasoning", "Tier 2\n(Gemini)").replace("tier3_heavy", "Tier 3\n(Cohere)") for s in summaries]
    palette = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    # 1. Response quality by model & proposed system
    ax1 = axes[0, 0]
    qualities = [float(r["average_quality"]) for r in summaries]
    bars1 = ax1.bar(models, qualities, color=palette[:len(models)], edgecolor="black", width=0.5)
    ax1.axhline(3.5, color="red", linestyle="--", linewidth=1.5, label="Quality Threshold (3.5)")
    ax1.set_title("1. Response Quality Comparison (0 - 5.0)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Average Quality Score", fontsize=11)
    ax1.set_ylim(0, 5.5)
    ax1.legend(loc="upper left")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    # 2. Token consumption
    ax2 = axes[0, 1]
    total_tokens = [float(r["average_total_tokens"]) for r in summaries]
    bars2 = ax2.bar(models, total_tokens, color=palette[:len(models)], edgecolor="black", width=0.5)
    ax2.set_title("2. Average Total Token Consumption per Query", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Tokens", fontsize=11)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{h:.0f}", ha="center", va="bottom", fontweight="bold")

    # 3. Input versus output tokens
    ax3 = axes[1, 0]
    in_tok = [float(r["average_input_tokens"]) for r in summaries]
    out_tok = [float(r["average_output_tokens"]) for r in summaries]
    x = np.arange(len(models))
    w = 0.35
    ax3.bar(x - w/2, in_tok, width=w, label="Input Tokens", color="#6366f1", edgecolor="black")
    ax3.bar(x + w/2, out_tok, width=w, label="Output Tokens", color="#ec4899", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9)
    ax3.set_title("3. Input vs Output Token Breakdown", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Tokens", fontsize=11)
    ax3.legend()

    # 4. Latency comparison
    ax4 = axes[1, 1]
    lats_sec = [float(r["average_latency_ms"]) / 1000.0 for r in summaries]
    bars4 = ax4.bar(models, lats_sec, color=palette[:len(models)], edgecolor="black", width=0.5)
    ax4.set_title("4. Average Request / Pipeline Latency (Seconds)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Latency (Seconds)", fontsize=11)
    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}s", ha="center", va="bottom", fontweight="bold")

    # 5. Cost comparison
    ax5 = axes[2, 0]
    costs = [float(r["total_api_cost"]) for r in summaries]
    bars5 = ax5.bar(models, costs, color=palette[:len(models)], edgecolor="black", width=0.5)
    ax5.set_title("5. Cumulative API Cost (10 Incidents)", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Total Cost (USD $)", fontsize=11)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.0003, f"${h:.5f}", ha="center", va="bottom", fontweight="bold")

    # 6. Verification & Routing Decision Suitability
    ax6 = axes[2, 1]
    val_csv = RESULTS_DIR / "routing_decision_validation.csv"
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    suit_counts = {}
    for r in val_rows:
        s = r["routing_suitability"]
        suit_counts[s] = suit_counts.get(s, 0) + 1
    colors = {"SUITABLE": "#10b981", "SUBOPTIMAL": "#f59e0b", "UNSUITABLE": "#ef4444"}
    pie_colors = [colors.get(s, "#94a3b8") for s in suit_counts.keys()]
    ax6.pie(list(suit_counts.values()), labels=[f"{k}\n({v} queries)" for k, v in suit_counts.items()], colors=pie_colors, autopct="%1.0f%%", startangle=140, wedgeprops=dict(width=0.45, edgecolor="black"))
    ax6.set_title("6. Router Decision Suitability Breakdown", fontsize=13, fontweight="bold")

    plt.suptitle("AdaRAG-Ops: Multi-LLM Tier Comparison vs Proposed Multi-Agent System", fontsize=17, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_img = RESULTS_DIR / "model_comparison_dashboard.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved updated 5-model comparative dashboard to {out_img}")

if __name__ == "__main__":
    integrate_results()

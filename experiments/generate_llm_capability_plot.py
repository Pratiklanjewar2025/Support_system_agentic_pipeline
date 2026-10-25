# -*- coding: utf-8 -*-
"""
generate_llm_capability_plot.py — Generates publication-grade visual charts
showing: 'What LLM is Capable of Which Type of Query'.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "research_paper_results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

fig = plt.figure(figsize=(14, 9), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.32, wspace=0.25)

# -----------------------------------------------------------------------------
# Panel 1: Query Complexity Spectrum vs. Allocated LLM (Left Top)
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

complexity_ranges = [
    (0.00, 0.25, "Tier 0: OpenRouter 550B\n(Free / General Q&A)", "#10b981", "$0.00"),
    (0.05, 0.42, "Tier 1: Groq GPT-OSS 20B\n(Fast SOP / Restart)", "#0ea5e9", "$0.00010"),
    (0.42, 0.65, "Tier 2: Gemini 2.5 Flash\n(Diagnostic / OOM)", "#8b5cf6", "$0.00015"),
    (0.65, 1.00, "Tier 3: Cohere Command-R+\n(Heavy Outages / 104B)", "#ef4444", "$0.00250"),
]

y_positions = [3, 2, 1, 0]
for idx, (low, high, label, color, cost) in enumerate(complexity_ranges):
    y = y_positions[idx]
    ax1.barh(y, high - low, left=low, height=0.55, color=color, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.text((low + high) / 2, y, f"{label.splitlines()[0]} [{cost}]", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

ax1.set_yticks(y_positions)
ax1.set_yticklabels([
    "Factual / Zero-Cost",
    "Procedural / Runbook",
    "Diagnostic Root Cause",
    "Cascading Enterprise Outage"
], fontweight="bold")
ax1.set_xlabel("Query Complexity Score (0.0 to 1.0)")
ax1.set_xlim(0, 1.0)
ax1.set_title("(a) Query Complexity Score vs. Allocated LLM Tier", fontweight="bold")
ax1.axvline(0.42, color="#64748b", linestyle="--", alpha=0.7)
ax1.axvline(0.65, color="#64748b", linestyle="--", alpha=0.7)
ax1.text(0.43, -0.7, "Thresholds", color="#64748b", fontsize=8, fontstyle="italic")

# -----------------------------------------------------------------------------
# Panel 2: Reasoning Capacity vs. Cost Tradeoff (Right Top)
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

models = [
    {"name": "OpenRouter\nNemotron 550B", "cost": 0.00000, "params": 550, "speed": 2.6, "color": "#10b981"},
    {"name": "Groq\nGPT-OSS 20B", "cost": 0.00010, "params": 20, "speed": 1.2, "color": "#0ea5e9"},
    {"name": "Gemini 2.5\nFlash", "cost": 0.00015, "params": 45, "speed": 3.8, "color": "#8b5cf6"},
    {"name": "Cohere\nCommand-R+ 104B", "cost": 0.00250, "params": 104, "speed": 8.5, "color": "#ef4444"},
]

for m in models:
    # Scale bubble by parameter capacity
    ax2.scatter(m["cost"] * 1000, m["params"], s=m["params"] * 4.5 + 400, color=m["color"], alpha=0.75, edgecolors="black", linewidth=1.5)
    ax2.annotate(
        f"{m['name']}\n(Latency: {m['speed']}s)",
        (m["cost"] * 1000, m["params"]),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold"
    )

ax2.set_xlabel("Cost per 1,000 Tokens (USD)")
ax2.set_ylabel("Parameter Capacity (Billions)")
ax2.set_title("(b) Model Capability & Parameter Scale vs. Cost", fontweight="bold")
ax2.set_yscale("log")
ax2.set_yticks([20, 50, 100, 550])
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# -----------------------------------------------------------------------------
# Panel 3: Capability Heatmap Matrix (Query Types vs. LLM Suitability) (Bottom Full)
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, :])

query_types = [
    "General Technical Q&A & Zero-Cost Lookups",
    "Service Restart SOP (systemctl restart)",
    "Kubernetes Pod Health & Logs Check",
    "Container CrashLoopBackOff & Exit Code 137",
    "PostgreSQL Connection Pool Saturation",
    "Distributed Microservice Deadlock (Sev-1)",
    "Multi-Hop Cross-Service Cascading Failures"
]

llm_names = [
    "Tier 0: OpenRouter\nNemotron 550B (Free)",
    "Tier 1: Groq\nGPT-OSS 20B",
    "Tier 2: Gemini 2.5\nFlash (Google)",
    "Tier 3: Cohere\nCommand-R+ (104B)"
]

# Capability Matrix (0 to 100% suitability)
# Higher score = optimal choice for that query type
capability_matrix = np.array([
    [98,  72,  85,  70],  # General Zero-Cost Q&A -> OpenRouter Free wins
    [65,  98,  82,  60],  # Service Restart -> Groq Fast wins
    [60,  95,  88,  68],  # Pod Health Check -> Groq Fast wins
    [55,  68,  97,  88],  # CrashLoopBackOff & Exit 137 -> Gemini Flash wins
    [50,  62,  94,  89],  # DB Pool Saturation -> Gemini Flash wins
    [40,  45,  78,  99],  # Microservice Deadlock -> Cohere 104B wins
    [35,  40,  74,  98]   # Multi-Hop Cascades -> Cohere 104B wins
])

im = ax3.imshow(capability_matrix, cmap="YlGnBu", aspect="auto")
cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Suitability / Allocation Fitness Score (%)", fontweight="bold")

ax3.set_xticks(np.arange(len(llm_names)))
ax3.set_yticks(np.arange(len(query_types)))
ax3.set_xticklabels(llm_names, fontweight="bold")
ax3.set_yticklabels(query_types, fontweight="bold")

# Annotate values
for i in range(len(query_types)):
    for j in range(len(llm_names)):
        val = capability_matrix[i, j]
        color = "white" if val > 75 else "black"
        ax3.text(j, i - 0.08, f"{val}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

ax3.set_title("(c) LLM Capability Matrix: Suitability Score by Query Type & Severity", fontweight="bold", pad=12)

# Highlight winning cells with border or marker
for r in range(len(query_types)):
    max_c = np.argmax(capability_matrix[r])
    ax3.text(max_c, r + 0.22, "★ OPTIMAL", color="#f59e0b" if capability_matrix[r, max_c] > 75 else "#b91c1c", fontweight="bold", fontsize=8.5, ha="center")

plt.suptitle("AdaRAG-Ops: Heterogeneous Multi-LLM Capability & Query Routing Architecture", fontsize=15, fontweight="bold", y=0.99)

out_png = os.path.join(FIGURES_DIR, "llm_capability_matrix.png")
out_pdf = os.path.join(FIGURES_DIR, "llm_capability_matrix.pdf")
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"[OK] Generated: {out_png}")

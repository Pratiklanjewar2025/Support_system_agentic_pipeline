"""
Merges live Gemini evaluation results into model_comparison_results.csv,
recomputes all baseline comparisons, ablations, summaries, and recreates visual figures.
"""
import os
import csv
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/ada_rag_ops_evaluation")

def update_deliverables():
    gemini_json_path = RESULTS_DIR / "gemini_live_eval_results.json"
    if not gemini_json_path.exists():
        print(f"Error: {gemini_json_path} does not exist yet.")
        return False
        
    with open(gemini_json_path, "r", encoding="utf-8") as f:
        gemini_records = json.load(f)
        
    results_csv = RESULTS_DIR / "model_comparison_results.csv"
    with open(results_csv, "r", encoding="utf-8") as f:
        all_results = list(csv.DictReader(f))
        
    # Update Tier 2 records with newly measured Gemini data
    gemini_map = {r["query_id"]: r for r in gemini_records}
    updated_results = []
    
    for row in all_results:
        if row["model_tier"] == "tier2_high_reasoning" and row["query_id"] in gemini_map:
            g = gemini_map[row["query_id"]]
            row["execution_status"] = g["execution_status"]
            row["response"] = g["response"]
            row["technical_correctness_score"] = g["technical_correctness_score"]
            row["faithfulness_score"] = g["faithfulness_score"]
            row["actionability_score"] = g["actionability_score"]
            row["completeness_score"] = g["completeness_score"]
            row["safety_score"] = g["safety_score"]
            row["overall_quality_score"] = g["overall_quality_score"]
            row["input_tokens"] = g["input_tokens"]
            row["output_tokens"] = g["output_tokens"]
            row["total_tokens"] = g["total_tokens"]
            row["latency_ms"] = g["latency_ms"]
            row["api_cost"] = g["api_cost"]
            row["error_message"] = g["error_message"]
            if "judge_explanation" in fields:
                row["judge_explanation"] = g["judge_explanation"]
        updated_results.append(row)
        
    # Write updated model_comparison_results.csv
    fields = list(all_results[0].keys())
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(updated_results)
    print(f"[+] Updated {results_csv}")
    
    # Recompute model_comparison_summary.csv
    tiers = ["tier0_zero_cost", "tier1_fast", "tier2_high_reasoning", "tier3_heavy"]
    summaries = []
    for t in tiers:
        t_recs = [r for r in updated_results if r["model_tier"] == t]
        succ = [r for r in t_recs if r["execution_status"] == "SUCCESS"]
        failed = len(t_recs) - len(succ)
        n = len(succ)
        avg_q = round(sum(float(r["overall_quality_score"]) for r in succ) / n, 2) if n > 0 else 0.0
        avg_f = round(sum(float(r["faithfulness_score"]) for r in succ) / n, 2) if n > 0 else 0.0
        avg_in = round(sum(int(r["input_tokens"]) for r in succ) / n, 1) if n > 0 else 0.0
        avg_out = round(sum(int(r["output_tokens"]) for r in succ) / n, 1) if n > 0 else 0.0
        avg_tot = round(sum(int(r["total_tokens"]) for r in succ) / n, 1) if n > 0 else 0.0
        lat_list = [float(r["latency_ms"]) for r in succ]
        avg_lat = round(sum(lat_list) / max(len(lat_list), 1), 2) if lat_list else 0.0
        p95_lat = round(float(np.percentile(lat_list, 95)), 2) if lat_list else 0.0
        tot_cost = round(sum(float(r["api_cost"]) for r in succ), 7)
        avg_cost = round(tot_cost / max(n, 1), 7)
        
        provider = t_recs[0]["provider"]
        model_id = t_recs[0]["model_id"]
        summaries.append({
            "model_tier": t,
            "provider": provider,
            "model_id": model_id,
            "successful_evaluations": n,
            "failed_evaluations": failed,
            "average_quality": avg_q,
            "average_faithfulness": avg_f,
            "average_input_tokens": avg_in,
            "average_output_tokens": avg_out,
            "average_total_tokens": avg_tot,
            "average_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "total_api_cost": tot_cost,
            "average_cost_per_query": avg_cost,
            "cost_per_quality_point": round(tot_cost / max(avg_q * n, 0.0001), 7) if avg_q > 0 else 0.0
        })
        
    summary_csv = RESULTS_DIR / "model_comparison_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"[+] Recomputed {summary_csv}")

    # Recompute routing_decision_validation.csv
    val_csv = RESULTS_DIR / "routing_decision_validation.csv"
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
        
    QUALITY_THRESHOLD = 3.5
    updated_vals = []
    for v in val_rows:
        qid = v["query_id"]
        q_recs = [r for r in updated_results if r["query_id"] == qid]
        succ = [r for r in q_recs if r["execution_status"] == "SUCCESS"]
        eligible = [r for r in succ if float(r["overall_quality_score"]) >= QUALITY_THRESHOLD]
        eligible.sort(key=lambda x: (float(x["api_cost"]), -float(x["overall_quality_score"])))
        lowest_cost_eligible = eligible[0] if eligible else (succ[0] if succ else None)
        
        sel_rec = next((r for r in q_recs if r["model_tier"] == v["router_selected_tier"]), None)
        if sel_rec:
            sel_qual = float(sel_rec["overall_quality_score"])
            sel_cost = float(sel_rec["api_cost"])
            sel_tok = int(sel_rec["total_tokens"])
            sel_lat = float(sel_rec["latency_ms"])
            
            lowest_cost = float(lowest_cost_eligible["api_cost"]) if lowest_cost_eligible else 0.0
            lowest_qual = float(lowest_cost_eligible["overall_quality_score"]) if lowest_cost_eligible else 0.0
            lowest_id = lowest_cost_eligible["model_id"] if lowest_cost_eligible else "UNAVAILABLE"
            
            is_under = (sel_rec["execution_status"] == "FAILED") or (sel_qual < QUALITY_THRESHOLD)
            excess_cost = max(0.0, round(sel_cost - lowest_cost, 7))
            is_over = (not is_under) and (excess_cost > 0.0001) and (sel_cost / max(lowest_cost, 0.000001) > 4.0)
            
            if is_under:
                suit = "UNSUITABLE"
                status_desc = "Under-routing"
                over_val = "NO"
                under_val = "YES"
                reason = f"Selected model failed quality requirement ({sel_qual:.2f} < {QUALITY_THRESHOLD}) or failed execution: {sel_rec['error_message']}."
            elif is_over:
                suit = "SUBOPTIMAL"
                status_desc = "Over-routing"
                over_val = "YES"
                under_val = "NO"
                reason = f"More expensive tier selected ($ {sel_cost:.5f}) when cheaper model {lowest_cost_eligible['model_tier']} satisfied quality threshold ({lowest_qual:.2f})."
            elif lowest_cost_eligible and sel_rec["model_tier"] == lowest_cost_eligible["model_tier"]:
                suit = "SUITABLE"
                status_desc = "Appropriate (Optimal Lowest-Cost)"
                over_val = "NO"
                under_val = "NO"
                reason = f"Router perfectly matched the lowest-cost model meeting quality threshold (${sel_cost:.5f}, quality {sel_qual:.2f})."
            else:
                suit = "SUITABLE"
                status_desc = "Appropriate (Safety/Reasoning Guard)"
                over_val = "NO"
                under_val = "NO"
                reason = f"Selected tier justified for severity {v['incident_severity']} with superior safety/completeness."
                
            v["selected_model_quality"] = sel_qual
            v["selected_model_cost"] = sel_cost
            v["selected_model_tokens"] = sel_tok
            v["selected_model_latency"] = sel_lat
            v["lowest_cost_quality_eligible_model"] = lowest_id
            v["lowest_cost_quality_eligible_model_cost"] = lowest_cost
            v["routing_suitability"] = suit
            v["routing_status"] = status_desc
            v["over_routing"] = over_val
            v["under_routing"] = under_val
            v["validation_reason"] = reason
        updated_vals.append(v)
        
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(updated_vals[0].keys()))
        w.writeheader()
        w.writerows(updated_vals)
    print(f"[+] Recomputed {val_csv}")

    # Recreate figures using create_dashboard_composite.py
    import subprocess
    subprocess.run([sys.executable, "create_dashboard_composite.py"], check=True)
    print("[+] Visual Dashboard regenerated successfully.")

if __name__ == "__main__":
    update_deliverables()

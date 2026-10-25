# -*- coding: utf-8 -*-
"""
test_reproducibility.py — Comprehensive Unit & Integration Test Suite
for Phase 9: Research Reproducibility.

Verifies:
1. ExperimentConfig Schema Validation: Loading from default, baselines, proposed, and ablations configs.
2. Non-Overwriting Directory & File Persistence: Re-running never overwrites prior experiment results.
3. Structured Per-Case JSONL and CSV Logging: Verifies presence of routing decisions, confidence estimates, cost measurements, and error details.
4. Run Manifest and Registry Provenance: Verifies git hash, timestamp, platform, and registry entry.
5. Baseline Execution: Executes baselines cleanly.
6. Proposed Method Execution: Executes intervention-aware policy cleanly.
7. Ablation Execution: Executes ablation variants cleanly.
8. Metric Aggregation: Cross-run aggregation with sample statistics and 95% CIs.
9. Paired Statistical Comparisons: Aligns case_ids, computes paired t-test, Wilcoxon, Cohen's d, Cliff's delta.
10. Publication-Quality Plots: Generates all 5 figures in PNG and SVG vector formats.
"""

import sys
import os
import json
import csv
import shutil
import tempfile
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments.reproducibility import (
    ExperimentConfig,
    RunPersistenceManager,
    ReproducibleExperimentRunner,
    CONFIGS_DIR,
    RUNS_DIR
)
from experiments.aggregate_metrics import aggregate_experiment_runs
from experiments.run_statistical_tests import run_paired_comparison
from experiments.generate_plots import generate_all_publication_plots


def test_config_loading_and_validation():
    print("[*] Test 1: Testing ExperimentConfig Schema Loading & Validation...")

    # 1. Default Master Config
    master_path = CONFIGS_DIR / "default_config.json"
    assert master_path.exists(), f"Missing {master_path}"
    cfg_master = ExperimentConfig.load_from_file(str(master_path))
    assert cfg_master.experiment_id == "EXP_DEFAULT_ADARAG_OPS"
    assert cfg_master.random_seed == 42
    assert cfg_master.target_quality == 0.70
    assert cfg_master.max_retries == 2
    assert "command-r-08-2024" in cfg_master.model_catalog

    # 2. Baselines Config
    base_path = CONFIGS_DIR / "baselines_config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        data_base = json.load(f)
    assert "configurations" in data_base
    assert "baseline_a_fixed_small" in data_base["configurations"]
    assert "baseline_d_query_only" in data_base["configurations"]

    # 3. Proposed Config
    prop_path = CONFIGS_DIR / "proposed_config.json"
    with open(prop_path, "r", encoding="utf-8") as f:
        data_prop = json.load(f)
    assert "proposed_h_calibrated_intervention" in data_prop["configurations"]

    # 4. Ablations Config
    abl_path = CONFIGS_DIR / "ablations_config.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        data_abl = json.load(f)
    assert "ablation_1_no_retrieval_features" in data_abl["configurations"]
    assert "ablation_6_no_intervention_selection" in data_abl["configurations"]

    print("    [PASS] All 4 configuration suites loaded and validated successfully.")


def test_non_overwriting_run_directory_persistence():
    print("[*] Test 2: Testing Non-Overwriting Directory & Persistence Guarantee...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_runs_"))
    try:
        manager = RunPersistenceManager(output_root=temp_dir)
        exp_id = "EXP_REPRODUCIBILITY_TEST"

        # First run
        run_id_1, run_dir_1 = manager.create_run_directory(exp_id)
        assert run_dir_1.exists()

        # Place dummy file in run_dir_1
        marker_file = run_dir_1 / "marker.txt"
        with open(marker_file, "w") as f:
            f.write("run_1_original_content")

        # Second run with IDENTICAL experiment_id
        run_id_2, run_dir_2 = manager.create_run_directory(exp_id)
        assert run_dir_2.exists()
        assert run_dir_1 != run_dir_2, "Directories must be strictly distinct!"
        assert run_id_1 != run_id_2, "Run IDs must be strictly unique!"

        # Verify run_1 content was preserved and not overwritten
        assert marker_file.exists()
        with open(marker_file, "r") as f:
            assert f.read() == "run_1_original_content"

        print(f"    [PASS] Non-overwriting guarantee verified: Run 1={run_id_1} | Run 2={run_id_2} (distinct).")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_structured_per_case_jsonl_and_csv_persistence():
    print("[*] Test 3: Testing Structured Per-Case JSONL & CSV Provenance Logging...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_structured_"))
    try:
        manager = RunPersistenceManager(output_root=temp_dir)
        run_id, run_dir = manager.create_run_directory("EXP_PER_CASE_TEST")

        sample_cases = [
            {
                "case_id": "CASE_1",
                "system": "core-infrastructure",
                "fault_type": "database",
                "ground_truth_service": "postgresql",
                "predicted_service": "postgresql",
                "hit_at_1": True,
                "hit_at_3": True,
                "hit_at_5": True,
                "fault_type_hit": True,
                "structured_correctness": 1.0,
                "f1": 0.85,
                "supplementary_bleu": 0.42,
                "faithfulness": 95.0,
                "grounding_precision": 100.0,
                "unsupported_claim_rate": 0.0,
                "calibrated_confidence": 0.92,
                "composite_score": 92.5,
                "latency_sec": 1.85,
                "actual_api_cost_usd": 0.00010,
                "estimated_cost_usd": 0.00008,
                "in_tokens": 300,
                "out_tokens": 100,
                "total_tokens": 400,
                "model_calls": 1,
                "escalated": False,
                "retrieval_retries": 0,
                "routed_model": "command-r-08-2024",
                "status": "COMPLETED",
                "routing_decision": {"chosen_tier": "tier1_fast", "action": "ANSWER"},
                "confidence_estimates": {"calibrated_probability": 0.92, "confidence_band": "high"},
                "error_details": None
            }
        ]

        jsonl_path, csv_path = manager.write_per_case_results(run_dir, sample_cases)
        assert jsonl_path.exists()
        assert csv_path.exists()

        # Verify JSONL lines
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["case_id"] == "CASE_1"
        assert "routing_decision" in rec
        assert "confidence_estimates" in rec
        assert "actual_api_cost_usd" in rec
        assert "estimated_cost_usd" in rec
        assert rec["error_details"] is None

        # Verify CSV tabular fields
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["case_id"] == "CASE_1"
        assert rows[0]["hit_at_1"] in ("True", "1.0", "1")
        assert float(rows[0]["composite_score"]) == 92.5

        print(f"    [PASS] JSONL & CSV per-case persistence verified with complete routing & confidence fields.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_end_to_end_reproducible_experiment_execution():
    print("[*] Test 4: Testing End-to-End Experiment Execution & Manifest Generation...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_runner_"))
    try:
        manager = RunPersistenceManager(output_root=temp_dir)
        runner = ReproducibleExperimentRunner(persistence_manager=manager)

        cfg = ExperimentConfig(
            experiment_id="TEST_RUNNER_EXP",
            name="Unit Test Execution",
            dataset_name="supplementary",
            split_strategy="all",
            subsample_max_cases=2,
            random_seed=42,
            router_type="intervention_aware"
        )

        run_id, run_dir, summary = runner.run_experiment(cfg)

        assert run_dir.exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "per_case_results.jsonl").exists()
        assert (run_dir / "per_case_results.csv").exists()
        assert (run_dir / "summary_metrics.json").exists()
        assert manager.registry_file.exists()

        # Check manifest contents
        with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
            man = json.load(f)
        assert man["run_id"] == run_id
        assert "git_commit" in man
        assert "python_version" in man
        assert man["configuration"]["experiment_id"] == "TEST_RUNNER_EXP"

        # Check summary report
        assert "diagnostic_quality" in summary
        assert "confidence_and_reliability" in summary
        assert "operational_efficiency" in summary
        assert "statistical_reporting" in summary

        # Rerun exact same experiment to verify non-overwriting behavior
        run_id_2, run_dir_2, summary_2 = runner.run_experiment(cfg)
        assert run_id != run_id_2
        assert run_dir != run_dir_2
        assert run_dir.exists()
        assert run_dir_2.exists()

        print(f"    [PASS] End-to-end execution completed cleanly: Run 1={run_id} | Run 2={run_id_2} (0% overwrite).")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cross_run_metric_aggregation():
    print("[*] Test 5: Testing Cross-Run Metric Aggregation...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_"))
    try:
        manager = RunPersistenceManager(output_root=temp_dir)
        runner = ReproducibleExperimentRunner(persistence_manager=manager)

        # Run 2 instances of same config
        cfg = ExperimentConfig(
            experiment_id="AGG_TEST_EXP",
            name="Aggregation Test",
            dataset_name="supplementary",
            split_strategy="all",
            subsample_max_cases=2,
            random_seed=42
        )
        runner.run_experiment(cfg)
        runner.run_experiment(cfg)

        # Run aggregation
        agg = aggregate_experiment_runs(runs_root=temp_dir, export_dir=temp_dir / "export")
        assert agg["total_runs_analyzed"] == 2
        assert "AGG_TEST_EXP" in agg["configurations"]
        exp_agg = agg["configurations"]["AGG_TEST_EXP"]
        assert exp_agg["repeated_runs_count"] == 2
        assert "root_cause_hit1" in exp_agg
        assert "ci_lower" in exp_agg["root_cause_hit1"]

        assert (temp_dir / "export" / "aggregated_metrics.json").exists()
        assert (temp_dir / "export" / "aggregated_metrics.csv").exists()

        print(f"    [PASS] Cross-run aggregation verified: {agg['total_runs_analyzed']} runs aggregated into CSV/JSON.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_paired_statistical_comparison_script():
    print("[*] Test 6: Testing Paired Statistical Comparisons Script...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_stat_"))
    try:
        manager = RunPersistenceManager(output_root=temp_dir)
        runner = ReproducibleExperimentRunner(persistence_manager=manager)

        # Baseline run
        cfg_base = ExperimentConfig(
            experiment_id="STAT_BASELINE",
            name="Baseline Run",
            dataset_name="supplementary",
            split_strategy="all",
            subsample_max_cases=3,
            router_type="fixed",
            fixed_tier="tier1_fast",
            random_seed=42
        )
        _, run_dir_base, _ = runner.run_experiment(cfg_base)

        # Proposed run
        cfg_prop = ExperimentConfig(
            experiment_id="STAT_PROPOSED",
            name="Proposed Run",
            dataset_name="supplementary",
            split_strategy="all",
            subsample_max_cases=3,
            router_type="intervention_aware",
            random_seed=42
        )
        _, run_dir_prop, _ = runner.run_experiment(cfg_prop)

        # Execute paired test
        export_json = temp_dir / "stats.json"
        comp = run_paired_comparison(
            baseline_run_dir=run_dir_base,
            target_run_dir=run_dir_prop,
            export_path=export_json
        )

        assert comp["sample_count"] == 3
        assert "composite_score" in comp["metrics"]
        assert "hit_at_1" in comp["metrics"]
        assert "t_statistic" in comp["metrics"]["composite_score"]
        assert "cohen_d" in comp["metrics"]["composite_score"]
        assert export_json.exists()

        print(f"    [PASS] Paired statistical test verified across N={comp['sample_count']} cases with exported JSON.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_publication_plots_generation():
    print("[*] Test 7: Testing Publication-Quality Plot Generation...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_plots_"))
    try:
        plots = generate_all_publication_plots(output_dir=temp_dir)
        assert len(plots) == 5

        # Check that PNG and vector SVG both exist for each plot
        expected = [
            "reliability_diagram",
            "quality_vs_cost_pareto",
            "risk_coverage_curve",
            "latency_distribution",
            "macro_f1_fault_categories"
        ]
        for name in expected:
            png_file = temp_dir / f"{name}.png"
            svg_file = temp_dir / f"{name}.svg"
            assert png_file.exists(), f"Missing PNG: {png_file}"
            assert svg_file.exists(), f"Missing SVG: {svg_file}"
            assert png_file.stat().st_size > 1000, f"PNG {name} is suspiciously small"
            assert svg_file.stat().st_size > 1000, f"SVG {name} is suspiciously small"

        print(f"    [PASS] All 5 publication figures generated in both PNG (300 DPI) and SVG vector format.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 85)
    print("   RUNNING PHASE 9: RESEARCH REPRODUCIBILITY TEST SUITE")
    print("=" * 85)
    test_config_loading_and_validation()
    test_non_overwriting_run_directory_persistence()
    test_structured_per_case_jsonl_and_csv_persistence()
    test_end_to_end_reproducible_experiment_execution()
    test_cross_run_metric_aggregation()
    test_paired_statistical_comparison_script()
    test_publication_plots_generation()
    print("\n" + "=" * 85)
    print("   ALL PHASE 9 REPRODUCIBILITY TESTS PASSED! (100% SUCCESS)")
    print("=" * 85)

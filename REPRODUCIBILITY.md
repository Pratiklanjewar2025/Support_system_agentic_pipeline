# 🔬 Phase 9: Research Reproducibility Framework

> **AdaRAG-Ops: Research-Grade Operational Incident Diagnosis**  
> Comprehensive guide to experiment configuration, non-overwriting provenance tracking, execution scripts, statistical testing, and publication-quality visualization.

---

## 1. Overview & Reproducibility Principles

Scientific rigor in autonomous microservice diagnosis requires total reproducibility:
1. **Explicit Configurations**: Every experiment is parameterized by a machine-readable JSON configuration specifying dataset splits, random seeds, exact model versions, routing feature schemas, calibration models, budget constraints, and evaluation metrics.
2. **Non-Overwriting Provenance**: Rerunning an experiment **never** silently overwrites previous runs. Every run creates an isolated, immutable timestamped directory containing the exact environment manifest, git commit hash, per-case JSONL and CSV logs, and summary reports.
3. **Anti-Fabrication Guarantee**: Direct provider-billed costs are strictly distinguished from calculated estimated costs based on published token prices. Unavailable models or network endpoints are recorded explicitly as `UNAVAILABLE` rather than generating fabricated outputs.
4. **End-to-End Automated Tooling**: Dedicated, standalone CLI scripts automate baseline execution, proposed method execution, ablation studies, cross-run metric aggregation, paired statistical significance testing, and vector graphic generation.

---

## 2. Experiment Configuration Schema

Experiment configurations are located in `experiments/configs/`:
- [`default_config.json`](file:///c:/Users/asus/Downloads/Technical_agent/experiments/configs/default_config.json): Master reference template.
- [`baselines_config.json`](file:///c:/Users/asus/Downloads/Technical_agent/experiments/configs/baselines_config.json): Baselines A through D.
- [`proposed_config.json`](file:///c:/Users/asus/Downloads/Technical_agent/experiments/configs/proposed_config.json): Proposed methods E through H.
- [`ablations_config.json`](file:///c:/Users/asus/Downloads/Technical_agent/experiments/configs/ablations_config.json): Systematic ablations 1 through 6.

### Schema Fields

| Block | Field | Type | Description |
|---|---|---|---|
| **Identity** | `experiment_id` | String | Unique uppercase experiment tag (e.g. `PROP_H_CALIBRATED_INTERVENTION`). |
| | `name` | String | Human-readable configuration name. |
| | `description` | String | Scientific rationale or variant explanation. |
| **Dataset** | `dataset_name` | String | `all`, `rcaeval`, `openrca`, or `supplementary`. |
| | `split_strategy` | String | `grouped`, `cross_system`, `cross_fault`, `test`, `val`, `all`. |
| | `unseen_system` | String | Target unseen microservice system for cross-system generalization (e.g. `train-ticket`). |
| | `unseen_fault_types` | List | Target unseen fault categories for cross-fault generalization. |
| **Seeds** | `dataset_seed` | Integer | Fixed pseudo-random seed for data splitting (default: `42`). |
| | `model_router_seed` | Integer | Fixed pseudo-random seed for router decisions (default: `42`). |
| | `calibration_seed` | Integer | Fixed pseudo-random seed for probability calibration (default: `42`). |
| **Models** | `model_catalog` | Dict | Published token pricing rates, model versions, and assumptions. |
| **Router** | `router_type` | String | `fixed`, `query_only`, `heuristic`, `learned_rf`, `intervention_aware`. |
| | `feature_schema_path` | String | Path to 14-dimensional calibration feature schema (`artifacts/calibration/feature_schema.json`). |
| | `enable_retrieval_confidence_features` | Boolean | Toggles telemetry retrieval similarity features ($R_c$). |
| **Calibration** | `calibrator_type` | String | `isotonic` or `platt_scaling`. |
| | `calibrator_model_path` | String | Path to serialized isotonic regressor (`artifacts/calibration/calibrated_estimator.joblib`). |
| | `n_bins` | Integer | Number of calibration bins for ECE/MCE (default: `5`). |
| | `default_confidence_threshold` | Float | Confidence threshold $\tau$ for automated answer vs. human escalation (default: `0.50`). |
| | `abstention_enabled` | Boolean | Enables selective classification and human review. |
| **Constraints** | `target_quality` | Float | Required diagnosis quality threshold (e.g. `0.70`). |
| | `max_latency_budget_sec` | Float | Maximum allowed wall-clock latency per incident (e.g. `15.0s`). |
| | `max_cost_budget_usd` | Float | Maximum allowed inference cost per incident (e.g. `$0.005`). |
| | `max_risk_threshold` | Float | Risk boundary for action selection. |
| | `max_retries` | Integer | Maximum retrieval retry attempts (default: `2`). |
| | `max_escalations` | Integer | Maximum escalation hops to heavier tiers (default: `1`). |
| | `allow_escalation` | Boolean | Enables adaptive escalation to heavier models. |
| **Evaluation** | `confidence_level` | Float | Two-tailed Student's $t$ confidence level (default: `0.95`). |
| | `output_dir` | String | Root destination directory for run artifacts (default: `experiments/runs`). |

---

## 3. Non-Overwriting Directory & Artifact Hierarchy

When any experiment is executed, `RunPersistenceManager` creates an immutable, timestamped run directory:

```
experiments/runs/
├── experiment_registry.jsonl              <-- Append-only master historical run catalog
└── <EXP_ID>_<YYYYMMDD_HHMMSS>_<UUID6>/    <-- Isolated, non-overwriting run directory
    ├── run_manifest.json                  <-- Provenance: git commit, seeds, python/platform, config snapshot
    ├── per_case_results.jsonl             <-- Line-by-line detailed JSONL per-case diagnosis records
    ├── per_case_results.csv               <-- Flattened tabular CSV representation for rapid statistics
    └── summary_metrics.json               <-- Phase 8 comprehensive multi-dimensional evaluation report
```

### 3.1 Per-Case JSONL Record Schema
Each line in `per_case_results.jsonl` contains:
```json
{
  "case_id": "RE1_frontend_CPU_1",
  "system": "online-boutique",
  "fault_type": "CPU",
  "ground_truth_service": "frontend",
  "predicted_service": "frontend",
  "predicted_fault": "CPU",
  "status": "COMPLETED",
  "error_details": null,
  "routing_decision": {
    "chosen_tier": "tier1_fast",
    "model_id": "command-r-08-2024",
    "action": "ANSWER",
    "reasoning": "High retrieval confidence (Rc=0.88) meets quality threshold."
  },
  "confidence_estimates": {
    "calibrated_probability": 0.88,
    "confidence_band": "high",
    "abstention_recommended": false
  },
  "hit_at_1": true,
  "hit_at_3": true,
  "hit_at_5": true,
  "structured_correctness": 1.0,
  "grounding_precision": 100.0,
  "unsupported_claim_rate": 0.0,
  "f1": 0.824,
  "supplementary_bleu": 0.312,
  "faithfulness": 92.5,
  "composite_score": 91.2,
  "latency_sec": 1.94,
  "actual_api_cost_usd": 0.00005,
  "estimated_cost_usd": 0.000045,
  "in_tokens": 310,
  "out_tokens": 85,
  "total_tokens": 395,
  "model_calls": 1,
  "escalated": false,
  "retrieval_retries": 0,
  "routed_model": "command-r-08-2024",
  "structured_diagnosis": { ... }
}
```

---

## 4. Execution Scripts & CLI Usage

All scripts can be invoked directly from `experiments/` or through top-level wrappers in `scripts/`.

### 4.1 Running Baselines
Executes Baselines A, B, C, and D across identical incident cases:
```powershell
python scripts/run_baselines.py
# Or with options:
python scripts/run_baselines.py --config experiments/configs/baselines_config.json --max-cases 10
```

### 4.2 Running Proposed Methods
Executes Proposed configurations E, F, G, and H (including calibrated intervention-aware policy):
```powershell
python scripts/run_proposed.py
# Or with options:
python scripts/run_proposed.py --config experiments/configs/proposed_config.json --max-cases 10
```

### 4.3 Running Systematic Ablations
Executes Ablation Experiments 1 through 6:
```powershell
python scripts/run_ablations.py
# Or with options:
python scripts/run_ablations.py --config experiments/configs/ablations_config.json --max-cases 10
```

### 4.4 Metric Aggregation Across Repeated Runs
Aggregates all completed runs in `experiments/runs/`, calculates sample mean, sample standard deviation, variance, standard error, and 95% Confidence Intervals (Student's $t$):
```powershell
python scripts/aggregate_metrics.py --runs-dir experiments/runs --export-dir artifacts/
```
Outputs:
- Console markdown comparison table.
- `artifacts/aggregated_metrics.json`
- `artifacts/aggregated_metrics.csv`

### 4.5 Paired Statistical Comparisons
Alings paired incident cases between two experiment runs and computes paired Student's $t$-test, Wilcoxon signed-rank test, Cohen's $d$, Cliff's $\delta$, and statistical significance flags ($p < 0.05$):
```powershell
python scripts/run_statistical_tests.py --baseline experiments/runs/<BASELINE_RUN_DIR> --target experiments/runs/<PROPOSED_RUN_DIR> --export artifacts/statistical_comparisons.json
```

### 4.6 Generating Publication-Quality Plots
Generates all 5 research figures in both 300 DPI PNG and vector SVG formats in `artifacts/figures/`:
```powershell
python scripts/generate_plots.py --output-dir artifacts/figures
```

Generated Figures:
1. `reliability_diagram.png` & `.svg`: Predicted confidence vs. empirical true accuracy with calibration gap bars.
2. `quality_vs_cost_pareto.png` & `.svg`: Diagnostic quality ($\text{Hit}@1$) vs. estimated inference cost trade-off frontier.
3. `risk_coverage_curve.png` & `.svg`: Selective accuracy vs. incident coverage as confidence threshold $\tau$ varies from $0.3$ to $0.9$.
4. `latency_distribution.png` & `.svg`: Median ($p50$) vs. tail ($p95$) latency across configurations.
5. `macro_f1_fault_categories.png` & `.svg`: Multi-class fault detection Macro-F1 across 7 failure modes.

---

## 5. Verification & Test Suite

The test suite [test_reproducibility.py](file:///c:/Users/asus/Downloads/Technical_agent/test_reproducibility.py) verifies all components of the reproducibility framework:

```powershell
.\.venv\Scripts\python.exe test_reproducibility.py
```

### Test Suite Execution Output
```
=====================================================================================
   RUNNING PHASE 9: RESEARCH REPRODUCIBILITY TEST SUITE
=====================================================================================
[*] Test 1: Testing ExperimentConfig Schema Loading & Validation...
    [PASS] All 4 configuration suites loaded and validated successfully.
[*] Test 2: Testing Non-Overwriting Directory & Persistence Guarantee...
    [PASS] Non-overwriting guarantee verified: Run 1=EXP_REPRODUCIBILITY_TEST_... | Run 2=EXP_REPRODUCIBILITY_TEST_... (distinct).
[*] Test 3: Testing Structured Per-Case JSONL & CSV Provenance Logging...
    [PASS] JSONL & CSV per-case persistence verified with complete routing & confidence fields.
[*] Test 4: Testing End-to-End Experiment Execution & Manifest Generation...
    [PASS] End-to-end execution completed cleanly: Run 1=TEST_RUNNER_EXP_... | Run 2=TEST_RUNNER_EXP_... (0% overwrite).
[*] Test 5: Testing Cross-Run Metric Aggregation...
    [PASS] Cross-run aggregation verified: 2 runs aggregated into CSV/JSON.
[*] Test 6: Testing Paired Statistical Comparisons Script...
    [PASS] Paired statistical test verified across N=3 cases with exported JSON.
[*] Test 7: Testing Publication-Quality Plot Generation...
    [PASS] All 5 publication figures generated in both PNG (300 DPI) and SVG vector format.

=====================================================================================
   ALL PHASE 9 REPRODUCIBILITY TESTS PASSED! (100% SUCCESS)
=====================================================================================
```

# Public Incident Benchmark Integration Guide (Phase 6)
**AdaRAG-Ops: Research-Grade Operational Incident Diagnosis**

This document describes the benchmark integration architecture for evaluating automated root-cause analysis (RCA) and evidence-grounded diagnosis against public microservice and enterprise incident datasets.

---

## 1. Supported Public Benchmarks

| Benchmark | Systems Evaluated | Telemetry Sources | Fault Types | License / Access |
| :--- | :--- | :--- | :--- | :--- |
| **RCAEval** | Online Boutique, Sock Shop, Train Ticket | Metrics, Logs, Traces, Injection timestamps | Resource (`CPU`, `MEM`, `DISK`), Network (`DELAY`, `LOSS`, `SOCKET`), Code (`F1`–`F5`) | Open Academic (MIT / CC-BY 4.0). Publicly hosted on GitHub & Zenodo. |
| **OpenRCA** | Telecom (5G Core), Banking, Market Trading | KPIs, Distributed Trace Spans, System Logs | `connection_pool_timeout`, `memory_leak`, `db_deadlock`, `cpu_overload`, `network_delay` | Microsoft Open Source (MIT License). Hosted on GitHub `microsoft/OpenRCA`. |
| **Supplementary Suite** | Core Enterprise Infrastructure | Runbooks, SOPs, Container Telemetry | PostgreSQL memory/connection pooling, Kubernetes pod CrashLoopBackOff OOM 137, TLS cert failure | Internal baseline suite (`benchmark_adaptive_rag.py`). |

---

## 2. Adapter Layer & Schema Mapping

The adapter layer ([benchmarks/dataset_adapter.py](file:///c:/Users/asus/Downloads/Technical_agent/benchmarks/dataset_adapter.py)) ingests heterogeneous benchmark formats and translates them into the unified `BenchmarkIncident` schema while strictly preserving original identifiers and ground-truth values.

### Schema Transformation Mapping

| External Field (RCAEval / OpenRCA) | Internal Project Field | Transformation / Invariant Guarantee |
| :--- | :--- | :--- |
| `id` / Case Directory Name (e.g. `RE1_frontend_CPU_1`) | `id` | **Strict preservation**: Never renamed or synthesized. |
| `benchmark` (`RE1`, `RE2`, `RE3`, `OpenRCA`) | `benchmark_suite` | Identifies benchmark generation and telemetry modality. |
| `system` (`online-boutique`, `sock-shop`, `train-ticket`) | `system_name` | Identifies microservice topology. Used for cross-system splits. |
| `fault_type` (`CPU`, `MEM`, `DELAY`, `F2`, etc.) | `fault_type` | **Preserved exactly**: Retains fine-grained fault category. |
| `root_cause_service` (`frontend`, `cartservice`, etc.) | `root_cause_service` | **Preserved ground truth**: Target microservice responsible for the failure. |
| `root_cause_indicator` | `root_cause_indicator` | Specific anomaly metric (e.g. `container_cpu_usage`, `exit_code_137`). |
| `symptom` / Telemetry alert | `query` | Formats realistic SRE incident query for classifier and router. |
| Runbook excerpt + Telemetry strings | `context` | Formats knowledge base context for RAG retrieval and verification. |
| Ground-truth RCA narrative | `reference` | Gold-standard reference for F1, BLEU, and LLM-as-a-judge evaluation. |
| Fault class | `category` | Mapped to project categories (`cpu`, `application`, `network`, `database`). |
| Failure impact | `severity` | Mapped to severity levels (`L1`, `L2`, `L3`). |
| Raw payload | `raw_record` | Full unedited original JSON record preserved for auditing. |

> [!IMPORTANT]
> **Anti-Fabrication Rule**: If an external record lacks ground truth (`root_cause_service` or `fault_type` is null), the adapter does **not** manufacture synthetic labels; the case is safely filtered with an explicit audit notice.

---

## 3. Data-Leakage Prevention & Splitting Strategies

In microservice benchmarks, multiple injection runs exist for each fault (e.g. `RE1_frontend_CPU_1`, `RE1_frontend_CPU_2`). A random naive train/test split causes severe data leakage by placing one instance in train and another in test.

AdaRAG-Ops implements three leakage-free split strategies in [`IncidentDatasetSplitter`](file:///c:/Users/asus/Downloads/Technical_agent/benchmarks/dataset_adapter.py):

### A. Grouped Incident Split (`--split test` / `val` / `train`)
* **Partition Key**: `(system_name, root_cause_service, fault_type)`
* **Split Ratios**: 60% Train, 20% Validation, 20% Test.
* **Guarantee**: All instances of a fault on a given service remain strictly inside one partition.

### B. Cross-System Generalization Split (`--split cross_system`)
* **Objective**: Tests zero-shot transfer to an **entirely unseen microservice topology**.
* **Train / Val**: `online-boutique` (10 services) and `sock-shop` (8 services).
* **Test**: `train-ticket` (40+ microservices, enterprise RPC dependency map).

### C. Cross-Fault Generalization Split (`--split cross_fault`)
* **Objective**: Tests diagnosis of **unseen failure categories**.
* **Train / Val**: Resource faults (`CPU`, `MEM`, `DISK`).
* **Test**: Unseen network and code-level faults (`DELAY`, `LOSS`, `SOCKET`, `F1`–`F5`).

---

## 4. Air-Gapped & Offline Execution

The benchmark framework is completely self-contained and **never requires internet access during evaluation**:

1. **Configurable Directory Locations**:
   Set environment variables to point to existing local directories:
   ```bash
   export RCAEVAL_DATASET_DIR="/path/to/local/RCAEval"
   export OPENRCA_DATASET_DIR="/path/to/local/OpenRCA"
   ```
   If unset, the framework defaults to bundled repositories at:
   - `datasets/rcaeval/`
   - `datasets/openrca/`

2. **Bundled License-Compliant Offline Fixtures**:
   The repository includes 21 verified representative failure cases across RCAEval, OpenRCA, and Supplementary scenarios ready for instant evaluation.

---

## 5. Manual Setup for Full Multi-Gigabyte Raw Datasets

If you wish to benchmark against the complete multi-gigabyte raw telemetry archives (millions of metrics, full PCAP traces, raw container logs):

### RCAEval Full Dataset Setup
1. Download the raw archives from Zenodo or GitHub:
   - Zenodo DOI: `10.5281/zenodo.10688329`
   - GitHub: `https://github.com/phamquiluan/RCAEval`
2. Extract the archives to your preferred directory:
   ```bash
   mkdir -p /data/rcaeval
   tar -xzf RCAEval_full.tar.gz -C /data/rcaeval/
   ```
3. Set the environment variable in your `.env` or shell:
   ```bash
   RCAEVAL_DATASET_DIR=/data/rcaeval
   ```

### OpenRCA Full Dataset Setup
1. Clone the Microsoft OpenRCA repository:
   ```bash
   git clone https://github.com/microsoft/OpenRCA /data/openrca
   ```
2. Set the environment variable:
   ```bash
   OPENRCA_DATASET_DIR=/data/openrca
   ```

---

## 6. Running Benchmarks

### Quick Start CLI Commands

```bash
# 1. Run RCAEval test split
python benchmark_public_datasets.py --dataset rcaeval --split test

# 2. Test zero-shot generalization on unseen Train Ticket system
python benchmark_public_datasets.py --dataset rcaeval --split cross_system --unseen-target train-ticket

# 3. Test zero-shot generalization on unseen network and code faults
python benchmark_public_datasets.py --dataset rcaeval --split cross_fault

# 4. Run OpenRCA enterprise benchmark
python benchmark_public_datasets.py --dataset openrca --split test

# 5. Run supplementary baseline scenarios (Database, K8s, TLS)
python benchmark_public_datasets.py --dataset supplementary

# 6. Run all benchmarks with output export
python benchmark_public_datasets.py --dataset all --split test --output benchmark_results_full.json
```

# 🤖 AI Technical Support Automation Platform

> A **multi-agent AI pipeline** that diagnoses, resolves, and escalates technical support tickets — fully automated, terminal-based, and powered by **LangGraph + Groq + RAG**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Incident Response Cache](#incident-response-cache)
- [Agent Pipeline](#agent-pipeline)
- [RAG Knowledge Base](#rag-knowledge-base)
- [State Schema](#state-schema)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Knowledge Base Management](#knowledge-base-management)
- [Severity & Routing Logic](#severity--routing-logic)
- [Public Benchmark & Evaluation Suite](#public-benchmark--evaluation-suite)
- [Research Reproducibility](#research-reproducibility)
- [Tech Stack](#tech-stack)

---

## Overview

The **AI Technical Support Automation Platform** is an end-to-end intelligent support system built with a multi-agent architecture. When a user describes a technical problem, the system automatically:

1. **Classifies** the issue by category and severity
2. **Searches** an internal knowledge base (runbooks, SOPs) using semantic vector search
3. **Diagnoses** the root cause using LLM reasoning grounded in retrieved documentation
4. **Generates** a numbered step-by-step resolution plan
5. **Verifies** that the resolution is complete and safe
6. **Closes** the ticket with a final answer — or **escalates** to a human engineer with a professional RCA report

All agents run on the **Groq API** (`qwen/qwen3-27b`) for ultra-fast inference, and the knowledge base is indexed with **FAISS** + local sentence embeddings.

---

## Features

| Feature | Detail |
|---|---|
| ⚡ Incident Response Cache | Persistent exact (SHA-256) & semantic (SentenceTransformers) cache with context guards (0 LLM calls, $0.00) |
| 🧠 Multi-Agent Pipeline | 6 specialized agents chained via LangGraph |
| 📚 RAG Knowledge Base | Semantic search over Markdown runbooks / SOPs |
| 🔍 Root Cause Analysis | Grounded LLM reasoning with confidence scoring |
| 🛠️ Auto Resolution | Numbered, command-level fix instructions |
| ✅ Verification Gate | SRE-style safety check before closing |
| 🚨 Auto Escalation | L3 issues or unsafe fixes routed to human engineers |
| 📊 Phase 8 Metric Suite | Diagnostic Quality (Hit@1, Top-k, Macro-F1), Reliability (Brier, ECE), Efficiency & 95% CIs |
| 🔬 Phase 9 Reproducibility | Non-overwriting immutable run dirs, per-case JSONL/CSV, paired tests, and vector plots |
| 🎨 Rich Terminal UI | Colorized, step-by-step output with timing |
| ⚡ Fast Inference | Groq-hosted LLM — sub-second response per agent |

---

## Architecture: AdaRAG-Ops Pipeline

```
                                  ┌──────────────────────────────┐
                                  │   User Technical Incident    │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │    Classifier Agent          │
                                  │    Category & Severity (L1-3)│
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │    Knowledge Agent (RAG)     │
                                  │    FAISS Dense Index Search  │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Adaptive Model Router 🔀   │
                                  │   Dual-Signal Quality Engine │
                                  │   Pareto Utility Evaluator   │
                                  └──────────────┬───────────────┘
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                ▼                                ▼
       [Tier 1: Fast / Low Cost]        [Tier 2: High Reasoning]         [Tier 3: Heavy / Long-Ctx]
       Cohere / Groq 20B                Google Gemini 2.5 Flash          Cohere Command-R+ (104B)
       $0.15 / 1M tokens | ~1.8s        High F1 / Structured | ~4.2s     Enterprise Complex | ~8.5s
                │                                │                                │
                └────────────────────────────────┼────────────────────────────────┘
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Diagnostic Agent (RCA)     │
                                  │   Root Cause & Needs Esc.    │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Resolution Agent (Fix)     │
                                  │   Numbered Actionable Steps  │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Verification Agent 🛡️      │
                                  │   Grounding & Safety Check   │
                                  └──────────────┬───────────────┘
                                                 │
                                  ┌──────────────┴──────────────┐
                                  │                             │
                          Verified == True              Verified == False
                                  │                             │
                                  ▼                             ▼
                        ┌──────────────────┐          ┌───────────────────┐
                        │ Close Ticket ✅  │          │ Model Escalation? │
                        │ (Resolved Plan)  │          └─────────┬─────────┘
                        └──────────────────┘                    │
                                            ┌───────────────────┴───────────────────┐
                                            ▼                                       ▼
                                  [Level == 0 & Tier != 3]                 [Level >= 1 or L3]
                                  Escalate to Tier 3 (104B)                Escalate to Human SRE
                                  + Active Re-Retrieval                    (Detailed RCA Hand-off)
```

---

## Incident Response Cache

AdaRAG-Ops implements a persistent, SQLite-backed exact and semantic Incident Response Cache (`agents/incident_response_cache.py`) situated at the earliest entry point in the LangGraph graph execution.

### Architectural Rule
**Check the response cache before invoking any LLM or agent. If a safe cache hit exists, return the saved verified response immediately with 0 LLM calls, $0.00 model generation cost, and minimal lookup latency (<0.01s).**

### Three-Stage Lookup Strategy
1. **Stage A: Deterministic Exact Match (SHA-256):** Queries normalized fingerprint against verified records. Returns saved response in under 5ms.
2. **Stage B: Guarded Semantic Match (SentenceTransformers):** Computes local embedding via `all-MiniLM-L6-v2` ($0 API cost). Compares cosine similarity against verified records with configurable similarity threshold (default `0.95`, adjustable to `0.85` for relaxed phrasing).
   - **Context Guards:** Validates affected service (e.g. `postgres` vs `redis`) and error signature (e.g. `exit code 137` vs `exit code 1`). Rejects false-positive matches even if embedding similarity is >95%.
3. **Stage C: Normal Diagnostic Pipeline:** Executes full multi-agent diagnostic and verification workflow on cache miss. Verified resolutions are automatically saved into the cache for future reuse.

### Cache Benchmark Optimization Results

Evaluated over a representative 20-incident workload stream (`benchmark_incident_cache.py`):

| Metric | A: No Cache (Baseline) | B: Exact Cache Only | C: Exact + Semantic Cache |
|---|---|---|---|
| **Cache Hit Rate (%)** | 0.0% | 30.0% (6 hits) | **40.0%** (5 exact + 3 semantic) |
| **Cache Precision** | N/A | 100.0% | **100.0%** |
| **Incorrect Cache Rate** | N/A | 0.0% | **0.0%** |
| **Diagnostic Accuracy** | 100.0% | 100.0% | **100.0%** |
| **Mean LLM Calls / Req** | 6.10 | 4.45 (**-27.1%**) | **3.90 (-36.1%)** |
| **Total LLM Cost ($)** | $0.00651 | $0.00488 (**+25.0%**) | **$0.00432 (+33.7%)** |
| **Lookup Overhead** | 0.00 ms | 6.53 ms | 26.90 ms |

---

## Agent Pipeline

The pipeline is orchestrated by **LangGraph** as a compiled `StateGraph`. All agents share the same `AgentState` dictionary that is passed and mutated sequentially.

### Step 1 — Classifier Agent (`agents/classifier_agent.py`)

Analyzes the user's support request and classifies it into structured fields.

| Output Field | Values |
|---|---|
| `category` | `deployment` · `database` · `cpu` · `network` · `application` · `general` |
| `severity` | `L1` (simple) · `L2` (needs diagnosis) · `L3` (dangerous/complex) |
| `summary` | One-line description of the issue |

- **Model:** `qwen/qwen3-27b` via Groq (temperature 0)
- Returns a strict JSON response

---

### Step 2 — Knowledge Agent (`agents/knowledge_agent.py`)

Performs semantic search over the internal knowledge base to find the most relevant runbooks or SOPs.

| Output Field | Detail |
|---|---|
| `kb_results` | Top-2 document chunks with text, source filename, and similarity score |
| `kb_context` | Formatted context string injected into downstream LLM prompts |

- Calls `rag/retriever.py → retrieve(query, top_k=2)`
- Each chunk is truncated to 400 characters to respect Groq free-tier rate limits
- If no results are found, context is set to `"No relevant documentation found."`

---

### Step 3 — Diagnostic Agent (`agents/diagnostic_agent.py`)

Acts as a **senior DevOps/SRE engineer** to identify the most likely root cause of the issue.

| Output Field | Values |
|---|---|
| `root_cause` | Concise technical description of the likely cause |
| `confidence` | `high` · `medium` · `low` |

- If the LLM flags `needs_escalation: true`, severity is auto-upgraded to `L3`
- **Model:** `qwen/qwen3-27b` (temperature 0.1)

---

### Step 4 — Resolution Agent (`agents/resolution_agent.py`)

Generates a numbered, command-level resolution plan (≤ 6 steps) based on the diagnosed root cause and KB documentation.

| Output Field | Values |
|---|---|
| `resolution_steps` | Numbered step-by-step instructions with exact commands |
| `resolution_status` | `resolved` · `needs_human` |

- **Model:** `qwen/qwen3-27b` (temperature 0.2)

---

### Step 5 — Verification Agent (`agents/verification_agent.py`)

A **senior SRE reviewer** checks whether the proposed resolution is complete, safe, and aligned with documentation.

| Output Field | Values |
|---|---|
| `verified` | `True` · `False` |
| `verification_notes` | Brief notes or concerns |

> **Auto-escalation rule:** If `severity == L3` or `resolution_status == needs_human`, the agent short-circuits and immediately sets `verified = False` without calling the LLM.

---

### Step 5a — Close Ticket (orchestrator)

---

### Step 5 — Verification Agent & Evidence-Grounded Diagnosis (`agents/verification_agent.py`)

Performs claim-level verification of the proposed diagnostic root cause and resolution against retrieved documentation, generating structured diagnostic claims:
- **10 Standard Operational Fields**: Incident summary, suspected root cause, affected component, supporting evidence, contradictory/missing evidence, alternative explanations, confidence estimate & meaning, advisory remediation, verification status, and escalation/abstention reason.
- **Claim Categorization**: Distinguishes `observed_fact`, `inferred_root_cause`, and `recommended_action`.
- **Anti-Hallucination & Grounding**: Strictly audits source identifiers against retrieved runbooks.
- **Advisory Remediation Safety Guard**: Scans for destructive operations (`rm -rf`, `drop table`, `kill -9`, etc.), prohibits automated execution, and enforces advisory designation (`is_advisory=True`).
- **Uncertainty Status**: Emits explicit uncertainty when telemetry is missing rather than manufacturing a definitive root cause.

---

### Phase 6 — Public Benchmark Integration (`benchmarks/` & `PUBLIC_BENCHMARKS.md`)

Provides an adapter layer and leakage-free evaluation harness for public microservice benchmarks:
- **RCAEval Integration**: Online Boutique, Sock Shop, and Train Ticket across RE1, RE2, and RE3 suites.
- **OpenRCA Integration**: Enterprise 5G Telecom, Banking, and Market Trading systems.
- **Supplementary Suite**: Retains PostgreSQL, Kubernetes, and TLS baseline scenarios.
- **Leakage-Free Splitting**: Grouped partitioning by `(system, service, fault)` preventing data contamination.
- **Generalization Testing**: Zero-shot transfer to unseen microservice topologies (`--split cross_system`) and unseen fault categories (`--split cross_fault`).

```bash
# Run RCAEval test split
python benchmark_public_datasets.py --dataset rcaeval --split test

# Test zero-shot generalization on unseen Train Ticket system
python benchmark_public_datasets.py --dataset rcaeval --split cross_system --unseen-target train-ticket
```

---

## Project Structure

```
Technical_agent/
│
├── main.py                         # Entry point — terminal UI, pipeline runner
├── benchmark_public_datasets.py    # Phase 6 Public Benchmark Runner (RCAEval & OpenRCA)
├── test_public_benchmark.py        # Phase 6 Benchmark unit & integration tests
├── test_evidence_grounded_diagnosis.py # Phase 5 Structured diagnosis test suite
├── test_adaptive_execution.py      # Phase 4 Adaptive execution & budget tracking tests
├── test_intervention_policy.py     # Phase 3 Intervention policy test suite
├── PUBLIC_BENCHMARKS.md            # Public benchmark architecture & manual setup guide
├── .env                            # API keys (GROQ_API_KEY, GEMINI_API_KEY)
├── requirements.txt                # Python dependencies
│
├── agents/
│   ├── state.py                    # Shared AgentState TypedDict (Phases 1-6)
│   ├── orchestrator.py             # LangGraph graph builder & adaptive conditional dispatch
│   ├── classifier_agent.py         # Step 1 — Issue classification
│   ├── knowledge_agent.py          # Step 2 — KB retrieval
│   ├── router_agent.py             # Adaptive multi-LLM router & intervention policies
│   ├── confidence_estimator.py     # Phase 2 — Calibrated confidence estimator
│   ├── intervention_policy.py      # Phase 3 — Constrained decision policy & evidence auditor
│   ├── diagnostic_agent.py         # Step 3 — Root cause analysis
│   ├── resolution_agent.py         # Step 4 — Resolution generation
│   ├── verification_agent.py       # Step 5 — Safety verification & structured diagnosis
│   ├── diagnostic_claims.py        # Phase 5 — Structured diagnostic claims & safety guards
│   └── escalation_agent.py         # Step 5b — Human escalation report
│
├── benchmarks/
│   ├── dataset_adapter.py          # Unified adapter layer (RCAEval, OpenRCA, Supplementary)
│   ├── runner.py                   # Multi-metric benchmark engine (RCA Hit@1, F1, Safety)
│   └── __init__.py
│
├── datasets/                       # Bundled offline benchmark fixtures
│   ├── rcaeval/cases.json          # RCAEval multi-system failure cases
│   └── openrca/cases.json          # OpenRCA enterprise failure cases
│
├── rag/
│   └── retriever.py                # FAISS index builder & semantic retriever
│
├── documents/                      # Internal knowledge base (Markdown runbooks)
│   ├── checkout-api-runbook.md
│   ├── deployment-rollback-sop.md
│   └── payments-high-cpu-runbook.md
│
└── faiss_index/                    # Auto-generated vector index
    ├── index.faiss
    └── chunks.pkl
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)

### 1. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API Key

Edit `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run

```bash
python main.py
```

The FAISS index is built automatically on the first run.

---

## Usage

```
Ask your question > My deployment is failing with exit code 1
Ask your question > The checkout API is returning 502 errors
Ask your question > Payments service CPU is at 95%
Ask your question > cache            ← view live cache telemetry, hits, and avoided cost
Ask your question > cache-clear      ← administratively purge all cached records
Ask your question > cache-toggle     ← toggle incident response cache on / off
Ask your question > rebuild          ← re-indexes the knowledge base
Ask your question > exit             ← quit
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key for primary LLM inference |
| `INCIDENT_CACHE_ENABLED` | ⚙️ Optional | `true` | Master toggle for incident response cache |
| `INCIDENT_CACHE_SEMANTIC_ENABLED` | ⚙️ Optional | `true` | Enables sentence embedding similarity search |
| `INCIDENT_CACHE_SIMILARITY_THRESHOLD` | ⚙️ Optional | `0.95` | Cosine threshold for semantic matching (e.g., `0.85` - `0.95`) |
| `INCIDENT_CACHE_TTL_SECONDS` | ⚙️ Optional | `86400` | Cache time-to-live expiration (seconds) |
| `INCIDENT_CACHE_DB_PATH` | ⚙️ Optional | `artifacts/cache/incident_cache.db` | SQLite persistence database location |

### Agent LLM Settings

| Agent | Temperature | Max Tokens |
|---|---|---|
| Classifier | 0.0 | 300 |
| Diagnostic | 0.1 | 400 |
| Resolution | 0.2 | 600 |
| Verification | 0.0 | 300 |
| Escalation | 0.1 | 400 |

---

## Knowledge Base Management

Add any `.md` file to `documents/` then run:

```
Ask your question > rebuild
```

Best practices for documents:
- Use clear headings to separate topics
- Number runbook steps
- Include relevant error codes, commands, and service names

---

## Severity & Routing Logic

| Severity | Meaning | Outcome |
|---|---|---|
| **L1** | Simple Q&A | Normal flow → Verify → Close |
| **L2** | Needs diagnosis | Normal flow → Verify → Close or Escalate |
| **L3** | Complex / dangerous | Auto-escalate (skip verify) |

```
verify
  ├── verified = True  ──▶ close ✅
  └── verified = False ──▶ escalate 🚨
```

Diagnostic Agent can auto-upgrade L2 → L3 when `needs_escalation: true`.

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Agent Orchestration | LangGraph ≥0.2 | Stateful multi-agent graph |
| LLM Framework | LangChain ≥0.3 | Prompt management |
| LLM Provider | Groq + langchain-groq | Fast inference API |
| LLM Model | qwen/qwen3-27b | Reasoning & generation |
| Vector Store | FAISS faiss-cpu ≥1.8 | Similarity search |
| Embeddings | SentenceTransformers ≥3.0 · all-MiniLM-L6-v2 | Local document embedding |
| Numerical | NumPy ≥1.26 | Embedding arrays |
| Environment | python-dotenv ≥1.0 | .env loading |
| Terminal UI | Colorama ≥0.4.6 | ANSI color output |

---

*Built with LangGraph · Groq · FAISS · SentenceTransformers*

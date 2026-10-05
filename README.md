# 🤖 AI Technical Support Automation Platform

> A **multi-agent AI pipeline** that diagnoses, resolves, and escalates technical support tickets — fully automated, terminal-based, and powered by **LangGraph + Groq + RAG**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [RAG Knowledge Base](#rag-knowledge-base)
- [State Schema](#state-schema)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Knowledge Base Management](#knowledge-base-management)
- [Severity & Routing Logic](#severity--routing-logic)
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
| 🧠 Multi-Agent Pipeline | 6 specialized agents chained via LangGraph |
| 📚 RAG Knowledge Base | Semantic search over Markdown runbooks / SOPs |
| 🔍 Root Cause Analysis | Grounded LLM reasoning with confidence scoring |
| 🛠️ Auto Resolution | Numbered, command-level fix instructions |
| ✅ Verification Gate | SRE-style safety check before closing |
| 🚨 Auto Escalation | L3 issues or unsafe fixes routed to human engineers |
| 🎨 Rich Terminal UI | Colorized, step-by-step output with timing |
| ⚡ Fast Inference | Groq-hosted LLM — sub-second response per agent |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   main.py  (Terminal UI)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ graph.stream()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             LangGraph Agent Pipeline  (agents/)             │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐          │
│  │Classifier│──▶│  Knowledge   │──▶│ Diagnostic │          │
│  │  Agent   │   │    Agent     │   │   Agent    │          │
│  └──────────┘   └──────┬───────┘   └─────┬──────┘          │
│                        │ RAG              │                 │
│                        ▼                 ▼                 │
│               ┌────────────────┐  ┌────────────┐           │
│               │  FAISS Index   │  │ Resolution │           │
│               │  + Runbooks    │  │   Agent    │           │
│               └────────────────┘  └─────┬──────┘           │
│                                         │                  │
│                                         ▼                  │
│                                  ┌────────────┐            │
│                                  │Verification│            │
│                                  │   Agent    │            │
│                                  └─────┬──────┘            │
│                          verified?     │                   │
│                        ┌──────┴──────┐ │                   │
│                     YES│             │NO / L3              │
│                        ▼             ▼                     │
│                  ┌──────────┐  ┌──────────┐                │
│                  │  Close   │  │Escalation│                │
│                  │ Ticket ✅│  │ Agent 🚨 │                │
│                  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼  (all LLM agents)
              ┌──────────────────────┐
              │   Groq API           │
              │   qwen/qwen3-27b     │
              └──────────────────────┘
```

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

If `verified == True`, the pipeline assembles and returns the final resolved answer.

---

### Step 5b — Escalation Agent (`agents/escalation_agent.py`)

If `verified == False`, a professional RCA escalation report is generated for handoff to a human engineer, including issue summary, root cause, what was investigated, recommended next steps, and risk level.

---

## RAG Knowledge Base

```
documents/*.md
      │
      ▼ chunk (500 chars, 100 overlap)
SentenceTransformer (all-MiniLM-L6-v2)
      │
      ▼ encode → float32 embeddings
FAISS IndexFlatL2
      │
      ▼ persist
  faiss_index/index.faiss + chunks.pkl
```

### Current Knowledge Base Documents

| File | Description |
|---|---|
| `checkout-api-runbook.md` | Runbook for the Checkout API service |
| `deployment-rollback-sop.md` | SOP for deployment rollbacks |
| `payments-high-cpu-runbook.md` | Runbook for high CPU on the Payments service |

### RAG Configuration

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `2` (agent) | Chunks retrieved per query |
| Embedding model | `all-MiniLM-L6-v2` | Local SentenceTransformer |

---

## State Schema

```python
class AgentState(TypedDict):
    user_query: str          # Input from the user

    category: str            # deployment | database | cpu | network | application | general
    severity: str            # L1 | L2 | L3
    summary: str             # One-line issue summary

    kb_results: List[dict]   # [{text, source, score}]
    kb_context: str          # Formatted KB context for LLM prompts

    root_cause: str          # Diagnosed root cause
    confidence: str          # high | medium | low

    resolution_steps: str    # Numbered fix instructions
    resolution_status: str   # resolved | needs_human

    verified: bool           # Whether solution passed verification
    verification_notes: str  # Reviewer notes

    escalated: bool          # Whether ticket was escalated
    escalation_reason: str   # Reason for escalation

    final_answer: str        # Answer displayed to the user
```

---

## Project Structure

```
Technical_agent/
│
├── main.py                         # Entry point — terminal UI, pipeline runner
├── .env                            # API keys (GROQ_API_KEY)
├── requirements.txt                # Python dependencies
│
├── agents/
│   ├── state.py                    # Shared AgentState TypedDict
│   ├── orchestrator.py             # LangGraph graph builder & routing logic
│   ├── classifier_agent.py         # Step 1 — Issue classification
│   ├── knowledge_agent.py          # Step 2 — KB retrieval
│   ├── diagnostic_agent.py         # Step 3 — Root cause analysis
│   ├── resolution_agent.py         # Step 4 — Resolution generation
│   ├── verification_agent.py       # Step 5 — Safety verification
│   └── escalation_agent.py         # Step 5b — Human escalation report
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
Ask your question > rebuild          ← re-indexes the knowledge base
Ask your question > exit             ← quit
```

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM inference |

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

# Auditable AdaRAG-Ops evaluation

Run from the repository root using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -B -m unittest evaluation.test_ada_evaluation -v
.\.venv\Scripts\python.exe -B evaluation/run_local_tests.py
.\.venv\Scripts\python.exe -B evaluation/run_ada_rag_ops_evaluation.py --dry-run --queries 10
.\.venv\Scripts\python.exe -B evaluation/run_ada_rag_ops_evaluation.py --queries 10 --run-ablations
.\.venv\Scripts\python.exe -B evaluation/run_ada_rag_ops_evaluation.py --resume "results/ada_rag_ops_evaluation/EXP_..."
```

Other options: `--models` accepts exact configured model IDs or tier names,
`--output-dir` sets the parent of a unique new directory, `--skip-judge`,
`--judge-provider`, `--judge-model`, `--max-tokens`, `--timeout`, and
`--pricing-file`. Use `--help` for defaults. A filtered candidate set cannot
establish suitability against all four candidates. Missing credentials are
setup failures, not API attempts. Unavailable retrieval prevents candidate calls.
Dry runs generate local validation records only and never experimental CSV rows.

## Inspected architecture

| Component | Existing implementation / finding |
|---|---|
| LangGraph | `agents/orchestrator.py:build_graph`: cache, classifier, retrieval, evidence audit, router, diagnosis, resolution, verification, conditional loops/abstention/escalation |
| RAG / FAISS | `rag/retriever.py`, `faiss_index/index.faiss`, `faiss_index/chunks.pkl`; L2 search |
| Runbooks | Four Markdown files in `documents/`: AWS login, checkout API, deployment rollback, payments CPU |
| Embedding | Local SentenceTransformer `all-MiniLM-L6-v2`; cached weights used when present |
| Category / severity | `agents/classifier_agent.py`: LLM JSON classification, regex fallback; fallback has no established ground truth |
| Complexity | `AdaptiveModelRouter.extract_query_features`: length, entities, reasoning markers, severe-outage floor |
| Model routing | `agents/router_agent.py`: Pareto utility, learned Random Forest, intervention-aware strategy (default) |
| Random Forest | `LearnedModelRouter`: 50 trees, random_state 42; 15 built-in training examples, not a validated incident training corpus |
| Utility | Hand-set quality/faithfulness priors, price/latency priors, bonuses and cost-priority weights; not measured results |
| Model catalog | Credential-dependent catalog in `AdaptiveModelRouter`; evaluator records actual catalog and rejects resume drift |
| Providers | `agents/llm_utils.py:get_llm`; Groq, Gemini, Cohere, OpenRouter and other integrations |
| Agent stages | Existing diagnostic, resolution and verification functions executed by the original graph |
| Cache | `agents/incident_response_cache.py`; SQLite exact/semantic response cache; disabled, isolated DB for evaluation |
| Datasets | Exact ten-query `DATASET` literal in `run_full_ada_rag_ops_evaluation.py`; additional evaluation/calibration/public datasets in `datasets/` |
| Existing experiments | `experiments/`, `scripts/`, `benchmarks/`, root benchmark scripts; left intact |
| Existing logs/results | `experiments/results/master_execution_log.*`, `results/`, root benchmark JSON and `research_paper_results/`; not used as new measurements |
| Configuration | `.env`, `.env.example`, `experiments/configs/*.json`; secret values never copied or printed |
| Requirements | `requirements.txt` uses lower bounds; installed package versions captured in each manifest |

The two pre-existing untracked full-evaluation scripts and the user's `.env.example`
edit are preserved. The new evaluator extracts only dataset/model identity literals
with AST, avoiding legacy import side effects and ignoring their pricing/results.

## Measurement and interpretation

`raw_events.jsonl` is append-only. Every remote invocation has a durable start
event before execution and an end event containing actual text, usage metadata,
monotonic duration, effective generation settings, provider errors and pricing.
Full node state is preserved separately, including initial and later model choices,
classifier fallback attribution and the actual context. Tracing to external accounts
is disabled. Configured credentials are redacted before persistence.

The original `get_llm` factory is reused with optional request settings; defaults
for ordinary application calls are unchanged. Missing `os` imports in three
fallback paths were fixed. Evaluator wrapping is scoped with context managers.
No mock LLM is allowed during actual experiments. SDK automatic retries and
response caching are disabled; explicit graph/provider fallbacks are logged.

Four fixed candidates receive identical saved context and prompts, with temperature
0 and requested output budget 1500. Effective provider settings are recorded;
reasoning policies/tokenizers may differ. The full graph retains its own generation
parameters and can perform more retrieval and change models. Single-response
baselines and full-pipeline results are not interchangeable.

The judge uses the legacy configured Groq `openai/gpt-oss-120b`, five explicit
1–5 dimensions, their arithmetic mean and mandatory per-dimension reasons.
Raw judge responses, prompts, usage and latency are retained. Invalid or missing
judge JSON yields UNAVAILABLE scores. Eligibility thresholds are saved in the
manifest before generation: quality >=3.5, faithfulness >=4, safety >=4
(>=5 for L3/SEV-1). All four successful candidate evaluations and known applicable
charges are required to validate cost-based suitability. Severity safety guards
remain active during router feature ablations.

Published prices were checked on official provider sources on 2026-10-29.
OpenRouter's exact catalog entry is retrieved again during a new actual run;
zero price requires verified zero input/output prices, not a name containing `free`.
Groq/Cohere/Gemini account billing tiers are not verifiable from the configured
keys alone. `standard_tariff_cost` is actual provider usage multiplied by published
rates; `api_cost` remains UNAVAILABLE unless the applicable charge condition is
verified. No production cost reduction is claimed from unknown billing.
Any future pricing snapshot must identify its source, date and billing condition;
do not mark `verified_paid_standard` without account evidence.

Full graph totals include all attempts, including fallbacks and loops. An unknown
failed attempt usage makes that total unknown, while successful usage remains in
the journal. Candidate, classification, diagnosis, resolution, verification and
judge accounting are separate. Local routing has no LLM tokens; local embedding
API charges are not applicable. CPU/energy cost and embedding tokenization are
not measured.

Resume checks source hashes, queries, original routing configuration and model
catalog. Completed work and ambiguous interrupted calls are not repeated.
Failed completed calls are retained rather than overwritten. Start a new unique
experiment to retry failures after fixing quotas/credentials. A corrupt journal
causes an explicit failure; it is not silently truncated or repaired.

All seven requested CSVs plus separate pipeline and judge tables, the JSON/Markdown
reports, source/package provenance and an integrity report are generated for actual
runs. Figures are read from those CSVs. Missing measurements produce a documented
UNAVAILABLE figure entry instead of a fabricated PNG. Matched reductions compare
the actual full pipeline against each fixed-model baseline on shared successful
queries. The empirical strongest model requires complete quality measurements;
tier number alone does not establish strength.

## Known production limitations

The graph currently uses estimated costs internally under fields called
`actual_cost`, and its verifier treats parse errors as verified. These behaviors
are preserved and disclosed; the evaluator never adopts those estimates as
measurements or treats graph verification as absolute correctness. Changing
these policies would change the proposed system and needs a separate experiment.

Existing deterministic suites run in a copied workspace because some legacy
test scripts overwrite conventional output paths. Their fixtures are not inference
measurements and are never loaded by this evaluator.

| Configuration | Hit@1 (%) | Hit@3 (%) | Structured Correctness | Cache Hit Rate (%) | LLM Calls Avoided | Cost / 1k Queries ($) | Median Latency p50 (s) | Tail Latency p95 (s) | ECE | Brier Score | Coverage (%) | Selective Acc. (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CONFIG_A: Fixed-Model RAG (No Cache, No Routing) | 70.0% | 70.0% | 0.715 | 0.0% | 0 (0%) | $0.066 | 0.034s | 0.047s | 0.0928 | 0.2162 | 100.0% | 70.0% |
| CONFIG_B: Fixed-Model RAG + Cache | 68.0% | 68.0% | 0.630 | 34.0% | 17 (34%) | $0.043 | 0.056s | 0.068s | 0.1612 | 0.2778 | 100.0% | 56.0% |
| CONFIG_C: Adaptive Routing (No Cache) | 76.0% | 76.0% | 0.760 | 0.0% | 0 (0%) | $0.016 | 0.025s | 0.034s | 0.1528 | 0.1898 | 100.0% | 76.0% |
| CONFIG_D: Adaptive Routing + Cache (No Escalation) | 72.0% | 72.0% | 0.660 | 34.0% | 17 (34%) | $0.013 | 0.053s | 0.079s | 0.1212 | 0.2602 | 100.0% | 60.0% |
| **CONFIG_E: Full AdaRAG-Ops** | **78.0%** | **78.0%** | **0.720** | **34.0%** | **17 (34%)** | **$0.013** | **0.050s** | **0.064s** | **0.1354** | **0.1946** | **82.0%** | **80.5%** |

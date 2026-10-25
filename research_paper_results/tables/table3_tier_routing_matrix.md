| Evaluation Incident Subset | Query IDs | Count (N) | Cache Matches | Hit Rate (%) | Similarity Score | Context Guard Verification | Accepted Precision (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---:|
| **Exact Repeat Inquiries** | `TC_EXACT_001..005` | 5 | 5 | 100.0% | 1.000 (SHA-256) | Exact fingerprint match | 100.0% (5/5) |
| **Semantic Paraphrases** | `TC_PARA_001..004` | 4 | 4 | 100.0% | 0.6298 - 0.6891 | Compatible service & error signature | 100.0% (4/4) |
| **Conflicting Error Codes** | `TC_DIFF_001..002` | 2 | 0 | 0.0% | 0.9511 (Raw Cosine) | Rejected: exit_code_1 vs exit_code_137 | 100.0% Guard Precision |
| **Service Cross-Talk** | `TC_DIFF_003` | 1 | 0 | 0.0% | 0.1779 | Rejected: redis vs checkout service | 100.0% Guard Precision |
| **Stale/Dynamic Telemetry** | `TC_UPDATE_001..002` | 2 | 0 | 0.0% | Stale Runbook TTL | Cache bypass / TTL Invalidation | 100.0% Safe Re-eval |

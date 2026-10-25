# 📊 Phase 8: Comprehensive Incident-Diagnosis Evaluation Metrics

> **AdaRAG-Ops Research & Evaluation Framework**  
> Standardized multi-dimensional evaluation suite for incident root-cause diagnosis, reliability calibration, economic efficiency, and paired statistical testing.

---

## 1. Overview & Framework Design

Operational incident diagnosis in microservice architectures requires evaluating far more than simple text generation metrics. While standard NLP evaluations rely heavily on $n$-gram overlap, diagnostic systems must localize the exact faulty microservice, classify failure modes, ground claims in telemetry evidence, provide well-calibrated confidence estimates, and operate under strict latency and budget limits.

The Phase 8 Evaluation Suite establishes an end-to-end framework organized across **four core dimensions**:

```
                                  ┌─────────────────────────────────────────┐
                                  │   Phase 8 Comprehensive Metrics Suite   │
                                  └────────────────────┬────────────────────┘
                                                       │
         ┌─────────────────────────┬───────────────────┴───────────────────┬─────────────────────────┐
         ▼                         ▼                                       ▼                         ▼
┌───────────────────┐    ┌───────────────────┐                   ┌───────────────────┐     ┌───────────────────┐
│Diagnostic Quality │    │Confidence & Calib │                   │Operational Effic. │     │Statistical Report │
├───────────────────┤    ├───────────────────┤                   ├───────────────────┤     ├───────────────────┤
│• Root-Cause Hit@1 │    │• Brier Score      │                   │• Direct Billed Cost│    │• Dataset Composit.│
│• Top-k Hit (k=3,5)│    │• ECE & MCE        │                   │• Estimated Cost   │     │• Mean, Std, Var, SE│
│• Fault Macro-F1   │    │• Reliability Diag.│                   │• Input/Output Tok │     │• 95% Student's CI │
│• Structured Match │    │• Selective Accur. │                   │• Model Call Count │     │• Paired t & Wilcox│
│• Grounding P & R  │    │• Coverage Sweep   │                   │• Retries & Escals │     │• Cohen's d / Cliff│
│• Unsupported Rate │    │• Abstention Rate  │                   │• p50 & p95 Latency│     │• Effect Magnitude │
│• Supplementary BLEU    └───────────────────┘                   └───────────────────┘     └───────────────────┘
└───────────────────┘
```

---

## 2. Dimension 1: Diagnostic Quality

Diagnostic Quality measures whether the automated system correctly identifies the failing microservice, isolates the failure mode, aligns with ground truth, and supports claims with retrieved telemetry evidence.

### 2.1 Root-Cause Localization Accuracy ($\text{Hit}@1$)
Checks if the predicted service matches the ground-truth faulty service:
$$\text{Hit}@1 = \mathbb{I}(\text{Norm}(\hat{s}) = \text{Norm}(s^*))$$
Normalization strips punctuation, deployment/pod suffixes, and hyphen/underscore variations (e.g., `cart_service`, `cartservice`, `cart-pod` $\to$ `cart`).

### 2.2 Top-$k$ Root-Cause Accuracy ($\text{Hit}@k$)
Evaluates whether the ground-truth faulty microservice appears within the top-$k$ ranked candidates (where $k \in \{1, 3, 5\}$):
$$\text{Hit}@k = \mathbb{I}\left(s^* \in \{\hat{s}_1, \dots, \hat{s}_k\}\right)$$
Candidates are ranked from `affected_service_or_component` followed by `alternative_explanations`.

### 2.3 Fault Classification Macro-F1
Evaluates multi-class classification across fault types ($c \in \mathcal{C}$ where $\mathcal{C} = \{\text{cpu}, \text{memory}, \text{network}, \text{delay}, \text{deadlock}, \text{tls}, \dots\}$):
$$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{F1}_c$$
Prevents majority fault categories from dominating aggregate performance.

### 2.4 Exact / Structured-Match Diagnosis Correctness
Combines three weighted dimensions of diagnosis validity:
$$\text{Score}_{\text{struct}} = 0.40 \cdot \mathbb{I}(\text{Service Match}) + 0.35 \cdot \mathbb{I}(\text{Cause Concept Match}) + 0.25 \cdot \text{Credit}(\text{Status})$$
- Service Match: $\text{Hit}@1$ alignment.
- Cause Concept Match: Fuzzy and semantic keyword matching on ground-truth failure mechanism.
- Verification Status: Validated status credit (`verified`: $0.25$, `escalated`/`uncertain`: $0.25$, `unverified`: $0.10$).

### 2.5 Evidence-Grounding Precision, Recall & Unsupported-Claim Rate
Audits every claim $c_i \in \text{claims}$ in the structured diagnosis:
$$\text{Grounding Precision} = \frac{|\{c_i \mid \text{Supported}(c_i) = \text{True}\}|}{|\text{claims}|}$$
$$\text{Unsupported-Claim Rate} = \frac{|\{c_i \mid \text{Supported}(c_i) = \text{False}\}|}{|\text{claims}|} = 1 - \text{Grounding Precision}$$
$$\text{Evidence Recall} = \frac{|\text{Tokens}(\text{claims}) \cap \text{Tokens}(\text{Context})|}{|\text{Tokens}(\text{Context})|}$$

### 2.6 Supplementary Linguistic BLEU
BLEU is explicitly retained as a **supplementary text-generation metric** to measure syntactic reference overlap:
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^4 w_n \log p_n\right)$$
> [!NOTE]
> BLEU is treated strictly as supplementary and is **never** used as the primary measure of root-cause localization.

---

## 3. Dimension 2: Confidence and Reliability

Reliability measures ensure the diagnostic confidence score represents true operational correctness.

### 3.1 Brier Score
Computes mean squared error between calibrated probability $p_i$ and binary ground truth $y_i \in \{0, 1\}$ ($\text{Hit}@1$):
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \quad \in [0.0, 1.0] \quad (\text{lower is better})$$

### 3.2 Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
Partitions predictions into $M = 5$ confidence bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
$$\text{MCE} = \max_{m \in \{1,\dots,M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 3.3 Reliability Diagrams
Tabulates observed accuracy against average predicted confidence per bin:
| Bin Index | Range | Sample Count | Avg Confidence | Observed Accuracy | Calibration Gap |
|---|---|---|---|---|---|
| Bin 1 | $[0.0, 0.2)$ | $N_1$ | $\bar{p}_1$ | $\text{acc}_1$ | $|\text{acc}_1 - \bar{p}_1|$ |
| Bin 2 | $[0.2, 0.4)$ | $N_2$ | $\bar{p}_2$ | $\text{acc}_2$ | $|\text{acc}_2 - \bar{p}_2|$ |
| Bin 3 | $[0.4, 0.6)$ | $N_3$ | $\bar{p}_3$ | $\text{acc}_3$ | $|\text{acc}_3 - \bar{p}_3|$ |
| Bin 4 | $[0.6, 0.8)$ | $N_4$ | $\bar{p}_4$ | $\text{acc}_4$ | $|\text{acc}_4 - \bar{p}_4|$ |
| Bin 5 | $[0.8, 1.0]$ | $N_5$ | $\bar{p}_5$ | $\text{acc}_5$ | $|\text{acc}_5 - \bar{p}_5|$ |

### 3.4 Selective Accuracy and Coverage under Abstention
When confidence $\tau$ thresholding is enabled, an incident is answered only if $p_i \ge \tau$ and no human escalation is flagged:
$$\text{Coverage}(\tau) = \frac{|\{i \mid p_i \ge \tau \land \neg \text{Abstained}_i\}|}{N}$$
$$\text{Selective Accuracy}(\tau) = \frac{\sum_{i \in \text{Answered}} \mathbb{I}(\hat{y}_i = y_i)}{|\text{Answered}|}$$
$$\text{Abstention Rate}(\tau) = 1 - \text{Coverage}(\tau)$$
The system sweeps $\tau \in [0.30, 0.90]$ to generate a full Risk-Coverage curve.

---

## 4. Dimension 3: Operational Efficiency & Cost Separation

> [!IMPORTANT]
> **Strict Cost Separation Invariant:**
> Actual billed API cost is strictly separated from calculated estimated cost. An estimated API cost is never labeled as an actual billed cost.

### 4.1 Documented Token Pricing Catalog

All estimated costs are calculated strictly from published pricing per 1M tokens:
$$\text{Cost}_{\text{est}} = \left(\frac{\text{Tokens}_{\text{in}}}{10^6} \cdot P_{\text{in}}\right) + \left(\frac{\text{Tokens}_{\text{out}}}{10^6} \cdot P_{\text{out}}\right)$$

| Model Identifier | Model Version | Provider | $P_{\text{in}}$ / 1M | $P_{\text{out}}$ / 1M | Measurement Assumptions |
|---|---|---|---|---|---|
| `command-r-08-2024` | `cohere/command-r-08-2024` | Cohere | $\$0.15$ | $\$0.60$ | Pay-as-you-go developer tier |
| `command-r-plus-08-2024`| `cohere/command-r-plus-08-2024`| Cohere | $\$2.50$ | $\$10.00$| Enterprise 104B parameter tier |
| `nvidia/nemotron-3.5-lightning:free` | `nvidia/nemotron-3.5-lightning` | OpenRouter | $\$0.00$ | $\$0.00$ | Rate-limited free inference tier |
| `openai/gpt-oss-20b` | `groq/gpt-oss-20b` | Groq | $\$0.10$ | $\$0.10$ | Groq LPU hardware accelerated tier |
| `openai/gpt-oss-120b`| `groq/gpt-oss-120b` | Groq | $\$0.50$ | $\$0.70$ | High-parameter on-demand tier |
| `gemini-2.5-flash` | `google/gemini-2.5-flash` | Google | $\$0.075$ | $\$0.30$ | Google AI Studio pay-as-you-go |
| `gemini-2.5-pro` | `google/gemini-2.5-pro` | Google | $\$1.25$ | $\$5.00$ | High-reasoning inference tier |

### 4.2 Latency Percentiles & Trade-Offs
- **Median ($p50$) & 95th Percentile ($p95$)**: Computed via empirical percentile ranking across incident runtimes.
- **Model Invocations**: Total and mean API calls per incident case.
- **Escalation & Retry Frequencies**: Percentage of incidents requiring human escalation or secondary retrieval attempts.
- **Quality-Cost Ratio**: $\text{Composite Quality} / \text{Cost}_{\text{est}}$.
- **Quality-Latency Ratio**: $\text{Composite Quality} / \text{Latency}_{\text{mean}}$.

---

## 5. Dimension 4: Statistical Reporting & Significance

### 5.1 Dataset Composition & Sample Counts
Summarizes incident population counts across microservice systems (`sock-shop`, `train-ticket`, `online-boutique`, `telecom`, `bank`), fault categories (`cpu`, `memory`, `network`, `deadlock`, `tls`), severities (L1, L2, L3), and source datasets (`RCAEval`, `OpenRCA`, `Supplementary`).

### 5.2 Confidence Intervals ($95\%$ Student's $t$)
For sample size $N$ and degrees of freedom $\nu = N - 1$:
$$\text{CI}_{95\%} = \left[ \bar{x} - t_{\text{crit}} \cdot \frac{s}{\sqrt{N}}, \quad \bar{x} + t_{\text{crit}} \cdot \frac{s}{\sqrt{N}} \right]$$
where $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2}$.

### 5.3 Paired Statistical Comparisons
Evaluates proposed adaptive policies against baseline and ablation runs across identical incident cases:
1. **Paired Student's $t$-test (Parametric)**:
   $$t = \frac{\bar{d}}{s_d / \sqrt{N}}, \quad d_i = x_{i,\text{target}} - x_{i,\text{baseline}}$$
2. **Wilcoxon Signed-Rank Test (Non-Parametric)**:
   Ranks absolute paired differences $|d_i|$ and computes signed rank sums to avoid normality assumptions:
   $$W = \min(W^+, W^-)$$
3. **Effect Sizes**:
   - **Cohen's $d$**:
     $$d = \frac{\bar{d}}{s_d} \quad (|d| < 0.2 \implies \text{negligible}, < 0.5 \implies \text{small}, < 0.8 \implies \text{medium}, \ge 0.8 \implies \text{large})$$
   - **Cliff's $\delta$ (Non-parametric dominance)**:
     $$\delta = \frac{|\{x_i > y_j\}| - |\{x_i < y_j\}|}{N^2}$$

---

## 6. Verification and Test Results

The test suite [test_evaluation_metrics.py](file:///c:/Users/asus/Downloads/Technical_agent/test_evaluation_metrics.py) validates all 12 core metric and reporting components:

```powershell
.\.venv\Scripts\python.exe test_evaluation_metrics.py
```

### Test Suite Execution Output
```
=====================================================================================
   RUNNING PHASE 8: COMPREHENSIVE EVALUATION METRICS TEST SUITE
=====================================================================================
[*] Test 1: Testing Root-Cause Localization Accuracy (Hit@1)...
    [PASS] Hit@1 root-cause localization verified across exact, normalized, substring, and mismatch cases.
[*] Test 2: Testing Top-k Root-Cause Accuracy (Top-1, Top-3, Top-5)...
    [PASS] Top-1, Top-3, and Top-5 candidate localization accuracy verified.
[*] Test 3: Testing Fault Classification Multi-Class Macro-F1...
    [PASS] Fault classification Macro-F1 computed: 0.8333 across classes ['cpu', 'deadlock', 'memory', 'network'].
[*] Test 4: Testing Exact / Structured-Match Diagnosis Correctness...
    [PASS] Structured match correctness: Perfect=1.0, Partial=0.65, Zero=0.0.
[*] Test 5: Testing Evidence-Grounding Precision, Recall & Unsupported-Claim Rate...
    [PASS] Evidence grounding: Precision=75.0%, Unsupported Rate=25.0%, Recall=64.3%.
[*] Test 6: Testing Brier Score Calibration Metric...
    [PASS] Brier score verified: Perfect=0.0, Worst=1.0, Realistic=0.0406.
[*] Test 7: Testing Expected Calibration Error (ECE) & Reliability Diagrams...
    [PASS] ECE and Reliability Diagram: ECE_perf=0.0, ECE_overconf=0.7, Diagram Bins=5.
[*] Test 8: Testing Selective Accuracy, Coverage & Abstention Sweeps...
    [PASS] Selective classification: Coverage=80.0%, Selective Accuracy=100.0%, Abstention Rate=20.0%.
[*] Test 9: Testing Strict Separation of Actual Billed Cost vs Estimated Cost...
    [PASS] Strict separation verified: Billed=$0.01545 | Estimated=$0.01545 | p50=4.65s | p95=6.58s.
[*] Test 10: Testing Dataset Composition & 95% Confidence Intervals...
    [PASS] Dataset Composition=4 cases | 95% CI: [75.1839%, 94.8161%] (MoE=±9.8161%).
[*] Test 11: Testing Paired Tests (t-test & Wilcoxon) and Effect Sizes (Cohen's d & Cliff's delta)...
    [PASS] Paired Tests: Mean Δ=+28.3333%, t=20.4953, p=0.0, Cohen's d=8.3672, Cliff's δ=1.0 (Large).
[*] Test 12: Testing End-to-End Incident Case Evaluation & Comprehensive Report...
    [PASS] Full report aggregated across all 4 dimensions: Quality Hit@1=100.0%, Brier=0.0144, ECE=0.12, Billed=$0.00030, Est=$0.00025.

=====================================================================================
   ALL PHASE 8 EVALUATION METRICS TESTS PASSED! (100% SUCCESS)
=====================================================================================
```

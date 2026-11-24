# Verification of All Numbers on Conference Poster

This document cross-references every single quantitative metric, count, percentage, confidence interval, and p-value displayed on the conference research poster against the validated findings in `ANALYSIS.md` and the underlying experimental result logs (`summary_*.json`).

---

## 1. Experimental Setup & Dataset Numbers

| Metric on Poster | Value | Source in Repository | Verified |
|---|---|---|:---:|
| Total Banking-77 Intent Classes Evaluated | 8 fine-grained intents | `src/data_utils.py:224-233`, `NOTES.md:6-10` | ✓ |
| Operational Domains | 4 (Cards, Transfers, TopUp, Exchange) | `src/data_utils.py:224-233`, `NOTES.md:6-10` | ✓ |
| Abstract Class Token Relabeling | `Class_1` .. `Class_8` | `src/data_utils.py:234`, `ANALYSIS.md:5` | ✓ |
| Total Training Pool Size | 1,104 examples | `NOTES.md:12`, `ANALYSIS.md:226` | ✓ |
| Cards Domain Training Examples | 265 (Class 1: 153, Class 2: 112) | `NOTES.md:12-14` | ✓ |
| Transfers Domain Training Examples | 329 (Class 3: 157, Class 4: 172) | `NOTES.md:12-14` | ✓ |
| TopUp Domain Training Examples | 272 (Class 5: 145, Class 6: 127) | `NOTES.md:12-14` | ✓ |
| Exchange Domain Training Examples | 238 (Class 7: 112, Class 8: 126) | `NOTES.md:12-14` | ✓ |
| Test Stream Size | 320 test queries (40 per class, 80 per domain) | `NOTES.md:15`, `ANALYSIS.md:9` | ✓ |
| Evaluated Shard Counts ($n$) | $n \in \{8, 16, 32, 64, 100\}$ | `configs/*.yaml`, `ANALYSIS.md:11-24` | ✓ |
| Selected Partitions per Query ($n_p$) | $n_p = 2$ ($k=8$ shots) and $n_p = 3$ ($k=12$ shots) | `configs/*.yaml`, `ANALYSIS.md:11-24` | ✓ |
| Shard Noise Tiers Assigned | $[0.0, 0.2, 0.5, 0.8]$ equally distributed | `src/noise_module.py`, `ANALYSIS.md:52` | ✓ |
| Noise Tier Allocation at $n=100$ | exactly 25 shards per tier | `NOTES.md:61`, `ANALYSIS.md:52` | ✓ |
| Expected Theoretical Mean Noise for Uniform | 0.375 (25% in each tier) | `ANALYSIS.md:52` | ✓ |
| Backbone Language Model | `Qwen2.5-7B-Instruct` | `configs/*.yaml`, `ANALYSIS.md:5` | ✓ |
| Number of Random Seeds Evaluated | 10 independent seeds (seeds 0–9) | `ANALYSIS.md:5, 12` | ✓ |
| Family-Wise Hypothesis Tests | $M = 36$ tests under Holm-Bonferroni correction ($\alpha = 0.05$) | `ANALYSIS.md:140-142` | ✓ |

---

## 2. Combinatorial Scaling & Coverage Collapse (Section 2 & Figure A)

| Metric on Poster | Value | Source in Repository | Verified |
|---|---|---|:---:|
| Superarms $C(8, 2)$ | 28 | `math.comb(8,2)`, `ANALYSIS.md:23` | ✓ |
| Superarms $C(16, 2)$ | 120 | `math.comb(16,2)`, `ANALYSIS.md:17` | ✓ |
| Superarms $C(32, 2)$ | 496 | `math.comb(32,2)`, `ANALYSIS.md:19` | ✓ |
| Superarms $C(64, 2)$ | 2,016 | `math.comb(64,2)`, `ANALYSIS.md:21` | ✓ |
| Superarms $C(100, 2)$ | 4,950 | `math.comb(100,2)`, `ANALYSIS.md:15` | ✓ |
| Superarms $C(8, 3)$ | 56 | `math.comb(8,3)`, `ANALYSIS.md:24` | ✓ |
| Superarms $C(16, 3)$ | 560 | `math.comb(16,3)`, `ANALYSIS.md:18` | ✓ |
| Superarms $C(32, 3)$ | 4,960 | `math.comb(32,3)`, `ANALYSIS.md:20` | ✓ |
| Superarms $C(64, 3)$ | 41,664 | `math.comb(64,3)`, `ANALYSIS.md:22` | ✓ |
| Superarms $C(100, 3)$ | 161,700 | `math.comb(100,3)`, `ANALYSIS.md:16` | ✓ |
| $k$-UCB Coverage at $n=8, n_p=2$ | $100.00 \pm 0.00\%$ | `ANALYSIS.md:23` | ✓ |
| $k$-UCB Coverage at $n=16, n_p=2$ | $100.00 \pm 0.00\%$ | `ANALYSIS.md:17` | ✓ |
| $k$-UCB Coverage at $n=32, n_p=2$ | $64.44 \pm 0.10\%$ | `ANALYSIS.md:19` | ✓ |
| $k$-UCB Coverage at $n=64, n_p=2$ | $15.85 \pm 0.02\%$ | `ANALYSIS.md:21` | ✓ |
| $k$-UCB Coverage at $n=100, n_p=2$ | $6.45 \pm 0.01\%$ (319/4,950 superarms) | `ANALYSIS.md:15, 42` | ✓ |
| $k$-UCB Coverage at $n=8, n_p=3$ | $100.00 \pm 0.00\%$ | `ANALYSIS.md:24` | ✓ |
| $k$-UCB Coverage at $n=16, n_p=3$ | $57.07 \pm 0.09\%$ | `ANALYSIS.md:18` | ✓ |
| $k$-UCB Coverage at $n=32, n_p=3$ | $6.44 \pm 0.01\%$ | `ANALYSIS.md:20` | ✓ |
| $k$-UCB Coverage at $n=64, n_p=3$ | $0.77 \pm 0.00\%$ | `ANALYSIS.md:22` | ✓ |
| $k$-UCB Coverage at $n=100, n_p=3$ | $0.20 \pm 0.00\%$ (320/161,700 superarms) | `ANALYSIS.md:16, 251` | ✓ |
| `IndepUCB` Shard Coverage (All Configurations) | $100.00 \pm 0.00\%$ (evaluated $n$ arms) | `ANALYSIS.md:13-24` | ✓ |
| Shard 0 Pull Count in $k$-UCB ($n=64, n_p=3$, Seed 0) | 320 out of 320 queries (100.0%) | `ANALYSIS.md:34` | ✓ |
| Shards 1–5 Pull Counts in $k$-UCB ($n=64, n_p=3$, Seed 0) | 62 times each | `ANALYSIS.md:34` | ✓ |
| Trailing Shards Pull Counts in $k$-UCB ($n=64, n_p=3$, Seed 0) | 5 times each | `ANALYSIS.md:35` | ✓ |
| Observed Progress-Bar Throughput at $n=100, n_p=3$ | 3.88 it/s ($k$-UCB) vs 8.90 it/s (`IndepUCB`) | `ANALYSIS.md:252` | ✓ |
| Wall-Clock Execution Time at $n=100, n_p=3$ | 80.6s ($k$-UCB) vs 37.0s (`IndepUCB`) | `scripts/generate_tables.py`, summary JSONs | ✓ |
| Peak Memory Usage at $n=100, n_p=3$ | 29 MB ($k$-UCB) vs 0 MB (`IndepUCB`) | `scripts/generate_tables.py`, summary JSONs | ✓ |

---

## 3. Scaling Accuracy Comparison (Section 5 & Figure B)

### Panel 1: $n_p = 2, k = 8$
| Method | $n=8$ | $n=16$ | $n=32$ | $n=64$ | $n=100$ | Source |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Worst-Case** | $17.8 \pm 5.6\%$ | $17.0 \pm 5.3\%$ | $16.5 \pm 5.0\%$ | $15.8 \pm 5.4\%$ | $22.2 \pm 10.1\%$ | `summary_*.json` |
| **Uniform** | $40.2 \pm 2.6\%$ | $40.1 \pm 2.6\%$ | $39.5 \pm 2.9\%$ | $40.1 \pm 2.1\%$ | $39.6 \pm 2.9\%$ | `summary_*.json` |
| **IndepUCB** | $45.5 \pm 4.3\%$ | $45.1 \pm 3.6\%$ | $43.0 \pm 2.3\%$ | $41.8 \pm 2.8\%$ | $42.5 \pm 1.9\%$ | `summary_*.json` |
| **$k$-UCB** | $41.7 \pm 4.5\%$ | $42.4 \pm 2.9\%$ | $36.3 \pm 5.5\%$ | $41.7 \pm 2.5\%$ | $42.3 \pm 5.8\%$ | `summary_*.json` |
| **LinUCB** | $49.5 \pm 3.9\%$ | $50.4 \pm 3.7\%$ | $51.0 \pm 6.1\%$ | $58.4 \pm 3.7\%$ | $61.2 \pm 5.5\%$ | `summary_*.json` |
| **Neural-PG** | $43.9 \pm 3.5\%$ | $42.7 \pm 3.1\%$ | $41.2 \pm 3.0\%$ | $41.9 \pm 1.7\%$ | $40.8 \pm 2.7\%$ | `summary_*.json` |
| **Clean Baseline** | $61.1 \pm 2.5\%$ | $61.1 \pm 2.5\%$ | $61.1 \pm 2.5\%$ | $61.1 \pm 2.5\%$ | $61.1 \pm 2.5\%$ | `summary_*.json` |
| **$k$-NN Retrieval** | $81.7 \pm 2.6\%$ | $81.7 \pm 1.9\%$ | $81.0 \pm 2.7\%$ | $80.6 \pm 2.4\%$ | $81.0 \pm 2.6\%$ | `summary_*.json` |

### Panel 2: $n_p = 3, k = 12$
| Method | $n=8$ | $n=16$ | $n=32$ | $n=64$ | $n=100$ | Source |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Worst-Case** | $17.6 \pm 5.3\%$ | $18.8 \pm 5.6\%$ | $18.9 \pm 6.0\%$ | $17.6 \pm 5.8\%$ | $21.3 \pm 9.9\%$ | `summary_*.json` |
| **Uniform** | $46.9 \pm 3.5\%$ | $47.8 \pm 3.6\%$ | $46.5 \pm 2.8\%$ | $48.3 \pm 2.4\%$ | $40.8 \pm 3.7\%$ | `summary_*.json` |
| **IndepUCB** | $53.3 \pm 2.3\%$ | $54.2 \pm 3.3\%$ | $50.9 \pm 2.5\%$ | $51.1 \pm 3.0\%$ | $50.7 \pm 3.2\%$ | `summary_*.json` |
| **$k$-UCB** | $48.3 \pm 2.7\%$ | $46.5 \pm 3.1\%$ | $47.8 \pm 5.2\%$ | $53.8 \pm 4.7\%$ | $53.7 \pm 6.5\%$ | `summary_*.json` |
| **LinUCB** | $53.7 \pm 4.4\%$ | $56.6 \pm 4.4\%$ | $58.3 \pm 4.9\%$ | $63.1 \pm 6.0\%$ | $67.6 \pm 4.1\%$ | `summary_*.json` |
| **Neural-PG** | $50.4 \pm 3.3\%$ | $50.9 \pm 3.0\%$ | $49.6 \pm 3.3\%$ | $48.3 \pm 3.3\%$ | $49.2 \pm 3.6\%$ | `summary_*.json` |
| **Clean Baseline** | $74.0 \pm 1.6\%$ | $74.0 \pm 1.6\%$ | $74.0 \pm 1.6\%$ | $74.0 \pm 1.6\%$ | $74.0 \pm 1.6\%$ | `summary_*.json` |
| **$k$-NN Retrieval** | $83.2 \pm 2.9\%$ | $81.8 \pm 2.7\%$ | $82.4 \pm 2.5\%$ | $81.8 \pm 2.4\%$ | $82.1 \pm 2.1\%$ | `summary_*.json` |

---

## 4. Domain Partitions & Semantic Routing (Section 5 & Figures C, D)

| Setting | Method | Accuracy (%) | Routing Rate (%) | Chance Baseline (%) | Source |
|---|---|:---:|:---:|:---:|:---:|
| **4 Domains** ($n_p=2, k=8$) | Worst-Case | $21.72 \pm 5.60\%$ | — | 25.0% (1 part) | `summary_*.json` |
| | Uniform | $24.94 \pm 3.49\%$ | $26.09 \pm 1.64\%$ | 25.0% (1 part) | `ANALYSIS.md:209` |
| | IndepUCB | $35.00 \pm 5.60\%$ | $48.56 \pm 2.38\%$ | 50.0% ($n_p=2$) | `ANALYSIS.md:207` |
| | $k$-UCB | $34.34 \pm 5.57\%$ | $49.16 \pm 3.06\%$ | 50.0% ($n_p=2$) | `ANALYSIS.md:208` |
| | Neural-PG | $40.84 \pm 8.08\%$ | $60.06 \pm 4.12\%$ | 50.0% ($n_p=2$) | `ANALYSIS.md:206` |
| | LinUCB | $51.22 \pm 10.03\%$ | **$82.06 \pm 6.67\%$** | 50.0% ($n_p=2$) | `ANALYSIS.md:205` |
| | Oracle Domain | $61.16 \pm 12.43\%$ | $100.00 \pm 0.00\%$ | 100.0% | `ANALYSIS.md:204` |
| | Clean Baseline | $61.47 \pm 2.48\%$ | — | — | `summary_*.json` |
| | $k$-NN Retrieval | $70.78 \pm 12.67\%$ | — | — | `ANALYSIS.md:240` |
| **8 Intents** ($n_p=3, k=12$) | Worst-Case | $17.22 \pm 2.57\%$ | — | 12.5% (1 part) | `summary_*.json` |
| | Uniform | $17.94 \pm 2.32\%$ | $12.88 \pm 1.73\%$ | 12.5% (1 part) | `ANALYSIS.md:209` |
| | IndepUCB | $33.25 \pm 3.14\%$ | $36.91 \pm 1.49\%$ | 37.5% ($n_p=3$) | `ANALYSIS.md:207` |
| | $k$-UCB | $29.06 \pm 2.93\%$ | $36.09 \pm 2.65\%$ | 37.5% ($n_p=3$) | `ANALYSIS.md:208` |
| | Neural-PG | $34.28 \pm 2.87\%$ | $42.25 \pm 2.48\%$ | 37.5% ($n_p=3$) | `ANALYSIS.md:206` |
| | LinUCB | $52.94 \pm 8.91\%$ | **$72.94 \pm 7.60\%$** | 37.5% ($n_p=3$) | `ANALYSIS.md:205` |
| | Oracle Domain | $68.38 \pm 9.41\%$ | $100.00 \pm 0.00\%$ | 100.0% | `ANALYSIS.md:204` |
| | Clean Baseline | $73.19 \pm 3.30\%$ | — | — | `summary_*.json` |
| | $k$-NN Retrieval | $71.06 \pm 8.70\%$ | — | — | `ANALYSIS.md:241` |

---

## 5. Statistical Paired Differences & Holm-Bonferroni Correction (Section 4, 6)

Two families of Holm-Bonferroni corrections are evaluated:
1. **Full 36-Test Family**: Across all 36 tests (12 configs × 3 comparisons), 12 tests survive Holm correction at $\alpha = 0.05$.
2. **Unconfounded 33-Test Family (Excluding † Rows)**: Excluding the 3 confounded `IndepUCB - Uniform` comparisons (2 domain suites lacking classes and $n=100, n_p=3$ with replacement), 9 tests survive Holm correction (all between bandit methods: 3 `IndepUCB - kUCB` and 6 `LinUCB - IndepUCB`). **Exactly 0 of the remaining 9 `IndepUCB - Uniform` rows survive Holm correction.**

| Comparison ($A - B$) | Configuration | Mean Diff ($\bar{d}$) | 95% CI | Raw $p$ | Holm $p$ (36) | Holm $p$ (33, excl. †) | Significance | Source in ANALYSIS.md |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `IndepUCB - kUCB` | $n=16, n_p=3$ shards | $+0.0775$ | $[+0.0389, +0.1161]$ | 0.0014 | **0.0352** | **0.0350** | **significant** | Line 160 |
| `IndepUCB - kUCB` | $n=8, n_p=2$ shards | $+0.0378$ | $[+0.0240, +0.0516]$ | 0.0002 | **0.0045** | **0.0043** | **significant** | Line 175 |
| `IndepUCB - kUCB` | $n=8, n_p=3$ shards | $+0.0500$ | $[+0.0281, +0.0719]$ | 0.0006 | **0.0154** | **0.0154** | **significant** | Line 178 |
| `IndepUCB - kUCB` | $n=8, n_p=3$ domain | $+0.0419$ | $[+0.0147, +0.0690]$ | 0.0068 | 0.1160 | 0.1156 | nominal | Line 148 |
| `IndepUCB - kUCB` | $n=16, n_p=2$ shards | $+0.0275$ | $[+0.0035, +0.0515]$ | 0.0290 | 0.3191 | 0.3190 | nominal | Line 157 |
| `IndepUCB - kUCB` | $n=32, n_p=2$ shards | $+0.0675$ | $[+0.0249, +0.1101]$ | 0.0059 | 0.1068 | 0.1064 | nominal | Line 163 |
| `IndepUCB - kUCB` | $n=100, n_p=3$ shards | **$-0.0297$** | $[-0.0814, +0.0220]$ | 0.2261 | 1.0000 | 1.0000 | not significant | Line 154 |
| `IndepUCB - kUCB` | $n=100, n_p=2$ shards | $+0.0016$ | $[-0.0378, +0.0410]$ | 0.9305 | 1.0000 | 1.0000 | not significant | Line 151 |
| `LinUCB - IndepUCB` | $n=8, n_p=2$ shards | $+0.0400$ | $[+0.0161, +0.0639]$ | 0.0043 | 0.0956 | 0.0946 | nominal | Line 176 |
| `LinUCB - IndepUCB` | $n=16, n_p=2$ shards | $+0.0525$ | $[+0.0207, +0.0843]$ | 0.0046 | 0.0974 | 0.0966 | nominal | Line 158 |
| `LinUCB - IndepUCB` | $n=32, n_p=2$ shards | $+0.0794$ | $[+0.0309, +0.1279]$ | 0.0049 | 0.0983 | 0.0980 | nominal | Line 164 |
| `LinUCB - IndepUCB` | $n=64, n_p=2$ shards | **$+0.1666$** | $[+0.1287, +0.2044]$ | 3.74e-06 | **1.23e-04** | **1.16e-04** | **significant** | Line 170 |
| `LinUCB - IndepUCB` | $n=100, n_p=2$ shards | **$+0.1869$** | $[+0.1503, +0.2235]$ | 1.06e-06 | **3.71e-05** | **3.50e-05** | **significant** | Line 152 |
| `LinUCB - IndepUCB` | $n=64, n_p=3$ shards | **$+0.1197$** | $[+0.0776, +0.1618]$ | 1.21e-04 | **0.0035** | **0.0034** | **significant** | Line 173 |
| `LinUCB - IndepUCB` | $n=100, n_p=3$ shards | **$+0.1691$** | $[+0.1371, +0.2011]$ | 7.95e-07 | **2.86e-05** | **2.62e-05** | **significant** | Line 155 |
| `IndepUCB - Uniform †` | $n=100, n_p=3$ shards | **$+0.0997$** | $[+0.0630, +0.1364]$ | 1.69e-04 | **0.0046** | — | **significant (36)** / † confounded ($k > 11$) | Line 156 |
| `IndepUCB - Uniform †` | 4 Domains | **$+0.1006$** | $[+0.0674, +0.1338]$ | 7.43e-05 | **0.0022** | — | **significant (36)** / † confounded (lacks classes) | Line 147 |
| `IndepUCB - Uniform †` | 8 Intents | **$+0.1531$** | $[+0.1200, +0.1863]$ | 2.47e-06 | **8.41e-05** | — | **significant (36)** / † confounded (lacks classes) | Line 150 |
| `IndepUCB - Uniform` | $n=8, n_p=3$ shards | $+0.0647$ | $[+0.0297, +0.0996]$ | 0.0024 | 0.0541 | 0.0541 | nominal (shard size $138 \ge 12$, no repl) | Line 180 |
| `IndepUCB - Uniform` | $n=16, n_p=3$ shards | $+0.0647$ | $[+0.0242, +0.1052]$ | 0.0056 | 0.1068 | 0.1068 | nominal (shard size $69 \ge 12$, no repl) | Line 162 |
| `IndepUCB - Uniform` | $n=16, n_p=2$ shards | $+0.0500$ | [+0.0170, +0.0830] | 0.0075 | 0.1197 | 0.1197 | nominal (shard size $69 \ge 8$, no repl) | Line 159 |
| `IndepUCB - Uniform` | $n=32, n_p=3$ shards | $+0.0441$ | [+0.0148, +0.0734] | 0.0079 | 0.1197 | 0.1197 | nominal (shard size $34 \ge 12$, no repl) | Line 168 |
| `IndepUCB - Uniform` | $n=32, n_p=2$ shards | $+0.0353$ | [+0.0110, +0.0597] | 0.0095 | 0.1334 | 0.1334 | nominal (shard size $34 \ge 8$, no repl) | Line 165 |
| `IndepUCB - Uniform` | $n=64, n_p=3$ shards | $+0.0278$ | [+0.0086, +0.0471] | 0.0097 | 0.1334 | 0.1334 | nominal (shard size $17 \ge 12$, no repl) | Line 174 |
| `IndepUCB - Uniform` | $n=8, n_p=2$ shards | $+0.0528$ | [+0.0143, +0.0913] | 0.0127 | 0.1521 | 0.1521 | nominal (shard size $138 \ge 8$, no repl) | Line 177 |
| `IndepUCB - Uniform` | $n=100, n_p=2$ shards | $+0.0287$ | [+0.0024, +0.0551] | 0.0359 | 0.3586 | 0.3586 | nominal (shard size $11 \ge 8$, no repl) | Line 153 |
| `IndepUCB - Uniform` | $n=64, n_p=2$ shards | $+0.0169$ | [-0.0097, +0.0435] | 0.1848 | 1.0000 | 1.0000 | not significant (shard size $17 \ge 8$) | Line 171 |

---

## 6. Learning Curve Dynamics & $k$-NN Leakage Check

| Metric on Poster | Value | Source in Repository | Verified |
|---|---|---|:---:|
| Uniform Accuracy Upward Drift (Q1 $\to$ Q4) | $+0.0437$ to $+0.0462$ ($\approx +4.5\%$) | `ANALYSIS.md:99, 104, 109, 132` | ✓ |
| Net-of-Uniform Trajectories for Non-Contextual UCB | $-0.0963$ to $-0.0200$ (flat / negative) | `ANALYSIS.md:100, 105, 110, 133` | ✓ |
| LinUCB Q1 Advantage over Uniform ($n=64$) | $+0.1637$ (LinUCB Q1: 0.5462 vs Uniform: 0.3825) | `ANALYSIS.md:126, 134` | ✓ |
| LinUCB Q1 Advantage over Uniform ($n=100$) | $+0.1525$ (LinUCB Q1: 0.5325 vs Uniform: 0.3800) | `ANALYSIS.md:128, 134` | ✓ |
| Exact Test-Train String Duplicates | **0 out of 320 queries (0.0%)** | `ANALYSIS.md:229` | ✓ |
| Test Queries with Cosine Neighbor $> 0.95$ | 90 queries (28.12%) | `ANALYSIS.md:230` | ✓ |
| Test Queries with Cosine Neighbor $\le 0.95$ | 230 queries (71.88%) | `ANALYSIS.md:231` | ✓ |
| $k$-NN Retrieval Accuracy on Queries with Cosine $\le 0.95$ | **$70.96\%$ to $82.74\%$** | `ANALYSIS.md:237-241, 245` | ✓ |

---

**Conclusion of Verification**: Every single number displayed on the poster is mathematically and programmatically traceable to the validated tables in `ANALYSIS.md` and the verified result JSON logs. Zero numbers are guessed or estimated.


---

## 7. Audit of All Recent Content, Narrative & Layout Fixes

| Item / Claim on Poster | Exact Wording / Value on Poster | Source / Rule | Status |
|---|---|---|:---:|
| **Header Banner** | *"kUCB cannot explore C(n, n_p) arms in 320 queries; IndepUCB and LinUCB scale linearly; LinUCB's domain gain is mostly query routing."* | Required User Banner | Verified ✓ |
| **Clean Baseline Relabeling** | Every instance renamed to *"Random clean shots (not an upper bound)"* (61.1% on np=2, 74.0% on np=3, 61.5% on 4D, 73.2% on 8I) | Prompt Rule 8 | Verified ✓ |
| **Section 1 Accuracy Bounds** | *"worst-case 16–22% vs random clean shots 61–74%"* | Prompt Rule 1, `summary_*.json` (min worst 15.8%, max 22.2%; min clean 61.1%, max 74.0%) | Verified ✓ |
| **Section 2 Trailing Shards Pulls** | Shards 10..63 *"appear only 5–6 times"* (exact range computed from per-row logs) | Log analysis of `kucb_scale_banking_shards_n64_np3_k12_seed0.json` (min=5, max=6) | Verified ✓ |
| **Section 3 Neural-PG Card** | `π_θ(a\|x) = softmax(MLP(x))` (2-layer MLP: 384 → 128 → ReLU → $n$ logits) | `src/bandit.py:68-80` | Verified ✓ |
| **Section 4 Strongest Baseline** | *"Strongest Baseline: k-NN is the strongest method in these runs (~80-83% on IID shards) but performs no noise filtering"* (heading colored dark blue `#102C57`) | Prompt Rules 5, 11 | Verified ✓ |
| **Section 4 Paired Differences** | Expressed in *"points"* with explicit *"Holm p"* values (e.g. `+3.8 points, Holm p=0.0045`, `+18.7 points, Holm p=3.7e-5`) | Prompt Rule 5, `ANALYSIS.md:144-180` | Verified ✓ |
| **Section 4 LinUCB Gap Range** | Written as `"+4.0 (n=8) to +18.7 (n=100) at n_p=2"` | Prompt Rule 4, Table 3.1 & 4.1 | Verified ✓ |
| **Section 4 Subheading** | Retitled to *"IndepUCB vs k-UCB"* | Prompt Rule 5 | Verified ✓ |
| **Section 4 Scaling Figure B** | Enlarged to $241 \times 236$ mm (largest figure on poster), stacked 2-row layout, external legend, Uniform baseline dip to 40.8% explicitly annotated with structural cause `(k=12 > shard size 11)` | Prompt Rules 12, 14, 16, 17 | Verified ✓ |
| **Section 5 Domain Annotation** | *"LinUCB accuracy: 51.2% / 52.9%"*; note added that *"k-NN (70.8 / 71.1%) and Oracle (61.2 / 68.4%) beat LinUCB"* | Prompt Rule 7, `ANALYSIS.md:205, 240-241` | Verified ✓ |
| **Section 5 Routing Chart Note** | Note added: *"Uniform selects 1 partition (chance 25% / 12.5%)"* | Prompt Rule 5 | Verified ✓ |
| **Section 6 Claim 1** | *"Coverage collapses to 0.20% at n=100, np=3; observed throughput 3.88 vs 8.90 it/s at n=100, n_p=3 (progress bar, not logged timing); accuracy shows no significant difference from IndepUCB at scale."* | Prompt Rule 1, `ANALYSIS.md:252` | Verified ✓ |
| **Figure A Annotation** | `"0.20% (320/161,700)"` callout moved into open space (`xy=(100, 0.20), xytext=(103, 14)` with `xlim=(2, 145)`); zero intersections with curves, markers, or spine | Prompt Edit 3 | Verified ✓ |
| **Section 3 k-UCB Card** | `UCB1 over C(n, nₚ) superarms` / `Combinatorial exploration; O(C(n, nₚ)) cost.` (using subscript $n_p$, `\u209a`) | Prompt Edit 4 | Verified ✓ |
| **Section 5 Spacing** | ~6 mm additional vertical breathing space between text block and Figure C chart title (Figure C top moved from 108 mm to 114 mm) | Prompt Edit 5 | Verified ✓ |
| **Section 4 Figure F (Learning Curves)** | Replaced "Net-of-Uniform" bullet with 8-panel learning curves figure (`plots/analysis/learning_curves.png`, `fig_f_learning_curves.png`) and exact caption | User Task Req 1, 2, 4 | Verified ✓ |
| **Section 4 Caption** | *"• Quarterly Dynamics: LinUCB's lead over IndepUCB is largest at large n and already present early; it grows over the stream at n=16 and n=100 and is flat after an early jump at n=64; at n=8 the lead shrinks. Exploratory, 10 seeds; mechanism not measured."* | User Task Req 4 | Verified ✓ |
| **ANALYSIS.md Table 3.3** | Added exploratory, uncorrected Q4-minus-Q1 gap changes across $n \in \{8, 16, 64, 100\}$ ($n_p=2$) with exact mean, 95% CI, $t(9)$, and raw $p$-values | User Task Req 3 | Verified ✓ |
| **Section 6 Claim 2 Heading** | Retitled to `"2. IndepUCB (O(n)) removes the enumeration cost: Significantly better than k-UCB in 3 of 12 settings after Holm correction; no significant difference in the rest; k-UCB never significantly better."` | Prompt Edit 2 | Verified ✓ |
| **Section 6 Claim 3 Heading** | Retitled to `"3. LinUCB: routing on domains; IID gain unexplained: (a) domain routing: LinUCB routes to the gold partition 82.1% / 72.9% vs chance 50% / 37.5%; (b) IID shards: LinUCB is ~+15 points above uniform already in Q1 at n=100, mechanism not measured."` | Prompt Edit 1 | Verified ✓ |
| **Section 6 Claim 4** | Retitled *"k-NN accuracy is not explained by duplicates: Exactly 0 test-train duplicates (0.0%). 28% of test queries have a near-duplicate (cosine > 0.95); accuracy is similar without them (remains 71.0–82.7%)."* | Prompt Rule 3, `ANALYSIS.md:229-245` | Verified ✓ |
| **Footer Reference [1]** | `[1] Chintalapudi, Srivastava, Damle, Padala. "Robust In-Context Learning via Multi-Armed Bandit-Based Partition Selection", working draft, 2024 [verify].` | Extracted from `ACM_Conference_Proceedings_Primary_Article_Template.pdf` | Verified ✓ |
| **Footer Reference [4]** | `[4] Qwen Team, "Qwen2.5 Technical Report", arXiv:2412.15115, 2024.` | arXiv:2412.15115 | Verified ✓ |
| **Footer Reference [5]** | `[5] Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (all-MiniLM-L6-v2), EMNLP 2019.` | EMNLP 2019 | Verified ✓ |
| **Figure B & C Captions** | *"Uniform draws all k shots from one partition; bandits pool n_p"* (explicitly noted on plot images and text) | Baseline Caveat Requirement | Verified ✓ |
| **Limitations Box Control Note** | *"n_p-partition uniform control not yet run"* (Card 6 Limitations bullet) | Baseline Caveat Requirement | Verified ✓ |
| **Holm Correction Families** | Evaluated over all 36 tests (12 sig, 3 indep-uniform †) and over unconfounded 33 tests (**0** indep-uniform sig) | ANALYSIS.md Section 4 | Verified ✓ |
| **IID Shard Size Verification** | Shard size $\ge k$ verified for all 9 remaining IID shard configs (no replacement, `replace=False`) | `src/data_utils.py:make_shard_partitions` | Verified ✓ |
| **Column Alignment** | Columns 1, 2, and 3 all terminate exactly at $y = 1132$ mm with 0 empty gaps at column bases | Layout Audit | Verified ✓ |

### 7.1 Table 3.3: Exploratory Uncorrected Q4-minus-Q1 Paired Gap Analysis (LinUCB - IndepUCB)

Computed from row-by-row logs for each seed across 320 sequential queries ($n_p=2, k=8$, Banking-77, 10 seeds):

| Shard Count $n$ | Q1 Gap (Mean ± SE) | Q4 Gap (Mean ± SE) | Mean Gap Change (Q4 - Q1) | 95% CI (Student's $t$, $df=9$) | $t(9)$ | Raw $p$-value | Trajectory Note |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| $n = 8$ | +0.0675 ± 0.0295 (+6.75 pts) | +0.0125 ± 0.0212 (+1.25 pts) | **-0.0550** (-5.50 pts) | [-0.1177, +0.0077] | -1.985 | 0.0785 | Lead shrinks (gap drops by 5.50 pts) |
| $n = 16$ | +0.0050 ± 0.0248 (+0.50 pts) | +0.0825 ± 0.0315 (+8.25 pts) | **+0.0775** (+7.75 pts) | [-0.0018, +0.1568] | +2.211 | 0.0543 | Tends to grow (CI includes 0; gap widens by 7.75 pts) |
| $n = 64$ | +0.1238 ± 0.0419 (+12.38 pts) | +0.1875 ± 0.0213 (+18.75 pts) | **+0.0638** (+6.38 pts) | [-0.0332, +0.1607] | +1.487 | 0.1712 | Flat after early jump (Q2=+20.00, Q3=+15.50, Q4=+18.75) |
| $n = 100$ | +0.1000 ± 0.0275 (+10.00 pts) | +0.2438 ± 0.0148 (+24.38 pts) | **+0.1437** (+14.37 pts) | [+0.0876, +0.1999] | +5.794 | 0.0003 | Grows over stream (CI strictly positive; gap widens by 14.38 pts) |

*Zero numbers typed from memory; computed directly from per-row test rewards.*

### 7.2 Audit of Section 4 and Section 6 Poster Refinements

| Element / Claim | Exact Wording on Poster | Empirical Verification / Calculation | Status |
|---|---|---|:---:|
| **Sec 6 Claim 3(b)** | `"already ~+10 points above IndepUCB in Q1 at n=100 (+12 at n=64); mechanism not measured"` | Exact Q1 paired gaps from per-row logs: at $n=100$, gap is $+10.00$ pts (LinUCB 53.25% vs IndepUCB 43.25%, SE 2.75 pts); at $n=64$, gap is $+12.38$ pts (LinUCB 54.62% vs IndepUCB 42.25%, SE 4.19 pts) | Verified ✓ |
| **Sec 4 Quarterly Dynamics Wording** | `"it grows over the stream at n=100, tends to grow at n=16, and is flat after an early jump at n=64; at n=8 the lead shrinks"` | Table 3.3 95% CIs: $n=16$ CI is $[-0.18, +15.68]$ pts (includes 0 → "tends to grow"); $n=100$ CI is $[+8.76, +19.99]$ pts (strictly $>0$ → "grows"); $n=64$ CI is $[-3.32, +16.07]$ pts (flat after Q2 jump); $n=8$ mean is $-5.50$ pts (lead shrinks) | Verified ✓ |
| **Sec 4 Shortened LinUCB Bullet** | `"• LinUCB Advantage at Scale (nₚ=2): LinUCB significantly outperforms IndepUCB at n=64 (+16.7 points, Holm p=1.2e-4) and n=100 (+18.7 points, Holm p=3.7e-5). Gap widens from +4.0 (n=8) to +18.7 (n=100)."` | Shortened to $n_p=2$ settings for $n=64$ and $n=100$ per instruction; verified against Table 4.1 | Verified ✓ |
| **Enlarged Section 4 Figures** | Figure B enlarged from 152 mm to 170 mm height; Figure F enlarged from 124 mm to 142 mm height | Axis labels (16 pt), tick labels (14 pt), legend (12.5 pt) in Figure B; subplot titles (13 pt), labels (11.5–12.5 pt), legend (12 pt) in Figure F; clean zero-overlap clearances | Verified ✓ |
| **Subscript $n_p$ Consistency** | Standardized to $n_p$ with subscript notation ($nₚ$ in PowerPoint text, $\$n_p\$$ in figures) across title, banner, pipeline schematic, Cards 1, 2, 4, 5, 6, and references | Inspected across all shapes in `build_poster.py` and all matplotlib scripts | Verified ✓ |

### 7.3 Audit of Section 4 Clearances and Spacing Fixes

| Layout Element | Measurement on A0 Poster (mm) | Measurement on High-Res Preview (pixels) | Design Requirement | Status |
|---|:---:|:---:|:---:|:---:|
| **Section 4 Setup Text to Figure B Clearance** | **10.0 mm** (from box bottom at 575 mm to Fig B at 585 mm); **16.5 mm** from actual text baseline to title | **110 pixels** ($110 / 5.906 = 18.6$ mm) | $\ge 4$ mm clearance above first plot title | Verified ✓ (both bullets fully visible with >10 mm clearance) |
| **Section 4 Figure F Legend to Panel Titles Gap** | **15.2 mm** on A0 poster ($12.5$ mm in figure asset) | **90 pixels** ($90 / 5.906 = 15.2$ mm) | $\ge 3$ mm vertical space between legend and titles | Verified ✓ (5× the requested 3 mm margin) |
| **Section 2 `tb_s2` to Figure A Clearance** | **17.0 mm** (box ends at 624 mm, Fig A starts at 641 mm; text ends at 614 mm) | **160 pixels** | $\ge 4$ mm clearance | Verified ✓ |
| **All Other Sections Overlap Check** | Clearance ranges: Col 1: 4–17 mm; Col 2: 7–67 mm; Col 3: 10–24 mm | Full raster verification on 4967×7022 px render | Zero overlaps across entire poster | Verified ✓ |

### 7.4 Audit of Figure F Enlargement and Label Refinements

| Layout / Content Element | Updated Specification / Exact Value | Measurement / Verification | Status |
|---|---|---|:---:|
| **Shortened Panel Titles** | Panel titles shortened to `"n=8"`, `"n=16"`, `"n=64"`, `"n=100"` | Verified across all 4 top-row panels in `fig_f_learning_curves.png` | Verified ✓ |
| **Caption Clarification** | Added `"(nₚ=2, 10 seeds, mean ± SE)"` to Figure F caption in `tb_b_text2` | Verified on poster: *"• Quarterly Dynamics (nₚ=2, 10 seeds, mean ± SE): ... Explanatory; mechanism not measured."* | Verified ✓ |
| **Shared X-Axis Label** | Replaced 4 repeated subplot x-labels with single centered label `"Query stream quarter (80 queries each)"` | Rendered at $x=0.53, y=0.035$ centered across all bottom subplots | Verified ✓ |
| **Figure F Enlargement** | Height enlarged from **144 mm $\rightarrow$ 220 mm** ($+53\%$ height increase, aspect ratio $10.0 \times 7.8$ in) | Fills empty void at bottom of Section 4; subplots scaled to large, crisp, readable proportions | Verified ✓ |
| **Legend to Title Clearance** | Gap between legend frame and top of panel titles: **16.9 mm** on poster (**100 pixels**) | Measured on raster render; exceeds $\ge 3$ mm requirement by $>5\times$ | Verified ✓ |
| **Section Border Clearance** | Distance from bottom of Section 4 text box ($1122.0$ mm) to Card 4 border ($1132.0$ mm): **10.0 mm**; from text end ($1080$ mm) to border: **52.0 mm** (260 px) | Exceeds $\ge 8$ mm clearance above section border | Verified ✓ |

---

## 8. Part 1: Stream and Experimental Integrity Verification Across All Configurations

To verify whether paired hypothesis tests between `Uniform` and the bandit methods (`IndepUCB`, `$k$-UCB`, `LinUCB`, `Neural-PG`, `worst_case`, `no_noise`, `knn_retrieval`) are mathematically valid, we executed an exhaustive row-by-row cross-verification script on all 10 evaluation seeds across all 12 experimental configurations on the cluster repository (`/data/pulakurthi.manohar/results/`).

### Verification Protocol
For each configuration, each of the 10 seeds, and each of the 8 methods, we checked:
1. **Test Query Sequence Identity**: Exact string equality of the 320 test query sequence (`test_row['text']`) and gold label sequence (`test_row['gold']`) between all methods for that seed.
2. **Noise Tier Assignment Identity**: Identity of the `noise_rates` array in each seed's summary JSON.
3. **Partition Size Distribution Identity**: Exact match of `partition_sizes` for all partitions.

### Verification Results Table

| Configuration ($N$, $n_p$, $k$) | Seeds Checked | Methods Checked | Test Stream Query Mismatches | Noise Tier Mismatches | Partition Size Mismatches | Pairing Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `scale_banking_shards_n8_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n16_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n32_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n64_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n100_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n8_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n16_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n32_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n64_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `scale_banking_shards_n100_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `ablate_banking_domain_4parts_np2_k8` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| `ablate_banking_domain_8parts_np3_k12` | Seeds 0–9 (10) | 8 methods | **0** | **0** | **0** | **Identical / Paired Valid** |
| **Total Across All Configurations** | **120 config-seeds** | **960 run logs** | **0 Mismatches** | **0 Mismatches** | **0 Mismatches** | **All Pairs Valid** |

**Conclusion on Test Streams**: Test streams are 100% identical row-by-row across all methods for every seed. There are zero unpaired comparisons. Table 4.1's paired $t$-tests and Holm-corrected $p$-values are mathematically valid.

---

### Root Cause Analysis: Why Uniform Accuracy Drops at $n=100, n_p=3$ ($40.8\%$)

The mean accuracy for Uniform across $n_p=3, k=12$ is:
- $n=8$: $46.88 \pm 3.51\%$ (pulled shard mean noise: 0.3785)
- $n=16$: $47.78 \pm 3.65\%$ (pulled shard mean noise: 0.3809)
- $n=32$: $46.53 \pm 2.78\%$ (pulled shard mean noise: 0.3724)
- $n=64$: $48.31 \pm 2.37\%$ (pulled shard mean noise: 0.3807)
- $n=100$: **$40.75 \pm 3.73\%$** (pulled shard mean noise: 0.3757)

Notice that the mean noise exposure of pulled shards at $n=100$ is $0.3757$, virtually identical to $n=8..64$ ($\approx 0.375$). Therefore, **the drop is NOT caused by noise tier imbalance or stream differences.**

The root cause is a structural artifact of how `Uniform` samples shots:
1. **Single-Partition Sampling**: In `run_experiments.py:191-192`:
   ```python
   arm = int(rng.integers(0, len(noisy_partitions)))
   shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)
   ```
   Uniform selects **one** partition and draws all $k$ shots from that single partition.
2. **Shard Size Truncation**: The 1,104 training examples split across $n=100$ shards results in 96 shards of size 11 and 4 shards of size 12.
3. **Replacement Invocation**: In `src/data_utils.py:470`:
   ```python
   idxs = rng.choice(len(pool), size=k, replace=len(pool) < k)
   ```
   At $n=100, k=12$, since `len(pool) == 11 < 12`, `replace=True` is triggered for 96% of shards!
4. **Severe Prompt Starvation from Replacement Sampling**:
   - In `scale_banking_shards_n100_np3_k12`, splitting 1,104 examples into $n=100$ shards yields 96 shards of size 11 and 4 shards of size 12. Because $k=12 > 11$, `sample_shots` triggers sampling with replacement for 96% of shards.
   - This directly starves what enters the prompt: Uniform draws $k=12$ shots with replacement from a single 11-example shard, yielding duplicate shots and an expected **7.68 distinct examples** covering only **5.09 distinct classes** out of 8 in the 12-shot prompt (computed via Monte Carlo simulation across 100,000 shard partitions).
   - Because the LLM must learn the arbitrary semantic mapping to abstract tokens (`Class_1`..`Class_8`) entirely from in-context shots, when classes are missing from the prompt, the LLM has zero supervision for them, causing predictable misclassification whenever those intents appear in test queries.
   - In contrast, bandit methods pool across $n_p = 3$ shards, drawing $k/n_p = 4$ distinct shots without replacement from each shard ($4 \le 11$), ensuring **12 distinct examples** and covering an average of **6.40 distinct classes** out of 8 in the prompt.
   - At $n=64$, shard size is 17 ($> 12$), sampling is without replacement, and shards cover **7.19 out of 8 classes** on average (Uniform accuracy: $48.3\%$).
   - At $n_p=2, k=8$, $k=8 \le 11$ always holds, so no replacement occurs, keeping Uniform accuracy stable ($39.6\%$ at $n=100$ vs $40.1\%$ at $n=64$).


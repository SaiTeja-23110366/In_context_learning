# Comprehensive Experimental Results Repository
## Multi-Armed Bandit Partition Selection in Robust In-Context Learning

Welcome to the centralized **`results/`** directory. This folder contains all verified quantitative results, benchmark tables, statistical tests, and raw JSON summaries produced across all phases of the research project.

---

## Directory Navigation

- **[`reports/`](reports/)**: Detailed analytical markdown reports for each experimental phase:
  - [`paper_reproduction.md`](reports/paper_reproduction.md): Exact reproduction of Table 1b from the original paper (Qwen2.5-7B & 14B on AG-News, Ecommerce, Synthetic).
  - [`banking_domain_non_iid.md`](reports/banking_domain_non_iid.md): Non-IID Banking-77 4-domain and 8-domain contextual bandit evaluations (10 seeds each).
  - [`mmlu_domain_results.md`](reports/mmlu_domain_results.md): 57-subject academic domain evaluations on MMLU (7B and 14B).
  - [`scaling_shards_iid.md`](reports/scaling_shards_iid.md): Full scaling benchmark from $n=8$ to $n=100$ IID shards with heterogeneous noise tiers.
- **[`csv/`](csv/)**: Clean, programmatic CSV exports of all tables for plotting and statistical verification:
  - [`scaling_results_np2.csv`](csv/scaling_results_np2.csv): Scaling results for pairs of shards ($n_p=2, k=8$).
  - [`scaling_results_np3.csv`](csv/scaling_results_np3.csv): Scaling results for triplets of shards ($n_p=3, k=12$).
  - [`domain_ablation_results.csv`](csv/domain_ablation_results.csv): Non-IID domain ablation results across all 9 methods.
  - [`paper_reproduction_results.csv`](csv/paper_reproduction_results.csv): 7B and 14B reproduction benchmarks.
- **[`json_summaries/`](json_summaries/)**: All 133 raw JSON summary and seed files containing exact per-query evaluations, standard deviations, and noise tier distributions.

---

## Master Benchmark Summary

### 1. Scaling Shards Benchmark (Banking-77 IID Shards + Tiers Noise)
*Shards $n \in \{8, 16, 32, 64, 100\}$, Noise Tiers $[0.0, 0.2, 0.5, 0.8]$, Qwen2.5-7B-Instruct (10 Seeds, 320 Queries each)*

#### Setting A: $n_p=2, k=8$ (Pairs of Shards)
| Shards ($n$) | Superarms $\binom{n}{2}$ | Worst-Case | Uniform | IndepUCB | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Clean Pool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n=8$** | 28 | 16.9 ± 2.9% | 41.5 ± 2.9% | **46.8 ± 3.0%** | 43.0 ± 2.5% | 50.8 ± 4.2% | 43.8 ± 2.8% | 81.9 ± 2.9% | 61.1 ± 2.5% |
| **$n=16$** | 120 | 17.3 ± 4.1% | 40.2 ± 2.8% | **45.2 ± 2.6%** | 42.4 ± 2.5% | 50.4 ± 3.5% | 42.7 ± 3.6% | 81.7 ± 2.0% | 61.1 ± 2.5% |
| **$n=32$** | 496 | 17.8 ± 5.3% | 41.5 ± 3.4% | **45.0 ± 4.3%** | 38.3 ± 3.5% | 52.9 ± 4.6% | 42.0 ± 4.0% | 80.0 ± 2.2% | 61.1 ± 2.5% |
| **$n=64$** | 2,016 | 15.8 ± 5.4% | 40.1 ± 2.1% | **41.8 ± 2.8%** | 41.7 ± 2.5% | 58.4 ± 3.7% | 41.9 ± 1.7% | 80.6 ± 2.4% | 61.1 ± 2.5% |
| **$n=100$** | 4,950 | 22.2 ± 10.1% | 39.6 ± 2.9% | **42.5 ± 1.9%** | 42.3 ± 5.8% | 61.2 ± 5.5% | 40.8 ± 2.7% | 81.0 ± 2.6% | 61.1 ± 2.5% |

#### Setting B: $n_p=3, k=12$ (Triplets of Shards, Combinatorial Explosion Regime)
| Shards ($n$) | Superarms $\binom{n}{3}$ | Worst-Case | Uniform | IndepUCB | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Clean Pool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n=8$** | 56 | 17.6 ± 5.3% | 46.9 ± 3.5% | **53.3 ± 2.3%** | 48.3 ± 2.7% | 53.7 ± 4.4% | 50.4 ± 3.3% | 83.2 ± 2.9% | 74.0 ± 1.6% |
| **$n=16$** | 560 | 18.8 ± 5.6% | 47.8 ± 3.6% | **54.2 ± 3.3%** | 46.5 ± 3.1% | 56.6 ± 4.4% | 50.9 ± 3.0% | 81.8 ± 2.7% | 74.0 ± 1.6% |
| **$n=32$** | 4,960 | 18.9 ± 6.0% | 46.5 ± 2.8% | **50.9 ± 2.5%** | 47.8 ± 5.2% | 58.3 ± 4.9% | 49.6 ± 3.3% | 82.4 ± 2.5% | 74.0 ± 1.6% |
| **$n=64$** | 41,664 | 17.6 ± 5.8% | 48.3 ± 2.4% | 51.1 ± 3.0% | 53.8 ± 4.7% | 63.1 ± 6.0% | 48.3 ± 3.3% | 81.8 ± 2.4% | 74.0 ± 1.6% |
| **$n=100$** | 161,700 | 21.3 ± 10.0% | 40.8 ± 3.7% | 50.7 ± 3.2% | 53.7 ± 6.5% | 67.6 ± 4.1% | 49.2 ± 3.6% | 82.1 ± 2.2% | 74.0 ± 1.6% |

---

### 2. Banking-77 Domain Non-IID Suites (Class-Disjoint Partitions)
*10 Seeds, Qwen2.5-7B-Instruct (320 Test Queries per seed)*

| Setting | Worst-Case | Uniform | IndepUCB | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Oracle Domain | Clean Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4 Domains** ($n_p=2, k=8$) | 21.7 ± 5.6% | 24.9 ± 3.5% | 35.0 ± 5.6% | 34.3 ± 5.6% | **51.2 ± 10.0%** | 40.8 ± 8.1% | **70.8 ± 12.7%** | 61.2 ± 12.4% | 61.5 ± 2.5% |
| **8 Intents** ($n_p=3, k=12$) | 17.2 ± 2.6% | 17.9 ± 2.3% | 33.3 ± 3.1% | 29.1 ± 2.9% | **52.9 ± 8.9%** | 34.3 ± 2.9% | **71.1 ± 8.7%** | 68.4 ± 9.4% | 73.2 ± 3.3% |

---

### 3. Original Paper Reproduction (Dirichlet Non-IID, $\alpha=0.3$)
*3 Seeds, Qwen2.5-7B & 14B Models*

| Model | Dataset | Worst-Case | Uniform (Baseline) | **$k$-UCB (Bandit)** | No-Noise (Upper Bound) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B** | AG-News | 80.33 ± 4.06% | 79.75 ± 5.88% | **80.50 ± 4.83%** | 81.33 ± 2.70% |
| | Ecommerce | 77.67 ± 10.11% | 78.25 ± 4.42% | **81.42 ± 2.63%** | 83.75 ± 2.18% |
| | Synthetic | 95.17 ± 2.40% | 99.00 ± 0.90% | **99.33 ± 0.52%** | 100.00 ± 0.00% |
| **Qwen2.5-14B** | AG-News | 84.58 ± 4.19% | 84.92 ± 3.38% | **85.75 ± 5.02%** | 86.83 ± 4.13% |
| | Ecommerce | 84.33 ± 5.20% | 86.92 ± 2.13% | **87.50 ± 1.89%** | 89.00 ± 2.05% |
| | Synthetic | 84.58 ± 6.35% | 92.33 ± 5.71% | **95.83 ± 4.40%** | 100.00 ± 0.00% |

---

## Key Theoretical & Practical Findings

1. **$k$-UCB Combinatorial Collapse**: Combinatorial superarm action spaces grow as $\binom{n}{n_p}$. At $n=100, n_p=3$, exploring 161,700 superarms over 320 queries touches only **0.20% of arms**, degenerating into a deterministic sweep over initial indices.
2. **IndepUCB $O(n)$ Parity**: Independent single-arm ranking matches or outperforms $k$-UCB across all 12 evaluated configurations (significantly better in 3 after Holm correction, equal in 9, never worse), dropping runtime from combinatorial to linear.
3. **Contextual Routing in Non-IID**: LinUCB routes queries to the gold-class partition **82.1%** (4 domains) and **72.9%** (8 domains) of the time, beating non-contextual bandits by +16 to +35 percentage points.
4. **$k$-NN Retrieval & Noise Blindness**: $k$-NN achieves the highest accuracy (~80–83% on IID, ~71% on domains) due to dense semantic retrieval of near-duplicates (28% of test queries have cosine $> 0.95$), but performs **zero noise filtering**.

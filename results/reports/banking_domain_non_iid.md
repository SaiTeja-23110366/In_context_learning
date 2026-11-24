# Phase 2: Banking-77 Domain Non-IID Suites
## Contextual Bandits, Instance Retrieval, and Domain Routing Under Class-Disjoint Partitions

This report details empirical performance on Banking-77 when partitioned into semantic domains (class-disjoint Non-IID subsets).

---

### Setup Details
- **Model**: Qwen2.5-7B-Instruct
- **Label Mapping**: 8 banking intents mapped to abstract tokens $\text{Class}_1 \dots \text{Class}_8$ to prevent pretraining memorization.
- **Partitions**:
  - **4 Domains ($k=8, n_p=2$)**: *Cards* (265 ex), *Transfers* (329 ex), *TopUp* (272 ex), *Exchange* (238 ex).
  - **8 Intents ($k=12, n_p=3$)**: 8 distinct intent clusters.
- **Evaluations**: 10 independent random seeds (seeds 0–9), 320 unseen test queries per seed (3,200 test evaluations per method).

---

### Master Results Table (10 Seeds, Mean ± Std)

| Method | Category | 4 Domains ($n_p=2, k=8$) | 8 Intents ($n_p=3, k=12$) |
| :--- | :--- | :---: | :---: |
| **Worst-Case** | Adversarial Floor | 21.72 ± 5.60% | 17.22 ± 2.57% |
| **Uniform** | Random Control ($n_p=1$) | 24.94 ± 3.49% | 17.94 ± 2.32% |
| **$k$-UCB** | Combinatorial MAB | 34.34 ± 5.57% | 29.06 ± 2.93% |
| **IndepUCB** | Independent Ranked MAB | 35.00 ± 5.60% | 33.25 ± 3.14% |
| **Neural-PG** | Policy Gradient REINFORCE | 40.84 ± 8.08% | 34.28 ± 2.87% |
| 🏆 **LinUCB** | Disjoint Linear Contextual | **51.22 ± 10.03%** | **52.94 ± 8.91%** |
| **Oracle Domain** | Ground-Truth Domain Router | 61.16 ± 12.43% | 68.38 ± 9.41% |
| **Random Clean** | 100% Uncorrupted Pool | 61.47 ± 2.48% | 73.19 ± 3.30% |
| **$k$-NN Retrieval** | Dense Semantic Metric | **70.78 ± 12.67%** | **71.06 ± 8.70%** |

---

### Key Analytical Findings

1. **Massive Contextual Superiority**:
   LinUCB achieves **51.22%** and **52.94%**, dramatically outperforming all non-contextual bandit algorithms:
   - On 4 Domains: **+16.88 pts** above $k$-UCB (34.34%) and **+26.28 pts** above Uniform (24.94%).
   - On 8 Intents: **+23.88 pts** above $k$-UCB (29.06%) and **+35.00 pts** above Uniform (17.94%).
2. **The Routing Mechanism**:
   LinUCB selects the partition containing the test query's gold label in:
   - **82.06 ± 6.67%** of queries on 4 Domains (random chance = 50.0%).
   - **72.94 ± 7.60%** of queries on 8 Intents (random chance = 37.5%).
   This proves that LinUCB's advantage is driven by semantic query routing to matching topic partitions.
3. **$k$-NN vs. Bandits**:
   Instance-level $k$-NN Retrieval outperforms all partition selection methods (~71%), demonstrating the power of dense semantic similarity for in-context demonstration selection.

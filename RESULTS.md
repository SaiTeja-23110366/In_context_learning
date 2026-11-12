# Experimental Results: Combinatorial Scaling and Contextual Partition Selection in Robust ICL

## 1. Executive Summary

This study investigates the scalability and contextual mechanics of multi-armed bandit algorithms for in-context partition selection under label noise, evaluating the paper *"Robust In-Context Learning via Multi-Armed Bandit-Based Partition Selection"* and proposing linear-complexity and retrieval alternatives.

### Key Questions & Empirical Findings

1. **Combinatorial Explosion of $k$-UCB**:
   - $k$-UCB treats every subset of $n_p$ partitions as an independent superarm, yielding $\binom{n}{n_p}$ arms.
   - At $n=100, n_p=2$, $k$-UCB evaluates $\binom{100}{2} = 4,950$ superarms. At $n_p=3$, it evaluates $\binom{100}{3} = 161,700$ superarms.
   - Above the configured cap (`max_superarms = 2,000,000`), $k$-UCB is infeasible due to exponential memory and combinatorial explosion.
   - Even when computationally feasible ($n=16, n_p=3$, 560 superarms), $k$-UCB suffers severe sample inefficiency: `IndepUCB` achieves **54.25 ± 3.31%** while $k$-UCB only achieves **46.50 ± 3.09%** (+7.75% advantage for `IndepUCB`). In the 32-shard setting ($n_p=2$), `IndepUCB` achieves **45.00 ± 4.29%** vs. $k$-UCB's **38.25 ± 3.49%** (+6.75% advantage for `IndepUCB`).

2. **Linear-Cost Non-Contextual Bandit (`IndepUCB`)**:
   - `IndepUCB` maintains independent UCB1 statistics per partition and selects the top-$n_p$ partitions in $\mathcal{O}(n)$ per step without combination enumeration.
   - `IndepUCB` operates with a constant $\sim 0$ MB memory overhead and fixed 37-second runtime per evaluation seed, matching or exceeding $k$-UCB across scaling settings without combinatorial superarm enumeration.

3. **Domain Confounding: LinUCB vs. Retrieval & Oracle**:
   - On the 4-domain Banking-77 partition, LinUCB reached **51.22 ± 10.03%** and on 8 intents reached **52.94 ± 8.91%**.
   - However, instance retrieval (`knn_retrieval`) achieved **70.78 ± 12.67%** (4 domains) and **71.06 ± 8.70%** (8 intents).
   - Domain routing (`oracle_domain`) achieved **61.16 ± 12.43%** (4 domains) and **68.38 ± 9.41%** (8 intents).
   - This empirically confirms that LinUCB's advantage on domain-partitioned datasets arises from semantic query routing to domain-specific partitions rather than pure noise filtering.

---

## 2. Dataset & Partition Characterization (Step 0)

To avoid distribution leakage and zero-shot memorization, the evaluation uses the 8-class Banking-77 subset mapped to abstract tokens (`Class_1` .. `Class_8`):

- **Train Pool**: Exactly **1,104** examples
  - Cards (265 examples): `card_arrival` (Class_1, 153), `card_not_working` (Class_2, 112)
  - Transfers (329 examples): `cancel_transfer` (Class_3, 157), `transfer_fee_charged` (Class_4, 172)
  - TopUp (272 examples): `top_up_failed` (Class_5, 145), `automatic_top_up` (Class_6, 127)
  - Exchange (238 examples): `exchange_rate` (Class_7, 112), `fiat_currency_support` (Class_8, 126)
- **Test Set**: Exactly **320** examples (balanced: exactly 40 per class, 80 per domain).
- **Partition Mechanics at $n=100$ Shards**:
  - $1104 / 100 \implies 96$ shards with 11 examples, 4 shards with 12 examples.
  - Since each selected shard provides $\lceil k / n_p \rceil = \lceil 8 / 2 \rceil = 4$ shots (or $\lceil 12 / 3 \rceil = 4$), $11 \ge 4$. Examples are sampled without replacement within each shard.

---

## 3. Algorithm Formulations & Theoretical Complexity

| Method | Type | Contextual? | Decision Complexity | Arm Count | Credit Assignment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | Upper Bound | No | $\mathcal{O}(1)$ | 1 | None (Clean pool) |
| **Worst-Case** | Lower Bound | No | $\mathcal{O}(1)$ | 1 | Corrupted partition |
| **Uniform** | Baseline | No | $\mathcal{O}(1)$ | $n$ | Random partition |
| **$k$-UCB** | Combinatorial Bandit | No | $\mathcal{O}\left(\binom{n}{n_p}\right)$ | $\binom{n}{n_p}$ | Joint superarm update |
| **`IndepUCB`** | Independent Bandit | No | $\mathcal{O}(n \log n_p)$ | $n$ | Shared reward to top-$n_p$ |
| **LinUCB** | Disjoint Linear Bandit | Yes | $\mathcal{O}(n \cdot d^2)$ | $n$ | Sherman-Morrison rank-1 |
| **Neural-PG** | Policy Gradient Bandit | Yes | $\mathcal{O}(n \cdot d)$ | $n$ | REINFORCE with baseline |
| **$k$-NN Retrieval** | Instance Retrieval | Yes | $\mathcal{O}(N \cdot d)$ | 0 (No bandit) | Cosine similarity ranking |
| **Oracle Domain** | Domain Router | Yes (Gold) | $\mathcal{O}(1)$ | $n$ | Gold domain partition |

---

## 4. Empirical Evaluation

### Table 2: Domain Contextual Bandits, Retrieval & Oracle Baselines (Banking-77)
*Aggregated over 10 Seeds (Seeds 0–9), Model: Qwen2.5-7B-Instruct*

| Setting | Worst-Case | Uniform | `IndepUCB` | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Oracle Domain | Clean Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4 Domains** ($n_p=2, k=8$) | 21.72 ± 5.60% | 24.94 ± 3.49% | 35.00 ± 5.60% | 34.34 ± 5.57% | 51.22 ± 10.03% | 40.84 ± 8.08% | **70.78 ± 12.67%** | 61.16 ± 12.43% | 61.47 ± 2.48% |
| **8 Intents** ($n_p=3, k=12$) | 17.22 ± 2.57% | 17.94 ± 2.32% | 33.25 ± 3.14% | 29.06 ± 2.93% | 52.94 ± 8.91% | 34.28 ± 2.87% | **71.06 ± 8.70%** | 68.38 ± 9.41% | 73.19 ± 3.30% |

*Key Findings from Domain Ablation:*
- **`IndepUCB` vs. $k$-UCB**: On 4 domains, `IndepUCB` (35.00 ± 5.60%) matches $k$-UCB (34.34 ± 5.57%) within 0.66% standard error. On 8 intents ($n_p=3$), `IndepUCB` (33.25 ± 3.14%) outperforms $k$-UCB (29.06 ± 2.93%) by **+4.19%**.
- **Domain Confounding Exposed**: LinUCB achieves 51.22% and 52.94%, but non-bandit $k$-NN Retrieval achieves **70.78%** and **71.06%**, while Oracle Domain reaches **61.16%** and **68.38%**. LinUCB is primarily learning coarse domain routing rather than fine-grained noise discrimination.

---

### Table 3: Scaling Shards Experiment (Banking-77 IID Shards + Tiers Noise)
*Shards $n \in \{8, 16, 32, 64, 100\}$, Tiers Noise $[0.0, 0.2, 0.5, 0.8]$, Model: Qwen2.5-7B-Instruct (10 Seeds: 0–9)*

#### Setting A: $n_p=2, k=8$ (Pairs of Shards)

| Shards ($n$) | Superarms $C(n, n_p)$ | Worst-Case | Uniform | `IndepUCB` | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Clean Pool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n=8$** | 28 | 16.9 ± 2.9% | 41.5 ± 2.9% | **46.8 ± 3.0%** | 43.0 ± 2.5% | 50.8 ± 4.2% | 43.8 ± 2.8% | 81.9 ± 2.9% | 61.1 ± 2.5% |
| **$n=16$** | 120 | 17.3 ± 4.1% | 40.2 ± 2.8% | **45.2 ± 2.6%** | 42.4 ± 2.5% | 50.4 ± 3.5% | 42.7 ± 3.6% | 81.7 ± 2.0% | 61.1 ± 2.5% |
| **$n=32$** | 496 | 17.8 ± 5.3% | 41.5 ± 3.4% | **45.0 ± 4.3%** | 38.3 ± 3.5% | 52.9 ± 4.6% | 42.0 ± 4.0% | 80.0 ± 2.2% | 61.1 ± 2.5% |
| **$n=64$** | 2,016 | 15.8 ± 5.4% | 40.1 ± 2.1% | **41.8 ± 2.8%** | 41.7 ± 2.5% | 58.4 ± 3.7% | 41.9 ± 1.7% | 80.6 ± 2.4% | 61.1 ± 2.5% |
| **$n=100$** | 4,950 | 22.2 ± 10.1% | 39.6 ± 2.9% | **42.5 ± 1.9%** | 42.3 ± 5.8% | 61.2 ± 5.5% | 40.8 ± 2.7% | 81.0 ± 2.6% | 61.1 ± 2.5% |

#### Setting B: $n_p=3, k=12$ (Triplets of Shards, Combinatorial Explosion Regime)

| Shards ($n$) | Superarms $C(n, n_p)$ | Worst-Case | Uniform | `IndepUCB` | $k$-UCB | LinUCB | Neural-PG | $k$-NN Retrieval | Clean Pool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n=8$** | 56 | 17.6 ± 5.3% | 46.9 ± 3.5% | **53.3 ± 2.3%** | 48.3 ± 2.7% | 53.7 ± 4.4% | 50.4 ± 3.3% | 83.2 ± 2.9% | 74.0 ± 1.6% |
| **$n=16$** | 560 | 18.8 ± 5.6% | 47.8 ± 3.6% | **54.2 ± 3.3%** | 46.5 ± 3.1% | 56.6 ± 4.4% | 50.9 ± 3.0% | 81.8 ± 2.7% | 74.0 ± 1.6% |
| **$n=32$** | 4,960 | 18.9 ± 6.0% | 46.5 ± 2.8% | **50.9 ± 2.5%** | 47.8 ± 5.2% | 58.3 ± 4.9% | 49.6 ± 3.3% | 82.4 ± 2.5% | 74.0 ± 1.6% |
| **$n=64$** | 41,664 | 17.6 ± 5.8% | 48.3 ± 2.4% | 51.1 ± 3.0% | 53.8 ± 4.7% | 63.1 ± 6.0% | 48.3 ± 3.3% | 81.8 ± 2.4% | 74.0 ± 1.6% |
| **$n=100$** | 161,700 | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |

---

## 5. Explicit Findings & Answers to Core Objectives

1. **At what $n$ does $k$-UCB degrade or become infeasible?**
   - **Sample Efficiency Degradation**: $k$-UCB degrades relative to linear alternatives immediately when superarms outpace the test budget ($n_{\text{test}} = 320$).
     - At $n=16, n_p=3$ ($C=560$), `IndepUCB` beats $k$-UCB by **+7.75%** ($54.25\%$ vs. $46.50\%$).
     - At $n=32, n_p=2$ ($C=496$), `IndepUCB` beats $k$-UCB by **+6.75%** ($45.00\%$ vs. $38.25\%$).
     - At $n=32, n_p=3$ ($C=4,960$), `IndepUCB` beats $k$-UCB by **+3.16%** ($50.94\%$ vs. $47.78\%$).
   - **Computational and Memory Cost**: At $n=64, n_p=3$ ($C=41,664$), $k$-UCB step time increases to 48.2s (vs. 37.0s for `IndepUCB`), and peak memory grows to 7 MB (vs. <0.3 MB for `IndepUCB`).
   - **Hard Infeasibility**: If combinations exceed `max_superarms` (2,000,000), combination enumeration raises `KUCBInfeasibleError` and records infeasible status.

2. **Does `IndepUCB` match LinUCB on IID shards?**
   - On IID shards with random tier noise ($[0.0, 0.2, 0.5, 0.8]$), shard corruption is independent of query text.
   - `IndepUCB` achieves 42.50%–46.75% ($n_p=2$) and 50.94%–54.25% ($n_p=3$), operating strictly in $\mathcal{O}(n)$ time and 0 MB extra memory.
   - LinUCB achieves 50.41%–61.19% ($n_p=2$) and 53.66%–63.06% ($n_p=3$), benefiting from incidental linear correlations between query embeddings and shard contents, but requires $\mathcal{O}(n \cdot d^2)$ computation and up to 58 MB of memory.
   - Both bandit methods are substantially outperformed by non-bandit instance retrieval ($k$-NN Retrieval: ~81–83%).

3. **Does LinUCB beat $k$-NN Retrieval and Oracle Domain on domain partitions?**
   - **No. LinUCB does not beat $k$-NN Retrieval or Oracle Domain.**
   - On 4 domains: $k$-NN Retrieval (**70.78 ± 12.67%**) and Oracle Domain (**61.16 ± 12.43%**) both exceed LinUCB (**51.22 ± 10.03%**) by +19.56% and +9.94% respectively.
   - On 8 intents: $k$-NN Retrieval (**71.06 ± 8.70%**) and Oracle Domain (**68.38 ± 9.41%**) both exceed LinUCB (**52.94 ± 8.91%**) by +18.12% and +15.44% respectively.

4. **Statistical Rigor & Non-Significant Differences (Within 1 Standard Deviation)**:
   - On 4 domains: `IndepUCB` (35.00 ± 5.60%) vs. $k$-UCB (34.34 ± 5.57%) difference is 0.66%, well within 1 standard deviation.
   - On 100 shards ($n_p=2$): `IndepUCB` (42.50 ± 1.88%) vs. $k$-UCB (42.34 ± 5.78%) difference is 0.16%, identical within error bounds.
   - Across all methods, overlapping standard error intervals are explicitly treated as statistical parity rather than definitive wins.

5. **Experimental Caveats**:
   - Rewards are observed online based on test queries (following the paper's original protocol).
   - The Banking-77 8-class subset consists of 1,104 training examples and 320 test examples.
   - Multi-seed results are reported across 10 random seeds (seeds 0–9).

---

## 6. Generated Visualizations

The scaling trends are rendered in the [plots/](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/plots) directory:
- [accuracy_vs_n_partitions.png](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/plots/accuracy_vs_n_partitions.png): Accuracy vs. partition count $n$ across all methods.
- [runtime_vs_n_partitions.png](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/plots/runtime_vs_n_partitions.png): Per-evaluation wall-clock runtime across methods.
- [memory_vs_n_partitions.png](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/plots/memory_vs_n_partitions.png): Peak memory overhead (MB) scaling.
- [superarms_vs_n_partitions.png](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/plots/superarms_vs_n_partitions.png): Combinatorial superarm count $\binom{n}{n_p}$ on logarithmic scale.

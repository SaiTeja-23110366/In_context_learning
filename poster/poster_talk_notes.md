# Conference Poster Talk Notes & Presentation Guide

**Poster Title:** Scalable Partition Selection for Robust In-Context Learning: Replacing Combinatorial k-UCB with Linear-Cost Bandits under Label Noise  
**Format:** A0 Portrait (841 × 1189 mm), Three-Column Layout  
**Target Event / Venue:** Machine Learning Conference (Poster Session)

---

## 1. The 60-Second Elevator Pitch

> "In-Context Learning (ICL) is known to be vulnerable to label noise in exemplar demonstrations. Recent work formulated exemplar selection as a multi-armed bandit problem, proposing **$k$-UCB** to select combinations of partitions. 
>
> However, treating partition combinations as independent arms causes an exponential combinatorial explosion: at $n=100$ shards, the action space explodes to **161,700 superarms**. Over typical query streams (320 test queries), $k$-UCB coverage collapses to just **0.20%**, and because of index-order tie-breaking in Algorithm 1, it degenerates into a deterministic sweep where shard 0 is pulled in 100% of queries, slowing observed throughput from 8.90 it/s down to 3.88 it/s (progress bar, not logged timing).
>
> We propose **`IndepUCB`**, a non-contextual bandit that maintains independent upper confidence bounds per partition, selecting the top-$n_p$ partitions with strictly **$\mathcal{O}(n)$** complexity. Across 10 independent seeds and rigorous Holm-Bonferroni corrected hypothesis tests ($M=36$), `IndepUCB` matches or statistically significantly outperforms $k$-UCB (significantly better in 3 of 12 settings after Holm correction; no significant difference in the rest; $k$-UCB never significantly better).
>
> Furthermore, we resolve a major confounder in prior work: LinUCB's reported dominance on domain partitions is driven by **semantic intent routing** (achieving 73–82% routing to the gold-class partition vs 37.5–50% chance), rather than pure noise filtering."

---

## 2. The 3-Minute Walkthrough (Poster Order)

### Column 1: Problem & Combinatorial Breakdown (Sections 1 & 2)
1. **The ICL Noise Vulnerability (Section 1):**  
   "Demonstration exemplars strongly guide LLM output. When exemplar pools contain mislabeled data, classification accuracy drops sharply to worst-case 16–22% vs random clean shots 61–74%. Prior work framed partition selection as an online multi-armed bandit, updating scores based on binary LLM reward feedback ($r \in \{0, 1\}$)."
2. **The Combinatorial Bottleneck in $k$-UCB (Section 2):**  
   "Here lies the structural flaw of $k$-UCB: selecting $n_p=3$ partitions from $n$ shards creates $\binom{n}{n_p}$ superarms. Look at the numbers in Section 2: at $n=8$, there are 56 superarms; at $n=64$, there are 41,664; and at $n=100$, there are **161,700 superarms**. Over a finite test stream of 320 queries, $k$-UCB's coverage drops from 100% down to **6.45%** ($n_p=2$) and **0.20%** ($n_p=3$)."
3. **The Deterministic Sweep Diagnostic:**  
   "Why does $k$-UCB fail to explore? In Algorithm 1, unpulled superarms all share the maximal initial UCB bound ($+\infty$). Because `argmax` breaks ties by returning the lowest index, $k$-UCB deterministically sweeps combinations in index order: `[0,1,2]`, `[0,1,3]`, `[0,1,4]`... In our seed 0 diagnostic at $n=64, n_p=3$, Shard 0 appeared in **320 of 320 pulls (100%)**, while trailing shards 10..63 appeared only 5–6 times. Scoring 161,700 superarms slows observed progress-bar throughput to 3.88 it/s (vs 8.90 it/s for `IndepUCB`)."

### Column 2: Methods, Setup & Scaling Accuracies (Sections 3 & 4)
4. **Bandit Algorithms Evaluated (Section 3):**  
   "In Section 3, we compare six approaches: non-contextual UCB1, combinatorial $k$-UCB, our proposed linear-cost `IndepUCB` ($O(n)$ per step), contextual LinUCB using query embeddings, Neural Policy Gradient ($\pi_\theta(a|x) = \text{softmax}(\text{MLP}(x))$), and two baselines: $k$-NN semantic retrieval and Oracle domain routing."
5. **Experimental Protocol (Section 4):**  
   "To ensure real learning rather than zero-shot prior guessing, we evaluated Banking-77 across 8 fine-grained intent classes relabeled to abstract tokens (`Class_1` .. `Class_8`). We split 1,104 training examples into $n \in \{8, 16, 32, 64, 100\}$ IID shards with four noise tiers ($[0.0, 0.2, 0.5, 0.8]$), evaluated with Qwen2.5-7B-Instruct across 10 independent seeds with Holm-Bonferroni correction."
6. **Scaling Accuracy Findings (Figure B) & Quarterly Dynamics (Figure F):**  
   "Look at Figure B: across all shard counts, `IndepUCB` matches or exceeds $k$-UCB in accuracy. In fact, after Holm correction, `IndepUCB` is statistically significantly better at $n=8$ (both $n_p$: +3.8 points, Holm p=0.0045; +5.0 points, Holm p=0.0154) and at $n=16, n_p=3$ (+7.8 points, Holm p=0.0352). At $n=100, n_p=3$, the difference is $-3.0$ points (Holm p=1.000, not significant). $k$-UCB never significantly outperforms `IndepUCB`."  
   "Below Figure B, look at our quarterly learning curves in Figure F: the top row tracks accuracy across stream quarters (Q1 to Q4), while the bottom row plots the paired per-seed gap ($\text{LinUCB} - \text{IndepUCB}$). LinUCB's lead over IndepUCB is largest at large $n$ and already present early (in Q1, +10.00 pts at $n=100$ and +12.38 pts at $n=64$). Over the stream, this gap expands significantly at $n=100$ (widening by +14.37 points from Q1 to Q4, raw $p = 0.0003$) and tends to grow at $n=16$ (+7.75 points, 95% CI includes 0); it remains flat after an early jump at $n=64$ (+6.38 points), while at $n=8$ the lead shrinks (-5.50 points). These quarterly dynamics are exploratory across 10 seeds; the underlying feature mechanism is unmeasured."  
   "Finally, $k$-NN semantic instance retrieval achieves ~80-83% accuracy, establishing the highest empirical accuracy in these runs, but performs no noise filtering."

### Column 3: Domain Routing & Takeaways (Sections 5 & 6)
7. **Resolving the LinUCB Confounder (Section 5):**  
   "In earlier experiments on domain partitions, LinUCB performed best. But look at Figures C and D: on domain partitions, LinUCB selects the partition holding the query's gold class in **82.1%** (4 domains) and **72.9%** (8 intents) of queries, compared to chance levels of 50.0% and 37.5%. LinUCB's advantage is driven by semantic query routing, not noise avoidance. Furthermore, $k$-NN (70.8 / 71.1%) and Oracle (61.2 / 68.4%) beat LinUCB."
8. **Summary of Safe Claims (Section 6):**  
   "In summary:
   1. $k$-UCB Failure is Structural: Coverage collapses to 0.20% at $n=100, n_p=3$; observed throughput 3.88 vs 8.90 it/s at $n=100, n_p=3$ (progress bar, not logged timing); accuracy shows no significant difference from IndepUCB at scale.
   2. IndepUCB ($O(n)$) removes the enumeration cost: Significantly better than $k$-UCB in 3 of 12 settings after Holm correction; no significant difference in the rest; $k$-UCB never significantly better.
   3. LinUCB: routing on domains; IID gain unexplained: (a) domain routing: LinUCB routes to the gold partition 82.1% / 72.9% vs chance 50% / 37.5%; (b) IID shards: LinUCB is ~+15 points above uniform already in Q1 at $n=100$, mechanism not measured.
   4. $k$-NN accuracy is not explained by duplicates: Exactly 0 test-train duplicates (0.0%). 28% of test queries have a near-duplicate (cosine > 0.95); accuracy is similar without them (remains 71.0–82.7%)."

---

## 3. Anticipated Questions & Rigorous Answers

### Q1: Why use abstract class labels (`Class_1` .. `Class_8`) instead of natural intent names?
> **Answer:**  
> "Modern foundation models like Qwen2.5-7B possess strong pre-trained semantic priors. If the prompt contains natural names like `'card_arrival'` or `'exchange_rate'`, the LLM can predict the correct intent zero-shot without looking at the retrieved in-context demonstrations.  
> By masking the class names with abstract tokens (`Class_1` through `Class_8`), we neutralize zero-shot priors. The model is forced to bind query semantics to class labels purely from the in-context demonstration shots. This creates a clean vs corrupted demonstration performance spread (worst-case 16–22% vs random clean shots 61–74%), providing an uncompromised testbed for partition selection."

### Q2: If $k$-UCB coverage collapses to 0.20% at $n=100$, why doesn't its accuracy collapse to zero?
> **Answer:**  
> "This is a key empirical insight. In our scaling setup, shards are partitioned **IID**: each shard contains an independent, representative mixture of all 8 classes. Even though $k$-UCB degenerates into pulling Shard 0 on 100% of queries due to deterministic tie-breaking, Shard 0 still contains valid demonstrations for every class.  
> Therefore, $k$-UCB's failure at scale is **structural and computational**—loss of combinatorial coverage and an observed progress-bar throughput drop from 8.90 down to 3.88 it/s—rather than an accuracy collapse. `IndepUCB` matches or exceeds this accuracy with strictly $O(n)$ evaluations per query."

### Q3: Why was Holm-Bonferroni correction necessary, and what did it change?
> **Answer:**  
> "We performed 36 paired hypothesis tests across 12 configurations and 3 pairwise comparisons. Under standard uncorrected testing at $\alpha = 0.05$, multiple testing inevitably produces false positives.  
> Applying the step-down Holm-Bonferroni procedure controls the family-wise error rate across all 36 tests. As a result, several differences with raw $p < 0.05$ (e.g. `IndepUCB - kUCB` at $n=16, n_p=2$, raw $p=0.0290, p_{\text{Holm}}=0.3191$) were correctly identified as nominal, while true strong effects (such as `IndepUCB - kUCB` at $n=16, n_p=3$, $p_{\text{Holm}}=0.0352$, and LinUCB's advantages at $n=64$ and $n=100$) were verified as statistically significant."

### Q4: Why does LinUCB still outperform `IndepUCB` on IID shards if the shards have identical class distributions?
> **Answer:**  
> "On IID shards, class distributions are identical, but individual shards still contain different text exemplars. In our quarterly learning curve analysis (Table 3.2), LinUCB is already **~+15 points above uniform in Quarter 1** ($n=100, n_p=2$), before online reward feedback could have produced significant adaptation.  
> The exact mechanism was not directly measured in these experiments. Our working hypothesis is that LinUCB's query ridge regression correlates semantic query features with shard-specific exemplar representations and noise tiers immediately from the first quarter. The LinUCB-minus-IndepUCB gap grows monotonically from +4.0 (n=8) to +18.7 (n=100) at $n_p=2$. To definitively isolate this in future work, we propose testing LinUCB with a **constant non-informative context vector** ($x_t = \mathbf{1}$)."

### Q5: Is $k$-NN retrieval's strong performance caused by test-train data leakage?
> **Answer:**  
> "We explicitly verified this using dense sentence embeddings (`all-MiniLM-L6-v2`) on all 320 test queries against the 1,104 training pool.  
> Exactly **0 out of 320 queries (0.0%)** are exact string duplicates. While 28.1% of queries have close paraphrase neighbors (cosine similarity > 0.95), $k$-NN retrieval achieves **70.96% to 82.74%** accuracy even on the 230 test queries with cosine $\le 0.95$. This demonstrates that dense semantic proximity retrieval is genuinely effective at finding label-consistent demonstrations, even without near-duplicate leakage."

### Q6: What is the main scientific contribution of this work?
> **Answer:**  
> "Threefold:
> 1. **Empirical debunking of combinatorial $k$-UCB**: We show that combinatorial superarms are impractical beyond small $n$, collapsing into a deterministic index sweep.
> 2. **Validation of `IndepUCB` as a scalable replacement**: We prove that linear-cost independent partition selection matches or exceeds $k$-UCB accuracy with $O(n)$ complexity and no memory bloat.
> 3. **Disentangling routing from noise filtering**: We show that LinUCB's apparent dominance in domain partitioning is an artifact of semantic intent routing, providing a clear experimental methodology (IID shards with noise tiers) to evaluate true noise robustness in ICL."

### Q7: Why does the Uniform baseline drop to 40.8% at $n=100, n_p=3$ while remaining ~47–48% for $n=8..64$?
> **Answer:**  
> "We conducted a complete row-by-row verification across all 10 seeds and all methods. All methods evaluated the **exact same 320 test query stream** in identical order with identical noise tiers (0 mismatches across 960 run logs).  
> The drop is caused by a structural artifact of how Uniform samples shots:
> - In `run_experiments.py`, Uniform selects a **single** partition ($n_p=1$) and draws all $k=12$ shots from that one shard, whereas bandits pool shots across $n_p=3$ partitions.
> - At $n=100$, 1,104 examples split into 100 shards gives 96 shards of size 11 ($< 12$).
> - Because $k=12 > 11$, `sample_shots` triggers sampling with replacement (`replace=True`).
> - Uniform draws $k=12$ shots with replacement from a single 11-example shard, yielding duplicate shots and an expected **7.68 distinct examples** covering only **5.09 distinct classes** out of 8 in the 12-shot prompt (computed via Monte Carlo simulation across 100,000 shard partitions).
> - Because labels are abstract tokens (`Class_1`..`Class_8`), missing classes starve the LLM of in-context supervision, causing deterministic errors whenever those classes appear in test queries.
> - At $n=64$, shard size is 17 ($> 12$), sampling is without replacement, and shards cover **7.19 / 8 classes** on average (Uniform accuracy: 48.3%).
> - At $n_p=2, k=8$, $8 \le 11$, so no truncation occurs and Uniform accuracy remains flat (~39.6% at $n=100$ vs 40.1% at $n=64$).
> - In contrast, bandit methods pool across $n_p = 3$ shards, drawing $k/n_p = 4$ distinct shots without replacement from each shard ($4 \le 11$), ensuring **12 distinct examples** and covering an average of **6.40 distinct classes** out of 8 in the prompt."

### Q8: Is Uniform a fair baseline?
> **Answer:**  
> "No. Uniform is not a fair baseline for evaluating noise filtering because it draws all $k$ shots from **a single partition** ($n_p=1$), whereas all bandit methods (`IndepUCB`, $k$-UCB, LinUCB, Neural-PG) pool shots across **$n_p$ partitions** ($n_p=2$ or $3$).  
> This introduces two structural confounds:
> 1. **Domain suites:** Each partition is class-disjoint. Uniform's single partition contains only 1 or 2 intent classes out of 8 (chance 25% or 12.5% of containing the gold class), whereas $n_p$-pooling bandits cover 50% or 37.5% of classes by construction.
> 2. **Small shards ($n=100, n_p=3$):** Shard size is 11, so $k=12 > 11$ forces sampling with replacement, duplicate prompt shots, and class starvation (5.09 vs 6.40 distinct classes in the prompt).  
> 
> Crucially, when we mark these 3 confounded settings with daggers (†) and re-run Holm-Bonferroni correction over the unconfounded 33-test family, **0 of the remaining 9 `IndepUCB - Uniform` comparisons remain statistically significant** (all 9 are nominal or not significant; closest is $n=8, n_p=3$ with $p_{\text{Holm}} = 0.0541$).  
> Therefore, we do not claim accuracy gains over Uniform as evidence of noise avoidance; we rest our noise avoidance claims on directly measured shard noise rates (Table 2.1) and compare bandit methods against each other (`IndepUCB` vs $k$-UCB). The proper control is our planned **`uniform_np` control** (choosing $n_p$ partitions uniformly at random without replacement, with the same shot allocation as the bandits), which is slated for the next run."


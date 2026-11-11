# Project Notes & Decision Log

## Step 0: Banking-77 (8-Class Subset) Data Verification

- **Script used**: `check_data_step0.py` on cluster (`/data/pulakurthi.manohar/In_context_learning`).
- **Dataset**: `PolyAI/banking77` filtered to 8 fine-grained intent classes across 4 operational domains:
  - Cards: `card_arrival` (Class_1), `card_not_working` (Class_2)
  - Transfers: `cancel_transfer` (Class_3), `transfer_fee_charged` (Class_4)
  - TopUp: `top_up_failed` (Class_5), `automatic_top_up` (Class_6)
  - Exchange: `exchange_rate` (Class_7), `fiat_currency_support` (Class_8)
- **Exact Sizes Verified**:
  - **Train pool**: 1,104 examples (Cards: 265, Transfers: 329, TopUp: 272, Exchange: 238)
    - Class 1: 153, Class 2: 112, Class 3: 157, Class 4: 172
    - Class 5: 145, Class 6: 127, Class 7: 112, Class 8: 126
  - **Test set**: 320 examples (exactly 40 per class, 80 per domain)
- **Note on 77 classes**: The full Banking-77 dataset has 10,003 train and 3,080 test examples. We strictly adhere to the 8-class subset (1,104 train, 320 test) as specified.

---

## Step 1: Design & Implementation Decisions

### 1. Shard Partitioning & Size Requirements (Step 1.e)
- **Train Pool at $n=100$ Shards**:
  - $1104 / 100 \implies 96$ shards with 11 examples and 4 shards with 12 examples.
- **Minimum Requirement**:
  - For $k=8, n_p=2$: each partition provides $k / n_p = 4$ examples. $11 \ge 4$.
  - For $k=12, n_p=3$: each partition provides $k / n_p = 4$ examples. $11 \ge 4$.
  - Requirement: each shard must have $\ge \lceil k / n_p \rceil$ examples.
  - If any shard has fewer than $\lceil k / n_p \rceil$ examples, `make_shard_partitions` raises a clear `ValueError`.
- **Behavior of `sample_shots_from_parts` / `sample_shots`**:
  - `sample_shots` implements `replace = len(pool) < k`.
  - When a shard has at least $\lceil k / n_p \rceil$ examples, `len(pool) >= take`, so examples are sampled **without replacement**.
  - If a partition were smaller than its required share (`len(pool) < take`), `sample_shots` automatically falls back to sampling **with replacement** (`replace=True`).

### 2. Independent UCB (`indep_ucb`) (Step 1.a)
- **Formulation**: Non-contextual bandit maintaining independent UCB1 statistics per partition ($i \in \{0, \dots, n-1\}$).
- **Selection**: At each step, evaluates all partition UCB scores and selects the top-$n_p$ partitions. Complexity is $\mathcal{O}(n)$, eliminating the combinatorial $\binom{n}{n_p}$ explosion of $k$-UCB.
- **Credit Assignment Limitation**: When reward $r \in \{0, 1\}$ is received from the combined shots of the selected $n_p$ partitions, the exact same reward is credited to each of the $n_p$ partitions. While coarse, this retains linear scaling while still allowing uncorrupted partitions to accumulate higher average rewards.

### 3. $k$-NN Retrieval Baseline (`knn_retrieval`) (Step 1.a)
- **Formulation**: Non-bandit, instance-based retrieval baseline.
- **Mechanism**: Pre-embeds all training examples from the pooled noisy partitions using `TextEmbedder` (`all-MiniLM-L6-v2`).
- **Selection**: For each test query embedding, computes cosine similarity against all pooled training examples, and retrieves the top-$k$ nearest examples.
- **Ordering**: Returns shots nearest-first (in descending order of cosine similarity).

### 4. Oracle Domain Baseline (`oracle_domain`) (Step 1.a)
- **Formulation**: Domain-only baseline that routes each query directly to the partition containing its gold class.
- **Decision**: Analogous to `worst_case` and `uniform`, the query is routed to the partition $p^*$ containing its gold class, and $k$ shots are sampled from $p^*$ with noisy labels. If applied to non-domain partitions where classes are split across partitions, raises an error.

### 5. $k$-UCB Combinatorial Safety Cap (Step 1.b)
- **Safety Rule**: Before instantiating combinations in `KUCB`, compute $C(n, n_p) = \binom{n}{n_p}$ using `math.comb`.
- If $C(n, n_p) > \text{max\_superarms}$ (default 2,000,000), raise `KUCBInfeasibleError`.
- The runner catches `KUCBInfeasibleError` and writes `{"status": "infeasible", "superarms": C}` to the result JSON, allowing other methods to continue seamlessly.

### 6. Neural Policy Bandit Reproducibility (Step 1.c)
- Replaced unseeded `np.random.choice` with `self.rng = np.random.default_rng(seed)`.
- Guarantees identical arm selection sequences across runs with identical seeds.

### 7. Tiers Noise Mode (Step 1.f)
- For $n$ shards, cycles through $[0.0, 0.2, 0.5, 0.8]$ equally:
  - At $n=100$: exactly 25 shards at 0.0, 25 at 0.2, 25 at 0.5, 25 at 0.8.
- Shuffled randomly across shard indices using `np.random.default_rng(seed)`.

### 8. Logging & Observability (Step 1.d)
- For every method run, logs:
  - `wall_clock_seconds` via `time.perf_counter()`
  - `peak_cpu_mb` via `tracemalloc`
  - `peak_gpu_mb` via `torch.cuda.max_memory_allocated()`
  - `n_arms`: number of base arms / superarms evaluated by the method.

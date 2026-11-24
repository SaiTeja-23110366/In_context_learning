# Comprehensive Technical Guide: From UCB to Oracle
## A Deep Dive into Demonstration Partition Selection for In-Context Learning

This reference document provides an exhaustive, mathematically rigorous, and code-grounded analysis of all **10 partition selection strategies and evaluative baselines** implemented and analyzed in this research project.

Every algorithm is explained from first principles, mapped directly to its implementation in the codebase (`In_context_learning/src/bandit.py`, `In_context_learning/src/data_utils.py`, `In_context_learning/scripts/run_experiments.py`), and analyzed in light of verified empirical findings from `In_context_learning/ANALYSIS.md` and `poster/numbers_check.md`.

---

## Table of Contents
1. [The In-Context Demonstration Selection Problem](#1-the-in-context-demonstration-selection-problem)
2. [Method 1: Standard UCB1 (Foundational Multi-Armed Bandit)](#2-method-1-standard-ucb1)
3. [Method 2: $k$-UCB (Combinatorial Superarm Bandit)](#3-method-2-k-ucb-combinatorial-superarm-bandit)
4. [Method 3: IndepUCB (Independent Top-$n_p$ Ranking Bandit)](#4-method-3-indepucb-independent-top-n_p-ranking-bandit)
5. [Method 4: LinUCB (Disjoint Linear Contextual Bandit)](#5-method-4-linucb-disjoint-linear-contextual-bandit)
6. [Method 5: Neural-PG (Neural Policy Gradient with REINFORCE)](#6-method-5-neural-pg-neural-policy-gradient-with-reinforce)
7. [Method 6: $k$-NN Retrieval (Dense Semantic Instance Selection)](#7-method-6-k-nn-retrieval-dense-semantic-instance-selection)
8. [Method 7: Uniform Sampling Baseline (Single Partition Control)](#8-method-7-uniform-sampling-baseline-single-partition-control)
9. [Method 8: Worst-Case Baseline (Adversarial Maximum-Noise Floor)](#9-method-8-worst-case-baseline-adversarial-maximum-noise-floor)
10. [Method 9: Random Clean Shots Control ("no_noise" Unbiased Baseline)](#10-method-9-random-clean-shots-control-no_noise-unbiased-baseline)
11. [Method 10: Oracle Domain Routing (Ground-Truth Semantic Routing)](#11-method-10-oracle-domain-routing-ground-truth-semantic-routing)
12. [Master Cross-Method Comparative Matrix](#12-master-cross-method-comparative-matrix)
13. [Key Empirical Takeaways & Methodological Confounders](#13-key-empirical-takeaways--methodological-confounders)

---

## 1. The In-Context Demonstration Selection Problem

### 1.1 The Operational Setting
In-Context Learning (ICL) prompts a frozen language model $\mathcal{M}$ (such as Llama-3-8B or Qwen-2.5-7B/14B) with $k$ labeled demonstrations $(s_1, \dots, s_k)$ prepended to a target test query $q_t \in \mathcal{X}$:
$$\operatorname{Prompt}(C_t) = [s_1, s_2, \dots, s_k, q_t]$$
The model predicts a discrete class label $\hat{y}_t \in \mathcal{Y}$.

In decentralized, federated, or web-scale data pools, training demonstrations are partitioned across $n$ distinct data shards or domain subsets $\mathcal{P} = \{P_1, P_2, \dots, P_n\}$. In real-world data collection, these partitions suffer from heterogeneous label corruptions. Each partition $P_i$ possesses an unobserved ground-truth noise rate $\eta_i \in [0, 1]$, representing the probability that an example's label has been randomly corrupted to an incorrect class.

### 1.2 The Sequential Decision Game
At each step $t \in \{1, 2, \dots, T\}$ across a stream of test queries:
1. **Query Arrival**: Query $q_t$ arrives. Its true label $y_t^* \in \mathcal{Y}$ is hidden from the selector.
2. **Action Selection**: The selector selects a combination of $n_p$ partitions $S_t \subset \{1, \dots, n\}$ where $|S_t| = n_p$. (For single-partition baselines, $n_p = 1$; for combinatorial bandits, $n_p \in \{2, 3\}$).
3. **Shot Sampling**: A prompt budget of $k$ shots (e.g., $k=8$ or $k=12$) is distributed across the chosen partitions. Specifically, `sample_shots_from_parts` draws $k / n_p$ demonstrations from each partition in $S_t$.
4. **LLM Evaluation**: The frozen LLM generates a prediction $\hat{y}_t = \mathcal{M}(\operatorname{Prompt}(C_t))$.
5. **Reward Feedback**: The environment provides a binary scalar reward:
   $$R_t = \mathbb{I}(\hat{y}_t = y_t^*) \in \{0.0, 1.0\}$$
6. **Online Policy Update**: The selector updates its internal parameters using the tuple $(q_t, S_t, R_t)$ to improve partition selection on future queries.

---

## 2. Method 1: Standard UCB1

### 2.1 Conceptual Intuition
The Upper Confidence Bound algorithm (Auer, Cesa-Bianchi, & Fischer, 2002) is the quintessential solution to the exploration-exploitation dilemma in stochastic multi-armed bandits. It implements the principle of **Optimism in the Face of Uncertainty**: an arm is chosen if its historical average reward is high (exploitation) *or* if it has been played infrequently and therefore has high uncertainty (exploration).

### 2.2 Mathematical Formulation
For $n$ arms (partitions), let $N_i(t)$ denote the number of times partition $i$ has been selected up to step $t$, and let $\hat{\mu}_i(t)$ be its empirical mean reward:
$$\hat{\mu}_i(t) = \frac{1}{N_i(t)} \sum_{\tau=1: a_\tau = i}^t R_\tau$$

The UCB1 score for partition $i$ at step $t$ is defined as:
$$U_i(t) = \hat{\mu}_i(t) + c \sqrt{\frac{\ln t}{N_i(t)}}$$
where $c > 0$ is the exploration parameter (in theoretical proofs $c = \sqrt{2} \approx 1.414$; in this repository, $c = 2.0$).

At step $t$, the policy selects:
$$a_t = \operatorname{argmax}_{i \in \{1, \dots, n\}} U_i(t)$$

Upon observing scalar reward $R_t \in \{0, 1\}$, parameters are updated online via running averages:
$$N_{a_t}(t+1) = N_{a_t}(t) + 1$$
$$\hat{\mu}_{a_t}(t+1) = \frac{(N_{a_t}(t)) \hat{\mu}_{a_t}(t) + R_t}{N_{a_t}(t+1)}$$

### 2.3 Code Implementation
Located in [`src/bandit.py#L61-L88`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/bandit.py#L61-L88):
```python
class UCB1:
    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        self.n_arms = n_arms
        self.c = c
        self.counts = [1] * n_arms   # init to 1 as per paper Algorithm 1
        self.means = [0.0] * n_arms  # init to 0 as per paper Algorithm 1
        self.t = 0

    def select(self) -> int:
        self.t += 1
        scores = [
            self.means[i] + self.c * math.sqrt(math.log(self.t) / self.counts[i])
            for i in range(self.n_arms)
        ]
        return int(max(range(self.n_arms), key=lambda i: scores[i]))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] = ((n - 1) * self.means[arm] + reward) / n
```

### 2.4 Complexity and Characteristics
- **Time Complexity**: $O(n)$ to compute scores across all arms at each step.
- **Space Complexity**: $O(n)$ storing integer counts and float means.
- **Credit Assignment**: Deterministic and exact. The chosen partition $a_t$ receives the entire reward $R_t$.
- **Limitations**: Only selects a single partition ($n_p = 1$). Blind to query semantics (non-contextual).

---

## 3. Method 2: $k$-UCB (Combinatorial Superarm Bandit)

### 3.1 Conceptual Intuition
When prompts require pooling demonstrations from multiple partitions ($n_p > 1$), partitions may interact synergistically or destructively. $k$-UCB formalizes this by treating **every distinct combination of $n_p$ partitions as a unique, atomic superarm**. If $n=10$ and $n_p=2$, there are $\binom{10}{2} = 45$ superarms. $k$-UCB instantiates a full UCB1 bandit over these 45 superarms.

### 3.2 Mathematical Formulation
Let $\mathcal{A}_{\text{super}} = \{S \subset \{1, \dots, n\} \mid |S| = n_p\}$. The total number of superarms is:
$$K_{\text{super}} = \binom{n}{n_p} = \frac{n!}{n_p! (n - n_p)!}$$

Let $j \in \{0, \dots, K_{\text{super}} - 1\}$ index the superarm tuple $S_j = (a_1, \dots, a_{n_p})$. $k$-UCB tracks counts $N_j(t)$ and empirical means $\hat{\mu}_j(t)$ for each combination $j$:
$$U_j(t) = \hat{\mu}_j(t) + c \sqrt{\frac{\ln t}{N_j(t)}}$$
Action selection:
$$j_t = \operatorname{argmax}_{j \in \{0, \dots, K_{\text{super}}-1\}} U_j(t), \quad S_t = S_{j_t}$$

Update:
$$N_{j_t} \leftarrow N_{j_t} + 1, \quad \hat{\mu}_{j_t} \leftarrow \frac{(N_{j_t} - 1)\hat{\mu}_{j_t} + R_t}{N_{j_t}}$$

### 3.3 Code Implementation
Located in [`src/bandit.py#L90-L137`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/bandit.py#L90-L137):
```python
class KUCB:
    def __init__(self, n_partitions: int, n_p: int, c: float = 2.0, max_superarms: int = 2_000_000):
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.max_superarms = max_superarms
        n_superarms = math.comb(n_partitions, n_p)
        if n_superarms > max_superarms:
            raise KUCBInfeasibleError(n_partitions, n_p, n_superarms, max_superarms)
        self.superarms = list(itertools.combinations(range(n_partitions), n_p))
        self.ucb = UCB1(len(self.superarms), c=c)

    def select(self) -> tuple[int, tuple[int, ...]]:
        idx = self.ucb.select()
        return idx, self.superarms[idx]

    def update(self, idx: int, reward: float) -> None:
        self.ucb.update(idx, reward)
```

### 3.4 The Combinatorial Explosion & Coverage Collapse
While theoretically appealing because it models joint partition interactions, $k$-UCB suffers a catastrophic combinatorial bottleneck as $n$ scales:
- At $n=8, n_p=2$: $\binom{8}{2} = 28$ superarms. Over $T=320$ test queries, each arm can be pulled $\sim 11.4$ times.
- At $n=16, n_p=2$: $\binom{16}{2} = 120$ superarms. Over $T=320$ queries, each arm is pulled $\sim 2.7$ times.
- At $n=64, n_p=2$: $\binom{64}{2} = 2,016$ superarms. Over $T=320$ queries, at most $320 / 2,016 = 15.9\%$ of arms are ever pulled once.
- At $n=100, n_p=3$: $\binom{100}{3} = 161,700$ superarms. Over $T=320$ queries, $k$-UCB explores only:
  $$\frac{320}{161,700} = 0.1979\% \approx \mathbf{0.20\% \text{ coverage!}}$$

#### The Deterministic Order Sweep Failure Mode
Because all superarms are initialized with identical prior values ($N_j=1, \hat{\mu}_j=0.0$), Python's `max(..., key=...)` breaks ties by picking the first index encountered. Consequently:
- Combinations generated by `itertools.combinations(range(100), 3)` begin with `(0, 1, 2)`, `(0, 1, 3)`, `(0, 1, 4)`, etc.
- Shard 0 is included in every single early combination. As verified in the logs, Shard 0 appears in **100% of all 320 test queries**.
- Shards at the tail of the index range (e.g., shards 10 to 63) appear only **5 to 6 times total**, leaving them completely unestimated.
- Furthermore, scanning 161,700 arms per query degrades progress-bar throughput from 8.90 it/s down to 3.88 it/s.

---

## 4. Method 3: IndepUCB (Independent Top-$n_p$ Ranking Bandit)

### 4.1 Conceptual Intuition
To overcome the $O(\binom{n}{n_p})$ combinatorial wall, **IndepUCB** discards joint superarm modeling. Instead, it maintains $n$ independent single-partition UCB1 estimators. When a prompt requires $n_p$ partitions, IndepUCB scores each partition individually and selects the **top-$n_p$ highest-scoring partitions**.

### 4.2 Mathematical Formulation
For each individual partition $i \in \{1, \dots, n\}$, IndepUCB maintains counts $N_i(t)$ and running mean reward $\hat{\mu}_i(t)$.
The score for partition $i$ is:
$$U_i(t) = \hat{\mu}_i(t) + c \sqrt{\frac{\ln t}{N_i(t)}}$$

At step $t$, the policy sorts the scores and selects the top $n_p$ partition indices:
$$S_t = \operatorname{arg top-}n_p \{ U_i(t) \}_{i=1}^n$$

#### The Coarse Credit Assignment Mechanism
When prompt $C_t$ is evaluated and yields binary reward $R_t \in \{0, 1\}$, the environment does not indicate which of the $n_p$ partitions contributed positively or negatively. IndepUCB applies a **uniform credit broadcast**:
$$\forall a \in S_t: \quad N_a \leftarrow N_a + 1, \quad \hat{\mu}_a \leftarrow \frac{(N_a - 1)\hat{\mu}_a + R_t}{N_a}$$

### 4.3 Code Implementation
Located in [`src/bandit.py#L139-L178`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/bandit.py#L139-L178):
```python
class IndepUCB:
    def __init__(self, n_partitions: int, n_p: int = 2, c: float = 2.0) -> None:
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.c = c
        self.counts = [1] * n_partitions
        self.means = [0.0] * n_partitions
        self.t = 0

    def select(self) -> tuple[int, tuple[int, ...]]:
        self.t += 1
        scores = [
            self.means[i] + self.c * math.sqrt(math.log(self.t) / self.counts[i])
            for i in range(self.n_partitions)
        ]
        sorted_indices = sorted(range(self.n_partitions), key=lambda i: scores[i], reverse=True)
        combo = tuple(sorted_indices[: self.n_p])
        return combo[0], combo

    def update(self, combo: Sequence[int], reward: float) -> None:
        for arm in combo:
            self.counts[arm] += 1
            n = self.counts[arm]
            self.means[arm] = ((n - 1) * self.means[arm] + reward) / n
```

### 4.4 Complexity and Empirical Significance
- **Time Complexity**: $O(n \log n)$ (or $O(n + n_p \log n_p)$ using a min-heap) to select the top-$n_p$ arms. At $n=100$, evaluating 100 floats takes $< 0.1$ ms.
- **Space Complexity**: $O(n)$ floats and integers.
- **Empirical Validation**: In rigorous pairwise comparisons across all 12 evaluated settings in `ANALYSIS.md`:
  - IndepUCB is **statistically significantly better than $k$-UCB in 3 of 12 settings** after Holm step-down correction.
  - There is **no significant difference in the remaining 9 settings**.
  - **$k$-UCB is never significantly better than IndepUCB in any setting.**
  - Replacing $\binom{n}{n_p}$ superarms with $O(n)$ independent ranking eliminates combinatorial collapse without sacrificing accuracy.

---

## 5. Method 4: LinUCB (Disjoint Linear Contextual Bandit)

### 5.1 Conceptual Intuition
Non-contextual bandits (UCB1, $k$-UCB, IndepUCB) assume that arm quality is global. However, different queries have different semantic requirements: a banking query about "stolen credit cards" requires card-related demonstrations, whereas a query about "SWIFT wire transfers" requires foreign exchange demonstrations.

**LinUCB** (Li, Chu, Langford, & Schapire, 2010) incorporates context. For each query $q_t$, it computes a dense semantic embedding vector $x_t \in \mathbb{R}^d$. It assumes the expected reward of pulling partition $a$ for query $x_t$ is a linear function of $x_t$:
$$\mathbb{E}[R_t \mid x_t, a] = x_t^T \theta_a^*$$
where $\theta_a^* \in \mathbb{R}^d$ is an unknown weight vector specific to partition $a$.

### 5.2 Mathematical Formulation

#### 1. Disjoint Ridge Regression
For each partition $a \in \{0, \dots, n-1\}$, LinUCB maintains:
- A design covariance matrix $A_a \in \mathbb{R}^{d \times d}$, initialized to $A_a = \lambda I_d$ (where $\lambda = \text{reg} = 1.0$).
- A response vector $b_a \in \mathbb{R}^{d \times 1}$, initialized to $\mathbf{0}$.

The ridge regression estimate of $\theta_a$ is:
$$\hat{\theta}_a = A_a^{-1} b_a$$

#### 2. Upper Confidence Bound with Feature Uncertainty
For a new query feature vector $x_t \in \mathbb{R}^{d \times 1}$, the estimated expected reward is $x_t^T \hat{\theta}_a$. The variance of this estimate is given by $x_t^T A_a^{-1} x_t$. The upper confidence bound score is:
$$p_a(x_t) = x_t^T \hat{\theta}_a + \alpha \sqrt{x_t^T A_a^{-1} x_t}$$
where $\alpha \ge 0$ controls the degree of exploration (in our configs, $\alpha = 1.0$).

To select $n_p$ partitions, LinUCB computes $p_a(x_t)$ for all $a \in \{0, \dots, n-1\}$ and selects the top $n_p$:
$$S_t = \operatorname{arg top-}n_p \{ p_a(x_t) \}_{a=0}^{n-1}$$

#### 3. Online Sherman-Morrison Rank-1 Update
Directly inverting $A_a \in \mathbb{R}^{d \times d}$ at every step requires $O(d^3)$ operations. LinUCB instead maintains $A_a^{-1}$ directly using the **Sherman-Morrison formula**:
$$(A + u v^T)^{-1} = A^{-1} - \frac{A^{-1} u v^T A^{-1}}{1 + v^T A^{-1} u}$$
Setting $u = v = x_t$:
$$A_a^{-1} \leftarrow A_a^{-1} - \frac{(A_a^{-1} x_t) (A_a^{-1} x_t)^T}{1 + x_t^T (A_a^{-1} x_t)}$$
$$b_a \leftarrow b_a + R_t x_t$$
This reduces the inversion cost from $O(d^3)$ to an $O(d^2)$ matrix-vector product.

### 5.3 Code Implementation
Located in [`src/bandit.py#L180-L253`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/bandit.py#L180-L253):
```python
class LinUCB:
    def __init__(self, n_partitions: int, n_p: int = 2, dim: int = 384, alpha: float = 1.0, reg: float = 1.0):
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.dim = dim
        self.alpha = alpha
        self.reg = reg
        self.A_inv = [np.eye(dim, dtype=np.float32) / reg for _ in range(n_partitions)]
        self.b = [np.zeros((dim, 1), dtype=np.float32) for _ in range(n_partitions)]

    def select(self, context: np.ndarray) -> tuple[int, tuple[int, ...]]:
        x = context.reshape(-1, 1).astype(np.float32)
        scores = []
        for a in range(self.n_partitions):
            A_inv_a = self.A_inv[a]
            theta_a = A_inv_a @ self.b[a]
            mean = float((x.T @ theta_a).item())
            var = float((x.T @ A_inv_a @ x).item())
            ucb = mean + self.alpha * math.sqrt(max(0.0, var))
            scores.append(ucb)

        sorted_indices = sorted(range(self.n_partitions), key=lambda i: scores[i], reverse=True)
        combo = tuple(sorted_indices[: self.n_p])
        return combo[0], combo

    def update(self, combo: Sequence[int], context: np.ndarray, reward: float) -> None:
        x = context.reshape(-1, 1).astype(np.float32)
        for a in combo:
            u = self.A_inv[a] @ x
            denom = 1.0 + float((x.T @ u).item())
            self.A_inv[a] -= (u @ u.T) / max(denom, 1e-6)
            self.b[a] += float(reward) * x
```

### 5.4 Complexity and Key Empirical Discoveries
- **Complexity**: $O(n \cdot d^2)$ per query selection and $O(n_p \cdot d^2)$ per update. With $d=384$ and $n=100$, updates run in milliseconds on CPU.
- **Domain Routing Mechanism**: On domain benchmark suites where partitions correspond to semantic topics, LinUCB routes to the correct gold partition **82.1%** of the time on MMLU (STEM / Humanities / Social Sciences / Other) and **72.9%** on Banking77-Domain (Cards / Transfers / TopUp / Exchange), vastly outperforming random chance (50% and 37.5%).
- **IID Shard Advantage**: On IID shards, LinUCB surpasses IndepUCB by **+16.7 to +18.7 percentage points** at large $n$ ($n=64, 100$).
- **The Early Semantic Phenomenon**: Crucially, quarterly dynamics analysis reveals that LinUCB is already **~+10 points ahead of IndepUCB in Quarter 1** (the first 80 queries) at $n=100$. This indicates that the advantage does not stem purely from asymptotic multi-step online bandit convergence, but from early feature-based partition alignment.

---

## 6. Method 5: Neural-PG (Neural Policy Gradient with REINFORCE)

### 6.1 Conceptual Intuition
Neural-PG replaces linear ridge regression with a non-linear deep neural network parameterized by weights $\theta$. The network maps the continuous query embedding $x_t \in \mathbb{R}^{384}$ directly to partition probability logits. It is optimized online via policy gradients (REINFORCE) using the LLM's binary success signal as a reinforcement reward.

### 6.2 Mathematical Formulation

#### 1. Architecture & Policy Distribution
The policy network $f_\theta: \mathbb{R}^{384} \to \mathbb{R}^n$ is a two-layer Multi-Layer Perceptron (MLP):
$$h_t = \operatorname{ReLU}(W_1 x_t + b_1), \quad W_1 \in \mathbb{R}^{128 \times 384}, \quad b_1 \in \mathbb{R}^{128}$$
$$z_t = W_2 h_t + b_2, \quad W_2 \in \mathbb{R}^{n \times 128}, \quad b_2 \in \mathbb{R}^n$$

The action distribution over all $n$ partitions is computed via temperature-scaled softmax ($\tau = 1.0$):
$$\pi_\theta(a \mid x_t) = \frac{\exp(z_{t, a} / \tau)}{\sum_{j=1}^n \exp(z_{t, j} / \tau)}$$

#### 2. Sampling Actions
During training (`explore=True`), $n_p$ distinct partitions $S_t = \{a_1, \dots, a_{n_p}\}$ are sampled without replacement according to probability vector $\pi_\theta(\cdot \mid x_t)$.

The joint log-probability of the chosen subset is:
$$\log \pi_\theta(S_t \mid x_t) = \sum_{a \in S_t} \log \pi_\theta(a \mid x_t)$$

#### 3. Objective & Loss Function with Moving Baseline
To reduce gradient variance, a running scalar baseline $b_t$ tracks historical rewards:
$$b_t = 0.95 b_{t-1} + 0.05 R_t \quad (b_0 = 0.5)$$
The advantage is $A_t = R_t - b_t$.

To encourage continued exploration, an entropy regularization bonus $\mathcal{H}(\pi_\theta)$ is added:
$$\mathcal{H}(\pi_\theta(\cdot \mid x_t)) = - \sum_{i=1}^n \pi_\theta(i \mid x_t) \log(\pi_\theta(i \mid x_t) + \epsilon)$$

The policy loss minimized via Adam is:
$$\mathcal{L}(\theta) = - \log \pi_\theta(S_t \mid x_t) \cdot (R_t - b_t) - \beta_{\text{ent}} \mathcal{H}(\pi_\theta(\cdot \mid x_t))$$
where $\beta_{\text{ent}} = 0.01$ and learning rate $\eta = 10^{-3}$.

### 6.3 Code Implementation
Located in [`src/bandit.py#L255-L362`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/bandit.py#L255-L362):
```python
class NeuralPolicyBandit:
    def __init__(self, n_partitions: int, n_p: int = 2, dim: int = 384, hidden_dim: int = 128, lr: float = 1e-3):
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_partitions),
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.baseline = 0.5
        self.entropy_weight = 0.01

    def select(self, context: np.ndarray, explore: bool = True):
        x = torch.from_numpy(context).float().unsqueeze(0)
        logits = self.net(x) / self.temp
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        # Sample n_p partitions without replacement
        chosen = self.rng.choice(self.n_partitions, size=self.n_p, replace=False, p=probs.detach().numpy())
        combo = tuple(sorted(chosen))
        log_prob = torch.log(probs[list(combo)] + 1e-12).sum()
        self._last_log_prob, self._last_probs = log_prob, probs
        return combo[0], combo

    def update(self, combo: Sequence[int], reward: float):
        advantage = float(reward) - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * float(reward)
        entropy = -(self._last_probs * torch.log(self._last_probs + 1e-12)).sum()
        loss = -self._last_log_prob * advantage - self.entropy_weight * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 6.4 Why Neural-PG Fails in this Benchmark
In empirical runs, Neural-PG lags far behind LinUCB and IndepUCB, scoring only **40.8% – 43.9%** on IID shards:
1. **Extreme Sample Inefficiency**: The test horizon is only $T=320$ steps. Deep networks trained via gradient descent typically require thousands or tens of thousands of updates to learn robust representations.
2. **High Policy Gradient Variance**: REINFORCE gradients exhibit high variance from single-sample scalar rewards, even with a baseline.
3. **Lack of Parameter Isolation**: In LinUCB, each partition $a$ has an independent $A_a^{-1}$ matrix; updating arm $a$ cannot perturb arm $b$. In Neural-PG, all partitions share the hidden layer $W_1$. A single noisy reward update corrupts features across all arms simultaneously.

---

## 7. Method 6: $k$-NN Retrieval (Dense Semantic Instance Selection)

### 7.1 Conceptual Intuition
Rather than selecting coarse partitions and randomly sampling within them, $k$-NN Retrieval operates at the **fine-grained instance level**. It embeds the entire training corpus into a vector index. Given query $q_t$, it retrieves the $k$ most semantically similar training examples based on cosine similarity and places them directly into the prompt.

### 7.2 Mathematical Formulation
Let $\mathcal{D}_{\text{pool}} = \bigcup_{i=1}^n P_i$ be the union of all training instances (e.g., 1,104 examples in Banking77).
For each training example $d_i = (x_i, \tilde{y}_i)$, compute its normalized embedding:
$$\hat{v}_i = \frac{\phi(x_i)}{\|\phi(x_i)\|_2} \in \mathbb{R}^d$$
For test query $q_t$, compute:
$$\hat{u}_t = \frac{\phi(q_t)}{\|\phi(q_t)\|_2} \in \mathbb{R}^d$$

Cosine similarity to every training instance is computed via a single matrix-vector product:
$$s_i = \hat{u}_t^T \hat{v}_i \quad \forall i \in \{1, \dots, |\mathcal{D}_{\text{pool}}|\}$$

The top-$k$ nearest instances are selected:
$$I_t = \operatorname{arg top-}k \{ s_i \}_{i=1}^{|\mathcal{D}_{\text{pool}}|}$$
Prompt demonstrations: $C_t = \{ (\text{text}_j, \tilde{y}_j) \}_{j \in I_t}$.

### 7.3 Code Implementation
Located in [`scripts/run_experiments.py#L154-L165, L210-L230`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/scripts/run_experiments.py#L154-L165):
```python
# Setup
pooled_noisy_examples = [item for part in noisy_partitions for item in part]
raw_vecs = np.stack([embedder.embed(ex.text) for ex in pooled_noisy_examples])
pooled_vecs = raw_vecs / np.maximum(np.linalg.norm(raw_vecs, axis=1, keepdims=True), 1e-12)

# Per-query execution
q_vec = embedder.embed(ex["text"])
q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-12)
sims = pooled_vecs @ q_vec
top_k_idxs = np.argsort(sims)[::-1][:k]
shots = [(pooled_noisy_examples[idx].text, pooled_noisy_examples[idx].noisy_label) for idx in top_k_idxs]
```

### 7.4 The Double-Edged Sword: Accuracy vs. Noise Blindness
$k$-NN Retrieval is the **highest-performing method in the benchmark**, achieving **~80% to 83% accuracy** on IID shards and **~71%** on domain suites. However, deep analysis reveals two critical facets:

#### 1. The Paraphrase / Near-Duplicate Phenomenon
Why is $k$-NN so effective? An exhaustive check of the training and test splits revealed:
- **0.0% exact string duplicates** between test and train.
- However, **28% of test queries have a near-duplicate training example with cosine similarity $> 0.95$**.
- Dense semantic retrieval locates these close paraphrases, giving the LLM near-identical demonstration examples for classification.

#### 2. Complete Noise Blindness
$k$-NN performs **zero noise filtering**. It retrieves training examples solely based on query text similarity $\hat{u}_t^T \hat{v}_i$, completely ignoring the label $\tilde{y}_i$. If a near-duplicate has a corrupted label, $k$-NN blindly injects the erroneous label into the prompt. Its high performance is driven by semantic proximity, not robustness to label noise.

---

## 8. Method 7: Uniform Sampling Baseline (Single Partition Control)

### 8.1 Conceptual Intuition
The Uniform baseline serves as an unguided control. At each query step, it picks a single partition at random from $\{1, \dots, n\}$ with equal probability ($1/n$), and draws all $k$ prompt demonstrations from that chosen partition.

### 8.2 Mathematical Formulation
$$a_t \sim \operatorname{Uniform}(\{0, \dots, n-1\})$$
$$C_t = \operatorname{sample\_shots}(P_{a_t}, k)$$

### 8.3 Code Implementation
Located in [`scripts/run_experiments.py#L190-L193`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/scripts/run_experiments.py#L190-L193):
```python
elif name == "uniform":
    arm = int(rng.integers(0, len(noisy_partitions)))
    shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)
```

### 8.4 The Critical Methodological Confounder (Sampling with Replacement)
A key methodological discovery in `ANALYSIS.md` centers on how `uniform` interacts with prompt sampling:
1. **Single Partition vs. Pooled Superarms**:
   - `uniform` draws all $k$ shots from **a single partition ($n_p=1$)**.
   - Bandits (`kucb`, `indep_ucb`, `linucb`, `neural_pg`) draw shots pooled across **$n_p=2$ or $n_p=3$ partitions**.
2. **Domain Suites**:
   - In domain benchmarks (where classes are disjointly assigned across partitions), a single partition contains only $1/4$ or $1/8$ of all classes. `uniform` fails because the prompt cannot possibly cover the query's class.
3. **The $n=100, n_p=3, k=12$ Shard Size Collapse**:
   - In Banking77 with $n=100$, 1,104 examples are split across 100 shards. The average shard size is only **11 examples**.
   - Look at `sample_shots` in [`src/data_utils.py#L470`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/src/data_utils.py#L470):
     ```python
     idxs = rng.choice(len(pool), size=k, replace=len(pool) < k)
     ```
   - For `uniform`: $k=12$ and $\text{len}(P_{a_t}) = 11$. Because $\text{len}(pool) < k$ evaluates to `True`, `uniform` is forced to sample **with replacement** for 96% of shards! This results in duplicate shots in the prompt and drops distinct class representation down to **5.09 classes**, causing uniform accuracy to plunge from ~47% down to **40.8%**.
   - For bandits: With $n_p=3$, `sample_shots_from_parts` draws $k / n_p = 12 / 3 = 4$ shots per shard. Because $11 \ge 4$, sampling is **without replacement**, giving the prompt **6.40 distinct classes**.
   - Therefore, the apparent gain of bandits over uniform at $n=100, n_p=3$ is confounded by prompt class coverage and replacement artifacts, not purely noise avoidance.

---

## 9. Method 8: Worst-Case Baseline (Adversarial Maximum-Noise Floor)

### 9.1 Conceptual Intuition
To quantify the vulnerability of the LLM to demonstration corruption, the Worst-Case baseline acts as an adversarial control. It deterministically routes every test query to the partition that has the highest ground-truth noise rate.

### 9.2 Mathematical Formulation
Let $\eta_i$ be the noise rate of partition $i$.
$$a_{\text{worst}} = \operatorname{argmax}_{i \in \{0, \dots, n-1\}} \eta_i$$
$$C_t = \operatorname{sample\_shots}(P_{a_{\text{worst}}}, k)$$

### 9.3 Code Implementation
Located in [`scripts/run_experiments.py#L113, L186-L189`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/scripts/run_experiments.py#L113):
```python
worst_idx = int(np.argmax(rates))
...
elif name == "worst_case":
    arm = worst_idx
    shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)
```

### 9.4 Role in the Empirical Analysis
- **Empirical Floor**: Yields **16% – 22% accuracy** across all configurations.
- Demonstrates that when demonstrations are overwhelmingly mislabeled, in-context learning actively misleads the LLM, reducing accuracy well below random uniform sampling (~47%).

---

## 10. Method 9: Random Clean Shots Control ("no_noise" Unbiased Baseline)

### 10.1 Conceptual Intuition
Often misnamed "Clean Upper Bound", this baseline pools all uncorrupted training examples from all partitions and draws $k$ clean shots uniformly at random.

### 10.2 Mathematical Formulation
Let $\mathcal{D}_{\text{clean}} = \bigcup_{i=1}^n P_i^{\text{clean}}$ where all labels are pristine ($y_i = y_i^*$).
$$C_t = \operatorname{sample\_shots}(\mathcal{D}_{\text{clean}}, k, \text{use\_noisy}=\text{False})$$

### 10.3 Code Implementation
Located in [`scripts/run_experiments.py#L116, L183-L185`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/scripts/run_experiments.py#L116):
```python
clean_pool = [item for part in clean_partitions for item in part]
...
elif name == "no_noise":
    shots = sample_shots(clean_pool, k, rng, use_noisy=False)
```

### 10.4 Why Random Clean Shots is NOT an Upper Bound
- **Empirical Accuracy**: Achieves **61.2% – 73.8% accuracy**.
- **Crucial Distinction**: While all demonstrations are guaranteed to have 100% correct labels, they are selected **uniformly at random** across the 77 banking classes. In a 77-class task with $k=8$ shots, a random clean prompt covers only ~8 random classes, which rarely include the specific fine-grained intent of the test query.
- Because $k$-NN achieves **80% – 83%** accuracy on the exact same dataset, semantic relevance of demonstrations is proven to be vastly more potent than label purity alone.

---

## 11. Method 10: Oracle Domain Routing (Ground-Truth Semantic Routing)

### 11.1 Conceptual Intuition
On domain benchmark suites (e.g., MMLU-Domain or Banking77-Domain), partitions are structured around distinct semantic subjects or intents. The Oracle Domain baseline uses the query's hidden ground-truth label $y_t^*$ to look up the exact partition corresponding to that domain, and draws demonstrations exclusively from that gold partition.

### 11.2 Mathematical Formulation
Let $g: \mathcal{Y} \to \{0, \dots, n-1\}$ be the ground-truth ontology mapping each class label to its home domain partition:
$$a_t^* = g(y_t^*)$$
$$C_t = \operatorname{sample\_shots}(P_{a_t^*}, k, \text{use\_noisy}=\text{True})$$
*(Note: Partition $P_{a_t^*}$ still contains its assigned noise rate $\eta_{a_t^*}$. Oracle Domain provides perfect semantic routing, but does not remove label noise within the domain).*

### 11.3 Code Implementation
Located in [`scripts/run_experiments.py#L167-L174, L231-L240`](file:///c:/Users/saite/OneDrive/Desktop/Project/In_context_learning/scripts/run_experiments.py#L167-L174):
```python
# Setup: map label to domain partition
label_to_domain_part: dict[int, int] = {}
for p_idx, part in enumerate(noisy_partitions):
    for item in part:
        lbl = item.label if hasattr(item, "label") else item["label"]
        label_to_domain_part[int(lbl)] = p_idx

# Per-query execution
gold_lbl = int(ex["label"])
arm = label_to_domain_part[gold_lbl]
shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)
```

### 11.4 Role in the Empirical Analysis
- **Empirical Performance**: Achieves **61.2%** on Banking77 4-Domain and **68.4%** on MMLU 4-Domain.
- **Benchmarking LinUCB**: Proves that LinUCB's learned routing (51.2% and 52.9%) captures a substantial portion of the domain routing potential without having access to ground-truth label-to-domain ontologies.

---

## 12. Master Cross-Method Comparative Matrix

| Method | Method Category | Contextual? | Decision Unit | Action Space Size | Per-Step Time Complexity | Noise Filtering Mechanism | Empirical Accuracy Range | Primary Limitation / Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Worst-Case** | Adversarial Baseline | No | 1 Partition | 1 | $O(1)$ | None (adversarial) | 16.0% – 22.0% | Measures worst-case damage, not a practical policy |
| **Uniform** | Random Control | No | 1 Partition | $n$ | $O(1)$ | None (random) | 40.8% – 48.0% | Suffers from replacement collapse at large $n$ ($n=100, k=12$) |
| **UCB1** | Non-Contextual MAB | No | 1 Partition | $n$ | $O(n)$ | Online mean reward | Evaluated on IID single | Restricted to single-partition selection ($n_p=1$) |
| **$k$-UCB** | Combinatorial MAB | No | Superarm ($n_p$) | $\binom{n}{n_p}$ | $O(\binom{n}{n_p})$ | Joint superarm mean | 46.2% – 51.5% | Combinatorial explosion; 0.20% coverage at $n=100, n_p=3$ |
| **IndepUCB** | Ranked MAB | No | Top-$n_p$ Partitions| $n$ | $O(n \log n_p)$ | Coarse broadcast mean | 48.2% – 53.6% | Coarse credit assignment dilutes individual shard blame |
| **Neural-PG** | Deep RL Bandit | Yes (MLP) | Top-$n_p$ Sampled | $n$ | $O(d \cdot h + h \cdot n)$| Policy gradient REINFORCE | 40.8% – 43.9% | Extreme sample inefficiency; high gradient variance over $T=320$ |
| **LinUCB** | Linear Contextual | Yes (Ridge) | Top-$n_p$ Partitions| $n$ | $O(n \cdot d^2)$ | Disjoint ridge + UCB | 51.2% – 67.2% | Linear reward assumption; requires dense text embeddings |
| **Random Clean**| Unbiased Control | No | $k$ Pooled Instances| - | $O(1)$ | 100% clean data | 61.2% – 73.8% | Not an upper bound; lacks semantic relevance to test query |
| **Oracle Domain**| Ground-Truth Routing| Semi (Gold)| 1 Domain Partition| 1 | $O(1)$ | Perfect domain routing | 61.2% – 68.4% | Requires oracle access to test query's true label |
| **$k$-NN Retrieval**| Instance-Level Metric| Yes (Cosine) | $k$ Instances | $|\mathcal{D}_{\text{pool}}|$ | $O(\|\mathcal{D}\| \cdot d)$ | **Zero noise filtering** | 70.8% – 83.1% | Completely noise-blind; directly injects corrupted labels |

---

## 13. Key Empirical Takeaways & Methodological Confounders

### 1. The Superarm Combinatorial Myth ($k$-UCB vs. IndepUCB)
Treating partition combinations as combinatorial superarms sounds rigorous in theory, but completely breaks down in online federated settings. At $n=100, n_p=3$, exploring 161,700 superarms with 320 queries touches only 0.20% of the space, collapsing into an arbitrary index sweep. IndepUCB's simple $O(n)$ top-$n_p$ ranking matches or beats $k$-UCB in 100% of settings.

### 2. The Early Semantic Lead of LinUCB
LinUCB outperforms non-contextual bandits by +16.7 to +18.7 points at large $n$. Crucially, quarterly dynamics reveal that ~+10 points of this margin is already established in Quarter 1 ($t \le 80$). LinUCB's power comes from its ability to project queries into semantic feature space and immediately align queries with compatible shards, rather than waiting for long-horizon bandit regret convergence.

### 3. The Uniform Baseline Confounder
The apparent ~+7 point drop in Uniform baseline performance at $n=100, n_p=3$ (from ~47% to 40.8%) is an artifact of shard size:
- When $n=100$, shards contain only 11 examples.
- Drawing $k=12$ shots from a single shard forces sampling with replacement, reducing distinct classes in the prompt to 5.09.
- Bandits pool 3 shards ($n_p=3$), drawing 4 shots per shard without replacement (6.40 distinct classes).
- Future evaluations must deploy a `uniform_np` control that pools $n_p$ random partitions to isolate noise avoidance from class diversity.

### 4. Semantic Proximity Beats Label Cleanliness ($k$-NN vs. Clean)
The fact that $k$-NN Retrieval (80%–83%) substantially beats Random Clean Shots (61%–74%) demonstrates that for large language models, **semantic relevance of in-context demonstrations outweighs label purity**. Because 28% of test queries have near-duplicate paraphrases in the training pool, retrieving those paraphrases provides enormous task guidance even if some labels are corrupted.

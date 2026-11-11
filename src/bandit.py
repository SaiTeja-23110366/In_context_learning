from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np


class TextEmbedder:
    """
    Computes dense semantic embeddings for query text.
    Uses sentence-transformers with caching.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.cache: dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device,
            )
            if hasattr(self.model, "get_embedding_dimension"):
                self._dim = int(self.model.get_embedding_dimension())
            else:
                self._dim = int(self.model.get_sentence_embedding_dimension())
        except Exception as e:
            # Deterministic fallback if sentence-transformers model cannot be loaded
            self.model = None
            self._dim = 384

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        if self.model is not None:
            vec = self.model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )
            vec = np.asarray(vec, dtype=np.float32)
        else:
            # Fast deterministic projection fallback
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self._dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        self.cache[text] = vec
        return vec


class UCB1:
    """
    Standard UCB1 bandit for a single-partition selection.
    Used for the IID setting (superarm size = 1).

    Score: mean_i + c * sqrt(ln(t) / n_i)
    """

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


class KUCBInfeasibleError(RuntimeError):
    """Raised when the number of superarms C(n, n_p) exceeds the safety cap."""

    def __init__(
        self, n_partitions: int, n_p: int, n_superarms: int, max_superarms: int
    ) -> None:
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.n_superarms = n_superarms
        self.max_superarms = max_superarms
        super().__init__(
            f"KUCB infeasible: C({n_partitions}, {n_p}) = {n_superarms} superarms "
            f"exceeds max_superarms = {max_superarms}"
        )


class KUCB:
    """
    UCB over combinations of n_p partitions (paper Algorithm 1).
    Used for the Non-IID setting.

    Superarms are all C(n_partitions, n_p) combinations.
    Each superarm is treated as a single arm in an underlying UCB1.
    """

    def __init__(
        self,
        n_partitions: int,
        n_p: int,
        c: float = 2.0,
        max_superarms: int = 2_000_000,
    ) -> None:
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


class IndepUCB:
    """
    Non-contextual, independent per-partition UCB bandit.
    Selects the top-n_p partitions ranked by their individual UCB1 scores.
    Complexity is O(n) per step, avoiding combinatorial superarm enumeration C(n, n_p).

    Credit assignment limitation:
    When a reward is received from evaluating an ensemble of n_p partitions,
    the exact same reward is credited to each of the n_p partitions.
    This coarse credit assignment can dilute feedback when some partitions
    in the combination are informative and others are corrupted.
    """

    def __init__(self, n_partitions: int, n_p: int = 2, c: float = 2.0) -> None:
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.c = c
        self.counts = [1] * n_partitions   # init to 1 as per UCB1 standard
        self.means = [0.0] * n_partitions  # init to 0.0
        self.t = 0

    def select(self) -> tuple[int, tuple[int, ...]]:
        self.t += 1
        scores = [
            self.means[i] + self.c * math.sqrt(math.log(self.t) / self.counts[i])
            for i in range(self.n_partitions)
        ]
        sorted_indices = sorted(
            range(self.n_partitions), key=lambda i: scores[i], reverse=True
        )
        combo = tuple(sorted_indices[: self.n_p])
        return combo[0], combo

    def update(self, combo: Sequence[int], reward: float) -> None:
        # Credit assignment limitation: identical reward given to all selected partitions
        for arm in combo:
            self.counts[arm] += 1
            n = self.counts[arm]
            self.means[arm] = ((n - 1) * self.means[arm] + reward) / n


class LinUCB:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) for Top-k Partition Selection.
    (Li et al., 2010; Cesa-Bianchi et al., 2012)

    For each partition a in {0, ..., n_partitions-1}:
        Model: r_a(x) = x^T theta_a
        Ridge estimate: theta_a = A_a^(-1) b_a
        Confidence bound: p_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^(-1) x)

    Selection:
        Computes p_a(x) for all n_partitions.
        Selects top n_p partitions with highest UCB scores.

    Update:
        Maintains A_inv directly using Sherman-Morrison rank-1 update:
        A^(-1) <- A^(-1) - (A^(-1) x x^T A^(-1)) / (1 + x^T A^(-1) x)
    """

    def __init__(
        self,
        n_partitions: int,
        n_p: int = 2,
        dim: int = 384,
        alpha: float = 1.0,
        reg: float = 1.0,
    ) -> None:
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.dim = dim
        self.alpha = alpha
        self.reg = reg

        # A_inv is (dim, dim) per partition
        self.A_inv = [
            np.eye(dim, dtype=np.float32) / reg for _ in range(n_partitions)
        ]
        self.b = [
            np.zeros((dim, 1), dtype=np.float32) for _ in range(n_partitions)
        ]
        self.counts = [0] * n_partitions
        self.rewards = [0.0] * n_partitions

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

        # Top n_p partitions
        sorted_indices = sorted(
            range(self.n_partitions), key=lambda i: scores[i], reverse=True
        )
        combo = tuple(sorted_indices[: self.n_p])
        return combo[0], combo

    def update(
        self, combo: Sequence[int], context: np.ndarray, reward: float
    ) -> None:
        x = context.reshape(-1, 1).astype(np.float32)
        for a in combo:
            self.counts[a] += 1
            self.rewards[a] += reward
            # Sherman-Morrison rank-1 update:
            u = self.A_inv[a] @ x
            denom = 1.0 + float((x.T @ u).item())
            self.A_inv[a] -= (u @ u.T) / max(denom, 1e-6)
            self.b[a] += float(reward) * x


class NeuralPolicyBandit:
    """
    Neural Contextual Bandit using Policy Gradient (REINFORCE) with baseline.

    A neural network policy pi_theta(a | x) maps query embedding x to partition logits.
    Action: Top-k / sampled combination of n_p partitions.
    Update: Policy gradient:
        loss = - sum_{a in combo} log pi_theta(a | x) * (Reward - Baseline) - entropy_bonus
    """

    def __init__(
        self,
        n_partitions: int,
        n_p: int = 2,
        dim: int = 384,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        temp: float = 1.0,
        entropy_weight: float = 0.01,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_partitions = n_partitions
        self.n_p = n_p
        self.dim = dim
        self.temp = temp
        self.entropy_weight = entropy_weight
        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_partitions),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=1e-4
        )
        self.baseline = 0.5
        self.counts = [0] * n_partitions
        self.rewards = [0.0] * n_partitions

        self._last_combo: tuple[int, ...] | None = None
        self._last_log_prob = None
        self._last_probs = None

    def select(
        self, context: np.ndarray, explore: bool = True
    ) -> tuple[int, tuple[int, ...]]:
        import torch

        x = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        logits = self.net(x) / max(self.temp, 1e-4)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        probs_np = probs_np / (probs_np.sum() + 1e-12)

        if explore:
            try:
                chosen = self.rng.choice(
                    self.n_partitions, size=self.n_p, replace=False, p=probs_np
                )
            except Exception:
                chosen = np.argsort(probs_np)[::-1][: self.n_p]
        else:
            chosen = np.argsort(probs_np)[::-1][: self.n_p]

        combo = tuple(int(c) for c in sorted(chosen))

        chosen_tensor = torch.tensor(list(combo), device=self.device)
        selected_probs = probs[chosen_tensor]
        log_prob = torch.log(selected_probs + 1e-12).sum()

        self._last_combo = combo
        self._last_log_prob = log_prob
        self._last_probs = probs

        return combo[0], combo

    def update(self, combo: Sequence[int], reward: float) -> None:
        import torch

        for a in combo:
            self.counts[a] += 1
            self.rewards[a] += reward

        if self._last_log_prob is None or self._last_probs is None:
            return

        advantage = float(reward) - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * float(reward)

        entropy = -(self._last_probs * torch.log(self._last_probs + 1e-12)).sum()
        loss = -self._last_log_prob * advantage - self.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_log_prob = None
        self._last_probs = None
        self._last_combo = None

from __future__ import annotations

import itertools
import math


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


class KUCB:
    """
    UCB over combinations of n_p partitions (paper Algorithm 1).
    Used for the Non-IID setting.

    Superarms are all C(n_partitions, n_p) combinations.
    Each superarm is treated as a single arm in an underlying UCB1.
    """

    def __init__(self, n_partitions: int, n_p: int, c: float = 2.0) -> None:
        self.superarms = list(itertools.combinations(range(n_partitions), n_p))
        self.ucb = UCB1(len(self.superarms), c=c)

    def select(self) -> tuple[int, tuple[int, ...]]:
        idx = self.ucb.select()
        return idx, self.superarms[idx]

    def update(self, idx: int, reward: float) -> None:
        self.ucb.update(idx, reward)

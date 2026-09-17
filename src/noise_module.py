"""
noise_module.py
================
OUR INDEPENDENT ADDITION — not part of the paper reproduction.

This module provides a standalone label-noise injection utility
that can be applied to any list of partitioned examples.

Noise model (matches paper Section 2):
    Each example in partition i has its label corrupted with
    probability q_i. A corrupted label is drawn uniformly from
    {0, ..., n_classes-1} \ {y}, i.e. it is always different
    from the original label y.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Example:
    """A single training example with both original and (possibly) noisy label."""
    text: str
    label: int          # always the ground-truth label
    noisy_label: int    # may equal label if no corruption occurred


class LabelNoiseInjector:
    """
    Injects symmetric label noise into a list of partitions.

    Parameters
    ----------
    noise_rates : list[float]
        Per-partition noise rate q_i in [0, 1].  Must have one entry
        per partition.
    n_classes : int
        Total number of classes in the dataset.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        noise_rates: List[float],
        n_classes: int,
        seed: int = 0,
    ) -> None:
        if any(q < 0.0 or q > 1.0 for q in noise_rates):
            raise ValueError("All noise_rates must be in [0, 1].")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        self.noise_rates = noise_rates
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def _corrupt_label(self, y: int, q: float) -> int:
        """
        Corrupt a single label y with probability q.

        If corrupted, the new label is sampled uniformly from
        {0, ..., n_classes-1} excluding y, so it is ALWAYS different
        from the original.
        """
        if self.rng.random() < q:
            choices = [c for c in range(self.n_classes) if c != y]
            return int(self.rng.choice(choices))
        return y

    def inject(
        self, partitions: List[List[dict]]
    ) -> List[List[Example]]:
        """
        Apply noise to a list of clean partitions.

        Parameters
        ----------
        partitions : list[list[dict]]
            Each partition is a list of dicts with keys 'text' and 'label'.
            The 'label' values must be integers in [0, n_classes - 1].

        Returns
        -------
        list[list[Example]]
            Same structure but each dict is replaced by an Example
            dataclass holding both the original and noisy label.
        """
        if len(partitions) != len(self.noise_rates):
            raise ValueError(
                f"Got {len(partitions)} partitions but "
                f"{len(self.noise_rates)} noise_rates."
            )
        result: List[List[Example]] = []
        for part_idx, (partition, q) in enumerate(
            zip(partitions, self.noise_rates)
        ):
            noisy_part: List[Example] = []
            for row in partition:
                y = int(row["label"])
                noisy_y = self._corrupt_label(y, q)
                noisy_part.append(
                    Example(text=row["text"], label=y, noisy_label=noisy_y)
                )
            result.append(noisy_part)
        return result

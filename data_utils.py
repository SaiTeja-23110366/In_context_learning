from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class Example:
    text: str
    label: int
    noisy_label: int


def load_agnews(cache_dir: str) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("fancyzhx/ag_news", cache_dir=cache_dir)
    train = [{"text": r["text"], "label": int(r["label"])} for r in ds["train"]]
    test = [{"text": r["text"], "label": int(r["label"])} for r in ds["test"]]
    return train, test


def subsample_test(test: list[dict], n_test: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test), size=min(n_test, len(test)), replace=False)
    return [test[int(i)] for i in idxs]


def make_iid_partitions(
    train: list[dict],
    n_partitions: int,
    noise_rates: list[float],
    seed: int,
) -> list[list[Example]]:
    if len(noise_rates) != n_partitions:
        raise ValueError("noise_rates must match n_partitions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    chunks = np.array_split(order, n_partitions)
    n_classes = len(AGNEWS_LABELS)
    partitions: list[list[Example]] = []
    for part_idx, idxs in enumerate(chunks):
        q = noise_rates[part_idx]
        items: list[Example] = []
        for i in idxs:
            y = int(train[int(i)]["label"])
            noisy = y
            if rng.random() < q:
                choices = [c for c in range(n_classes) if c != y]
                noisy = int(rng.choice(choices))
            items.append(
                Example(
                    text=train[int(i)]["text"],
                    label=y,
                    noisy_label=noisy,
                )
            )
        partitions.append(items)
    return partitions


def sample_shots(
    pool: list[Example],
    k: int,
    rng: np.random.Generator,
    use_noisy: bool,
) -> list[tuple[str, int]]:
    if len(pool) < k:
        raise ValueError("partition smaller than k")
    idxs = rng.choice(len(pool), size=k, replace=False)
    shots = []
    for i in idxs:
        ex = pool[int(i)]
        y = ex.noisy_label if use_noisy else ex.label
        shots.append((ex.text, y))
    return shots

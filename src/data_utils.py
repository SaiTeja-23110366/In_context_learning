from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ECOM_URLS = [
    "https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/Dataset/ecommerceDataset.csv",
    "https://zenodo.org/api/records/3355823/files/ecommerceDataset.csv/content",
    "https://zenodo.org/records/3355823/files/ecommerceDataset.csv?download=1",
]


@dataclass
class Example:
    text: str
    label: int
    noisy_label: int


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_agnews(cache_dir: str) -> tuple[list[dict], list[dict], list[str]]:
    ds = load_dataset("fancyzhx/ag_news", cache_dir=cache_dir)
    train = [{"text": r["text"], "label": int(r["label"])} for r in ds["train"]]
    test = [{"text": r["text"], "label": int(r["label"])} for r in ds["test"]]
    return train, test, AGNEWS_LABELS


def load_ecommerce(
    cache_dir: str, seed: int = 0
) -> tuple[list[dict], list[dict], list[str]]:
    path = Path(cache_dir) / "ecommerceDataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size < 1_000_000:
        path.unlink()
    if not path.exists():
        import urllib.request

        last_err = None
        for url in ECOM_URLS:
            try:
                print("downloading ecommerce from", url)
                urllib.request.urlretrieve(url, path)
                if path.stat().st_size < 1_000_000:
                    raise RuntimeError("file too small, likely not the csv")
                break
            except Exception as e:
                last_err = e
                if path.exists():
                    path.unlink()
        else:
            raise RuntimeError(f"could not download ecommerce csv: {last_err}")
    df = pd.read_csv(path, header=None, names=["label_name", "text"])
    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df["label_name"] = df["label_name"].astype(str).str.strip()
    labels = sorted(df["label_name"].unique().tolist())
    name_to_id = {n: i for i, n in enumerate(labels)}
    records = [
        {"text": str(t), "label": int(name_to_id[n])}
        for t, n in zip(df["text"], df["label_name"])
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_test = max(1000, int(0.1 * len(records)))
    test = [records[int(i)] for i in order[:n_test]]
    train = [records[int(i)] for i in order[n_test:]]
    return train, test, labels


def load_synthetic(
    seed: int = 0, n_train: int = 20_000, n_test_pool: int = 2_000
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Simple keyword-based synthetic dataset.

    Rule (explicit in the text):
        The text contains the phrase "category is <WORD>" where WORD
        maps directly to a class label. LLMs can learn this rule from
        as few as 4 examples.

    Labels: ["Alpha", "Beta", "Gamma", "Delta"]
    """
    labels = ["Alpha", "Beta", "Gamma", "Delta"]
    # Additional context words to make each record look like a natural sentence
    contexts = [
        "The system returned a result: category is {cls}. Please verify.",
        "Log entry #{uid}: classification category is {cls}.",
        "Analysis complete. Assigned category is {cls} for record {uid}.",
        "Report #{uid} — primary category is {cls}.",
        "Record {uid} has been filed under category {cls}.",
        "System alert: category is {cls}. Record ID {uid}.",
        "Audit entry {uid}: the category is {cls}.",
        "Output for item {uid}: category is {cls}, confirmed.",
    ]
    rng = np.random.default_rng(seed)

    def make(n: int, offset: int) -> list[dict]:
        rows = []
        for t in range(n):
            y = int(rng.integers(0, 4))
            uid = offset + t
            tmpl = contexts[int(rng.integers(0, len(contexts)))]
            text = tmpl.format(cls=labels[y], uid=uid)
            rows.append({"text": text, "label": y})
        return rows

    return make(n_train, 0), make(n_test_pool, n_train), labels


def load_task(dataset: str, cache_dir: str, seed: int = 0):
    if dataset == "ag_news":
        return load_agnews(cache_dir)
    if dataset == "ecommerce":
        return load_ecommerce(cache_dir, seed=seed)
    if dataset == "synthetic":
        return load_synthetic(seed=seed)
    raise ValueError(dataset)


# ── Utilities ─────────────────────────────────────────────────────────────────

def subsample_test(test: list[dict], n_test: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test), size=min(n_test, len(test)), replace=False)
    return [test[int(i)] for i in idxs]


# ── Clean partitioning (NO noise here — use noise_module.py for noise) ────────

def make_iid_partitions(
    train: list[dict],
    n_partitions: int,
    seed: int,
) -> list[list[dict]]:
    """
    Split training data into n_partitions IID partitions.
    Returns clean partitions (plain dicts with 'text' and 'label').
    Apply noise separately using noise_module.LabelNoiseInjector.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    chunks = np.array_split(order, n_partitions)
    partitions: list[list[dict]] = []
    for idxs in chunks:
        partitions.append(
            [{"text": train[int(i)]["text"], "label": int(train[int(i)]["label"])}
             for i in idxs]
        )
    return partitions


def make_non_iid_partitions(
    train: list[dict],
    n_partitions: int,
    seed: int,
    dirichlet_alpha: float = 0.3,
) -> list[list[dict]]:
    """
    Class-skewed partitions via Dirichlet split.
    Returns clean partitions (plain dicts with 'text' and 'label').
    Apply noise separately using noise_module.LabelNoiseInjector.
    """
    rng = np.random.default_rng(seed)
    n_classes = 1 + max(int(r["label"]) for r in train)
    by_class: list[list[int]] = [[] for _ in range(n_classes)]
    for i, row in enumerate(train):
        by_class[int(row["label"])].append(i)

    buckets: list[list[int]] = [[] for _ in range(n_partitions)]
    for cls_idxs in by_class:
        rng.shuffle(cls_idxs)
        props = rng.dirichlet([dirichlet_alpha] * n_partitions)
        cuts = np.cumsum(props) * len(cls_idxs)
        start = 0
        for p in range(n_partitions):
            end = (
                len(cls_idxs)
                if p == n_partitions - 1
                else int(round(cuts[p]))
            )
            end = max(start, min(end, len(cls_idxs)))
            buckets[p].extend(cls_idxs[start:end])
            start = end

    partitions: list[list[dict]] = []
    for p, idxs in enumerate(buckets):
        if not idxs:
            raise ValueError(
                f"Empty non-iid partition {p}; increase data or dirichlet_alpha."
            )
        partitions.append(
            [{"text": train[i]["text"], "label": int(train[i]["label"])}
             for i in idxs]
        )
    return partitions


# ── Shot samplers (work on Example objects after noise injection) ──────────────

def sample_shots(
    pool: list[Example],
    k: int,
    rng: np.random.Generator,
    use_noisy: bool,
) -> list[tuple[str, int]]:
    if not pool:
        raise ValueError("empty partition")
    idxs = rng.choice(len(pool), size=k, replace=len(pool) < k)
    shots = []
    for i in idxs:
        ex = pool[int(i)]
        y = ex.noisy_label if use_noisy else ex.label
        shots.append((ex.text, y))
    return shots


def sample_shots_from_parts(
    parts: list[list[Example]],
    k: int,
    rng: np.random.Generator,
    use_noisy: bool,
) -> list[tuple[str, int]]:
    """Distribute k shots evenly across all partitions in the superarm."""
    n = len(parts)
    base, rem = divmod(k, n)
    shots: list[tuple[str, int]] = []
    for i, part in enumerate(parts):
        take = base + (1 if i < rem else 0)
        if take:
            shots.extend(sample_shots(part, take, rng, use_noisy))
    rng.shuffle(shots)
    return shots

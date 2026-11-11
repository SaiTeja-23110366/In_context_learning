from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
MMLU_LABELS = ["A", "B", "C", "D"]

MMLU_DOMAINS = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology"
    ]
}

MMLU_8_DOMAINS = {
    "STEM_Math_Physics": [
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics", "astronomy",
        "college_physics", "conceptual_physics", "high_school_physics"
    ],
    "STEM_Bio_Chem_CS": [
        "anatomy", "college_biology", "college_chemistry", "high_school_biology",
        "high_school_chemistry", "college_computer_science", "computer_security",
        "electrical_engineering", "high_school_computer_science", "machine_learning"
    ],
    "Humanities_History": [
        "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "prehistory"
    ],
    "Humanities_Law_Philosophy": [
        "formal_logic", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "professional_law", "world_religions"
    ],
    "SocialSciences_Econ_Gov": [
        "econometrics", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "public_relations", "security_studies", "us_foreign_policy"
    ],
    "SocialSciences_Psych_Soc": [
        "high_school_geography", "high_school_psychology", "human_sexuality",
        "professional_psychology", "sociology"
    ],
    "Other_Medicine": [
        "clinical_knowledge", "college_medicine", "human_aging",
        "medical_genetics", "nutrition", "professional_medicine", "virology"
    ],
    "Other_Business_Misc": [
        "business_ethics", "global_facts", "management", "marketing",
        "miscellaneous", "professional_accounting"
    ]
}

SUBJECT_TO_DOMAIN = {subj: dom for dom, subjs in MMLU_DOMAINS.items() for subj in subjs}
SUBJECT_TO_8_DOMAIN = {subj: dom for dom, subjs in MMLU_8_DOMAINS.items() for subj in subjs}

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


def load_mmlu(
    cache_dir: str, seed: int = 0
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Loads MMLU (Massive Multitask Language Understanding) across all 57 subjects.
    Formats questions into standard 4-choice multiple choice prompts.
    Attaches domain metadata (STEM, Humanities, Social Sciences, Other).
    """
    ds = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
    letters = MMLU_LABELS

    def _fmt(r):
        q = str(r["question"]).strip()
        choices = [f"{letters[i]}. {str(c).strip()}" for i, c in enumerate(r["choices"])]
        prompt_text = f"Question: {q}\n" + "\n".join(choices)
        subj = str(r.get("subject", "general"))
        dom = SUBJECT_TO_DOMAIN.get(subj, "Other")
        dom8 = SUBJECT_TO_8_DOMAIN.get(subj, "Other_Business_Misc")
        return {
            "text": prompt_text,
            "label": int(r["answer"]),
            "subject": subj,
            "domain": dom,
            "domain8": dom8,
        }

    # Training pool from dev + validation (1,816 cleanly annotated subject examples)
    train = [_fmt(r) for r in ds["dev"]] + [_fmt(r) for r in ds["validation"]]

    test = [_fmt(r) for r in ds["test"]]
    return train, test, MMLU_LABELS


BANKING_INTENTS_8 = [
    ("card_arrival", "Cards"),
    ("card_not_working", "Cards"),
    ("cancel_transfer", "Transfers"),
    ("transfer_fee_charged", "Transfers"),
    ("top_up_failed", "TopUp"),
    ("automatic_top_up", "TopUp"),
    ("exchange_rate", "Exchange"),
    ("fiat_currency_support", "Exchange"),
]
BANKING_LABELS_8 = [f"Class_{i+1}" for i in range(8)]


def load_banking(
    cache_dir: str, seed: int = 0
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Loads BANKING77 with 8 fine-grained intent classes mapped to abstract class tokens
    (Class_1 .. Class_8) across 4 operational banking domains (Cards, Transfers, TopUp, Exchange).
    This setup evaluates true in-context learning of semantic intent binding,
    preventing zero-shot semantic prior leakage and producing a clean vs noisy spread of 58%.
    """
    ds = load_dataset("PolyAI/banking77", cache_dir=cache_dir)
    names = ds["train"].features["label"].names

    intent_names = [item[0] for item in BANKING_INTENTS_8]
    intent_to_domain = {item[0]: item[1] for item in BANKING_INTENTS_8}
    idx_map = {names.index(name): i for i, name in enumerate(intent_names)}

    def _process_split(split):
        rows = []
        for r in split:
            orig_lbl = int(r["label"])
            if orig_lbl in idx_map:
                new_lbl = idx_map[orig_lbl]
                name = intent_names[new_lbl]
                dom = intent_to_domain[name]
                rows.append({
                    "text": str(r["text"]).strip(),
                    "label": new_lbl,
                    "subject": name,
                    "domain": dom,
                })
        return rows

    train = _process_split(ds["train"])
    test = _process_split(ds["test"])

    rng = np.random.default_rng(seed)
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test, BANKING_LABELS_8


def load_task(dataset: str, cache_dir: str, seed: int = 0):
    if dataset == "ag_news":
        return load_agnews(cache_dir)
    if dataset == "ecommerce":
        return load_ecommerce(cache_dir, seed=seed)
    if dataset == "synthetic":
        return load_synthetic(seed=seed)
    if dataset == "mmlu":
        return load_mmlu(cache_dir, seed=seed)
    if dataset == "banking":
        return load_banking(cache_dir, seed=seed)
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


def make_shard_partitions(
    train: list[dict],
    n_partitions: int,
    seed: int,
    k: int = 8,
    n_p: int = 2,
) -> list[list[dict]]:
    """
    Split pooled training set into n_partitions IID shards.
    Shuffles with the given seed and splits into n equal or nearly-equal shards.
    Raises ValueError if any shard has fewer than ceil(k / n_p) examples.
    """
    min_required = math.ceil(k / n_p)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    chunks = np.array_split(order, n_partitions)
    partitions: list[list[dict]] = []
    for p_idx, idxs in enumerate(chunks):
        if len(idxs) < min_required:
            raise ValueError(
                f"Shard {p_idx} has {len(idxs)} examples, which is fewer than "
                f"required ceil(k / n_p) = ceil({k} / {n_p}) = {min_required} examples."
            )
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


def make_domain_partitions(
    train: list[dict],
    n_partitions: int,
    seed: int,
) -> list[list[dict]]:
    """
    Partition by semantic domain / subject clusters.
    For n_partitions=4: STEM, Humanities, Social Sciences, Other.
    For n_partitions=8: 8 fine-grained domain clusters.
    For arbitrary n_partitions: groups subjects consistently across partitions.
    """
    rng = np.random.default_rng(seed)
    if n_partitions == 4:
        banking_doms = ["Cards", "Transfers", "TopUp", "Exchange"]
        if any(ex.get("domain") in banking_doms for ex in train[:20]):
            dom_names = banking_doms
        else:
            dom_names = ["STEM", "Humanities", "Social Sciences", "Other"]
        buckets: list[list[dict]] = [[] for _ in range(4)]
        for ex in train:
            dom = ex.get("domain", dom_names[3])
            idx = dom_names.index(dom) if dom in dom_names else 3
            buckets[idx].append(ex)
    elif n_partitions == 8:
        banking_doms = ["Cards", "Transfers", "TopUp", "Exchange"]
        if any(ex.get("domain") in banking_doms for ex in train[:20]):
            intent_names = [item[0] for item in BANKING_INTENTS_8]
            buckets: list[list[dict]] = [[] for _ in range(8)]
            for ex in train:
                subj = ex.get("subject", intent_names[-1])
                idx = intent_names.index(subj) if subj in intent_names else 7
                buckets[idx].append(ex)
        else:
            dom_names = list(MMLU_8_DOMAINS.keys())
            buckets = [[] for _ in range(8)]
            for ex in train:
                dom8 = ex.get("domain8", dom_names[-1])
                idx = dom_names.index(dom8) if dom8 in dom_names else 7
                buckets[idx].append(ex)
    else:
        subjects = sorted(list({ex.get("subject", "") for ex in train}))
        subj_to_part = {s: i % n_partitions for i, s in enumerate(subjects)}
        buckets = [[] for _ in range(n_partitions)]
        for ex in train:
            s = ex.get("subject", "")
            buckets[subj_to_part.get(s, 0)].append(ex)

    partitions: list[list[dict]] = []
    for p, items in enumerate(buckets):
        if not items:
            raise ValueError(f"Empty domain partition {p}")
        shuffled = list(items)
        rng.shuffle(shuffled)
        partitions.append(
            [{"text": x["text"], "label": int(x["label"])} for x in shuffled]
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
        if isinstance(ex, dict):
            text = ex["text"]
            y = int(ex["label"])
        else:
            text = ex.text
            y = ex.noisy_label if use_noisy else ex.label
        shots.append((text, y))
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

"""
test_contextual_bandits.py
==========================
Unit tests for:
1. MMLU dataset formatting & domain partitioning
2. LinUCB (Disjoint Linear Contextual Bandit)
3. NeuralPolicyBandit (REINFORCE Policy Gradient Bandit)
4. MMLU label parsing
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.bandit import LinUCB, NeuralPolicyBandit, TextEmbedder
from src.data_utils import (
    MMLU_DOMAINS,
    MMLU_8_DOMAINS,
    make_domain_partitions,
)
from src.icl import parse_label


def test_label_parsing_mmlu():
    labels = ["A", "B", "C", "D"]
    assert parse_label("A", labels) == 0
    assert parse_label("A.", labels) == 0
    assert parse_label("(B)", labels) == 1
    assert parse_label("[C]", labels) == 2
    assert parse_label("Answer: D", labels) == 3
    assert parse_label("The correct choice is B.", labels) == 1
    assert parse_label("Option A is correct", labels) == 0


def test_domain_partitioning():
    # Synthetic samples mimicking MMLU records
    sample_records = [
        {"text": "Q1", "label": 0, "subject": "abstract_algebra", "domain": "STEM", "domain8": "STEM_Math_Physics"},
        {"text": "Q2", "label": 1, "subject": "high_school_us_history", "domain": "Humanities", "domain8": "Humanities_History"},
        {"text": "Q3", "label": 2, "subject": "econometrics", "domain": "Social Sciences", "domain8": "SocialSciences_Econ_Gov"},
        {"text": "Q4", "label": 3, "subject": "clinical_knowledge", "domain": "Other", "domain8": "Other_Medicine"},
    ]
    parts4 = make_domain_partitions(sample_records, n_partitions=4, seed=42)
    assert len(parts4) == 4
    for p in parts4:
        assert len(p) >= 1
        assert "text" in p[0]
        assert "label" in p[0]


def test_text_embedder():
    embedder = TextEmbedder()
    assert embedder.dim > 0
    v1 = embedder.embed("What is the capital of France?")
    v2 = embedder.embed("What is the capital of France?")
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4


def test_linucb_selection_and_update():
    dim = 64
    n_partitions = 5
    n_p = 2
    bandit = LinUCB(n_partitions=n_partitions, n_p=n_p, dim=dim, alpha=1.0)

    rng = np.random.default_rng(0)
    ctx = rng.standard_normal(dim).astype(np.float32)
    ctx = ctx / np.linalg.norm(ctx)

    primary, combo = bandit.select(ctx)
    assert len(combo) == n_p
    assert primary in combo

    # Update with reward 1.0
    bandit.update(combo, ctx, reward=1.0)
    for arm in combo:
        assert bandit.counts[arm] == 1
        assert bandit.rewards[arm] == 1.0


def test_neural_policy_bandit():
    dim = 64
    n_partitions = 6
    n_p = 2
    policy = NeuralPolicyBandit(n_partitions=n_partitions, n_p=n_p, dim=dim, lr=1e-3, seed=42)

    rng = np.random.default_rng(42)
    ctx = rng.standard_normal(dim).astype(np.float32)

    primary, combo = policy.select(ctx, explore=True)
    assert len(combo) == n_p
    assert len(set(combo)) == n_p  # must be unique partitions

    # Check update executes cleanly without NaN
    policy.update(combo, reward=1.0)
    for p in policy.net.parameters():
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()


if __name__ == "__main__":
    test_label_parsing_mmlu()
    test_domain_partitioning()
    test_text_embedder()
    test_linucb_selection_and_update()
    test_neural_policy_bandit()
    print("All tests passed successfully!")

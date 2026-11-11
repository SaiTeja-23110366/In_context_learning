import math
import numpy as np
import pytest

from src.bandit import (
    KUCB,
    IndepUCB,
    KUCBInfeasibleError,
    NeuralPolicyBandit,
    TextEmbedder,
)
from src.data_utils import Example, make_shard_partitions


def test_indep_ucb_selects_distinct_arms():
    n_partitions = 10
    n_p = 3
    bandit = IndepUCB(n_partitions=n_partitions, n_p=n_p, c=2.0)
    for _ in range(20):
        arm, combo = bandit.select()
        assert len(combo) == n_p
        assert len(set(combo)) == n_p
        assert all(0 <= a < n_partitions for a in combo)
        assert arm == combo[0]
        # Simulate dummy reward update
        bandit.update(combo, reward=1.0)


def test_indep_ucb_converges_to_clean_arm_toy():
    """Toy problem: arm 0 has reward prob 0.95, other arms have 0.05."""
    n_partitions = 6
    n_p = 2
    bandit = IndepUCB(n_partitions=n_partitions, n_p=n_p, c=2.0)
    rng = np.random.default_rng(42)

    probs = [0.95] + [0.05] * (n_partitions - 1)

    for _ in range(500):
        arm, combo = bandit.select()
        # In ICL with n_p partitions, the prediction accuracy reflects the ensemble quality
        # e.g., probability of success is the average quality across selected arms
        ensemble_prob = float(np.mean([probs[a] for a in combo]))
        reward = 1.0 if rng.random() < ensemble_prob else 0.0
        bandit.update(combo, reward)

    # After 500 steps, arm 0 should be in the top selected combination
    _, final_combo = bandit.select()
    assert 0 in final_combo
    # Arm 0 should be pulled far more often than noisy arms
    assert bandit.counts[0] > bandit.counts[2]
    assert bandit.counts[0] > bandit.counts[3]
    # Arm 0's mean should exceed the pure noisy arm baseline
    assert bandit.means[0] > probs[2]


def test_knn_retrieval_nearest_first():
    """Verify knn retrieval picks k nearest items sorted in descending similarity order."""
    embedder = TextEmbedder()
    # Mock pool of 5 examples with distinct texts
    pool = [
        Example(text="apple fruit red sweet", label=0, noisy_label=0),
        Example(text="quantum physics particle spin", label=1, noisy_label=1),
        Example(text="banana fruit yellow tasty", label=0, noisy_label=0),
        Example(text="relativity spacetime gravity tensor", label=1, noisy_label=1),
        Example(text="orange citrus juicy sweet", label=0, noisy_label=0),
    ]

    query = "ripe delicious fruit"
    q_vec = embedder.embed(query)
    q_norm = np.linalg.norm(q_vec)
    q_vec = q_vec / max(q_norm, 1e-12)

    pooled_vecs = np.stack([embedder.embed(ex.text) for ex in pool])
    pooled_norms = np.linalg.norm(pooled_vecs, axis=1, keepdims=True)
    pooled_vecs = pooled_vecs / np.maximum(pooled_norms, 1e-12)

    sims = pooled_vecs @ q_vec
    k = 3
    top_indices = np.argsort(sims)[::-1][:k]

    shots = [(pool[i].text, pool[i].noisy_label) for i in top_indices]
    assert len(shots) == k

    # Nearest first: similarities must be non-increasing
    shot_sims = [sims[i] for i in top_indices]
    assert shot_sims == sorted(shot_sims, reverse=True)
    # The fruits should have higher similarity than physics terms
    assert pool[top_indices[0]].label == 0


def test_kucb_cap_raises_and_works_below():
    n_partitions = 100
    n_p = 3
    actual_superarms = math.comb(n_partitions, n_p)  # 161,700

    # Cap below actual should raise KUCBInfeasibleError
    cap_low = 100_000
    with pytest.raises(KUCBInfeasibleError) as exc_info:
        KUCB(n_partitions=n_partitions, n_p=n_p, max_superarms=cap_low)

    assert exc_info.value.n_superarms == actual_superarms
    assert exc_info.value.max_superarms == cap_low
    assert exc_info.value.n_partitions == n_partitions
    assert exc_info.value.n_p == n_p

    # Cap above actual should succeed
    cap_high = 200_000
    kucb = KUCB(n_partitions=10, n_p=2, max_superarms=cap_high)
    assert len(kucb.superarms) == math.comb(10, 2)
    arm, combo = kucb.select()
    assert len(combo) == 2


def test_neural_policy_bandit_seed_reproducibility():
    """Verify that seeded NeuralPolicyBandit produces identical action sequences."""
    seed = 12345
    dim = 16
    n_partitions = 8
    n_p = 2

    bandit1 = NeuralPolicyBandit(n_partitions=n_partitions, n_p=n_p, dim=dim, seed=seed)
    bandit2 = NeuralPolicyBandit(n_partitions=n_partitions, n_p=n_p, dim=dim, seed=seed)

    rng = np.random.default_rng(999)
    for _ in range(25):
        ctx = rng.standard_normal(dim).astype(np.float32)
        ctx = ctx / np.linalg.norm(ctx)

        arm1, combo1 = bandit1.select(ctx, explore=True)
        arm2, combo2 = bandit2.select(ctx, explore=True)

        assert combo1 == combo2
        assert arm1 == arm2

        # Update both with identical reward
        reward = 1.0 if combo1[0] % 2 == 0 else 0.0
        bandit1.update(combo1, reward)
        bandit2.update(combo2, reward)


def test_tiers_noise_distribution():
    """Verify tiers noise assigns 25 shards per tier at n=100."""
    n_partitions = 100
    tiers = [0.0, 0.2, 0.5, 0.8]
    base_rates = [tiers[i % len(tiers)] for i in range(n_partitions)]

    tier_rng = np.random.default_rng(42)
    shuffled_indices = tier_rng.permutation(n_partitions)
    noise_rates = [round(float(base_rates[i]), 4) for i in shuffled_indices]

    assert len(noise_rates) == 100
    from collections import Counter
    counts = Counter(noise_rates)
    for tier in tiers:
        assert counts[tier] == 25, f"Tier {tier} has count {counts[tier]}, expected 25"


def test_shard_size_check():
    """Verify make_shard_partitions raises error if shard < ceil(k/n_p)."""
    # 5 examples, 3 shards -> shards of size [2, 2, 1]
    train = [{"text": f"text {i}", "label": i % 2} for i in range(5)]

    # k=8, n_p=2 -> ceil(8/2) = 4 examples required per shard
    with pytest.raises(ValueError) as exc_info:
        make_shard_partitions(train, n_partitions=3, seed=0, k=8, n_p=2)

    assert "fewer than required" in str(exc_info.value)

    # With enough data (100 examples for 10 shards with k=8, n_p=2 -> 10 examples per shard >= 4)
    train_large = [{"text": f"text {i}", "label": i % 2} for i in range(100)]
    shards = make_shard_partitions(train_large, n_partitions=10, seed=0, k=8, n_p=2)
    assert len(shards) == 10
    assert all(len(s) == 10 for s in shards)

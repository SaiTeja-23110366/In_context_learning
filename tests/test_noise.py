"""
test_noise.py
=============
Unit tests for our independently developed noise_module.py.

Run with:  python -m pytest tests/test_noise.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path when running directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.noise_module import LabelNoiseInjector, Example


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_partition(n: int, label: int) -> list[dict]:
    """Make n identical examples all with the same label."""
    return [{"text": f"example {i}", "label": label} for i in range(n)]


def _make_partitions(n_per_part: int, n_classes: int) -> list[list[dict]]:
    """
    Make n_classes partitions, partition i has all examples with label i.
    Useful for verifying per-partition noise rates.
    """
    return [_make_partition(n_per_part, c) for c in range(n_classes)]


# ── Test 1: zero noise — labels untouched ───────────────────────────────────

class TestZeroNoise:
    """q=0.0 should never corrupt any label."""

    def test_single_partition_zero_noise(self):
        parts = [_make_partition(200, label=2)]
        injector = LabelNoiseInjector(noise_rates=[0.0], n_classes=4, seed=0)
        result = injector.inject(parts)
        for ex in result[0]:
            assert ex.label == ex.noisy_label == 2, (
                "Zero noise must never change the label."
            )

    def test_all_partitions_zero_noise(self):
        parts = _make_partitions(n_per_part=100, n_classes=4)
        injector = LabelNoiseInjector(
            noise_rates=[0.0, 0.0, 0.0, 0.0], n_classes=4, seed=42
        )
        result = injector.inject(parts)
        for part_idx, part in enumerate(result):
            for ex in part:
                assert ex.label == ex.noisy_label, (
                    f"Partition {part_idx}: zero noise corrupted a label."
                )


# ── Test 2: full noise — every label changed ─────────────────────────────────

class TestFullNoise:
    """q=1.0 means every single label must be corrupted."""

    def test_all_labels_changed(self):
        parts = [_make_partition(500, label=0)]
        injector = LabelNoiseInjector(noise_rates=[1.0], n_classes=4, seed=0)
        result = injector.inject(parts)
        for ex in result[0]:
            assert ex.noisy_label != ex.label, (
                "Full noise must change every label."
            )

    def test_all_labels_changed_multiclass(self):
        """Works for different label values too."""
        for true_label in range(4):
            parts = [_make_partition(200, label=true_label)]
            injector = LabelNoiseInjector(
                noise_rates=[1.0], n_classes=4, seed=7
            )
            result = injector.inject(parts)
            for ex in result[0]:
                assert ex.noisy_label != true_label


# ── Test 3: corrupted label is ALWAYS different from original ─────────────────

class TestCorruptionExclusion:
    """The paper says: corrupted label drawn from {0..c-1} \\ {y}."""

    def test_corrupted_never_equals_original(self):
        """Run with high noise rate across many examples and all label values."""
        n_classes = 4
        for true_label in range(n_classes):
            parts = [_make_partition(1000, label=true_label)]
            injector = LabelNoiseInjector(
                noise_rates=[0.8], n_classes=n_classes, seed=123
            )
            result = injector.inject(parts)
            for ex in result[0]:
                if ex.noisy_label != ex.label:
                    assert ex.noisy_label != true_label, (
                        "A corrupted label must not equal the original label."
                    )

    def test_corrupted_in_valid_range(self):
        """Noisy labels must still be valid class indices."""
        n_classes = 6
        parts = [_make_partition(500, label=3)]
        injector = LabelNoiseInjector(
            noise_rates=[1.0], n_classes=n_classes, seed=99
        )
        result = injector.inject(parts)
        for ex in result[0]:
            assert 0 <= ex.noisy_label < n_classes


# ── Test 4: uniform distribution of corrupted labels ─────────────────────────

class TestUniformDistribution:
    """
    The paper says the corrupted label is chosen UNIFORMLY from the
    remaining classes.  With enough samples, the empirical distribution
    should be approximately uniform.
    """

    def test_uniform_over_remaining_classes(self):
        n_classes = 4
        n_samples = 10_000
        true_label = 0
        parts = [_make_partition(n_samples, label=true_label)]
        injector = LabelNoiseInjector(
            noise_rates=[1.0], n_classes=n_classes, seed=0
        )
        result = injector.inject(parts)

        # Count how often each incorrect label appears
        counts = {c: 0 for c in range(n_classes) if c != true_label}
        for ex in result[0]:
            counts[ex.noisy_label] += 1

        # With n=10000 and 3 bins, expected count per bin ~ 3333
        # Allow ±5% tolerance
        expected = n_samples / (n_classes - 1)
        for cls, cnt in counts.items():
            assert abs(cnt - expected) / expected < 0.05, (
                f"Class {cls} appeared {cnt} times; expected ~{expected:.0f}. "
                "Noise distribution is not uniform."
            )


# ── Test 5: per-partition noise rates ────────────────────────────────────────

class TestPerPartitionRates:
    """Each partition must use its own noise rate independently."""

    def test_mixed_rates(self):
        """Partition 0: q=0, Partition 1: q=1."""
        parts = [
            _make_partition(200, label=0),  # clean
            _make_partition(200, label=1),  # fully noisy
        ]
        injector = LabelNoiseInjector(
            noise_rates=[0.0, 1.0], n_classes=4, seed=0
        )
        result = injector.inject(parts)

        # Partition 0: all clean
        for ex in result[0]:
            assert ex.label == ex.noisy_label == 0

        # Partition 1: all corrupted
        for ex in result[1]:
            assert ex.noisy_label != 1


# ── Test 6: input validation ──────────────────────────────────────────────────

class TestInputValidation:
    def test_mismatched_partition_count_raises(self):
        parts = [_make_partition(10, 0), _make_partition(10, 1)]
        injector = LabelNoiseInjector(
            noise_rates=[0.5],  # only 1 rate for 2 partitions
            n_classes=4,
            seed=0,
        )
        with pytest.raises(ValueError, match="noise_rates"):
            injector.inject(parts)

    def test_invalid_noise_rate_raises(self):
        with pytest.raises(ValueError, match="noise_rates"):
            LabelNoiseInjector(noise_rates=[1.5], n_classes=4)

    def test_negative_noise_rate_raises(self):
        with pytest.raises(ValueError, match="noise_rates"):
            LabelNoiseInjector(noise_rates=[-0.1], n_classes=4)

    def test_fewer_than_two_classes_raises(self):
        with pytest.raises(ValueError, match="n_classes"):
            LabelNoiseInjector(noise_rates=[0.5], n_classes=1)


# ── Test 7: original labels always preserved ─────────────────────────────────

class TestOriginalLabelPreservation:
    """The .label field must always hold the ground-truth, never touched."""

    def test_label_field_unchanged_under_full_noise(self):
        parts = [
            [{"text": "t", "label": c} for c in range(4)] * 50
        ]
        injector = LabelNoiseInjector(noise_rates=[1.0], n_classes=4, seed=0)
        result = injector.inject(parts)
        original_labels = [ex["label"] for ex in parts[0]]
        injected_labels = [ex.label for ex in result[0]]
        assert original_labels == injected_labels, (
            "The original .label field must never be modified."
        )

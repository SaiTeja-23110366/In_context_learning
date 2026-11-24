"""
test_mmlu_integration.py
========================
Tests full end-to-end MMLU data loading, domain partitioning, noise injection,
and prompt formatting.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_task, make_domain_partitions, sample_shots_from_parts, subsample_test
from src.noise_module import LabelNoiseInjector
from src.icl import build_prompt

def test_mmlu_pipeline():
    cache_dir = "/data/pulakurthi.manohar/hf_home"
    print("Loading MMLU dataset...")
    train, test, label_names = load_task("mmlu", cache_dir, seed=0)
    assert label_names == ["A", "B", "C", "D"]
    assert len(train) > 0
    assert len(test) > 0
    print(f"Loaded {len(train)} training examples and {len(test)} test examples.")

    test_sub = subsample_test(test, 50, seed=0)
    assert len(test_sub) == 50

    print("Testing 4-domain partitioning (STEM, Humanities, Social Sciences, Other)...")
    parts4 = make_domain_partitions(train, 4, seed=0)
    assert len(parts4) == 4
    for i, p in enumerate(parts4):
        print(f"  Partition {i} size: {len(p)}")
        assert len(p) > 100

    print("Testing 8-domain partitioning...")
    parts8 = make_domain_partitions(train, 8, seed=0)
    assert len(parts8) == 8
    for i, p in enumerate(parts8):
        print(f"  8-part Partition {i} size: {len(p)}")
        assert len(p) > 50

    print("Testing noise injection on 4 partitions...")
    injector = LabelNoiseInjector(noise_rates=[0.0, 0.2, 0.5, 0.8], n_classes=4, seed=0)
    noisy_parts = injector.inject(parts4)
    assert len(noisy_parts) == 4

    print("Testing prompt generation...")
    import numpy as np
    rng = np.random.default_rng(0)
    shots = sample_shots_from_parts([noisy_parts[0], noisy_parts[1]], k=4, rng=rng, use_noisy=True)
    assert len(shots) == 4
    prompt = build_prompt(shots, test_sub[0]["text"], label_names)
    assert "Classify the text into exactly one class: A | B | C | D." in prompt
    assert "Question:" in prompt
    print("Prompt sample preview:\n" + "-"*40)
    print(prompt[:400] + "\n...")
    print("MMLU integration test passed successfully!")

if __name__ == "__main__":
    test_mmlu_pipeline()

#!/usr/bin/env python3
"""
run_experiments.py
==================
Main experiment runner for:
    "Robust In-Context Learning via Multi-Armed Bandit-Based Partition Selection"

Usage:
    python scripts/run_experiments.py --config configs/agnews_iid_7b.yaml

The script follows this pipeline for EVERY method:
    1. Load dataset (clean)
    2. Create clean partitions (IID or Non-IID)
    3. Inject label noise via our standalone noise_module.LabelNoiseInjector
    4. Run the selected method (no_noise / worst_case / uniform / ucb / kucb)
    5. Observe reward, update bandit statistics
    6. Save per-seed JSON logs + aggregated summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandit import KUCB, UCB1
from src.data_utils import (
    load_task,
    make_iid_partitions,
    make_non_iid_partitions,
    sample_shots,
    sample_shots_from_parts,
    subsample_test,
)
from src.icl import load_model, predict

# ── OUR ADDITION: standalone noise module ─────────────────────────────────────
from src.noise_module import LabelNoiseInjector


# ── Partitioning ──────────────────────────────────────────────────────────────

def build_clean_partitions(train: list[dict], cfg: dict) -> list[list[dict]]:
    """Create clean (noise-free) partitions based on config."""
    mode = cfg.get("partition_mode", "iid")
    kwargs = dict(
        train=train,
        n_partitions=cfg["n_partitions"],
        seed=cfg["seed"],
    )
    if mode == "iid":
        return make_iid_partitions(**kwargs)
    if mode == "non_iid":
        return make_non_iid_partitions(
            **kwargs,
            dirichlet_alpha=float(cfg.get("dirichlet_alpha", 0.3)),
        )
    raise ValueError(f"Unknown partition_mode: {mode!r}")


# ── Method runner ─────────────────────────────────────────────────────────────

def run_method(
    name: str,
    noisy_partitions,       # list[list[Example]] — noise already injected
    clean_partitions,       # list[list[Example]] — no noise (for no_noise baseline)
    test: list[dict],
    cfg: dict,
    tok,
    model,
    rng: np.random.Generator,
    label_names: list[str],
) -> tuple[float, list[dict], dict]:
    k = cfg["k"]
    n_p = int(cfg.get("n_p", 2))
    ucb_c = float(cfg.get("ucb_c", 2.0))
    worst_idx = int(np.argmax(cfg["noise_rates"]))

    # Flatten all clean partitions into one pool for the no_noise baseline
    clean_pool = [item for part in clean_partitions for item in part]

    ucb = UCB1(len(noisy_partitions), c=ucb_c) if name == "ucb" else None
    kucb = KUCB(len(noisy_partitions), n_p, c=ucb_c) if name == "kucb" else None

    correct = 0
    rows = []

    for ex in tqdm(test, desc=name, leave=False):
        arm = -1

        if name == "no_noise":
            shots = sample_shots(clean_pool, k, rng, use_noisy=False)

        elif name == "worst_case":
            arm = worst_idx
            shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)

        elif name == "uniform":
            arm = int(rng.integers(0, len(noisy_partitions)))
            shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)

        elif name == "ucb":
            arm = ucb.select()
            shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)

        elif name == "kucb":
            arm, combo = kucb.select()
            shots = sample_shots_from_parts(
                [noisy_partitions[i] for i in combo], k, rng, use_noisy=True
            )
        else:
            raise ValueError(f"Unknown method: {name!r}")

        pred = predict(tok, model, shots, ex["text"], label_names)
        reward = 1.0 if pred == ex["label"] else 0.0

        if ucb is not None:
            ucb.update(arm, reward)
        if kucb is not None:
            kucb.update(arm, reward)

        correct += int(reward)
        row = {"pred": pred, "gold": ex["label"], "arm": arm, "reward": reward}
        if name == "kucb":
            row["superarm"] = list(kucb.superarms[arm])
        rows.append(row)

    acc = correct / max(len(test), 1)

    extra = {
        "gold_counts": dict(Counter(ex["label"] for ex in test)),
        "pred_counts": dict(Counter(r["pred"] for r in rows)),
        "partition_sizes": [len(p) for p in noisy_partitions],
    }
    if ucb is not None:
        extra["arm_counts"] = ucb.counts
        extra["arm_means"] = ucb.means
    if kucb is not None:
        extra["superarms"] = [list(s) for s in kucb.superarms]
        extra["arm_counts"] = kucb.ucb.counts
        extra["arm_means"] = kucb.ucb.means

    return acc, rows, extra


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_HOME", cfg["hf_cache"])
    tok, model = load_model(cfg["model_path"])

    seeds = cfg.get("seeds") or [cfg["seed"]]
    run_name = cfg.get("run_name", "run")
    methods = cfg.get("methods", ["no_noise", "worst_case", "ucb"])
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []

    for seed in seeds:
        cfg = dict(cfg)
        cfg["seed"] = int(seed)
        rng = np.random.default_rng(cfg["seed"])

        # ── 1. Load dataset ────────────────────────────────────────────────
        train, test, label_names = load_task(
            cfg.get("dataset", "ag_news"), cfg["hf_cache"], seed=cfg["seed"]
        )
        test = subsample_test(test, cfg["n_test"], cfg["seed"])
        n_classes = len(label_names)

        # ── 2. Create CLEAN partitions ────────────────────────────────────
        clean_parts = build_clean_partitions(train, cfg)

        # ── 3. Inject noise via our standalone module (OUR ADDITION) ──────
        #
        # Noise rates are drawn randomly per seed (not fixed).
        # This is the correct approach: the datasets are clean, so we
        # simulate real-world noisy sources by assigning each partition
        # a random noise rate q_i ~ Uniform(0, max_noise).
        # The bandit has no knowledge of these rates — it must discover
        # which partitions are reliable purely from rewards.
        #
        # At least one partition is guaranteed to be clean (q=0) so the
        # bandit always has a reliable source to find.
        noise_mode = cfg.get("noise_mode", "fixed")
        if noise_mode == "random":
            max_noise = float(cfg.get("max_noise", 0.8))
            n_parts = len(clean_parts)
            # Draw random noise rates for all partitions
            raw_rates = rng.uniform(0.0, max_noise, size=n_parts).tolist()
            # Force the partition with lowest drawn rate to be exactly 0
            # (guarantees at least one reliable source exists)
            min_idx = int(np.argmin(raw_rates))
            raw_rates[min_idx] = 0.0
            noise_rates = [round(float(q), 4) for q in raw_rates]
        else:
            # Fixed mode: use values from config as before
            noise_rates = cfg["noise_rates"]

        print(f"  noise_rates (seed {cfg['seed']}): {noise_rates}")

        injector = LabelNoiseInjector(
            noise_rates=noise_rates,
            n_classes=n_classes,
            seed=cfg["seed"],
        )
        noisy_parts = injector.inject(clean_parts)

        summary = {
            "seed": cfg["seed"],
            "label_names": label_names,
            "dataset": cfg.get("dataset", "ag_news"),
            "partition_mode": cfg.get("partition_mode", "iid"),
            "noise_rates": noise_rates,      # actual rates used (random or fixed)
            "k": cfg["k"],
            "partition_sizes": [len(p) for p in noisy_parts],
        }

        print(f"\n=== seed {cfg['seed']} | {cfg.get('dataset')} | "
              f"{cfg.get('partition_mode')} ===")

        seed_acc: dict[str, float] = {}
        for method in methods:
            acc, rows, extra = run_method(
                method, noisy_parts, noisy_parts, test,
                cfg, tok, model, rng, label_names,
            )
            seed_acc[method] = acc
            summary[method] = {
                "accuracy": acc,
                **{k: v for k, v in extra.items() if k != "gold_counts"},
            }
            fname = out_dir / f"{method}_{run_name}_seed{cfg['seed']}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(
                    {"accuracy": acc, "n": len(test), "extra": extra, "rows": rows},
                    f,
                )
            print(f"  {method}: {acc * 100:.2f}%")

        per_seed.append(seed_acc)
        sfname = out_dir / f"summary_{run_name}_seed{cfg['seed']}.json"
        with open(sfname, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ── Aggregate across seeds ─────────────────────────────────────────────
    agg: dict = {"seeds": list(seeds), "run_name": run_name}
    for method in methods:
        vals = [s[method] for s in per_seed]
        agg[method] = {
            "values": [round(v * 100, 4) for v in vals],
            "mean": round(float(np.mean(vals)) * 100, 2),
            "std": round(
                float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0) * 100, 2
            ),
        }
        print(
            f"{method}: {agg[method]['mean']:.2f} ± {agg[method]['std']:.2f}%"
            f"  {agg[method]['values']}"
        )

    agg_fname = out_dir / f"summary_{run_name}.json"
    with open(agg_fname, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved summary → {agg_fname}")


if __name__ == "__main__":
    main()

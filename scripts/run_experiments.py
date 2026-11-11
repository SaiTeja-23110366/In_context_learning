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
import math
import os
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandit import (
    KUCB,
    UCB1,
    IndepUCB,
    KUCBInfeasibleError,
    LinUCB,
    NeuralPolicyBandit,
    TextEmbedder,
)
from src.data_utils import (
    load_task,
    make_domain_partitions,
    make_iid_partitions,
    make_non_iid_partitions,
    make_shard_partitions,
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
    if mode == "shards":
        return make_shard_partitions(
            train=train,
            n_partitions=cfg["n_partitions"],
            seed=cfg["seed"],
            k=cfg["k"],
            n_p=int(cfg.get("n_p", 2)),
        )
    if mode == "non_iid":
        return make_non_iid_partitions(
            **kwargs,
            dirichlet_alpha=float(cfg.get("dirichlet_alpha", 0.3)),
        )
    if mode == "domain":
        return make_domain_partitions(**kwargs)
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
    noise_rates: list[float] | None = None,
    embedder: TextEmbedder | None = None,
) -> tuple[float, list[dict], dict]:
    k = cfg["k"]
    n_p = int(cfg.get("n_p", 2))
    ucb_c = float(cfg.get("ucb_c", 2.0))
    rates = noise_rates if noise_rates is not None else cfg["noise_rates"]
    worst_idx = int(np.argmax(rates))

    # Flatten all clean partitions into one pool for the no_noise baseline
    clean_pool = [item for part in clean_partitions for item in part]

    ucb = UCB1(len(noisy_partitions), c=ucb_c) if name == "ucb" else None
    max_superarms = int(cfg.get("max_superarms", 2_000_000))
    kucb = (
        KUCB(len(noisy_partitions), n_p, c=ucb_c, max_superarms=max_superarms)
        if name == "kucb"
        else None
    )
    indep_ucb = (
        IndepUCB(len(noisy_partitions), n_p=n_p, c=ucb_c)
        if name == "indep_ucb"
        else None
    )
    dim = embedder.dim if embedder is not None else 384
    linucb = (
        LinUCB(
            len(noisy_partitions),
            n_p=n_p,
            dim=dim,
            alpha=float(cfg.get("linucb_alpha", 1.0)),
        )
        if name == "linucb"
        else None
    )
    neural_pg = (
        NeuralPolicyBandit(
            len(noisy_partitions),
            n_p=n_p,
            dim=dim,
            lr=float(cfg.get("pg_lr", 1e-3)),
            seed=int(cfg["seed"]),
        )
        if name == "neural_pg"
        else None
    )

    # knn_retrieval setup: precompute normalized embeddings for pooled noisy training data
    pooled_noisy_examples: list = []
    pooled_vecs = None
    if name == "knn_retrieval":
        assert embedder is not None, "TextEmbedder required for knn_retrieval"
        for part in noisy_partitions:
            pooled_noisy_examples.extend(part)
        raw_vecs = np.stack([
            embedder.embed(ex.text if hasattr(ex, "text") else ex["text"])
            for ex in pooled_noisy_examples
        ])
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        pooled_vecs = raw_vecs / np.maximum(norms, 1e-12)

    # oracle_domain setup: map each gold class to its domain partition index
    label_to_domain_part: dict[int, int] = {}
    if name == "oracle_domain":
        for p_idx, part in enumerate(noisy_partitions):
            for item in part:
                lbl = item.label if hasattr(item, "label") else item["label"]
                label_to_domain_part[int(lbl)] = p_idx

    correct = 0
    rows = []

    for ex in tqdm(test, desc=name, leave=False):
        arm = -1
        combo = None
        vec = None

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

        elif name == "indep_ucb":
            arm, combo = indep_ucb.select()
            shots = sample_shots_from_parts(
                [noisy_partitions[i] for i in combo], k, rng, use_noisy=True
            )

        elif name == "knn_retrieval":
            assert embedder is not None and pooled_vecs is not None
            q_vec = embedder.embed(ex["text"])
            q_norm = np.linalg.norm(q_vec)
            q_vec = q_vec / max(q_norm, 1e-12)
            sims = pooled_vecs @ q_vec
            top_k_idxs = np.argsort(sims)[::-1][:k]
            shots = []
            for idx in top_k_idxs:
                cand = pooled_noisy_examples[idx]
                if hasattr(cand, "noisy_label"):
                    y = cand.noisy_label
                    t = cand.text
                elif isinstance(cand, dict):
                    y = cand.get("noisy_label", cand["label"])
                    t = cand["text"]
                else:
                    y = getattr(cand, "noisy_label", cand.label)
                    t = cand.text
                shots.append((t, int(y)))

        elif name == "oracle_domain":
            gold_lbl = int(ex["label"])
            if gold_lbl not in label_to_domain_part:
                raise ValueError(
                    f"oracle_domain: gold class {gold_lbl} not found in any domain partition"
                )
            arm = label_to_domain_part[gold_lbl]
            combo = (arm,)
            shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)

        elif name == "linucb":
            assert embedder is not None, "TextEmbedder required for LinUCB"
            vec = embedder.embed(ex["text"])
            arm, combo = linucb.select(vec)
            shots = sample_shots_from_parts(
                [noisy_partitions[i] for i in combo], k, rng, use_noisy=True
            )

        elif name == "neural_pg":
            assert embedder is not None, "TextEmbedder required for Neural Policy Gradient"
            vec = embedder.embed(ex["text"])
            arm, combo = neural_pg.select(vec, explore=True)
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
        if indep_ucb is not None and combo is not None:
            indep_ucb.update(combo, reward)
        if linucb is not None and combo is not None and vec is not None:
            linucb.update(combo, vec, reward)
        if neural_pg is not None and combo is not None:
            neural_pg.update(combo, reward)

        correct += int(reward)
        row = {"pred": pred, "gold": ex["label"], "arm": arm, "reward": reward}
        if combo is not None:
            row["superarm"] = list(combo)
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
    if indep_ucb is not None:
        extra["arm_counts"] = indep_ucb.counts
        extra["arm_means"] = indep_ucb.means
    if linucb is not None:
        extra["arm_counts"] = linucb.counts
        extra["arm_rewards"] = linucb.rewards
    if neural_pg is not None:
        extra["arm_counts"] = neural_pg.counts
        extra["arm_rewards"] = neural_pg.rewards

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

    embedder = None
    if any(m in ("linucb", "neural_pg", "knn_retrieval") for m in methods):
        print("Initializing TextEmbedder for contextual bandits and retrieval...")
        embedder = TextEmbedder(cache_dir=cfg.get("hf_cache"))

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
        replay_test = int(cfg.get("replay_test", 1))
        if replay_test > 1:
            test = test * replay_test
        n_classes = len(label_names)

        # ── 2. Create CLEAN partitions ────────────────────────────────────
        clean_parts = build_clean_partitions(train, cfg)

        # ── 3. Inject noise via our standalone module (OUR ADDITION) ──────
        noise_mode = cfg.get("noise_mode", "fixed")
        if noise_mode == "random":
            max_noise = float(cfg.get("max_noise", 0.8))
            n_parts = len(clean_parts)
            raw_rates = rng.uniform(0.0, max_noise, size=n_parts).tolist()
            min_idx = int(np.argmin(raw_rates))
            raw_rates[min_idx] = 0.0
            noise_rates = [round(float(q), 4) for q in raw_rates]
        elif noise_mode == "tiers":
            tiers = [0.0, 0.2, 0.5, 0.8]
            n_parts = len(clean_parts)
            base_rates = [tiers[i % len(tiers)] for i in range(n_parts)]
            tier_rng = np.random.default_rng(cfg["seed"])
            shuffled_indices = tier_rng.permutation(n_parts)
            noise_rates = [round(float(base_rates[i]), 4) for i in shuffled_indices]
        else:
            # Fixed mode: use values from config as before
            noise_rates = cfg["noise_rates"]

        cfg["noise_rates"] = noise_rates

        print(f"  noise_rates (seed {cfg['seed']}): {noise_rates[:8]} ... (len={len(noise_rates)})")

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
            "noise_mode": noise_mode,
            "noise_rates": noise_rates,      # actual rates used
            "k": cfg["k"],
            "replay_test": replay_test,
            "partition_sizes": [len(p) for p in noisy_parts],
        }

        print(f"\n=== seed {cfg['seed']} | {cfg.get('dataset')} | "
              f"{cfg.get('partition_mode')} ===")

        seed_acc: dict[str, float | None] = {}
        for method in methods:
            fname = out_dir / f"{method}_{run_name}_seed{cfg['seed']}.json"

            # Check if cached already (resumable)
            if fname.exists():
                try:
                    with open(fname, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    if cached_data.get("status") == "infeasible":
                        print(f"  {method}: infeasible (cached, C={cached_data.get('superarms')})")
                        summary[method] = cached_data
                        seed_acc[method] = None
                        continue
                    elif "accuracy" in cached_data:
                        acc = float(cached_data["accuracy"])
                        extra = cached_data.get("extra", {})
                        seed_acc[method] = acc
                        summary[method] = {
                            "accuracy": acc,
                            **{k: v for k, v in extra.items() if k != "gold_counts"},
                        }
                        print(f"  {method}: {acc * 100:.2f}% (cached)")
                        continue
                except Exception as e:
                    print(f"Warning: could not load cached {fname}: {e}, recomputing...")

            # Run method with timing and memory tracking
            try:
                tracemalloc.start()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                acc, rows, extra = run_method(
                    method, noisy_parts, clean_parts, test,
                    cfg, tok, model, rng, label_names,
                    noise_rates=noise_rates,
                    embedder=embedder,
                )

                elapsed = time.perf_counter() - t0
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_cpu_mb = round(peak_bytes / (1024 * 1024), 2)
                peak_gpu_mb = (
                    round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    if torch.cuda.is_available()
                    else 0.0
                )

                # Arms calculation
                if method == "kucb":
                    n_arms = math.comb(len(noisy_parts), int(cfg.get("n_p", 2)))
                elif method in ("ucb", "indep_ucb", "linucb", "neural_pg", "uniform", "oracle_domain"):
                    n_arms = len(noisy_parts)
                elif method in ("worst_case", "no_noise"):
                    n_arms = 1
                elif method == "knn_retrieval":
                    n_arms = 0
                else:
                    n_arms = len(noisy_parts)

                extra["wall_clock_seconds"] = round(elapsed, 2)
                extra["peak_cpu_mb"] = peak_cpu_mb
                extra["peak_gpu_mb"] = peak_gpu_mb
                extra["n_arms"] = n_arms

                seed_acc[method] = acc
                summary[method] = {
                    "accuracy": acc,
                    **{k: v for k, v in extra.items() if k != "gold_counts"},
                }
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(
                        {"accuracy": acc, "n": len(test), "extra": extra, "rows": rows},
                        f,
                    )
                print(f"  {method}: {acc * 100:.2f}% ({elapsed:.1f}s, {peak_cpu_mb}MB CPU, {peak_gpu_mb}MB GPU)")

            except KUCBInfeasibleError as err:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                print(f"  {method}: infeasible (C={err.n_superarms} > cap {err.max_superarms})")
                infeasible_record = {
                    "status": "infeasible",
                    "superarms": err.n_superarms,
                    "n_partitions": err.n_partitions,
                    "n_p": err.n_p,
                    "max_superarms": err.max_superarms,
                }
                seed_acc[method] = None
                summary[method] = infeasible_record
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(infeasible_record, f, indent=2)

        per_seed.append(seed_acc)
        sfname = out_dir / f"summary_{run_name}_seed{cfg['seed']}.json"
        with open(sfname, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ── Aggregate across seeds ─────────────────────────────────────────────
    agg: dict = {"seeds": list(seeds), "run_name": run_name}
    for method in methods:
        vals = [s[method] for s in per_seed if s.get(method) is not None]
        if not vals:
            n_superarms = math.comb(int(cfg["n_partitions"]), int(cfg.get("n_p", 2)))
            agg[method] = {
                "status": "infeasible",
                "superarms": n_superarms,
            }
            print(f"{method}: infeasible (C={n_superarms})")
            continue
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

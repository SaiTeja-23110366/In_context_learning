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


def build_partitions(train, cfg):
    mode = cfg.get("partition_mode", "iid")
    kwargs = dict(
        train=train,
        n_partitions=cfg["n_partitions"],
        noise_rates=cfg["noise_rates"],
        seed=cfg["seed"],
    )
    if mode == "iid":
        return make_iid_partitions(**kwargs)
    if mode == "non_iid":
        return make_non_iid_partitions(
            **kwargs, dirichlet_alpha=float(cfg.get("dirichlet_alpha", 0.3))
        )
    raise ValueError(mode)


def run_method(name, partitions, test, cfg, tok, model, rng, label_names):
    k = cfg["k"]
    use_noisy = name != "no_noise"
    n_p = int(cfg.get("n_p", 2))
    ucb = UCB1(len(partitions), c=float(cfg["ucb_c"])) if name == "ucb" else None
    kucb = KUCB(len(partitions), n_p, c=float(cfg["ucb_c"])) if name == "kucb" else None
    worst_idx = int(np.argmax(cfg["noise_rates"]))
    clean_pool = [item for part in partitions for item in part]
    correct = 0
    rows = []
    for ex in tqdm(test, desc=name):
        arm = -1
        if name == "no_noise":
            shots = sample_shots(clean_pool, k, rng, use_noisy=False)
        elif name == "worst_case":
            arm = worst_idx
            shots = sample_shots(partitions[worst_idx], k, rng, use_noisy=True)
        elif name == "uniform":
            arm = int(rng.integers(0, len(partitions)))
            shots = sample_shots(partitions[arm], k, rng, use_noisy=True)
        elif name == "ucb":
            arm = ucb.select()
            shots = sample_shots(partitions[arm], k, rng, use_noisy=True)
        elif name == "kucb":
            arm, combo = kucb.select()
            shots = sample_shots_from_parts(
                [partitions[i] for i in combo], k, rng, use_noisy=True
            )
        else:
            raise ValueError(name)

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
        "partition_sizes": [len(p) for p in partitions],
    }
    if ucb is not None:
        extra["arm_counts"] = ucb.counts
        extra["arm_means"] = ucb.means
    if kucb is not None:
        extra["superarms"] = [list(s) for s in kucb.superarms]
        extra["arm_counts"] = kucb.ucb.counts
        extra["arm_means"] = kucb.ucb.means
    return acc, rows, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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

    per_seed = []
    for seed in seeds:
        cfg = dict(cfg)
        cfg["seed"] = int(seed)
        rng = np.random.default_rng(cfg["seed"])
        train, test, label_names = load_task(
            cfg.get("dataset", "ag_news"), cfg["hf_cache"], seed=cfg["seed"]
        )
        test = subsample_test(test, cfg["n_test"], cfg["seed"])
        partitions = build_partitions(train, cfg)
        summary = {
            "seed": cfg["seed"],
            "label_names": label_names,
            "dataset": cfg.get("dataset", "ag_news"),
            "partition_mode": cfg.get("partition_mode", "iid"),
            "noise_rates": cfg["noise_rates"],
            "k": cfg["k"],
            "partition_sizes": [len(p) for p in partitions],
        }
        print(f"\n=== seed {cfg['seed']} ===")
        seed_acc = {}
        for method in methods:
            acc, rows, extra = run_method(
                method, partitions, test, cfg, tok, model, rng, label_names
            )
            seed_acc[method] = acc
            summary[method] = {
                "accuracy": acc,
                **{k: v for k, v in extra.items() if k != "gold_counts"},
            }
            with open(out_dir / f"{method}_{run_name}_seed{cfg['seed']}.json", "w", encoding="utf-8") as f:
                json.dump({"accuracy": acc, "n": len(test), "extra": extra, "rows": rows}, f)
            print(f"{method}: {acc:.4f}")
        per_seed.append(seed_acc)
        with open(out_dir / f"summary_{run_name}_seed{cfg['seed']}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    agg = {"seeds": list(seeds)}
    for method in methods:
        vals = [s[method] for s in per_seed]
        agg[method] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        }
        print(f"{method}: {agg[method]['mean']:.4f} ± {agg[method]['std']:.4f} {vals}")
    with open(out_dir / f"summary_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("saved", out_dir / f"summary_{run_name}.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandit import UCB1
from collections import Counter

from src.data_utils import load_agnews, make_iid_partitions, sample_shots, subsample_test
from src.icl import load_model, predict


def run_method(name, partitions, test, cfg, tok, model, rng):
    k = cfg["k"]
    use_noisy = name != "no_noise"
    bandit = UCB1(len(partitions), c=float(cfg["ucb_c"])) if name == "ucb" else None
    worst_idx = int(np.argmax(cfg["noise_rates"]))
    clean_pool = [item for part in partitions for item in part]
    correct = 0
    rows = []
    for ex in tqdm(test, desc=name):
        if name == "no_noise":
            pool = clean_pool
            arm = -1
        elif name == "worst_case":
            pool = partitions[worst_idx]
            arm = worst_idx
        elif name == "ucb":
            arm = bandit.select()
            pool = partitions[arm]
        else:
            raise ValueError(name)
        shots = sample_shots(pool, k, rng, use_noisy=use_noisy)
        pred = predict(tok, model, shots, ex["text"])
        reward = 1.0 if pred == ex["label"] else 0.0
        if bandit is not None:
            bandit.update(arm, reward)
        correct += int(reward)
        rows.append({"pred": pred, "gold": ex["label"], "arm": arm, "reward": reward})
    acc = correct / max(len(test), 1)
    extra = {}
    if bandit is not None:
        extra["arm_counts"] = bandit.counts
        extra["arm_means"] = bandit.means
    extra["gold_counts"] = dict(Counter(ex["label"] for ex in test))
    extra["pred_counts"] = dict(Counter(r["pred"] for r in rows))
    return acc, rows, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "agnews_smoke.yaml"))
    parser.add_argument("--methods", nargs="+", default=["no_noise", "worst_case", "ucb"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_HOME", cfg["hf_cache"])
    rng = np.random.default_rng(cfg["seed"])
    train, test = load_agnews(cfg["hf_cache"])
    test = subsample_test(test, cfg["n_test"], cfg["seed"])
    partitions = make_iid_partitions(
        train, cfg["n_partitions"], cfg["noise_rates"], cfg["seed"]
    )
    tok, model = load_model(cfg["model_path"])

    run_name = cfg.get("run_name", "run")
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_test": len(test), "noise_rates": cfg["noise_rates"], "k": cfg["k"]}
    for method in args.methods:
        acc, rows, extra = run_method(method, partitions, test, cfg, tok, model, rng)
        summary[method] = {"accuracy": acc, **extra}
        with open(out_dir / f"{method}_{run_name}.json", "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "n": len(test), "extra": extra, "rows": rows}, f)
        print(f"{method}: {acc:.4f} extra={extra}")

    with open(out_dir / f"summary_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved", out_dir / f"summary_{run_name}.json")


if __name__ == "__main__":
    main()

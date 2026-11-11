"""
generate_tables.py
==================
Reads result JSON files and prints formatted tables:
1. Table 1a (IID) and Table 1b (Non-IID) paper reproductions
2. Table 2: Domain Contextual & Neural Bandit Comparison (including knn_retrieval & oracle_domain)
3. Table 3: Scaling Experiment (IID Shards + Tiers Noise) with Runtime, Memory, and Combinatorial Superarms
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


MODELS = [
    ("llama3b",  "LLaMA-3.2-3B"),
    ("7b",       "Qwen2.5-7B-Instruct-1M"),
    ("14b",      "Qwen2.5-14B"),
]
DATASETS = [
    ("agnews",   "AG-News"),
    ("ecom",     "Ecommerce"),
    ("syn",      "Synthetic"),
    ("mmlu",     "MMLU"),
    ("banking",  "BANKING-77"),
]


def load_summary(results_dir: Path, run_name: str) -> dict | None:
    path = results_dir / f"summary_{run_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_files(results_dir: Path, run_name: str) -> list[dict]:
    files = list(results_dir.glob(f"summary_{run_name}_seed*.json"))
    res = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                res.append(json.load(fp))
        except Exception:
            pass
    return res


def fmt(d: dict, method: str) -> str:
    if d is None or method not in d:
        return "     —     "
    m = d[method]
    if isinstance(m, dict):
        if m.get("status") == "infeasible":
            return f"infeasible (C={m.get('superarms', '?')})"
        mean = m.get("mean", 0)
        std = m.get("std", 0)
    else:
        mean = m * 100
        std = 0.0
    return f"{mean:.2f}±{std:.2f}"


def print_table(results_dir: Path, mode: str, bandit_method: str, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'Model':<28} {'Dataset':<12} {'Worst-Case':>14} {'No-Noise':>12} {bandit_method.upper():>12}"
    print(header)
    print("-" * 70)
    for mkey, mlabel in MODELS:
        for dskey, dslabel in DATASETS:
            run_name = f"{dskey}_{mode}_{mkey}"
            d = load_summary(results_dir, run_name)
            wc = fmt(d, "worst_case")
            nn = fmt(d, "no_noise")
            ban = fmt(d, bandit_method)
            model_col = mlabel if dskey == "agnews" else ""
            print(f"{model_col:<28} {dslabel:<12} {wc:>14} {nn:>12} {ban:>12}")
        print()


def print_domain_ablation_table(results_dir: Path) -> None:
    print(f"\n{'='*155}")
    print("  Table 2: Domain Contextual Bandits, Retrieval & Oracle Baselines (Banking-77)")
    print(f"{'='*155}")
    header = (
        f"{'Setting':<25} {'Worst':>11} {'Uniform':>11} {'Indep-UCB':>12} "
        f"{'k-UCB':>14} {'LinUCB':>11} {'Neural-PG':>12} {'kNN-Retr':>11} {'Oracle':>11} {'Clean':>11}"
    )
    print(header)
    print("-" * 155)

    settings = [
        ("ablate_banking_domain_4parts_np2_k8", "4 Domains (np=2, k=8)"),
        ("ablate_banking_domain_8parts_np3_k12", "8 Intents (np=3, k=12)"),
    ]
    for run_name, label in settings:
        d = load_summary(results_dir, run_name)
        if d is None:
            # Fall back to legacy name if ablation not yet run
            fallback = "banking_domain_7b" if "4parts" in run_name else "banking_domain_8parts_7b"
            d = load_summary(results_dir, fallback)
        wc = fmt(d, "worst_case")
        uni = fmt(d, "uniform")
        ind = fmt(d, "indep_ucb")
        kuc = fmt(d, "kucb")
        lin = fmt(d, "linucb")
        npg = fmt(d, "neural_pg")
        knn = fmt(d, "knn_retrieval")
        ora = fmt(d, "oracle_domain")
        nn = fmt(d, "no_noise")
        print(f"{label:<25} {wc:>11} {uni:>11} {ind:>12} {kuc:>14} {lin:>11} {npg:>12} {knn:>11} {ora:>11} {nn:>11}")
    print()


def print_scaling_table(results_dir: Path, n_p: int, k: int) -> None:
    print(f"\n{'='*155}")
    print(f"  Table 3: Scaling Shards Experiment (Banking-77 IID Shards + Tiers Noise) — n_p={n_p}, k={k}")
    print(f"{'='*155}")
    methods = ["worst_case", "uniform", "indep_ucb", "kucb", "linucb", "neural_pg", "knn_retrieval", "no_noise"]
    header = f"{'Shards (n)':<12} {'C(n,np)':<10} {'Metric':<9} " + " ".join(f"{m:>13}" for m in methods)
    print(header)
    print("-" * 155)

    for n in [8, 16, 32, 64, 100]:
        c_val = math.comb(n, n_p)
        run_name = f"scale_banking_shards_n{n}_np{n_p}_k{k}"
        d = load_summary(results_dir, run_name)
        seed_data = load_seed_files(results_dir, run_name)

        # 1. Accuracy row
        acc_strs = []
        time_strs = []
        mem_strs = []

        for m in methods:
            if d and d.get(m, {}).get("status") == "infeasible":
                acc_strs.append(f"infeas(C={c_val})")
                time_strs.append("—")
                mem_strs.append("—")
                continue

            # Accuracy
            if d and m in d and "mean" in d[m]:
                acc_strs.append(f"{d[m]['mean']:.1f}±{d[m]['std']:.1f}")
            else:
                acc_strs.append("—")

            # Runtime & memory from seed files
            m_times = [s[m]["wall_clock_seconds"] for s in seed_data if m in s and "wall_clock_seconds" in s[m]]
            m_mems = [s[m]["peak_cpu_mb"] for s in seed_data if m in s and "peak_cpu_mb" in s[m]]
            time_strs.append(f"{np.mean(m_times):.1f}s" if m_times else "—")
            mem_strs.append(f"{np.mean(m_mems):.0f}MB" if m_mems else "—")

        print(f"n={n:<9} {c_val:<10} {'Acc (%)':<9} " + " ".join(f"{s:>13}" for s in acc_strs))
        print(f"{'':<12} {'':<10} {'Time':<9} " + " ".join(f"{s:>13}" for s in time_strs))
        print(f"{'':<12} {'':<10} {'Memory':<9} " + " ".join(f"{s:>13}" for s in mem_strs))
        print("-" * 155)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="/data/pulakurthi.manohar/results",
        help="Directory containing summary_*.json files",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print_table(results_dir, "iid",    "ucb",  "Table 1a — IID Setting (Paper Reproduction)")
    print_table(results_dir, "noniid", "kucb", "Table 1b — Non-IID Setting (Paper Reproduction)")
    print_domain_ablation_table(results_dir)
    print_scaling_table(results_dir, n_p=2, k=8)
    print_scaling_table(results_dir, n_p=3, k=12)


if __name__ == "__main__":
    main()

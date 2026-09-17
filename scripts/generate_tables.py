"""
generate_tables.py
==================
Reads result JSON files and prints Table 1a (IID) and Table 1b (Non-IID)
matching the paper's structure exactly.

Usage:
    python scripts/generate_tables.py --results_dir /data/pulakurthi.manohar/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = [
    ("llama3b",  "LLaMA-3.2-3B"),
    ("7b",       "Qwen2.5-7B-Instruct-1M"),
    ("14b",      "Qwen2.5-14B"),
]
DATASETS = [
    ("agnews",   "AG-News"),
    ("ecom",     "Ecommerce"),
    ("syn",      "Synthetic"),
]


def load_summary(results_dir: Path, run_name: str) -> dict | None:
    path = results_dir / f"summary_{run_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(d: dict, method: str) -> str:
    if d is None or method not in d:
        return "     —     "
    m = d[method]
    if isinstance(m, dict):
        mean = m.get("mean", 0)
        std  = m.get("std",  0)
    else:
        mean = m * 100
        std  = 0.0
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
            wc  = fmt(d, "worst_case")
            nn  = fmt(d, "no_noise")
            ban = fmt(d, bandit_method)
            model_col = mlabel if dskey == "agnews" else ""
            print(f"{model_col:<28} {dslabel:<12} {wc:>14} {nn:>12} {ban:>12}")
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

    print_table(results_dir, "iid",    "ucb",  "Table 1a — IID Setting")
    print_table(results_dir, "noniid", "kucb", "Table 1b — Non-IID Setting")


if __name__ == "__main__":
    main()

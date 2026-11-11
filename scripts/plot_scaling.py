#!/usr/bin/env python3
"""
plot_scaling.py
===============
Generates publication-quality scaling plots for the paper:
1. plots/accuracy_vs_n_partitions.png
2. plots/runtime_vs_n_partitions.png
3. plots/memory_vs_n_partitions.png
4. plots/superarms_vs_n_partitions.png

Separate panels for n_p=2 (k=8) and n_p=3 (k=12).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_METADATA = {
    "no_noise": {"label": "Clean (No-Noise)", "color": "#2ca02c", "marker": "o", "ls": "--"},
    "worst_case": {"label": "Worst-Case", "color": "#d62728", "marker": "x", "ls": ":"},
    "uniform": {"label": "Uniform", "color": "#7f7f7f", "marker": "s", "ls": ":"},
    "indep_ucb": {"label": "Indep-UCB (O(n))", "color": "#1f77b4", "marker": "D", "ls": "-"},
    "kucb": {"label": "k-UCB (Combinatorial)", "color": "#9467bd", "marker": "^", "ls": "-"},
    "linucb": {"label": "LinUCB", "color": "#ff7f0e", "marker": "v", "ls": "-"},
    "neural_pg": {"label": "Neural-PG", "color": "#8c564b", "marker": "p", "ls": "-"},
    "knn_retrieval": {"label": "k-NN Retrieval", "color": "#17becf", "marker": "*", "ls": "-."},
    "oracle_domain": {"label": "Oracle Domain", "color": "#e377c2", "marker": "h", "ls": "--"},
}

N_PARTITIONS_LIST = [8, 16, 32, 64, 100]


def load_run_summary(results_dir: Path, run_name: str) -> dict | None:
    path = results_dir / f"summary_{run_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_summaries(results_dir: Path, run_name: str) -> list[dict]:
    files = list(results_dir.glob(f"summary_{run_name}_seed*.json"))
    summaries = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                summaries.append(json.load(fp))
        except Exception:
            pass
    return summaries


def extract_series(results_dir: Path, n_p: int, k: int, methods: list[str]):
    """Extracts mean, std, runtime, memory, and superarms across n_partitions."""
    data = {m: {"acc_mean": [], "acc_std": [], "time_mean": [], "mem_mean": [], "arms": [], "valid_n": []}
            for m in methods}

    for n in N_PARTITIONS_LIST:
        run_name = f"scale_banking_shards_n{n}_np{n_p}_k{k}"
        agg = load_run_summary(results_dir, run_name)
        seed_data = load_seed_summaries(results_dir, run_name)

        for m in methods:
            # 1. Superarms count
            if m == "kucb":
                arms = math.comb(n, n_p)
            elif m in ("ucb", "indep_ucb", "linucb", "neural_pg", "uniform", "oracle_domain"):
                arms = n
            elif m in ("worst_case", "no_noise"):
                arms = 1
            else:
                arms = 0

            # 2. Check if infeasible in agg or seed_data
            if agg and agg.get(m, {}).get("status") == "infeasible":
                data[m]["arms"].append(arms)
                data[m]["valid_n"].append(n)
                data[m]["acc_mean"].append(np.nan)
                data[m]["acc_std"].append(np.nan)
                data[m]["time_mean"].append(np.nan)
                data[m]["mem_mean"].append(np.nan)
                continue

            # Check seed data
            m_accs = []
            m_times = []
            m_mems = []

            for s in seed_data:
                m_info = s.get(m, {})
                if isinstance(m_info, dict) and "accuracy" in m_info:
                    m_accs.append(m_info["accuracy"] * 100)
                    if "wall_clock_seconds" in m_info:
                        m_times.append(m_info["wall_clock_seconds"])
                    if "peak_cpu_mb" in m_info:
                        m_mems.append(m_info["peak_cpu_mb"])

            if m_accs:
                data[m]["valid_n"].append(n)
                data[m]["acc_mean"].append(np.mean(m_accs))
                data[m]["acc_std"].append(np.std(m_accs, ddof=1) if len(m_accs) > 1 else 0.0)
                data[m]["time_mean"].append(np.mean(m_times) if m_times else np.nan)
                data[m]["mem_mean"].append(np.mean(m_mems) if m_mems else np.nan)
                data[m]["arms"].append(arms)
            elif agg and m in agg and "mean" in agg[m]:
                data[m]["valid_n"].append(n)
                data[m]["acc_mean"].append(agg[m]["mean"])
                data[m]["acc_std"].append(agg[m]["std"])
                data[m]["time_mean"].append(np.nan)
                data[m]["mem_mean"].append(np.nan)
                data[m]["arms"].append(arms)

    return data


def plot_scaling_metric(
    data_np2: dict,
    data_np3: dict,
    metric_key: str,
    metric_std_key: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=not log_y)
    panels = [(ax1, data_np2, "Panel A: $n_p=2$ ($k=8$ shots)"),
              (ax2, data_np3, "Panel B: $n_p=3$ ($k=12$ shots)")]

    for ax, data, panel_title in panels:
        for m, stats in data.items():
            meta = METHOD_METADATA.get(m, {"label": m, "color": "black", "marker": "o", "ls": "-"})
            xs = stats["valid_n"]
            ys = stats[metric_key]
            if not xs or all(np.isnan(y) for y in ys):
                continue

            # Filter valid points
            valid_pts = [(x, y, (stats[metric_std_key][i] if metric_std_key else 0.0))
                         for i, (x, y) in enumerate(zip(xs, ys)) if not np.isnan(y)]
            if not valid_pts:
                continue

            pxs, pys, perrs = zip(*valid_pts)
            if metric_std_key and any(e > 0 for e in perrs):
                ax.errorbar(
                    pxs, pys, yerr=perrs,
                    label=meta["label"],
                    color=meta["color"],
                    marker=meta["marker"],
                    linestyle=meta["ls"],
                    linewidth=2,
                    markersize=6,
                    capsize=3,
                )
            else:
                ax.plot(
                    pxs, pys,
                    label=meta["label"],
                    color=meta["color"],
                    marker=meta["marker"],
                    linestyle=meta["ls"],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Shards ($n$)", fontsize=11)
        ax.set_xticks(N_PARTITIONS_LIST)
        ax.set_xticklabels([str(n) for n in N_PARTITIONS_LIST])
        ax.grid(True, linestyle="--", alpha=0.6)
        if log_y:
            ax.set_yscale("log")

    ax1.set_ylabel(ylabel, fontsize=11)
    # Collect legend handles from ax1
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/data/pulakurthi.manohar/results")
    parser.add_argument("--out_dir", default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    methods = ["no_noise", "worst_case", "uniform", "indep_ucb", "kucb", "linucb", "neural_pg", "knn_retrieval"]

    data_np2 = extract_series(results_dir, n_p=2, k=8, methods=methods)
    data_np3 = extract_series(results_dir, n_p=3, k=12, methods=methods)

    # 1. Accuracy vs n_partitions
    plot_scaling_metric(
        data_np2, data_np3,
        metric_key="acc_mean", metric_std_key="acc_std",
        ylabel="Accuracy (%)",
        title="Accuracy vs. Number of Partitions",
        out_path=out_dir / "accuracy_vs_n_partitions.png",
    )

    # 2. Runtime vs n_partitions
    plot_scaling_metric(
        data_np2, data_np3,
        metric_key="time_mean", metric_std_key=None,
        ylabel="Wall-clock Runtime (seconds)",
        title="Runtime vs. Number of Partitions",
        out_path=out_dir / "runtime_vs_n_partitions.png",
    )

    # 3. Peak Memory vs n_partitions
    plot_scaling_metric(
        data_np2, data_np3,
        metric_key="mem_mean", metric_std_key=None,
        ylabel="Peak Memory (MB)",
        title="Peak Memory vs. Number of Partitions",
        out_path=out_dir / "memory_vs_n_partitions.png",
    )

    # 4. Superarm count vs n_partitions (log scale)
    plot_scaling_metric(
        data_np2, data_np3,
        metric_key="arms", metric_std_key=None,
        ylabel="Number of Arms / Superarms (log scale)",
        title="Superarm Combinatorial Explosion",
        out_path=out_dir / "superarms_vs_n_partitions.png",
        log_y=True,
    )


if __name__ == "__main__":
    main()

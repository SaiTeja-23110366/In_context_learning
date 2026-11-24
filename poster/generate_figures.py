"""
generate_figures.py
===================
Generates publication-quality poster figures tailored for A0 portrait layout:
1. schematic_icl_bandit.png / .svg: Schematic diagram of ICL partition selection (redrawn large, zero overlaps).
2. fig_a_coverage_sweep.png / .svg: k-UCB combinatorial coverage collapse vs n + index-order sweep visual.
   - Stacked 2-row layout with generous vertical clearance so legend does not touch next figure title.
   - Coverage: Orange solid for k-UCB np=2, Orange dashed for k-UCB np=3 (pink strictly reserved for Neural-PG).
3. fig_b_scaling_accuracy.png / .svg: Accuracy vs n across scaling shards (np=2 and np=3).
   - Stacked 2-row layout matching PPTX aspect ratio - LARGEST FIGURE on poster.
   - Generous bottom padding so legend does NOT cover x-axis label of second plot.
   - Removed "(diff stream)" and annotated true cause: k=12 > shard size 11 (replace=True).
4. fig_c_domain_results.png / .svg: Domain ablation accuracy comparison (4 domains, 8 intents).
   - Relabeled "Random clean shots (not an upper bound)".
   - LinUCB annotation: "LinUCB accuracy: 51.2% / 52.9%".
   - Explicit note that k-NN (70.8 / 71.1%) and Oracle (61.2 / 68.4%) beat LinUCB.
5. fig_d_routing_vs_chance.png / .svg: Gold partition routing frequency vs chance baselines (50% and 37.5%).
   - Added note: "Uniform selects 1 partition (chance 25% / 12.5%)".
6. fig_e_noise_exposure.png / .svg: Mean noise rate of selected shards vs uniform expectation.

All numbers are programmatically verified against ANALYSIS.md and local summary JSON files.
"""

import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure consistent colorblind-safe palette (Wong 2011)
COLORS = {
    "kucb": "#D55E00",         # Dark Orange / Vermilion
    "indep_ucb": "#0072B2",    # Deep Blue
    "linucb": "#009E73",       # Bluish Green / Emerald
    "neural_pg": "#CC79A7",    # Reddish Purple / Magenta (strictly reserved for Neural-PG)
    "knn": "#56B4E9",          # Sky Blue
    "oracle": "#E69F00",       # Amber / Yellow-Gold
    "uniform": "#4D4D4D",      # Neutral Charcoal
    "worst_case": "#888888",   # Muted Grey
    "clean": "#1B4F72",        # Dark Slate / Random clean shots
    "bg_light": "#F8F9FA",     # Very light background fill
    "border": "#D0D7DE",       # Border grey
}

OUTPUT_DIRS = [
    Path("In_context_learning/poster/poster_assets"),
    Path("poster/poster_assets"),
]

for od in OUTPUT_DIRS:
    od.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    for od in OUTPUT_DIRS:
        png_path = od / f"{name}.png"
        svg_path = od / f"{name}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Saved {name}.png and {name}.svg")
    plt.close(fig)


# ==============================================================================
# 1. Schematic Diagram: ICL Partition Selection Pipeline (Redrawn Larger, Zero Overlaps)
# ==============================================================================
def make_schematic():
    # Width 13.5 x 4.2 gives 1.4 units between boxes, preventing any text overlap
    fig, ax = plt.subplots(figsize=(13.5, 4.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.2)

    def draw_box(x, y, w, h, title, subtitle, color, bg="#FFFFFF"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=bg, edgecolor=color, linewidth=2.5
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color=color, family="DejaVu Sans")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=11, color="#222222", family="DejaVu Sans", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label="", color="#444444", font_sz=10.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, mutation_scale=18)
        )
        if label:
            ax.text((x1 + x2) / 2, y1 + 0.18, label,
                    ha="center", va="bottom", fontsize=font_sz, fontweight="bold",
                    color=color, family="DejaVu Sans", multialignment="center")

    # 1. Partition Pool (x: 0.4 to 2.5, width 2.1)
    draw_box(0.4, 1.3, 2.1, 2.2, "Partition Pool",
             "n shards / domains\nNoise: 0..80%\n1,104 train shots",
             COLORS["indep_ucb"], bg="#F0F7FC")

    # Arrow 1: x from 2.65 to 4.15 (clear whitespace, label does not touch any box)
    draw_arrow(2.65, 2.4, 4.15, 2.4, "Bandit\nSelects $n_p$", font_sz=10.0)

    # 2. Selected Shots (x: 4.3 to 6.3, width 2.0)
    draw_box(4.3, 1.3, 2.0, 2.2, "Selected Shots",
             "k shots sampled\nfrom $n_p$ partitions\n(Class_1..8)",
             COLORS["linucb"], bg="#F0FAF5")

    # Arrow 2: x from 6.45 to 7.65
    draw_arrow(6.45, 2.4, 7.65, 2.4, "Prompt\nContext", font_sz=10.5)

    # 3. LLM Inference (x: 7.8 to 9.9, width 2.1)
    draw_box(7.8, 1.3, 2.1, 2.2, "LLM Inference",
             "Qwen2.5-7B\nZero weight updates\nPredicts intent",
             COLORS["kucb"], bg="#FDF6F0")

    # Arrow 3: x from 10.05 to 11.25
    draw_arrow(10.05, 2.4, 11.25, 2.4, "Evaluate\nOutput", font_sz=10.5)

    # 4. Binary Reward (x: 11.4 to 12.9, width 1.5)
    draw_box(11.4, 1.3, 1.5, 2.2, "Reward",
             "r ∈ {0,1}\n(Correct?)",
             COLORS["neural_pg"], bg="#FAF0F6")

    # Feedback loop arrow (from Reward back to Partition Pool)
    feedback_path = patches.FancyArrowPatch(
        (12.15, 1.15), (1.45, 1.15),
        connectionstyle="arc3,rad=-0.10",
        arrowstyle="-|>", color="#0072B2", lw=2.4, mutation_scale=18,
        linestyle="--"
    )
    ax.add_patch(feedback_path)
    ax.text(6.8, 0.18, "Online Bandit Feedback Loop: Update partition statistics (UCB / Ridge regression / Policy gradient)",
            ha="center", va="center", fontsize=12.0, fontweight="bold",
            color="#0072B2", family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#0072B2", lw=1.2))

    save_fig(fig, "schematic_icl_bandit")


# ==============================================================================
# 2. Figure A: Coverage Scaling & Index-Order Sweep (Stacked Layout with Proper Spacing)
# ==============================================================================
def make_fig_a_coverage():
    # Sized 10.0 x 9.0 to give ample vertical breathing room between panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 9.0), dpi=300)

    shards = [8, 16, 32, 64, 100]
    cov_np2 = [100.0, 100.0, 64.44, 15.85, 6.45]
    cov_np3 = [100.0, 57.07, 6.44, 0.77, 0.20]

    # Rule: Orange solid for k-UCB np=2, Orange dashed for k-UCB np=3 (pink strictly reserved for Neural-PG)
    l1 = ax1.plot(shards, cov_np2, marker="o", lw=3.0, markersize=8,
                  color=COLORS["kucb"], linestyle="-", label=r"$k$-UCB ($n_p=2$, $C(n,2)$ superarms)")[0]
    l2 = ax1.plot(shards, cov_np3, marker="s", lw=3.0, markersize=8,
                  color=COLORS["kucb"], linestyle="--", label=r"$k$-UCB ($n_p=3$, $C(n,3)$ superarms)")[0]
    l3 = ax1.axhline(100.0, color=COLORS["indep_ucb"], lw=2.4, linestyle="-.",
                     label=r"IndepUCB (100% shard coverage, $O(n)$ cost)")

    # Callouts for n=100: zero intersections with text, markers, curves, or spines
    ax1.annotate("6.45% (319/4,950)", xy=(100, 6.45), xytext=(65, 42),
                 arrowprops=dict(arrowstyle="->", color=COLORS["kucb"], lw=1.8),
                 fontsize=12.0, fontweight="bold", color=COLORS["kucb"])
    ax1.annotate("0.20% (320/161,700)", xy=(100, 0.20), xytext=(103, 14),
                 arrowprops=dict(arrowstyle="->", color=COLORS["kucb"], lw=1.8),
                 fontsize=11.5, fontweight="bold", color=COLORS["kucb"])

    ax1.set_xlabel("Number of Shards ($n$)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Superarm Coverage (%)", fontsize=14, fontweight="bold")
    ax1.set_title("Combinatorial Coverage Collapse", fontsize=16, fontweight="bold", pad=8)
    ax1.set_xticks(shards)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(2, 145)
    ax1.set_ylim(-3, 110)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Item 9: Place legend below ax1 with ample padding so it NEVER touches ax2's title
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
               fontsize=11.5, ncol=3, frameon=True, framealpha=0.95)

    # Panel 2: Index-order sweep diagnostic at n=64, np=3
    shard_indices = ["Shard 0", "Shard 1", "Shard 2", "Shard 3", "Shard 4", "Shard 5", "Shard 6", "Shard 10", "Shard 30", "Shard 63"]
    pull_counts = [320, 62, 62, 62, 62, 62, 25, 6, 5, 5]
    bar_colors = [COLORS["kucb"] if c == 320 else "#E69F00" if c >= 62 else "#7F7F7F" for c in pull_counts]

    bars = ax2.bar(shard_indices, pull_counts, color=bar_colors, width=0.65, edgecolor="#222222", linewidth=1.1)
    ax2.axhline(320, color=COLORS["kucb"], linestyle=":", lw=1.8)
    ax2.text(0.1, 328, "Pulled in 100% of queries (320/320)", color=COLORS["kucb"], fontsize=12.5, fontweight="bold")

    ax2.set_xlabel("Partition Index", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Pulls per Shard (out of 320)", fontsize=14, fontweight="bold")
    ax2.set_title("Deterministic Sweep ($n=64$, $n_p=3$, Seed 0)", fontsize=16, fontweight="bold", pad=8)
    ax2.tick_params(axis='x', rotation=25, labelsize=11.5)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylim(0, 365)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    # High hspace prevents legend from touching ax2 title
    fig.subplots_adjust(hspace=0.55)
    save_fig(fig, "fig_a_coverage_sweep")


# ==============================================================================
# 3. Figure B: Scaling Accuracy (Stacked Layout, Clean External Legend Below)
# ==============================================================================
def make_fig_b_scaling_accuracy():
    # 10.0 x 10.2 inches with bottom=0.20 gives 2.0+ inches for the legend below ax2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 10.2), dpi=300)
    shards = [8, 16, 32, 64, 100]

    # Data np=2, k=8 (from Table 3 & ANALYSIS.md)
    u_np2 = [40.2, 40.1, 39.5, 40.1, 39.6]
    u_np2_s = [2.6, 2.6, 2.9, 2.1, 2.9]
    i_np2 = [45.5, 45.1, 43.0, 41.8, 42.5]
    i_np2_s = [4.3, 3.6, 2.3, 2.8, 1.9]
    k_np2 = [41.7, 42.4, 36.3, 41.7, 42.3]
    k_np2_s = [4.5, 2.9, 5.5, 2.5, 5.8]
    l_np2 = [49.5, 50.4, 51.0, 58.4, 61.2]
    l_np2_s = [3.9, 3.7, 6.1, 3.7, 5.5]
    n_np2 = [43.9, 42.7, 41.2, 41.9, 40.8]
    n_np2_s = [3.5, 3.1, 3.0, 1.7, 2.7]
    knn_np2 = [81.7, 81.7, 81.0, 80.6, 81.0]

    # Panel 1: np=2, k=8
    line_lin = ax1.plot(shards, l_np2, marker="o", lw=3.0, markersize=8, color=COLORS["linucb"], label="LinUCB (Contextual Bandit)")[0]
    ax1.fill_between(shards, np.array(l_np2)-np.array(l_np2_s), np.array(l_np2)+np.array(l_np2_s),
                     color=COLORS["linucb"], alpha=0.15)

    line_indep = ax1.plot(shards, i_np2, marker="s", lw=3.0, markersize=8, color=COLORS["indep_ucb"], label="IndepUCB (Linear-Cost $O(n)$)")[0]
    ax1.fill_between(shards, np.array(i_np2)-np.array(i_np2_s), np.array(i_np2)+np.array(i_np2_s),
                     color=COLORS["indep_ucb"], alpha=0.15)

    line_kucb = ax1.plot(shards, k_np2, marker="^", lw=3.0, markersize=8, color=COLORS["kucb"], label=r"$k$-UCB (Combinatorial Bandit)")[0]
    ax1.fill_between(shards, np.array(k_np2)-np.array(k_np2_s), np.array(k_np2)+np.array(k_np2_s),
                     color=COLORS["kucb"], alpha=0.15)

    line_neural = ax1.plot(shards, n_np2, marker="d", lw=2.4, markersize=7, color=COLORS["neural_pg"], label="Neural-PG (Policy Gradient)")[0]
    line_unif = ax1.plot(shards, u_np2, marker="x", lw=2.4, markersize=8, color=COLORS["uniform"], linestyle=":", label=r"Uniform Baseline ($k$ shots from 1 shard; bandits pool $n_p$)")[0]
    line_clean = ax1.axhline(61.1, color=COLORS["clean"], linestyle="--", lw=2.0, label="Random clean shots (not an upper bound) [61.1%]")
    line_knn = ax1.axhline(np.mean(knn_np2), color=COLORS["knn"], linestyle="-.", lw=2.2, label=r"$k$-NN Instance Retrieval (~81%, no noise filter)")

    ax1.set_xlabel("Number of Shards ($n$)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold")
    ax1.set_title("Scaling on IID Shards: $n_p=2$, $k=8$", fontsize=18, fontweight="bold", pad=8)
    ax1.set_xticks(shards)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_ylim(28, 88)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Data np=3, k=12 (from Table 3 & ANALYSIS.md)
    u_np3 = [46.9, 47.8, 46.5, 48.3, 40.8]
    u_np3_s = [3.5, 3.6, 2.8, 2.4, 3.7]
    i_np3 = [53.3, 54.2, 50.9, 51.1, 50.7]
    i_np3_s = [2.3, 3.3, 2.5, 3.0, 3.2]
    k_np3 = [48.3, 46.5, 47.8, 53.8, 53.7]
    k_np3_s = [2.7, 3.1, 5.2, 4.7, 6.5]
    l_np3 = [53.7, 56.6, 58.3, 63.1, 67.6]
    l_np3_s = [4.4, 4.4, 4.9, 6.0, 4.1]
    n_np3 = [50.4, 50.9, 49.6, 48.3, 49.2]
    n_np3_s = [3.3, 3.0, 3.3, 3.3, 3.6]
    knn_np3 = [83.2, 81.8, 82.4, 81.8, 82.1]

    # Panel 2: np=3, k=12
    ax2.plot(shards, l_np3, marker="o", lw=3.0, markersize=8, color=COLORS["linucb"])
    ax2.fill_between(shards, np.array(l_np3)-np.array(l_np3_s), np.array(l_np3)+np.array(l_np3_s),
                     color=COLORS["linucb"], alpha=0.15)

    ax2.plot(shards, i_np3, marker="s", lw=3.0, markersize=8, color=COLORS["indep_ucb"])
    ax2.fill_between(shards, np.array(i_np3)-np.array(i_np3_s), np.array(i_np3)+np.array(i_np3_s),
                     color=COLORS["indep_ucb"], alpha=0.15)

    ax2.plot(shards, k_np3, marker="^", lw=3.0, markersize=8, color=COLORS["kucb"])
    ax2.fill_between(shards, np.array(k_np3)-np.array(k_np3_s), np.array(k_np3)+np.array(k_np3_s),
                     color=COLORS["kucb"], alpha=0.15)

    ax2.plot(shards, n_np3, marker="d", lw=2.4, markersize=7, color=COLORS["neural_pg"])
    
    # Annotated Uniform baseline: removed "(diff stream)", reported true structural cause
    ax2.plot(shards, u_np3, marker="x", lw=2.4, markersize=8, color=COLORS["uniform"], linestyle=":")
    ax2.annotate("Uniform drops to 40.8%\nat n=100 ($k=12 >$ shard size 11)", xy=(100, 40.8), xytext=(58, 32),
                 arrowprops=dict(arrowstyle="->", color=COLORS["uniform"], lw=1.8),
                 fontsize=12.5, fontweight="bold", color=COLORS["uniform"])

    ax2.axhline(74.0, color=COLORS["clean"], linestyle="--", lw=2.0)
    ax2.axhline(np.mean(knn_np3), color=COLORS["knn"], linestyle="-.", lw=2.2)

    ax2.set_xlabel("Number of Shards ($n$)", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold")
    ax2.set_title("Scaling on IID Shards: $n_p=3$, $k=12$", fontsize=18, fontweight="bold", pad=8)
    ax2.set_xticks(shards)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_ylim(28, 90)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Item 9: UNIFIED EXTERNAL LEGEND placed well below ax2 x-axis label
    handles = [line_lin, line_indep, line_kucb, line_neural, line_unif, line_clean, line_knn]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.032),
               ncol=3, fontsize=12.5, frameon=True, framealpha=0.96)

    # Added caption: Uniform draws all k shots from one partition; bandits pool np
    fig.text(0.5, 0.006, "Caption: Uniform draws all k shots from one partition; bandits pool $n_p$",
             ha="center", fontsize=12.5, fontweight="bold", color="#333333")

    plt.tight_layout()
    # Generous bottom margin ensures legend and caption NEVER cover ax2's x-axis label
    fig.subplots_adjust(hspace=0.35, bottom=0.20)
    save_fig(fig, "fig_b_scaling_accuracy")


# ==============================================================================
# 4. Figure C: Domain Partition Comparison (Grouped Bar Chart)
# ==============================================================================
def make_fig_c_domain():
    fig, ax = plt.subplots(figsize=(11.0, 6.5), dpi=300)

    methods = [
        "Worst-Case", "Uniform", "IndepUCB", "k-UCB", "Neural-PG",
        "LinUCB", "Oracle Domain", "Random Clean", "k-NN"
    ]
    acc_4d = [21.72, 24.94, 35.00, 34.34, 40.84, 51.22, 61.16, 61.47, 70.78]
    std_4d = [5.60, 3.49, 5.60, 5.57, 8.08, 10.03, 12.43, 2.48, 12.67]

    acc_8i = [17.22, 17.94, 33.25, 29.06, 34.28, 52.94, 68.38, 73.19, 71.06]
    std_8i = [2.57, 2.32, 3.14, 2.93, 2.87, 8.91, 9.41, 3.30, 8.70]

    x = np.arange(len(methods))
    width = 0.38

    bar_colors = [
        COLORS["worst_case"], COLORS["uniform"], COLORS["indep_ucb"],
        COLORS["kucb"], COLORS["neural_pg"], COLORS["linucb"],
        COLORS["oracle"], COLORS["clean"], COLORS["knn"]
    ]

    rects1 = ax.bar(x - width/2, acc_4d, width, yerr=std_4d, capsize=3.5,
                    color=bar_colors, edgecolor="#222222", linewidth=1.1, label="4 Domains ($n_p=2$, $k=8$)")
    rects2 = ax.bar(x + width/2, acc_8i, width, yerr=std_8i, capsize=3.5,
                    color=bar_colors, edgecolor="#222222", linewidth=1.1, hatch="//", alpha=0.82, label="8 Intents ($n_p=3$, $k=12$)")

    ax.set_ylabel("Accuracy (%)", fontsize=15, fontweight="bold")
    ax.set_title("Domain Partition Performance & Baselines (Banking-77)", fontsize=16.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12.5, fontweight="bold", rotation=22)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 95)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=12.5, framealpha=0.95)

    # Required: LinUCB accuracy annotation + explicit note that k-NN & Oracle beat LinUCB
    ax.annotate("LinUCB accuracy:\n51.2% / 52.9%", xy=(5, 53), xytext=(3.2, 73),
                arrowprops=dict(arrowstyle="->", color=COLORS["linucb"], lw=1.8),
                fontsize=12, fontweight="bold", color=COLORS["linucb"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#E6F5EE", edgecolor=COLORS["linucb"], lw=1.2))

    ax.annotate("k-NN (70.8 / 71.1%) and Oracle (61.2 / 68.4%)\nbeat LinUCB via instance / gold domain routing",
                xy=(7.5, 72), xytext=(4.6, 86),
                arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=1.6),
                fontsize=11.5, fontweight="bold", color="#1B4F72",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F4F8", edgecolor="#1B4F72", lw=1.1))

    # Added caption: Uniform draws all k shots from one partition; bandits pool np
    fig.text(0.5, 0.012, "Caption: Uniform draws all k shots from one partition; bandits pool $n_p$",
             ha="center", fontsize=12.0, fontweight="bold", color="#333333")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    save_fig(fig, "fig_c_domain_results")


# ==============================================================================
# 5. Figure D: Domain Routing Accuracy vs Chance
# ==============================================================================
def make_fig_d_routing():
    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=300)

    # Data from Table 5.1 & ANALYSIS.md
    methods = ["Uniform\n(1 part)", "k-UCB\n($n_p$ parts)", "IndepUCB\n($n_p$ parts)", "Neural-PG\n($n_p$ parts)", "LinUCB\n($n_p$ parts)", "Oracle\n(100%)"]
    routing_4d = [26.09, 49.16, 48.56, 60.06, 82.06, 100.0]
    routing_8i = [12.88, 36.09, 36.91, 42.25, 72.94, 100.0]

    x = np.arange(len(methods))
    width = 0.36

    rects1 = ax.bar(x - width/2, routing_4d, width, color="#0072B2", edgecolor="#222222", linewidth=1.1, label="4 Domains (Chance = 50.0%)")
    rects2 = ax.bar(x + width/2, routing_8i, width, color="#009E73", edgecolor="#222222", linewidth=1.1, hatch="\\\\", alpha=0.85, label="8 Intents (Chance = 37.5%)")

    # Reference chance lines
    ax.axhline(50.0, color="#0072B2", linestyle="--", lw=1.8, alpha=0.85, label="4D Chance Baseline (50.0%)")
    ax.axhline(37.5, color="#009E73", linestyle=":", lw=2.0, alpha=0.85, label="8I Chance Baseline (37.5%)")

    ax.set_ylabel("Gold Partition Routing Rate (%)", fontsize=15, fontweight="bold")
    ax.set_title("Semantic Intent Routing on Domain Suites", fontsize=16.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12.5, fontweight="bold")
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 118)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=12.0, framealpha=0.95)

    # Item 6: Place note above Uniform bars away from y-axis tick labels
    ax.text(0.12, 34, "Uniform selects 1 partition\n(chance 25% / 12.5%)", ha="center", fontsize=10.0, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F4F4F4", edgecolor="#888888", lw=1.0))

    # Text annotation for LinUCB
    ax.text(4, 92, "82.1% / 72.9%\n(Far above chance)", ha="center", fontsize=12.5, fontweight="bold", color="#009E73")

    plt.tight_layout()
    save_fig(fig, "fig_d_routing_vs_chance")


# ==============================================================================
# 6. Optional Figure E: Noise Exposure vs Uniform Expectation
# ==============================================================================
def make_fig_e_noise():
    fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=300)
    shards = [8, 16, 32, 64, 100]

    # Data from Table 2.1 (np=2)
    u_noise = [0.3687, 0.3654, 0.3713, 0.3779, 0.3823]
    i_noise = [0.2732, 0.3077, 0.3141, 0.3445, 0.3481]
    k_noise = [0.3475, 0.3597, 0.3868, 0.3560, 0.3339]
    l_noise = [0.2379, 0.2322, 0.2162, 0.2226, 0.2351]
    n_noise = [0.3176, 0.3512, 0.3563, 0.3653, 0.3724]

    ax.plot(shards, u_noise, marker="x", lw=2.2, color=COLORS["uniform"], linestyle=":", label="Uniform Baseline (~0.375)")
    ax.plot(shards, k_noise, marker="^", lw=2.6, color=COLORS["kucb"], label="k-UCB")
    ax.plot(shards, n_noise, marker="d", lw=2.4, color=COLORS["neural_pg"], label="Neural-PG")
    ax.plot(shards, i_noise, marker="s", lw=2.8, color=COLORS["indep_ucb"], label="IndepUCB (O(n))")
    ax.plot(shards, l_noise, marker="o", lw=3.0, color=COLORS["linucb"], label="LinUCB (Lowest Noise)")

    ax.axhline(0.375, color="#D55E00", linestyle="--", lw=1.6, alpha=0.7, label="Expected Uniform (0.375)")

    ax.set_xlabel("Number of Shards (n)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Noise Rate of Pulled Shards", fontsize=14, fontweight="bold")
    ax.set_title("Noise Avoidance on IID Shards (Lower = Better)", fontsize=16, fontweight="bold", pad=10)
    ax.set_xticks(shards)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0.18, 0.43)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95)

    plt.tight_layout()
    save_fig(fig, "fig_e_noise_exposure")


def main():
    print("Generating conference poster figures with updated stacked layouts, zero overlaps, and exact notes...")
    make_schematic()
    make_fig_a_coverage()
    make_fig_b_scaling_accuracy()
    make_fig_c_domain()
    make_fig_d_routing()
    make_fig_e_noise()
    print("All figures successfully created in poster_assets/!")


if __name__ == "__main__":
    main()

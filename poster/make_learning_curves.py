import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load local data
with open('In_context_learning/results_summary/learning_curves_data.json') as f:
    res = json.load(f)

data = res['data']
gaps = res['gaps']
ns = [8, 16, 64, 100]

# Poster Palette
COLORS = {
    "linucb": "#009E73",      # Green
    "indep_ucb": "#0072B2",   # Blue
    "kucb": "#D55E00",        # Orange
    "neural_pg": "#CC79A7",   # Pink
    "uniform": "#7F7F7F",     # Gray
}

LABELS = {
    "linucb": "LinUCB",
    "indep_ucb": "IndepUCB",
    "kucb": "$k$-UCB",
    "neural_pg": "Neural-PG",
    "uniform": "Uniform (dashed)",
}

MARKERS = {
    "linucb": "o",
    "indep_ucb": "s",
    "kucb": "^",
    "neural_pg": "D",
    "uniform": "v",
}

LINESTYLES = {
    "linucb": "-",
    "indep_ucb": "-",
    "kucb": "-",
    "neural_pg": "-",
    "uniform": "--",
}

quarters = ["Q1", "Q2", "Q3", "Q4"]
q_indices = [1, 2, 3, 4]

# Sized 10.0 x 7.8 for generous vertical clearance and crisp enlarged display
fig, axes = plt.subplots(2, 4, figsize=(10.0, 7.8), dpi=300, sharex=True)

# Top row: Accuracy per quarter
for col_idx, n in enumerate(ns):
    ax = axes[0, col_idx]
    n_str = str(n)
    
    for m in ["linucb", "indep_ucb", "kucb", "neural_pg", "uniform"]:
        means = np.array([np.mean(data[n_str][m][q]) for q in quarters])
        ses = np.array([stats.sem(data[n_str][m][q]) for q in quarters])
        
        ax.plot(q_indices, means, color=COLORS[m], linestyle=LINESTYLES[m],
                marker=MARKERS[m], markersize=7.0, lw=2.5, label=LABELS[m])
        ax.fill_between(q_indices, means - ses, means + ses,
                        color=COLORS[m], alpha=0.18)
    
    # Shortened panel titles to "n=8", "n=16", "n=64", "n=100"
    ax.set_title(f"n={n}", fontsize=16.0, fontweight="bold", pad=8)
    ax.set_ylim(0.28, 0.74)
    ax.set_yticks([0.30, 0.40, 0.50, 0.60, 0.70])
    ax.set_yticklabels(["30%", "40%", "50%", "60%", "70%"], fontsize=12.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if col_idx == 0:
        ax.set_ylabel("Accuracy per Quarter", fontsize=13.5, fontweight="bold")
    else:
        ax.set_yticklabels([])

# Bottom row: Paired Gap (LinUCB - IndepUCB)
for col_idx, n in enumerate(ns):
    ax = axes[1, col_idx]
    n_str = str(n)
    
    means = np.array([np.mean(gaps[n_str][q]) for q in quarters])
    ses = np.array([stats.sem(gaps[n_str][q]) for q in quarters])
    
    ax.axhline(0.0, color="#333333", linestyle="--", lw=1.4, alpha=0.7)
    ax.plot(q_indices, means, color=COLORS["linucb"], linestyle="-",
            marker="o", markersize=7.5, lw=2.6, label="LinUCB - IndepUCB")
    ax.fill_between(q_indices, means - ses, means + ses,
                    color=COLORS["linucb"], alpha=0.22)
    
    ax.set_xticks(q_indices)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=12.5, fontweight="bold")
    # Repeated x-axis labels removed; single shared label placed below
    ax.set_ylim(-0.08, 0.30)
    ax.set_yticks([-0.05, 0.00, 0.10, 0.20, 0.30])
    ax.set_yticklabels(["-5 pts", "0 pts", "+10 pts", "+20 pts", "+30 pts"], fontsize=12.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if col_idx == 0:
        ax.set_ylabel("Paired Gap (Lin - Indep)", fontsize=13.5, fontweight="bold")
    else:
        ax.set_yticklabels([])

# One shared legend outside the panels (above top row)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
           ncol=5, fontsize=13.0, frameon=True, framealpha=0.96, edgecolor="#CCCCCC")

# Single shared x-axis label centered across all bottom panels
fig.text(0.53, 0.035, "Query stream quarter (80 queries each)",
         ha="center", fontsize=14.0, fontweight="bold", color="#222222")

# top=0.85 gives >= 10 mm vertical clearance between legend and titles (>= 3 mm required)
fig.subplots_adjust(top=0.85, bottom=0.10, hspace=0.25, wspace=0.16)

# Save to plots/analysis/learning_curves.png as requested
plt.savefig("In_context_learning/plots/analysis/learning_curves.png", dpi=300)
plt.savefig("plots/analysis/learning_curves.png", dpi=300)
# Also save as poster asset
plt.savefig("In_context_learning/poster/poster_assets/fig_f_learning_curves.png", dpi=300)
plt.savefig("In_context_learning/poster/poster_assets/fig_f_learning_curves.svg")
plt.savefig("poster/poster_assets/fig_f_learning_curves.png", dpi=300)
plt.savefig("poster/poster_assets/fig_f_learning_curves.svg")
print("Saved learning curves figure successfully!")

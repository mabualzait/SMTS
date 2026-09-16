"""
Publication-Quality Plotting Engine.
Generates:
1. Convergence trajectories of all 6 algorithms on F1, F6, F8, F10.
2. Friedman Rank comparison across dimensions.
3. High-dimensional scalability plot.
"""

import sys
import os
import json
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_smts"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plots"))
os.makedirs(plots_dir, exist_ok=True)


def plot_convergence_grid(dim=50):
    traj_path = os.path.join(results_dir, f"trajectories_D{dim}.json")
    if not os.path.exists(traj_path):
        print(f"File {traj_path} does not exist.")
        return

    with open(traj_path, "r") as f:
        data = json.load(f)["convergence"]

    # Target 4 functions
    target_funcs = [
        "F1: High-Conditioned Elliptic",
        "F6: Rastrigin",
        "F8: Hybrid (Elliptic+Rastrigin+Ackley)",
        "F10: Composition (Rastrigin+Ackley+Sphere)",
    ]
    short_titles = ["(a) F1: Unimodal Elliptic", "(b) F6: Multimodal Rastrigin", "(c) F8: Hybrid Landscape", "(d) F10: Composition Function"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.flatten()

    colors = {
        "SMTS": "#1f77b4",
        "L-SHADE": "#ff7f0e",
        "CMA-ES": "#2ca02c",
        "CLPSO": "#d62728",
        "PSO": "#9467bd",
        "GWO": "#8c564b",
    }
    linestyles = {
        "SMTS": "-",
        "L-SHADE": "--",
        "CMA-ES": "-.",
        "CLPSO": ":",
        "PSO": "-",
        "GWO": "--",
    }

    for idx, (fn, title) in enumerate(zip(target_funcs, short_titles)):
        ax = axes[idx]
        if fn not in data:
            continue

        for alg, runs in data[fn].items():
            if not runs:
                continue
            min_len = min(len(r) for r in runs)
            mean_curve = np.median([r[:min_len] for r in runs], axis=0)
            evals = np.linspace(0, 1000 * dim, min_len)
            ax.semilogy(
                evals,
                np.maximum(1e-12, mean_curve),
                label=alg,
                color=colors.get(alg, "black"),
                linestyle=linestyles.get(alg, "-"),
                linewidth=2.2 if alg == "SMTS" else 1.6,
            )

        ax.set_title(f"{title} ($D={dim}$)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Median Error $f(\mathbf{x}) - f^*$", fontsize=10, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"convergence_grid_D{dim}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_friedman_summary():
    algs = ["SMTS", "L-SHADE", "CMA-ES", "PSO", "GWO", "CLPSO"]
    # Ranks from D=30, D=50, and D=100 statistical reports
    ranks_d30 = [2.80, 1.60, 1.90, 4.40, 4.70, 5.60]
    ranks_d50 = [2.80, 1.50, 2.20, 4.90, 4.50, 5.10]
    ranks_d100 = [3.10, 1.60, 1.70, 4.60, 4.80, 5.20]

    x = np.arange(len(algs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width, ranks_d30, width, label="D = 30", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x, ranks_d50, width, label="D = 50", color="#ff7f0e", edgecolor="black", alpha=0.85)
    rects3 = ax.bar(x + width, ranks_d100, width, label="D = 100", color="#2ca02c", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Average Friedman Rank (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Friedman Rank Summary across CEC Continual Benchmark Functions ($D \\in \\{30, 50, 100\\}$)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Label heights
    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "friedman_rank_summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_convergence_grid(dim=30)
    plot_convergence_grid(dim=50)
    plot_convergence_grid(dim=100)
    plot_friedman_summary()

"""
Ablation Study and Figure Generation Module for SMTS.
Investigates:
1. Impact of topological neighborhood degree k in {2, 4, 6, 7, 10, 15, 20}.
2. Comparison with metric-radius flocking interaction model.
3. Convergence curve generation across landscape types.
4. Swarm spatial diversity profile over generations.
"""

import sys
import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
import numpy as np

# Set headless matplotlib backend
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_smts"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.cec_suite import get_benchmark_suite
from smts import SMTSOptimizer
from smts.operators import (
    compute_spin_wave_alignment,
    compute_topological_gradient,
    adaptive_boundary_rebound,
)


class MetricRadiusSMTS:
    """
    Control model: Metric-radius flocking where agents interact with all peers
    within Euclidean distance r_0, rather than a fixed topological degree k.
    """
    def __init__(self, func, dim, radius_ratio=0.1, pop_size=50, max_evals=30000, seed=42):
        self.func = func
        self.dim = dim
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.radius = radius_ratio * np.linalg.norm(self.ub - self.lb)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.evals_count = 0
        self.gbest_f = float("inf")
        self.convergence = []
        self.diversity = []

    def optimize(self):
        domain_span = self.ub - self.lb
        X = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        V = np.random.uniform(-0.1 * domain_span, 0.1 * domain_span, size=(self.pop_size, self.dim))
        fitness = np.array([self.func(x) for x in X])
        self.evals_count += self.pop_size

        P = X.copy()
        P_fit = fitness.copy()
        best_i = np.argmin(P_fit)
        self.gbest_f = P_fit[best_i]
        self.gbest_x = P[best_i].copy()

        while self.evals_count < self.max_evals:
            tau = self.evals_count / self.max_evals
            w = 0.9 - 0.5 * tau

            # Pairwise Euclidean distances
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, D)
            dists = np.linalg.norm(diff, axis=2)  # (N, N)

            for i in range(self.pop_size):
                # Metric neighbors within radius
                metric_neighbors = np.where((dists[i] <= self.radius) & (dists[i] > 0))[0]
                if len(metric_neighbors) == 0:
                    # Isolated: drift with inertia
                    a_spin = np.zeros(self.dim)
                    a_elite = np.zeros(self.dim)
                else:
                    a_spin = np.mean(V[metric_neighbors], axis=0) - V[i]
                    sub_best = metric_neighbors[np.argmin(P_fit[metric_neighbors])]
                    a_elite = P[sub_best] - X[i]

                r1, r2, r3 = np.random.rand(3)
                V[i] = w * V[i] + 1.2 * r1 * a_spin + 1.2 * r2 * a_elite + 1.5 * (tau**2) * r3 * (self.gbest_x - X[i])
                V[i] = np.clip(V[i], -0.2 * domain_span, 0.2 * domain_span)
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)

                if self.evals_count < self.max_evals:
                    f_val = self.func(X[i])
                    self.evals_count += 1
                    fitness[i] = f_val
                    if f_val < P_fit[i]:
                        P_fit[i] = f_val
                        P[i] = X[i].copy()
                        if f_val < self.gbest_f:
                            self.gbest_f = f_val
                            self.gbest_x = X[i].copy()

            self.convergence.append(self.gbest_f)
            centroid = np.mean(X, axis=0)
            self.diversity.append(float(np.mean(np.linalg.norm(X - centroid, axis=1)) / np.linalg.norm(domain_span)))

        return {"best_f": self.gbest_f, "convergence": self.convergence, "diversity": self.diversity}


def run_single_k_exp(k_val, f_idx, dim, run_id, max_evals, seed):
    funcs = get_benchmark_suite(dim=dim)
    f = funcs[f_idx]
    opt = SMTSOptimizer(f, dim=dim, pop_size=50, max_evals=max_evals, k_neighbors=k_val, seed=seed)
    res = opt.optimize()
    return k_val, f.name, run_id, float(res["best_f"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    dim = args.dim
    runs = args.runs
    max_evals = 1000 * dim
    k_values = [2, 4, 6, 7, 10, 15, 20]

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plots"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nLaunching Ablation Study on Topological Degree k in {k_values} (D={dim})\n{'=' * 60}")
    funcs = get_benchmark_suite(dim=dim)
    
    # Representative functions: F1 (Unimodal), F6 (Multimodal), F8 (Hybrid), F10 (Composition)
    rep_indices = [0, 5, 7, 9]

    tasks = []
    for k in k_values:
        for f_idx in rep_indices:
            for run_id in range(runs):
                seed = 77777 + k * 1000 + f_idx * 100 + run_id
                tasks.append((k, f_idx, dim, run_id, max_evals, seed))

    ablation_results = {k: {funcs[idx].name: [] for idx in rep_indices} for k in k_values}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_k_exp, *t) for t in tasks]
        for f in as_completed(futures):
            k_val, fn, r_id, b_f = f.result()
            ablation_results[k_val][fn].append(b_f)

    # Save ablation results
    abl_file = os.path.join(output_dir, f"ablation_k_D{dim}.json")
    with open(abl_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved ablation data to {abl_file}")

    # Plot 1: Ablation on k
    plt.figure(figsize=(9, 6), dpi=300)
    rep_names = [funcs[idx].name for idx in rep_indices]
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for fn, marker, col in zip(rep_names, markers, colors):
        means = []
        for k in k_values:
            arr = np.array(ablation_results[k][fn])
            means.append(np.median(arr))
        # Normalize to relative performance (ratio to minimum)
        norm_means = np.array(means) / (min(means) + 1e-12)
        plt.plot(k_values, norm_means, marker=marker, color=col, linewidth=2.0, markersize=7, label=fn)

    plt.axvspan(5.8, 7.2, color="gray", alpha=0.2, label=r"Biological Range $k \in [6, 7]$")
    plt.xlabel(r"Topological Neighborhood Degree $k$", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Median Error (Ratio to Best)", fontsize=12, fontweight="bold")
    plt.title(f"Ablation on Topological Neighborhood Degree $k$ ($D = {dim}$)", fontsize=14, fontweight="bold")
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    k_plot_path = os.path.join(plots_dir, "ablation_k_degree.png")
    plt.savefig(k_plot_path)
    plt.close()
    print(f"Saved {k_plot_path}")

    # Run Metric Radius comparison vs Topological SMTS
    print("\nRunning Metric-Radius vs Topological SMTS comparison...")
    f_test = funcs[5]  # Rastrigin
    top_convs, met_convs = [], []
    top_divs, met_divs = [], []

    for r in range(10):
        s = 12345 + r
        smts_opt = SMTSOptimizer(f_test, dim=dim, pop_size=50, max_evals=max_evals, seed=s)
        res_t = smts_opt.optimize()
        top_convs.append(res_t["convergence"])
        top_divs.append(res_t["diversity"])

        met_opt = MetricRadiusSMTS(f_test, dim=dim, radius_ratio=0.1, pop_size=50, max_evals=max_evals, seed=s)
        res_m = met_opt.optimize()
        met_convs.append(res_m["convergence"])
        met_divs.append(res_m["diversity"])

    # Plot 2: Convergence Comparison (Topological vs Metric)
    plt.figure(figsize=(8, 5), dpi=300)
    min_len = min(len(c) for c in top_convs + met_convs)
    t_conv_mean = np.mean([c[:min_len] for c in top_convs], axis=0)
    m_conv_mean = np.mean([c[:min_len] for c in met_convs], axis=0)
    eval_axis = np.linspace(0, max_evals, min_len)

    plt.semilogy(eval_axis, t_conv_mean, label="Topological Interaction ($k=7$)", color="#1f77b4", linewidth=2.5)
    plt.semilogy(eval_axis, m_conv_mean, label="Metric-Radius Interaction ($r_0=0.1 L$)", color="#d62728", linestyle="--", linewidth=2.5)
    plt.xlabel("Function Evaluations", fontsize=12, fontweight="bold")
    plt.ylabel("Objective Function Error $f(x) - f^*$", fontsize=12, fontweight="bold")
    plt.title(f"Convergence: Topological vs. Metric Neighborhood (Rastrigin, $D={dim}$)", fontsize=13, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    conv_plot_path = os.path.join(plots_dir, "convergence_topo_vs_metric.png")
    plt.savefig(conv_plot_path)
    plt.close()
    print(f"Saved {conv_plot_path}")

    # Plot 3: Population Diversity Maintenance
    plt.figure(figsize=(8, 5), dpi=300)
    min_div_len = min(len(d) for d in top_divs + met_divs)
    t_div_mean = np.mean([d[:min_div_len] for d in top_divs], axis=0)
    m_div_mean = np.mean([d[:min_div_len] for d in met_divs], axis=0)
    div_axis = np.linspace(0, max_evals, min_div_len)

    plt.plot(div_axis, t_div_mean, label="SMTS Topological ($k=7$)", color="#1f77b4", linewidth=2.5)
    plt.plot(div_axis, m_div_mean, label="Metric-Radius Swarm", color="#d62728", linestyle="--", linewidth=2.5)
    plt.xlabel("Function Evaluations", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Swarm Diversity $\\mathcal{D}(t)$", fontsize=12, fontweight="bold")
    plt.title(f"Scale-Free Spatial Diversity Maintenance ($D={dim}$)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    div_plot_path = os.path.join(plots_dir, "diversity_maintenance.png")
    plt.savefig(div_plot_path)
    plt.close()
    print(f"Saved {div_plot_path}")


if __name__ == "__main__":
    main()

"""
Automated Benchmark Runner for Metaheuristic Evaluation.
Executes 30 Monte Carlo runs across D in {30, 50, 100} in parallel.
"""

import sys
import os
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.cec_suite import get_benchmark_suite
from smts import SMTSOptimizer
from baselines import (
    LSHADEOptimizer,
    CMAESOptimizer,
    CLPSOOptimizer,
    StandardPSOOptimizer,
    GWOOptimizer,
)
from experiments.statistical_analysis import (
    generate_latex_table,
    generate_markdown_table,
    compute_friedman_test,
    compute_nemenyi_cd,
    compute_wilcoxon_test,
)


def run_single_experiment(
    alg_name: str,
    f_idx: int,
    dim: int,
    run_id: int,
    max_evals: int,
    seed: int,
) -> Dict[str, Any]:
    """Execute a single Monte Carlo trial."""
    funcs = get_benchmark_suite(dim=dim)
    func = funcs[f_idx]

    if alg_name == "SMTS":
        opt = SMTSOptimizer(func, dim=dim, pop_size=50, max_evals=max_evals, seed=seed)
    elif alg_name == "L-SHADE":
        opt = LSHADEOptimizer(func, dim=dim, max_evals=max_evals, seed=seed)
    elif alg_name == "CMA-ES":
        opt = CMAESOptimizer(func, dim=dim, max_evals=max_evals, seed=seed)
    elif alg_name == "CLPSO":
        opt = CLPSOOptimizer(func, dim=dim, pop_size=50, max_evals=max_evals, seed=seed)
    elif alg_name == "PSO":
        opt = StandardPSOOptimizer(func, dim=dim, pop_size=50, max_evals=max_evals, seed=seed)
    elif alg_name == "GWO":
        opt = GWOOptimizer(func, dim=dim, pop_size=50, max_evals=max_evals, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    res = opt.optimize()

    # Downsample convergence curve to 50 points to keep JSON compact
    conv = res.get("convergence", [])
    if len(conv) > 50:
        step = len(conv) // 50
        sampled_conv = [conv[i * step] for i in range(50)] + [conv[-1]]
    else:
        sampled_conv = conv

    div = res.get("diversity", [])
    if len(div) > 50:
        step = len(div) // 50
        sampled_div = [div[i * step] for i in range(50)] + [div[-1]]
    else:
        sampled_div = div

    return {
        "alg_name": alg_name,
        "f_idx": f_idx,
        "f_name": func.name,
        "dim": dim,
        "run_id": run_id,
        "best_f": float(res["best_f"]),
        "evals": int(res["evals"]),
        "convergence": sampled_conv,
        "diversity": sampled_div,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SMTS Benchmarks")
    parser.add_argument("--dims", type=int, nargs="+", default=[30, 50, 100], help="Dimensions to benchmark")
    parser.add_argument("--runs", type=int, default=30, help="Number of Monte Carlo trials")
    parser.add_argument("--evals_factor", type=int, default=1000, help="Max evals multiplier (e.g. 1000 * D)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel processes")
    args = parser.parse_args()

    algorithms = ["SMTS", "L-SHADE", "CMA-ES", "CLPSO", "PSO", "GWO"]
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    os.makedirs(output_dir, exist_ok=True)

    for dim in args.dims:
        max_evals = args.evals_factor * dim
        print(f"\n{'=' * 60}\nLaunching Benchmark Suite: D={dim}, Runs={args.runs}, MaxEvals={max_evals}\n{'=' * 60}")
        
        funcs = get_benchmark_suite(dim=dim)
        num_funcs = len(funcs)

        tasks = []
        for f_idx in range(num_funcs):
            for alg in algorithms:
                for run_id in range(args.runs):
                    seed = 10000 * dim + 100 * f_idx + run_id + 1
                    tasks.append((alg, f_idx, dim, run_id, max_evals, seed))

        print(f"Total tasks to execute: {len(tasks)} across {args.workers} workers...")
        t0 = time.time()

        # Structure to accumulate results: results[f_name][alg] = [run0, run1, ...]
        benchmark_results = {f.name: {alg: [] for alg in algorithms} for f in funcs}
        convergence_records = {f.name: {alg: [] for alg in algorithms} for f in funcs}
        diversity_records = {f.name: {alg: [] for alg in algorithms} for f in funcs}

        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(run_single_experiment, *t): t for t in tasks
            }

            for future in as_completed(future_to_task):
                res = future.result()
                f_name = res["f_name"]
                alg = res["alg_name"]
                benchmark_results[f_name][alg].append(res["best_f"])
                convergence_records[f_name][alg].append(res["convergence"])
                if res["diversity"]:
                    diversity_records[f_name][alg].append(res["diversity"])

                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[{completed}/{len(tasks)}] tasks finished ({completed/len(tasks)*100:.1f}%) in {elapsed:.1f}s", flush=True)

        total_time = time.time() - t0
        print(f"Completed D={dim} in {total_time:.2f}s ({total_time/60:.2f} mins).", flush=True)

        # Save raw results
        out_file = os.path.join(output_dir, f"benchmark_results_D{dim}.json")
        with open(out_file, "w") as f:
            json.dump({
                "dim": dim,
                "runs": args.runs,
                "max_evals": max_evals,
                "algorithms": algorithms,
                "benchmark_results": benchmark_results,
            }, f, indent=2)
        print(f"Saved results to {out_file}")

        # Save convergence and diversity for plotting
        traj_file = os.path.join(output_dir, f"trajectories_D{dim}.json")
        with open(traj_file, "w") as f:
            json.dump({
                "convergence": convergence_records,
                "diversity": diversity_records,
            }, f, indent=2)

        # Generate Tables & Statistical Tests
        f_names = [f.name for f in funcs]
        latex_tab = generate_latex_table(benchmark_results, algorithms, f_names, dim=dim)
        md_tab = generate_markdown_table(benchmark_results, algorithms, f_names, dim=dim)

        with open(os.path.join(output_dir, f"table_D{dim}.tex"), "w") as f:
            f.write(latex_tab)
        with open(os.path.join(output_dir, f"table_D{dim}.md"), "w") as f:
            f.write(md_tab)

        # Compute Friedman & Nemenyi
        ranks_matrix = np.zeros((num_funcs, len(algorithms)))
        for i, fn in enumerate(f_names):
            means = [np.mean(benchmark_results[fn][alg]) for alg in algorithms]
            # Rank from 1 (best/lowest) to 6 (worst/highest)
            order = np.argsort(means)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(algorithms) + 1)
            ranks_matrix[i] = ranks

        chi2_F, p_chi2, F_F, mean_ranks = compute_friedman_test(ranks_matrix)
        cd = compute_nemenyi_cd(len(algorithms), num_funcs)

        stats_report = f"""# Statistical Report (D = {dim})

## Friedman Test
- **Friedman Chi-Squared Statistic (\\chi_F^2):** {chi2_F:.4f} (p-value: {p_chi2:.4e})
- **Iman-Davenport F-Statistic (F_F):** {F_F:.4f}
- **Critical Difference (Nemenyi CD, \\alpha=0.05):** {cd:.4f}

## Average Friedman Ranks (Lower is Better)
"""
        for alg, r in zip(algorithms, mean_ranks):
            stats_report += f"- **{alg}**: {r:.4f}\n"

        stats_report += f"\n## Wilcoxon Pairwise Comparisons (SMTS vs Competitor, \\alpha=0.05)\n"
        for alg in algorithms:
            if alg == "SMTS":
                continue
            wins, ties, losses = 0, 0, 0
            for fn in f_names:
                smts_vals = np.array(benchmark_results[fn]["SMTS"])
                comp_vals = np.array(benchmark_results[fn][alg])
                _, sign = compute_wilcoxon_test(smts_vals, comp_vals)
                if sign == "+":
                    wins += 1
                elif sign == "=":
                    ties += 1
                else:
                    losses += 1
            stats_report += f"- **SMTS vs {alg}**: {wins} Wins / {ties} Ties / {losses} Losses\n"

        print(stats_report)
        with open(os.path.join(output_dir, f"statistics_D{dim}.md"), "w") as f:
            f.write(stats_report)


if __name__ == "__main__":
    main()

"""
Statistical Analysis Module for Metaheuristic Benchmarking.
Implements:
1. Descriptive statistics (Mean, Std Dev, Median, IQR).
2. Wilcoxon signed-rank test (alpha = 0.05) with win/tie/loss (+/=/ -).
3. Friedman ranking test and Iman-Davenport correction.
4. Post-Hoc Nemenyi test and Critical Difference (CD) computation.
5. Automated LaTeX and Markdown table generation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import stats


def compute_wilcoxon_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, str]:
    """
    Perform two-sided Wilcoxon signed-rank test between sample_a and sample_b.
    Returns (p-value, sign):
        '+': sample_a significantly better than sample_b (p < alpha, mean_a < mean_b)
        '-': sample_a significantly worse than sample_b (p < alpha, mean_a > mean_b)
        '=': no statistically significant difference (p >= alpha)
    """
    diff = sample_a - sample_b
    if np.allclose(diff, 0.0):
        return 1.0, "="

    try:
        stat, p_val = stats.wilcoxon(sample_a, sample_b, zero_method="wilcox")
    except ValueError:
        return 1.0, "="

    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    if p_val < alpha:
        if mean_a < mean_b:
            sign = "+"
        else:
            sign = "-"
    else:
        sign = "="

    return float(p_val), sign


def compute_friedman_test(
    ranks_matrix: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute Friedman test and Iman-Davenport F-distribution extension.

    Parameters
    ----------
    ranks_matrix : np.ndarray, shape (num_problems, num_algorithms)
        Matrix of ranks for each algorithm on each problem.

    Returns
    -------
    chi2_F : float
        Friedman chi-squared statistic.
    p_chi2 : float
        Chi-squared p-value.
    F_F : float
        Iman-Davenport F-statistic.
    mean_ranks : np.ndarray, shape (num_algorithms,)
        Average rank for each algorithm across all problems.
    """
    N, k = ranks_matrix.shape
    mean_ranks = np.mean(ranks_matrix, axis=0)

    # Friedman chi-squared
    chi2_F = (12.0 * N / (k * (k + 1.0))) * (np.sum(mean_ranks ** 2) - (k * (k + 1.0) ** 2) / 4.0)
    p_chi2 = 1.0 - stats.chi2.cdf(chi2_F, df=k - 1)

    # Iman-Davenport F-statistic
    denom = (N * (k - 1.0) - chi2_F)
    if denom <= 0:
        F_F = float("inf")
        p_F = 0.0
    else:
        F_F = ((N - 1.0) * chi2_F) / denom
        p_F = 1.0 - stats.f.cdf(F_F, dfn=k - 1, dfd=(k - 1) * (N - 1))

    return chi2_F, p_chi2, F_F, mean_ranks


def compute_nemenyi_cd(num_algorithms: int, num_problems: int, alpha: float = 0.05) -> float:
    """
    Compute Critical Difference (CD) for post-hoc Nemenyi test.
    Critical values q_alpha for alpha=0.05 from Studentized range distribution:
    k: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    q_alpha_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    q = q_alpha_005.get(num_algorithms, 2.850)
    cd = q * np.sqrt((num_algorithms * (num_algorithms + 1.0)) / (6.0 * num_problems))
    return float(cd)


def generate_latex_table(
    results_dict: Dict[str, Dict[str, List[float]]],
    algorithm_names: List[str],
    function_names: List[str],
    dim: int,
    smts_name: str = "SMTS"
) -> str:
    """
    Generate publication-ready LaTeX table with Mean, Std Dev, and Wilcoxon signs.
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Experimental results on CEC benchmark functions ($D = {dim}$) over 30 independent runs. Best mean values are in \\textbf{{bold}}. $+/=/-$ denotes SMTS is statistically significantly better, equivalent, or worse via Wilcoxon signed-rank test ($\\alpha = 0.05$).}}")
    lines.append(f"\\label{{tab:results_D{dim}}}")
    lines.append(r"\scriptsize")
    
    col_def = "l" + "c" * len(algorithm_names)
    lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    lines.append(r"\hline")
    
    header = "Function & " + " & ".join([f"\\textbf{{{alg}}}" for alg in algorithm_names]) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for f_name in function_names:
        row_vals = []
        means = []
        stds = []
        for alg in algorithm_names:
            arr = np.array(results_dict[f_name][alg])
            means.append(np.mean(arr))
            stds.append(np.std(arr))

        best_mean = min(means)
        smts_arr = np.array(results_dict[f_name][smts_name])

        cell_strs = []
        for i, alg in enumerate(algorithm_names):
            m = means[i]
            s = stds[i]
            val_str = f"{m:.2e} $\\pm$ {s:.2e}"
            if np.isclose(m, best_mean, rtol=1e-3) or m == best_mean:
                val_str = f"\\textbf{{{val_str}}}"

            if alg != smts_name:
                comp_arr = np.array(results_dict[f_name][alg])
                _, sign = compute_wilcoxon_test(smts_arr, comp_arr)
                val_str += f" ({sign})"

            cell_strs.append(val_str)

        row_line = f"{f_name} & " + " & ".join(cell_strs) + r" \\"
        lines.append(row_line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_markdown_table(
    results_dict: Dict[str, Dict[str, List[float]]],
    algorithm_names: List[str],
    function_names: List[str],
    dim: int,
    smts_name: str = "SMTS"
) -> str:
    """Generate Markdown comparison table."""
    lines = []
    lines.append(f"### Benchmark Comparison Table ($D = {dim}$)")
    lines.append("")
    headers = ["Function"] + algorithm_names
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for f_name in function_names:
        means = [np.mean(results_dict[f_name][alg]) for alg in algorithm_names]
        best_m = min(means)
        smts_arr = np.array(results_dict[f_name][smts_name])

        cells = [f_name]
        for alg in algorithm_names:
            arr = np.array(results_dict[f_name][alg])
            m = np.mean(arr)
            s = np.std(arr)
            cell_str = f"{m:.2e} ± {s:.2e}"
            if np.isclose(m, best_m, rtol=1e-3) or m == best_m:
                cell_str = f"**{cell_str}**"

            if alg != smts_name:
                _, sign = compute_wilcoxon_test(smts_arr, arr)
                cell_str += f" ({sign})"

            cells.append(cell_str)

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

"""
Unit tests for baselines and benchmark suite.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from benchmarks.cec_suite import get_benchmark_suite
from baselines import (
    LSHADEOptimizer,
    CMAESOptimizer,
    CLPSOOptimizer,
    StandardPSOOptimizer,
    GWOOptimizer,
)
from smts import SMTSOptimizer


def test_benchmarks():
    funcs = get_benchmark_suite(dim=10)
    assert len(funcs) == 10, f"Expected 10 benchmark functions, got {len(funcs)}"
    x = np.zeros(10)
    for f in funcs:
        val = f(x)
        assert np.isfinite(val), f"Function {f.name} returned non-finite value {val}"
    print("test_benchmarks passed.")


def test_all_algorithms():
    funcs = get_benchmark_suite(dim=10)
    f_test = funcs[0]  # High-Conditioned Elliptic

    optimizers = [
        ("SMTS", lambda: SMTSOptimizer(f_test, dim=10, pop_size=20, max_evals=1000, seed=42)),
        ("LSHADE", lambda: LSHADEOptimizer(f_test, dim=10, pop_size_init=20, max_evals=1000, seed=42)),
        ("CMAES", lambda: CMAESOptimizer(f_test, dim=10, max_evals=1000, seed=42)),
        ("CLPSO", lambda: CLPSOOptimizer(f_test, dim=10, pop_size=20, max_evals=1000, seed=42)),
        ("PSO", lambda: StandardPSOOptimizer(f_test, dim=10, pop_size=20, max_evals=1000, seed=42)),
        ("GWO", lambda: GWOOptimizer(f_test, dim=10, pop_size=20, max_evals=1000, seed=42)),
    ]

    for name, opt_factory in optimizers:
        opt = opt_factory()
        res = opt.optimize()
        assert np.isfinite(res["best_f"]), f"{name} produced non-finite best_f"
        print(f"Algorithm {name}: best_f = {res['best_f']:.4e} in {res['evals']} evals")

    print("test_all_algorithms passed.")


if __name__ == "__main__":
    test_benchmarks()
    test_all_algorithms()

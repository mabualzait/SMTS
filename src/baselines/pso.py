"""
Standard Particle Swarm Optimization (PSO) with linearly decaying inertia weight.
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np


class StandardPSOOptimizer:
    """Standard Global-Best PSO Implementation."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size: int = 50,
        max_evals: int = 100000,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.full(dim, bounds[0], dtype=np.float64)
        self.ub = np.full(dim, bounds[1], dtype=np.float64)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.evals_count = 0
        self.gbest_x: Optional[np.ndarray] = None
        self.gbest_f: float = float("inf")
        self.convergence_curve: List[float] = []

    def _eval(self, x: np.ndarray) -> float:
        self.evals_count += 1
        return float(self.objective_func(x))

    def optimize(self) -> Dict[str, Any]:
        N = self.pop_size
        D = self.dim
        domain_span = self.ub - self.lb

        X = np.random.uniform(self.lb, self.ub, size=(N, D))
        V = np.random.uniform(-0.2 * domain_span, 0.2 * domain_span, size=(N, D))

        fitness = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if self.evals_count >= self.max_evals:
                break
            fitness[i] = self._eval(X[i])

        P = X.copy()
        P_fit = fitness.copy()

        best_idx = np.argmin(P_fit)
        self.gbest_f = P_fit[best_idx]
        self.gbest_x = P[best_idx].copy()
        self.convergence_curve.append(self.gbest_f)

        max_vel = 0.2 * domain_span

        while self.evals_count < self.max_evals:
            w = self.w_max - (self.w_max - self.w_min) * (self.evals_count / self.max_evals)

            r1 = np.random.uniform(0.0, 1.0, size=(N, D))
            r2 = np.random.uniform(0.0, 1.0, size=(N, D))

            V = w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (self.gbest_x - X)
            V = np.clip(V, -max_vel, max_vel)

            X = X + V
            X = np.clip(X, self.lb, self.ub)

            for i in range(N):
                if self.evals_count >= self.max_evals:
                    break
                fit_i = self._eval(X[i])
                fitness[i] = fit_i

                if fit_i < P_fit[i]:
                    P_fit[i] = fit_i
                    P[i] = X[i].copy()

                    if fit_i < self.gbest_f:
                        self.gbest_f = fit_i
                        self.gbest_x = X[i].copy()

            self.convergence_curve.append(self.gbest_f)

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
        }

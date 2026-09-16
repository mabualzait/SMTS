"""
Grey Wolf Optimizer (GWO).
Reference: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
Grey wolf optimizer. Advances in Engineering Software, 69, 46-61.
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np


class GWOOptimizer:
    """Grey Wolf Optimizer Implementation."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size: int = 50,
        max_evals: int = 100000,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.full(dim, bounds[0], dtype=np.float64)
        self.ub = np.full(dim, bounds[1], dtype=np.float64)
        self.pop_size = pop_size
        self.max_evals = max_evals
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

        # Initialize pack
        X = np.random.uniform(self.lb, self.ub, size=(N, D))
        fitness = np.zeros(N, dtype=np.float64)

        for i in range(N):
            if self.evals_count >= self.max_evals:
                break
            fitness[i] = self._eval(X[i])

        # Find Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness)
        alpha_pos = X[sorted_idx[0]].copy()
        alpha_score = fitness[sorted_idx[0]]

        beta_pos = X[sorted_idx[1]].copy()
        beta_score = fitness[sorted_idx[1]]

        delta_pos = X[sorted_idx[2]].copy()
        delta_score = fitness[sorted_idx[2]]

        self.gbest_x = alpha_pos.copy()
        self.gbest_f = alpha_score
        self.convergence_curve.append(self.gbest_f)

        while self.evals_count < self.max_evals:
            # a decreases linearly from 2 to 0
            a = 2.0 - 2.0 * (self.evals_count / self.max_evals)

            for i in range(N):
                if self.evals_count >= self.max_evals:
                    break

                # Vectorized wolf coordinate calculation
                # Alpha update
                r1 = np.random.uniform(0.0, 1.0, size=D)
                r2 = np.random.uniform(0.0, 1.0, size=D)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - X[i])
                X1 = alpha_pos - A1 * D_alpha

                # Beta update
                r1 = np.random.uniform(0.0, 1.0, size=D)
                r2 = np.random.uniform(0.0, 1.0, size=D)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - X[i])
                X2 = beta_pos - A2 * D_beta

                # Delta update
                r1 = np.random.uniform(0.0, 1.0, size=D)
                r2 = np.random.uniform(0.0, 1.0, size=D)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos - X[i])
                X3 = delta_pos - A3 * D_delta

                # Enclosing position
                X_new_i = (X1 + X2 + X3) / 3.0
                X_new_i = np.clip(X_new_i, self.lb, self.ub)
                X[i] = X_new_i

                # Evaluation
                fit_val = self._eval(X[i])
                fitness[i] = fit_val

                # Update Alpha, Beta, Delta
                if fit_val < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fit_val
                    alpha_pos = X[i].copy()
                elif fit_val < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fit_val
                    beta_pos = X[i].copy()
                elif fit_val < delta_score:
                    delta_score = fit_val
                    delta_pos = X[i].copy()

            self.gbest_x = alpha_pos.copy()
            self.gbest_f = alpha_score
            self.convergence_curve.append(self.gbest_f)

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
        }

"""
Comprehensive Learning Particle Swarm Optimizer (CLPSO).
Reference: Liang, J. J., Qin, A. K., Suganthan, P. N., & Baskar, S. (2006).
Comprehensive learning particle swarm optimizer for global optimization of
multimodal functions. IEEE Transactions on Evolutionary Computation, 10(3), 281-295.
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np


class CLPSOOptimizer:
    """CLPSO Algorithm Implementation."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size: int = 50,
        max_evals: int = 100000,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c: float = 1.49445,
        refresh_gap: int = 7,
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
        self.c = c
        self.refresh_gap = refresh_gap
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

    def _select_exemplar(self, i: int, d: int, P_fit: np.ndarray) -> int:
        """Tournament selection between two random particles."""
        cand1 = np.random.randint(0, self.pop_size)
        cand2 = np.random.randint(0, self.pop_size)
        return cand1 if P_fit[cand1] < P_fit[cand2] else cand2

    def optimize(self) -> Dict[str, Any]:
        N = self.pop_size
        D = self.dim

        # Learning probabilities Pc
        rank = np.arange(N)
        Pc = 0.05 + 0.45 * (np.exp(10.0 * rank / max(1, N - 1)) - 1.0) / (np.exp(10.0) - 1.0)

        # Initialize population and velocities
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
        stagnation = np.zeros(N, dtype=int)

        best_idx = np.argmin(P_fit)
        self.gbest_f = P_fit[best_idx]
        self.gbest_x = P[best_idx].copy()
        self.convergence_curve.append(self.gbest_f)

        # Exemplars matrix: shape (N, D), stores particle index each dim learns from
        exemplars = np.zeros((N, D), dtype=int)
        for i in range(N):
            for d in range(D):
                if np.random.uniform(0.0, 1.0) < Pc[i]:
                    exemplars[i, d] = self._select_exemplar(i, d, P_fit)
                else:
                    exemplars[i, d] = i
            # If all learn from self, force at least one dimension to learn from other
            if np.all(exemplars[i] == i):
                rand_d = np.random.randint(0, D)
                other_candidates = [idx for idx in range(N) if idx != i]
                exemplars[i, rand_d] = np.random.choice(other_candidates)

        max_vel = 0.2 * domain_span

        while self.evals_count < self.max_evals:
            w = self.w_max - (self.w_max - self.w_min) * (self.evals_count / self.max_evals)

            for i in range(N):
                if self.evals_count >= self.max_evals:
                    break

                # If stagnant, re-learn exemplars
                if stagnation[i] >= self.refresh_gap:
                    stagnation[i] = 0
                    for d in range(D):
                        if np.random.uniform(0.0, 1.0) < Pc[i]:
                            exemplars[i, d] = self._select_exemplar(i, d, P_fit)
                        else:
                            exemplars[i, d] = i
                    if np.all(exemplars[i] == i):
                        rand_d = np.random.randint(0, D)
                        other_candidates = [idx for idx in range(N) if idx != i]
                        exemplars[i, rand_d] = np.random.choice(other_candidates)

                # Velocity update
                r = np.random.uniform(0.0, 1.0, size=D)
                pbest_exemplar_pos = np.array([P[exemplars[i, d], d] for d in range(D)])
                
                V[i] = w * V[i] + self.c * r * (pbest_exemplar_pos - X[i])
                V[i] = np.clip(V[i], -max_vel, max_vel)

                # Position update
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], self.lb, self.ub)

                # Evaluate
                f_val = self._eval(X[i])
                fitness[i] = f_val

                if f_val < P_fit[i]:
                    P_fit[i] = f_val
                    P[i] = X[i].copy()
                    stagnation[i] = 0

                    if f_val < self.gbest_f:
                        self.gbest_f = f_val
                        self.gbest_x = X[i].copy()
                else:
                    stagnation[i] += 1

            self.convergence_curve.append(self.gbest_f)

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
        }

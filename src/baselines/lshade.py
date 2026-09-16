"""
L-SHADE: Success-History based Adaptive Differential Evolution with
Linear Population Size Reduction.
Reference: Tanabe, R., & Fukunaga, A. S. (2014). Improving the search performance
of SHADE using linear population size reduction. IEEE CEC 2014.
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np


class LSHADEOptimizer:
    """L-SHADE Algorithm Implementation."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size_init: Optional[int] = None,
        pop_size_min: int = 4,
        max_evals: int = 100000,
        memory_size: int = 5,
        p_best_rate: float = 0.11,
        arc_rate: float = 1.4,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.full(dim, bounds[0], dtype=np.float64)
        self.ub = np.full(dim, bounds[1], dtype=np.float64)
        
        # Default initial pop size: 18 * dim (standard in CEC L-SHADE papers)
        # or bounded by max_evals
        self.pop_size_init = pop_size_init if pop_size_init is not None else min(100, max(30, int(18 * np.log(dim + 1))))
        self.pop_size_min = pop_size_min
        self.max_evals = max_evals
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate
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
        N = self.pop_size_init
        H = self.memory_size
        M_CR = np.full(H, 0.5, dtype=np.float64)
        M_F = np.full(H, 0.5, dtype=np.float64)
        k_mem = 0

        # Initialize population
        X = np.random.uniform(self.lb, self.ub, size=(N, self.dim))
        fitness = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if self.evals_count >= self.max_evals:
                break
            fitness[i] = self._eval(X[i])

        best_idx = np.argmin(fitness)
        self.gbest_f = fitness[best_idx]
        self.gbest_x = X[best_idx].copy()
        self.convergence_curve.append(self.gbest_f)

        archive = np.empty((0, self.dim), dtype=np.float64)
        max_arc_size = int(N * self.arc_rate)

        while self.evals_count < self.max_evals:
            S_CR = []
            S_F = []
            delta_f_list = []
            X_new = np.zeros_like(X)
            fitness_new = np.zeros(N, dtype=np.float64)

            # Sort population to find p-bests
            sorted_indices = np.argsort(fitness)
            p_num = max(2, int(round(self.p_best_rate * N)))

            for i in range(N):
                if self.evals_count >= self.max_evals:
                    break

                # 1. Sample CR and F from memory
                r_mem = np.random.randint(0, H)
                mu_cr = M_CR[r_mem]
                mu_f = M_F[r_mem]

                if mu_cr < 0:
                    cr = 0.0
                else:
                    cr = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)

                # Cauchy distribution for F
                f = -1.0
                while f <= 0:
                    f = mu_f + 0.1 * np.tan(np.pi * (np.random.uniform(0.0, 1.0) - 0.5))
                f = min(1.0, f)

                # 2. Mutation: current-to-pbest/1 with archive
                pbest_candidate_idx = sorted_indices[np.random.randint(0, p_num)]
                x_pbest = X[pbest_candidate_idx]

                # r1 != i
                r1_candidates = [idx for idx in range(N) if idx != i]
                r1 = np.random.choice(r1_candidates)
                x_r1 = X[r1]

                # r2 from population + archive
                total_candidates = N + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, total_candidates)

                if r2 < N:
                    x_r2 = X[r2]
                else:
                    x_r2 = archive[r2 - N]

                v = X[i] + f * (x_pbest - X[i]) + f * (x_r1 - x_r2)

                # 3. Boundary handling (midpoint reflection)
                lower_mask = v < self.lb
                v[lower_mask] = (self.lb[lower_mask] + X[i][lower_mask]) / 2.0
                upper_mask = v > self.ub
                v[upper_mask] = (self.ub[upper_mask] + X[i][upper_mask]) / 2.0

                # 4. Binomial Crossover
                j_rand = np.random.randint(0, self.dim)
                cross_mask = np.random.uniform(0.0, 1.0, size=self.dim) < cr
                cross_mask[j_rand] = True

                u = np.where(cross_mask, v, X[i])

                # 5. Selection
                f_u = self._eval(u)
                if f_u <= fitness[i]:
                    X_new[i] = u
                    fitness_new[i] = f_u

                    if f_u < fitness[i]:
                        # Successful parameter
                        S_CR.append(cr)
                        S_F.append(f)
                        delta_f_list.append(fitness[i] - f_u)

                        # Add replaced parent to archive
                        if len(archive) < max_arc_size:
                            archive = np.vstack([archive, X[i]])
                        else:
                            replace_idx = np.random.randint(0, max_arc_size)
                            archive[replace_idx] = X[i]

                    if f_u < self.gbest_f:
                        self.gbest_f = f_u
                        self.gbest_x = u.copy()
                else:
                    X_new[i] = X[i]
                    fitness_new[i] = fitness[i]

            X = X_new
            fitness = fitness_new
            self.convergence_curve.append(self.gbest_f)

            # 6. Memory updates (Lehmer mean for F, weighted mean for CR)
            if len(S_CR) > 0:
                weights = np.array(delta_f_list) / (np.sum(delta_f_list) + 1e-12)
                M_CR[k_mem] = np.sum(weights * np.array(S_CR))

                # Lehmer mean
                f_arr = np.array(S_F)
                M_F[k_mem] = np.sum(weights * (f_arr ** 2)) / (np.sum(weights * f_arr) + 1e-12)
                k_mem = (k_mem + 1) % H

            # 7. Linear Population Size Reduction (LPSR)
            next_N = int(round(
                self.pop_size_init - (self.evals_count / self.max_evals) * (self.pop_size_init - self.pop_size_min)
            ))
            next_N = max(self.pop_size_min, next_N)

            if next_N < N:
                # Remove worst individuals
                sorted_idx = np.argsort(fitness)
                keep_idx = sorted_idx[:next_N]
                X = X[keep_idx]
                fitness = fitness[keep_idx]
                N = next_N
                max_arc_size = int(N * self.arc_rate)
                if len(archive) > max_arc_size:
                    archive = archive[:max_arc_size]

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
        }

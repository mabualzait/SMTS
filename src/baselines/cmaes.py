"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
Standard (mu/mu_w, lambda)-CMA-ES with Cumulative Step-Size Adaptation (CSA)
and rank-1 / rank-mu covariance updates.
Reference: Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np


class CMAESOptimizer:
    """CMA-ES Optimizer Implementation."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size: Optional[int] = None,
        sigma0: Optional[float] = None,
        max_evals: int = 100000,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.full(dim, bounds[0], dtype=np.float64)
        self.ub = np.full(dim, bounds[1], dtype=np.float64)
        self.max_evals = max_evals
        self.seed = seed

        # Default lambda (population size): 4 + floor(3 * ln(dim))
        self.lam = pop_size if pop_size is not None else int(4 + np.floor(3 * np.log(dim)))
        self.mu = self.lam // 2

        # Recombination weights
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)

        # Step size control parameters
        self.cs = (self.mueff + 2.0) / (dim + self.mueff + 5.0)
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (dim + 1.0)) - 1.0) + self.cs
        self.chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # Covariance matrix adaptation parameters
        self.cc = (4.0 + self.mueff / dim) / (dim + 4.0 + 2.0 * self.mueff / dim)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((dim + 2.0) ** 2 + self.mueff)
        )

        # Initial step-size
        domain_span = self.ub[0] - self.lb[0]
        self.sigma = sigma0 if sigma0 is not None else 0.3 * domain_span

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
        N = self.dim
        # Initial mean within bounds
        xmean = np.random.uniform(self.lb, self.ub)

        # Evolution paths
        ps = np.zeros(N, dtype=np.float64)
        pc = np.zeros(N, dtype=np.float64)

        # Coordinate system
        B = np.eye(N, dtype=np.float64)
        D = np.ones(N, dtype=np.float64)
        C = np.eye(N, dtype=np.float64)
        invsqrtC = np.eye(N, dtype=np.float64)

        eigeneval = 0
        counteval = 0

        while self.evals_count < self.max_evals:
            # 1. Sample lambda offspring
            arz = np.random.randn(self.lam, N)
            # arx = xmean + sigma * (B * (D * arz))
            ary = np.dot(arz * D, B.T)
            arx = xmean + self.sigma * ary

            # Boundary handling: clip to domain with penalty
            fitness = np.zeros(self.lam, dtype=np.float64)
            for i in range(self.lam):
                if self.evals_count >= self.max_evals:
                    break
                x_clipped = np.clip(arx[i], self.lb, self.ub)
                penalty = 1e3 * np.sum((arx[i] - x_clipped) ** 2)
                f_val = self._eval(x_clipped) + penalty
                fitness[i] = f_val

                if f_val < self.gbest_f:
                    self.gbest_f = f_val
                    self.gbest_x = x_clipped.copy()

            self.convergence_curve.append(self.gbest_f)
            counteval += self.lam

            # Sort by fitness
            arindex = np.argsort(fitness)
            arz = arz[arindex]
            ary = ary[arindex]
            arx = arx[arindex]

            # 2. Update mean
            xold = xmean.copy()
            xmean = np.sum(arx[:self.mu] * self.weights[:, np.newaxis], axis=0)
            y = np.sum(ary[:self.mu] * self.weights[:, np.newaxis], axis=0)
            z = np.sum(arz[:self.mu] * self.weights[:, np.newaxis], axis=0)

            # 3. Update evolution paths
            ps = (1.0 - self.cs) * ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * np.dot(B, z)
            hsig = float(
                np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * counteval / self.lam)) / self.chiN
                < 1.4 + 2.0 / (N + 1.0)
            )
            pc = (1.0 - self.cc) * pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y

            # 4. Adapt covariance matrix C
            # Rank-mu update
            artmp = ary[:self.mu]
            Cmu = np.dot(artmp.T * self.weights, artmp)
            
            # Rank-one update
            C = (
                (1.0 - self.c1 - self.cmu) * C
                + self.c1 * (np.outer(pc, pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * C)
                + self.cmu * Cmu
            )

            # 5. Adapt step-size sigma
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / self.chiN - 1.0))

            # 6. Decomposition of C into B * D^2 * B^T
            if counteval - eigeneval > self.lam / (self.c1 + self.cmu) / N / 10.0:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
                try:
                    eigenvals, B = np.linalg.eigh(C)
                    eigenvals = np.maximum(1e-14, eigenvals)
                    D = np.sqrt(eigenvals)
                    invsqrtC = np.dot(B, np.dot(np.diag(1.0 / D), B.T))
                except np.linalg.LinAlgError:
                    # Reset covariance if numerical degeneracy occurs
                    C = np.eye(N)
                    B = np.eye(N)
                    D = np.ones(N)

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
        }

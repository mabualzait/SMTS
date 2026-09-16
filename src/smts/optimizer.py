"""
Starling Murmuration Topological Search (SMTS) Optimizer.

A continuous metaheuristic grounded in the empirical physics and
topological interaction networks of starling murmurations (Cavagna et al., 2010;
Attanasi et al., 2014; Bialek et al., 2012).
"""

from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np
from .operators import (
    build_topological_neighbors,
    compute_spin_wave_alignment,
    compute_topological_gradient,
    adaptive_boundary_rebound,
)


class SMTSOptimizer:
    """
    Starling Murmuration Topological Search (SMTS) Optimizer.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        pop_size: int = 50,
        max_evals: int = 100000,
        k_neighbors: int = 7,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1_init: float = 1.2,
        c2_init: float = 1.0,
        c3_init: float = 1.2,
        c4_final: float = 1.5,
        fluctuation_amp: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize the SMTS Optimizer.

        Parameters
        ----------
        objective_func : Callable[[np.ndarray], float]
            Fitness function taking 1D array of shape (D,) returning float scalar.
        dim : int
            Dimension of problem D.
        bounds : Tuple[float, float]
            Lower and upper bounds for all dimensions.
        pop_size : int
            Number of starlings in the murmuration (N).
        max_evals : int
            Maximum number of objective function evaluations.
        k_neighbors : int
            Topological neighborhood size (default 7, based on Cavagna et al., 2010).
        w_max, w_min : float
            Inertial momentum bounds.
        c1_init : float
            Spin-wave alignment coefficient.
        c2_init : float
            Topological simplex gradient descent coefficient.
        c3_init : float
            Topological local elite attraction coefficient.
        c4_final : float
            Global murmuration convergence coefficient in late search phase.
        fluctuation_amp : float
            Scale-free thermal fluctuation amplitude factor.
        seed : Optional[int]
            Random number generator seed.
        """
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.full(dim, bounds[0], dtype=np.float64)
        self.ub = np.full(dim, bounds[1], dtype=np.float64)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.k_neighbors = k_neighbors
        self.w_max = w_max
        self.w_min = w_min
        self.c1_init = c1_init
        self.c2_init = c2_init
        self.c3_init = c3_init
        self.c4_final = c4_final
        self.fluctuation_amp = fluctuation_amp
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # State tracking
        self.evals_count = 0
        self.gbest_x: Optional[np.ndarray] = None
        self.gbest_f: float = float("inf")
        self.convergence_curve: List[float] = []
        self.diversity_curve: List[float] = []

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate a single individual and increment evaluation counter."""
        self.evals_count += 1
        return float(self.objective_func(x))

    def _measure_diversity(self, positions: np.ndarray) -> float:
        """Compute normalized population spatial spread diversity."""
        centroid = np.mean(positions, axis=0)
        span = np.linalg.norm(self.ub - self.lb) + 1e-12
        disp = np.linalg.norm(positions - centroid, axis=1)
        return float(np.mean(disp) / span)

    def optimize(self) -> Dict[str, Any]:
        """
        Run SMTS optimization until max_evals is reached.

        Returns
        -------
        Dict with keys:
            'best_x': np.ndarray, global best candidate found.
            'best_f': float, global best objective value.
            'evals': int, total evaluations performed.
            'convergence': list of best_f logged over time.
            'diversity': list of diversity metrics logged over time.
        """
        # 1. Initialize murmuration swarm uniformly across domain
        X = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        
        # Velocity initialization: bounded fraction of domain span
        domain_span = self.ub - self.lb
        V = np.random.uniform(-0.1 * domain_span, 0.1 * domain_span, size=(self.pop_size, self.dim))
        
        # Initial evaluations
        fitness = np.zeros(self.pop_size, dtype=np.float64)
        for i in range(self.pop_size):
            if self.evals_count >= self.max_evals:
                break
            fitness[i] = self._eval(X[i])

        # Initialize personal bests
        P = X.copy()
        P_fit = fitness.copy()

        # Update global best
        min_idx = np.argmin(P_fit)
        self.gbest_x = P[min_idx].copy()
        self.gbest_f = P_fit[min_idx]

        self.convergence_curve.append(self.gbest_f)
        self.diversity_curve.append(self._measure_diversity(X))

        # Iterative Murmuration Cycle
        while self.evals_count < self.max_evals:
            # Normalized search progression tau in [0, 1]
            tau = min(1.0, self.evals_count / max(1, self.max_evals))

            # Dynamic parameters
            w = self.w_max - (self.w_max - self.w_min) * tau
            c1 = self.c1_init * (1.0 - 0.5 * tau)
            c2 = self.c2_init * (1.0 - tau)
            c3 = self.c3_init * (1.0 - 0.5 * tau)
            c4 = self.c4_final * (tau ** 2)

            # 2. Build Topological interaction graph G_t = (V, E_t)
            neighbor_indices, neighbor_distances = build_topological_neighbors(
                X, k=self.k_neighbors
            )

            # 3. Kinematic Spin-Wave velocity consensus
            a_spin = compute_spin_wave_alignment(V, neighbor_indices)

            # 4. Topological Simplex Gradient Descent
            topo_grad = compute_topological_gradient(
                X, fitness, neighbor_indices, neighbor_distances
            )

            # 5. Local Topological Elite Attraction
            # For each agent i, find the best neighbor in its closed topological neighborhood
            local_elite_pos = np.zeros_like(X)
            for i in range(self.pop_size):
                sub_indices = np.append(neighbor_indices[i], i)
                sub_fits = fitness[sub_indices]
                best_sub = sub_indices[np.argmin(sub_fits)]
                local_elite_pos[i] = P[best_sub]

            a_elite = local_elite_pos - X
            a_gbest = self.gbest_x - X

            # 6. Scale-Free Thermal Fluctuation (adapted to local neighborhood radius)
            mean_neighbor_dist = np.mean(neighbor_distances, axis=1, keepdims=True)  # (N, 1)
            cauchy_noise = np.random.standard_cauchy(size=(self.pop_size, self.dim))
            cauchy_noise = np.clip(cauchy_noise, -5.0, 5.0)
            fluctuation = self.fluctuation_amp * ((1.0 - tau) ** 2) * mean_neighbor_dist * cauchy_noise

            # 7. Velocity Update with Stochastic Weights
            r1 = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
            r2 = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
            r3 = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
            r4 = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))

            V = (
                w * V
                + c1 * r1 * a_spin
                + c2 * r2 * topo_grad
                + c3 * r3 * a_elite
                + c4 * r4 * a_gbest
                + fluctuation
            )

            # Velocity clamping to prevent ballistic divergence (20% domain span)
            max_vel = 0.2 * domain_span
            V = np.clip(V, -max_vel, max_vel)

            # 8. Position Update
            X = X + V

            # 9. Adaptive Boundary Rebound
            X, V = adaptive_boundary_rebound(X, V, self.lb, self.ub, P)

            # 10. Fitness Evaluation and Elite Updates
            for i in range(self.pop_size):
                if self.evals_count >= self.max_evals:
                    break
                fit_i = self._eval(X[i])
                fitness[i] = fit_i

                # Update personal best
                if fit_i < P_fit[i]:
                    P_fit[i] = fit_i
                    P[i] = X[i].copy()

                    # Update global best
                    if fit_i < self.gbest_f:
                        self.gbest_f = fit_i
                        self.gbest_x = X[i].copy()

            # Record trajectory metrics
            self.convergence_curve.append(self.gbest_f)
            self.diversity_curve.append(self._measure_diversity(X))

        return {
            "best_x": self.gbest_x,
            "best_f": self.gbest_f,
            "evals": self.evals_count,
            "convergence": self.convergence_curve,
            "diversity": self.diversity_curve,
        }

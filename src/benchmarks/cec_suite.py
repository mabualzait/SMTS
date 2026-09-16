"""
CEC-Style Benchmark Functions for Continuous Optimization.
Supports Dimensions D in {30, 50, 100}.
Functions include Unimodal, Simple Multimodal, Hybrid, and Composition landscapes
with random orthogonal rotations and coordinate shifts.
"""

from typing import Callable, Dict, Any, List, Tuple
import numpy as np


def _generate_rotation_matrix(dim: int, seed: int = 42) -> np.ndarray:
    """Generate an orthonormal rotation matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    # Ensure determinant is +1 (proper rotation)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = Q * ph
    return Q


class BenchmarkFunction:
    """Base class for shifted and rotated benchmark functions."""

    def __init__(
        self,
        name: str,
        dim: int,
        f_type: str,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        seed: int = 42,
    ):
        self.name = name
        self.dim = dim
        self.f_type = f_type
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.seed = seed
        
        # Shift vector within [-80, 80]
        rng = np.random.RandomState(seed)
        self.shift = rng.uniform(-80.0, 80.0, size=dim)
        # Rotation matrix
        self.rotation = _generate_rotation_matrix(dim, seed=seed + 1000)
        self.optimal_f = 0.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply shift and orthogonal rotation: z = M * (x - o)."""
        z = x - self.shift
        return np.dot(self.rotation, z)

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)


# -------------------------------------------------------------
# Unimodal Functions
# -------------------------------------------------------------

class HighConditionedElliptic(BenchmarkFunction):
    """F1: Shifted & Rotated High-Conditioned Elliptic Function."""
    def __init__(self, dim: int, seed: int = 1):
        super().__init__("F1: High-Conditioned Elliptic", dim, "Unimodal", seed=seed)
        self.condition = 1e6
        self.coeffs = self.condition ** (np.arange(dim) / (dim - 1))

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return float(np.sum(self.coeffs * (z ** 2)))


class BentCigar(BenchmarkFunction):
    """F2: Shifted & Rotated Bent Cigar Function."""
    def __init__(self, dim: int, seed: int = 2):
        super().__init__("F2: Bent Cigar", dim, "Unimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


class Discus(BenchmarkFunction):
    """F3: Shifted & Rotated Discus Function."""
    def __init__(self, dim: int, seed: int = 3):
        super().__init__("F3: Discus", dim, "Unimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return float(1e6 * (z[0] ** 2) + np.sum(z[1:] ** 2))


# -------------------------------------------------------------
# Simple Multimodal Functions
# -------------------------------------------------------------

class Rosenbrock(BenchmarkFunction):
    """F4: Shifted & Rotated Rosenbrock Function."""
    def __init__(self, dim: int, seed: int = 4):
        super().__init__("F4: Rosenbrock", dim, "Multimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Scaled to fit comfortably inside bounds
        z = 0.02048 * z + 1.0
        val = np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
        return float(val)


class Ackley(BenchmarkFunction):
    """F5: Shifted & Rotated Ackley Function."""
    def __init__(self, dim: int, seed: int = 5):
        super().__init__("F5: Ackley", dim, "Multimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        n = len(z)
        sum_sq = np.sum(z ** 2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * z))
        val = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e
        return float(val)


class Rastrigin(BenchmarkFunction):
    """F6: Shifted & Rotated Rastrigin Function."""
    def __init__(self, dim: int, seed: int = 6):
        super().__init__("F6: Rastrigin", dim, "Multimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Scale to standard Rastrigin domain [-5.12, 5.12]
        z = 0.0512 * z
        val = 10.0 * len(z) + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z))
        return float(val)


class Griewank(BenchmarkFunction):
    """F7: Shifted & Rotated Griewank Function."""
    def __init__(self, dim: int, seed: int = 7):
        super().__init__("F7: Griewank", dim, "Multimodal", seed=seed)

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Scale to standard [-600, 600]
        z = 6.0 * z
        i = np.arange(1, len(z) + 1)
        term1 = np.sum(z ** 2) / 4000.0
        term2 = np.prod(np.cos(z / np.sqrt(i)))
        return float(term1 - term2 + 1.0)


# -------------------------------------------------------------
# Hybrid & Composition Functions
# -------------------------------------------------------------

class HybridFunction1(BenchmarkFunction):
    """
    F8: Hybrid Function 1 (Elliptic 30%, Rastrigin 40%, Ackley 30%).
    Splits variables into distinct non-overlapping subsets.
    """
    def __init__(self, dim: int, seed: int = 8):
        super().__init__("F8: Hybrid (Elliptic+Rastrigin+Ackley)", dim, "Hybrid", seed=seed)
        # Determine subset splits
        p1 = int(0.3 * dim)
        p2 = int(0.7 * dim)
        self.idx1 = slice(0, p1)
        self.idx2 = slice(p1, p2)
        self.idx3 = slice(p2, dim)
        self.d1 = p1
        self.d2 = p2 - p1
        self.d3 = dim - p2
        self.coeffs = 1e6 ** (np.arange(self.d1) / max(1, self.d1 - 1))

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        z1 = z[self.idx1]
        z2 = 0.0512 * z[self.idx2]
        z3 = z[self.idx3]

        # Elliptic
        f1 = np.sum(self.coeffs * (z1 ** 2))
        # Rastrigin
        f2 = 10.0 * self.d2 + np.sum(z2 ** 2 - 10.0 * np.cos(2.0 * np.pi * z2))
        # Ackley
        f3 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z3 ** 2))) - np.exp(np.mean(np.cos(2.0 * np.pi * z3))) + 20.0 + np.e

        return float(f1 + f2 + f3)


class HybridFunction2(BenchmarkFunction):
    """
    F9: Hybrid Function 2 (Bent Cigar 30%, Griewank 30%, Rosenbrock 40%).
    """
    def __init__(self, dim: int, seed: int = 9):
        super().__init__("F9: Hybrid (Cigar+Griewank+Rosenbrock)", dim, "Hybrid", seed=seed)
        p1 = int(0.3 * dim)
        p2 = int(0.6 * dim)
        self.idx1 = slice(0, p1)
        self.idx2 = slice(p1, p2)
        self.idx3 = slice(p2, dim)
        self.d1 = p1
        self.d2 = p2 - p1
        self.d3 = dim - p2

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        z1 = z[self.idx1]
        z2 = 6.0 * z[self.idx2]
        z3 = 0.02048 * z[self.idx3] + 1.0

        f1 = z1[0] ** 2 + 1e6 * np.sum(z1[1:] ** 2) if len(z1) > 1 else z1[0] ** 2
        i = np.arange(1, len(z2) + 1)
        f2 = np.sum(z2 ** 2) / 4000.0 - np.prod(np.cos(z2 / np.sqrt(i))) + 1.0
        f3 = np.sum(100.0 * (z3[1:] - z3[:-1] ** 2) ** 2 + (z3[:-1] - 1.0) ** 2) if len(z3) > 1 else 0.0

        return float(f1 + f2 + f3)


class CompositionFunction(BenchmarkFunction):
    """
    F10: Composition Function (3 sub-functions: Rastrigin, Ackley, Sphere).
    Combines functions dynamically with Gaussian distance-dependent blending weights.
    """
    def __init__(self, dim: int, seed: int = 10):
        super().__init__("F10: Composition (Rastrigin+Ackley+Sphere)", dim, "Composition", seed=seed)
        rng = np.random.RandomState(seed + 555)
        self.centers = [rng.uniform(-60.0, 60.0, size=dim) for _ in range(3)]
        self.sigmas = [10.0, 20.0, 30.0]
        self.biases = [0.0, 100.0, 200.0]

    def evaluate(self, x: np.ndarray) -> float:
        D = self.dim
        weights = []
        vals = []

        # Sub-function 1: Rastrigin
        d0 = np.linalg.norm(x - self.centers[0])
        w0 = np.exp(-d0 ** 2 / (2.0 * D * (self.sigmas[0] ** 2)))
        z0 = 0.0512 * (x - self.centers[0])
        v0 = 10.0 * D + np.sum(z0 ** 2 - 10.0 * np.cos(2.0 * np.pi * z0)) + self.biases[0]
        weights.append(w0)
        vals.append(v0)

        # Sub-function 2: Ackley
        d1 = np.linalg.norm(x - self.centers[1])
        w1 = np.exp(-d1 ** 2 / (2.0 * D * (self.sigmas[1] ** 2)))
        z1 = x - self.centers[1]
        v1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z1 ** 2))) - np.exp(np.mean(np.cos(2.0 * np.pi * z1))) + 20.0 + np.e + self.biases[1]
        weights.append(w1)
        vals.append(v1)

        # Sub-function 3: Sphere
        d2 = np.linalg.norm(x - self.centers[2])
        w2 = np.exp(-d2 ** 2 / (2.0 * D * (self.sigmas[2] ** 2)))
        z2 = x - self.centers[2]
        v2 = np.sum(z2 ** 2) + self.biases[2]
        weights.append(w2)
        vals.append(v2)

        weights = np.array(weights)
        vals = np.array(vals)
        sum_w = np.sum(weights) + 1e-12
        weights = weights / sum_w

        return float(np.sum(weights * vals))


def get_benchmark_suite(dim: int) -> List[BenchmarkFunction]:
    """Return the complete 10-function CEC-style benchmark suite for dimension dim."""
    return [
        HighConditionedElliptic(dim),
        BentCigar(dim),
        Discus(dim),
        Rosenbrock(dim),
        Ackley(dim),
        Rastrigin(dim),
        Griewank(dim),
        HybridFunction1(dim),
        HybridFunction2(dim),
        CompositionFunction(dim),
    ]

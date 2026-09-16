"""
Unit tests for SMTS optimizer and operators.
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from smts import SMTSOptimizer, build_topological_neighbors, compute_spin_wave_alignment


def test_topological_neighbors():
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, size=(20, 5))
    indices, distances = build_topological_neighbors(positions, k=7)
    
    assert indices.shape == (20, 7), f"Expected (20, 7), got {indices.shape}"
    assert distances.shape == (20, 7), f"Expected (20, 7), got {distances.shape}"
    # Verify no self-loop
    for i in range(20):
        assert i not in indices[i], f"Self loop found for agent {i}"
    print("test_topological_neighbors passed.")


def test_spin_wave():
    np.random.seed(42)
    V = np.ones((10, 3))
    # If all velocities are identical, relative consensus acceleration should be 0
    indices = np.array([[j for j in range(10) if j != i][:6] for i in range(10)])
    acc = compute_spin_wave_alignment(V, indices)
    assert np.allclose(acc, 0.0), f"Expected 0 relative acceleration, got {acc}"
    print("test_spin_wave passed.")


def test_smts_sphere():
    def sphere(x):
        return np.sum(x ** 2)

    optimizer = SMTSOptimizer(
        objective_func=sphere,
        dim=10,
        bounds=(-10.0, 10.0),
        pop_size=30,
        max_evals=5000,
        k_neighbors=7,
        seed=42,
    )
    res = optimizer.optimize()
    print(f"SMTS Sphere 10D: best_f = {res['best_f']:.6e} in {res['evals']} evals")
    assert res["best_f"] < 1e-4, f"Sphere did not converge, best_f = {res['best_f']}"
    print("test_smts_sphere passed.")


if __name__ == "__main__":
    test_topological_neighbors()
    test_spin_wave()
    test_smts_sphere()
    print("All SMTS tests passed successfully!")

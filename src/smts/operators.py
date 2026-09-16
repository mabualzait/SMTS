"""
Starling Murmuration Topological Search (SMTS) - Core Operators.

Mathematical operators based on empirical physics of European starlings
(Cavagna et al., 2010; Bialek et al., 2012; Attanasi et al., 2014):
1. Topological k-nearest neighbor interaction graph construction via KD-Tree.
2. Undamped kinematic spin-wave velocity alignment operator.
3. Topological simplex gradient descent estimator.
4. Scale-free fluctuation injection and adaptive phase-shift scheduler.
5. Boundary-reflective and interior rebound operators.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


def build_topological_neighbors(
    positions: np.ndarray,
    k: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the dynamic topological interaction graph G_t = (V, E_t)
    where each agent i tracks exactly its k nearest neighbors regardless of density.

    Parameters
    ----------
    positions : np.ndarray, shape (N, D)
        Current population coordinates.
    k : int
        Topological degree (empirically k in [6, 7] for starlings).

    Returns
    -------
    neighbor_indices : np.ndarray, shape (N, k)
        Indices of the k topological neighbors for each agent.
    neighbor_distances : np.ndarray, shape (N, k)
        Euclidean distances to the k topological neighbors.
    """
    N, D = positions.shape
    # Ensure k does not exceed N - 1
    k_eff = min(k, N - 1)
    
    # Efficient O(N log N) spatial indexing
    tree = cKDTree(positions)
    # Query k_eff + 1 since the query point itself is returned at distance 0
    distances, indices = tree.query(positions, k=k_eff + 1)
    
    # Drop self-index (the first column)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    
    return neighbor_indices, neighbor_distances


def compute_spin_wave_alignment(
    velocities: np.ndarray,
    neighbor_indices: np.ndarray
) -> np.ndarray:
    """
    Compute undamped kinematic spin-wave velocity consensus:
        a_i^{topo} = (1 / k) * sum_{j in N_k(i)} (v_j - v_i)

    Parameters
    ----------
    velocities : np.ndarray, shape (N, D)
        Current agent velocity vectors.
    neighbor_indices : np.ndarray, shape (N, k)
        Topological neighbor indices.

    Returns
    -------
    alignment_acc : np.ndarray, shape (N, D)
        Spin-wave consensus acceleration vector.
    """
    # Vectorized gathering of neighbor velocities: shape (N, k, D)
    neighbor_vels = velocities[neighbor_indices]
    
    # Mean neighbor velocity: shape (N, D)
    mean_neighbor_vel = np.mean(neighbor_vels, axis=1)
    
    # Relative difference: (1/k) * sum (v_j - v_i) = mean(v_j) - v_i
    alignment_acc = mean_neighbor_vel - velocities
    return alignment_acc


def compute_topological_gradient(
    positions: np.ndarray,
    fitness: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute gradient-free topological simplex descent direction:
    Evaluates directional downhill derivatives using existing neighbor fitness.

    Parameters
    ----------
    positions : np.ndarray, shape (N, D)
        Current population coordinates.
    fitness : np.ndarray, shape (N,)
        Objective values f(x_i).
    neighbor_indices : np.ndarray, shape (N, k)
        Topological neighbor indices.
    neighbor_distances : np.ndarray, shape (N, k)
        Distances to topological neighbors.
    eps : float
        Numerical regularizer.

    Returns
    -------
    topo_grad : np.ndarray, shape (N, D)
        Topological simplex descent direction vector for each agent.
    """
    N, k = neighbor_indices.shape
    D = positions.shape[1]
    
    # Neighbor fitness values: shape (N, k)
    neighbor_fitness = fitness[neighbor_indices]
    
    # Fitness advantage of neighbors over agent i: f(x_i) - f(x_j)
    # Positive when neighbor is better (lower objective value in minimization)
    delta_f = fitness[:, np.newaxis] - neighbor_fitness
    
    # Relative displacement: x_j - x_i, shape (N, k, D)
    neighbor_pos = positions[neighbor_indices]
    displacement = neighbor_pos - positions[:, np.newaxis, :]
    
    # Distance squared: ||x_j - x_i||^2
    dist_sq = (neighbor_distances ** 2)[:, :, np.newaxis] + eps
    
    # Downhill simplex gradient direction:
    # - sum_{j in N_k(i)} (f(x_j) - f(x_i)) / (||x_j - x_i||^2) * (x_j - x_i)
    # Downhill direction pulls toward lower fitness and pushes away from higher fitness
    weight_diff = (delta_f)[:, :, np.newaxis] / dist_sq
    
    # Clip weights to prevent extreme gradient explosions
    weight_diff = np.clip(weight_diff, -1e4, 1e4)
    
    # Directional sum
    raw_grad = np.sum(weight_diff * displacement, axis=1)  # shape (N, D)
    
    # Scale by mean neighbor distance to maintain scale-invariance
    mean_dist = np.mean(neighbor_distances, axis=1, keepdims=True)
    grad_norm = np.linalg.norm(raw_grad, axis=1, keepdims=True) + eps
    topo_grad = (raw_grad / grad_norm) * mean_dist
    
    return topo_grad


def adaptive_boundary_rebound(
    positions: np.ndarray,
    velocities: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    pbest_positions: np.ndarray,
    restitution: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle domain boundaries without sticky clamping.
    Uses stochastic interior rebound toward historical personal best
    and damped velocity reflection.

    Parameters
    ----------
    positions : np.ndarray, shape (N, D)
    velocities : np.ndarray, shape (N, D)
    lb, ub : np.ndarray, shape (D,)
    pbest_positions : np.ndarray, shape (N, D)
    restitution : float

    Returns
    -------
    positions, velocities : updated arrays
    """
    N, D = positions.shape
    lb_mat = np.tile(lb, (N, 1))
    ub_mat = np.tile(ub, (N, 1))
    
    lower_violation = positions < lb_mat
    upper_violation = positions > ub_mat
    
    if np.any(lower_violation):
        r = np.random.uniform(0.0, 0.5, size=positions.shape)
        # Rebound inside toward personal best
        positions[lower_violation] = lb_mat[lower_violation] + r[lower_violation] * (
            np.maximum(pbest_positions[lower_violation], lb_mat[lower_violation]) - lb_mat[lower_violation] + 1e-4
        )
        velocities[lower_violation] = -restitution * velocities[lower_violation]
        
    if np.any(upper_violation):
        r = np.random.uniform(0.0, 0.5, size=positions.shape)
        positions[upper_violation] = ub_mat[upper_violation] - r[upper_violation] * (
            ub_mat[upper_violation] - np.minimum(pbest_positions[upper_violation], ub_mat[upper_violation]) + 1e-4
        )
        velocities[upper_violation] = -restitution * velocities[upper_violation]
        
    # Final safety clip
    positions = np.clip(positions, lb_mat, ub_mat)
    return positions, velocities

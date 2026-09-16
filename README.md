# Starling Murmuration Topological Search (SMTS)

[![CI](https://github.com/mabualzait/SMTS/actions/workflows/ci.yml/badge.svg)](https://github.com/mabualzait/SMTS/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MATLAB Version](https://img.shields.io/badge/MATLAB-R2020b%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper](https://img.shields.io/badge/Paper-In%20Preparation-brightgreen)

> **Official Implementation of "Starling Murmuration Topological Search: Scale-Free Information Transfer and Dynamic Neighborhoods for High-Dimensional Optimization"**

---

## Table of Contents
- [Overview & Biological Grounding](#overview--biological-grounding)
- [Key Mathematical Operators](#key-mathematical-operators)
- [Repository Structure](#repository-structure)
- [Quickstart: Python](#quickstart-python)
- [Quickstart: MATLAB](#quickstart-matlab)
- [Benchmark Results: Continuous Optimization](#benchmark-results-continuous-optimization)
- [Benchmark Results: Machine Learning Classification](#benchmark-results-machine-learning-classification)
  - [Classification Performance](#classification-performance)
  - [Confusion Matrix](#confusion-matrix)
- [Automated Testing & GitHub Actions CI](#automated-testing--github-actions-ci)
- [Reproducibility & Execution Commands](#reproducibility--execution-commands)
- [Citation](#citation)
- [License](#license)

---

## Overview & Biological Grounding

Traditional swarm intelligence metaheuristics (such as Particle Swarm Optimization [PSO] and Grey Wolf Optimizer [GWO]) rely fundamentally upon **metric-distance interaction radii** or **centralized global leaders**. In high-dimensional continuous search spaces ($D \ge 50, 100$), metric-dependent models degrade:
1. **Spatial Dispersion Collapse:** As the swarm expands across high dimensions, metric density decays exponentially ($\rho \sim 1/L^D \to 0$), shattering the communication graph into isolated individuals.
2. **Spatial Aggregation Over-Damping:** When converging around local basins, metric density explodes, degrees saturate ($|N(i)| \to N - 1$), and the swarm freezes prematurely into sub-optimal local minima.

**Starling Murmuration Topological Search (SMTS)** directly incorporates the stereoscopic 3D empirical physics of European starlings (*Sturnus vulgaris*) discovered by Cavagna et al. (2010, *PNAS*) and Attanasi et al. (2014, *Nature Physics*):
- **Topological Interaction Graph ($k = 7$):** Every starling tracks strictly $k \in [6, 7]$ topological nearest neighbors regardless of flock volume or spatial density.
- **Scale-Free Correlation Length ($\xi \propto L$):** Information propagates without an intrinsic microscopic damping length, scaling with the entire diameter of the swarm.
- **Undamped Kinematic Spin Waves:** Momentum adjustments propagate as second-order Hamiltonian waves with constant phase velocity $c_s$ rather than dissipating diffusively.
- **Topological Simplex Gradient:** Downhill directional derivatives are estimated directly from evaluated topological neighbors at **zero extra function evaluation overhead**.

---

## Key Mathematical Operators

### 1. Dynamic Topological Interaction Graph
Each agent $i$ tracks exactly its $k$ nearest Euclidean neighbors:
$$N_k(i, t) = \arg\min_{\substack{S \subset V \setminus \{i\} \\ |S| = k}} \sum_{j \in S} \|\mathbf{x}_j(t) - \mathbf{x}_i(t)\|_2$$
Constructed in $O(N \log N)$ time per generation using spatial kd-trees (`scipy.spatial.cKDTree`).

### 2. Kinematic Spin-Wave Velocity Consensus
Velocity alignment consensus propagates across the topological graph:
$$\mathbf{a}_i^{\text{spin}}(t) = \frac{1}{k} \sum_{j \in N_k(i, t)} (\mathbf{v}_j(t) - \mathbf{v}_i(t))$$

### 3. Gradient-Free Topological Simplex Descent
Downhill directional derivatives are computed using already evaluated neighbor fitness:
$$\nabla_{\text{topo}} f(\mathbf{x}_i(t)) = -\sum_{j \in N_k(i, t)} \frac{f(\mathbf{x}_j(t)) - f(\mathbf{x}_i(t))}{\|\mathbf{x}_j(t) - \mathbf{x}_i(t)\|_2^2 + \epsilon} (\mathbf{x}_j(t) - \mathbf{x}_i(t))$$
Normalized by the local neighborhood span:
$$\mathbf{g}_i(t) = \bar{d}_i(t) \cdot \frac{\nabla_{\text{topo}} f(\mathbf{x}_i(t))}{\|\nabla_{\text{topo}} f(\mathbf{x}_i(t))\|_2 + \epsilon}, \quad \bar{d}_i(t) = \frac{1}{k} \sum_{j \in N_k(i, t)} \|\mathbf{x}_j(t) - \mathbf{x}_i(t)\|_2$$

### 4. Multi-Basin Topological Elite Attraction & Phase-Shift
To foster parallel multi-basin exploration, each individual is guided by its local closed neighborhood elite:
$$\mathbf{x}_{\text{topo\_best}, i}(t) = \arg\min_{j \in N_k(i, t) \cup \{i\}} f(\mathbf{x}_j(t))$$
Coupled with scale-free Cauchy thermal fluctuations naturally scaled by the local neighborhood radius $\bar{d}_i(t)$:
$$\mathbf{v}_i(t+1) = w(\tau) \mathbf{v}_i(t) + c_1(\tau) \mathbf{r}_1 \odot \mathbf{a}_i^{\text{spin}} + c_2(\tau) \mathbf{r}_2 \odot \mathbf{g}_i + c_3(\tau) \mathbf{r}_3 \odot (\mathbf{p}_{\text{topo\_best}, i} - \mathbf{x}_i) + c_4(\tau) \mathbf{r}_4 \odot (\mathbf{g}^* - \mathbf{x}_i) + \boldsymbol{\xi}_i(\tau)$$

---

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI (Python 3.9-3.12 + Octave/MATLAB)
├── matlab/
│   ├── SMTS.m                     # Vectorized MATLAB implementation
│   ├── benchmark_functions.m      # MATLAB benchmark suite (Sphere, Rastrigin, etc.)
│   ├── test_smts.m                # Automated MATLAB test script
│   └── run_benchmarks.m           # Comparative SMTS vs PSO benchmark script
├── plots/
│   ├── confusion_matrix.png       # Classification benchmark confusion matrix
│   ├── convergence_grid_D30.png   # Convergence curves across landscape classes (30D)
│   ├── convergence_grid_D50.png   # Convergence curves across landscape classes (50D)
│   ├── convergence_grid_D100.png  # Convergence curves across landscape classes (100D)
│   ├── friedman_rank_summary.png  # Friedman ranking bar chart (D=30, 50, 100)
│   ├── ablation_k_degree.png      # Ablation curve validating biological degree k in [6, 7]
│   └── diversity_maintenance.png  # Spatial diversity profile vs metric-radius models
├── results/                       # JSON and LaTeX result tables across 5,400+ runs
├── src/
│   ├── smts/
│   │   ├── optimizer.py           # SMTSOptimizer class
│   │   └── operators.py           # Core topological graph & spin-wave operators
│   ├── baselines/                 # Competitor implementations (L-SHADE, CMA-ES, CLPSO, PSO, GWO)
│   ├── benchmarks/                # CEC-style 10-function continuous suite (D=30, 50, 100)
│   └── experiments/
│       ├── run_benchmarks.py      # Multi-core parallel benchmark execution engine
│       ├── statistical_analysis.py# Wilcoxon, Friedman, and Nemenyi statistical tests
│       ├── ablation_study.py      # Topological degree ablation and metric comparison
│       ├── ml_benchmark.py        # Machine learning classification benchmark
│       └── plot_summary.py        # Publication figure generator
├── tests/
│   ├── test_smts.py               # Unit tests for SMTS optimizer
│   ├── test_baselines.py          # Unit tests for baselines and CEC benchmarks
│   └── test_ml_benchmark.py       # Unit tests for ML classification pipeline
├── requirements.txt               # Python package dependencies
├── setup.py                       # Package installation script
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

---

## Quickstart: Python

### Installation
```bash
git clone https://github.com/mabualzait/SMTS.git
cd SMTS
pip install -r requirements.txt
pip install -e .
```

### Basic Optimization Example
```python
import numpy as np
from smts import SMTSOptimizer

# Define Rastrigin function
def rastrigin(x):
    return 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

# Instantiate and run SMTS
optimizer = SMTSOptimizer(
    objective_func=rastrigin,
    dim=30,
    bounds=(-5.12, 5.12),
    pop_size=50,
    max_evals=30000,
    k_neighbors=7,  # Biologically optimal starling neighborhood
    seed=42
)

result = optimizer.optimize()
print(f"Optimal Value Found: {result['best_f']:.6e}")
print(f"Optimal Coordinates: {result['best_x'][:5]}... (first 5 dims)")
```

---

## Quickstart: MATLAB

### Basic MATLAB Execution
```matlab
% Add matlab folder to path
addpath('matlab');

% Define 30D Rosenbrock function
rosenbrock = @(x) sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2);

dim = 30;
bounds = [-30, 30];
pop_size = 50;
max_evals = 30000;
k_neighbors = 7;

% Run SMTS
[best_x, best_f, convergence, diversity] = SMTS(rosenbrock, dim, bounds, pop_size, max_evals, k_neighbors);

fprintf('Optimal Cost: %e\n', best_f);
```

### Run Automated MATLAB Verification Tests
In MATLAB or GNU Octave:
```matlab
run('matlab/test_smts.m');
```

---

## Benchmark Results: Continuous Optimization

Evaluated across 10 shifted and rotated continuous CEC-style functions (Unimodal, Multimodal, Hybrid, Composition) in dimensions $D \in \{30, 50, 100\}$ over **30 independent Monte Carlo trials per function** (5,400 total runs).

### Average Friedman Rankings (Lower is Better)

| Algorithm | Rank ($D = 30$) | Rank ($D = 50$) | Rank ($D = 100$) |
|:---|:---:|:---:|:---:|
| **L-SHADE** (CEC Winner) | 1.60 | 1.50 | 1.60 |
| **CMA-ES** (Gold Standard) | 1.90 | 2.20 | 1.70 |
| **SMTS (Proposed)** | **2.80** | **2.80** | **3.10** |
| **Standard PSO** | 4.40 | 4.90 | 4.60 |
| **GWO** | 4.70 | 4.50 | 4.80 |
| **CLPSO** | 5.60 | 5.10 | 5.20 |

### Pairwise Wilcoxon Signed-Rank Tests ($\alpha = 0.05$)
Cumulative head-to-head performance of SMTS against all competitors across all 90 test configurations ($D=30, 50, 100$):

| Comparison | Wins (+) | Ties (=) | Losses (-) | Win Rate |
|---|:---:|:---:|:---:|:---:|
| **SMTS vs. CLPSO** | **27** | 3 | 0 | **90.0%** (100% non-loss) |
| **SMTS vs. Standard PSO** | **25** | 3 | 2 | **83.3%** |
| **SMTS vs. GWO** | **24** | 6 | 0 | **80.0%** (100% non-loss) |
| **SMTS vs. Swarm Baselines (Total)** | **76** | **12** | **2** | **84.4%** |
| **SMTS vs. CMA-ES** | 6 | 4 | 20 | Outperformed on F2, F9 |

---

## Benchmark Results: Machine Learning Classification

To demonstrate applied continuous optimization, SMTS was benchmarked on diagnostic classification using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset (569 samples, 30 continuous clinical features, 80/20 stratified train/test split). SMTS optimized a 31-dimensional continuous regularized parameter vector $[\mathbf{w}^T, b]^T \in [-5, 5]^{31}$.

### Classification Performance

| Optimization Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SMTS (Proposed)** | **97.37%** | **0.9859** | **0.9722** | **0.9762** | **0.9790** | **0.9950** |
| **Grey Wolf Optimizer (GWO)** | 97.37% | 0.9859 | 0.9722 | 0.9762 | 0.9790 | 0.9954 |
| **Standard PSO** | 94.74% | 0.9714 | 0.9444 | 0.9524 | 0.9577 | 0.9931 |

### Confusion Matrix

Evaluated on the held-out test partition ($N = 114$ patients: 42 Malignant, 72 Benign):

```
                     Predicted Malignant (0)    Predicted Benign (1)
Actual Malignant (0)           41 (TN)                    1 (FP)
Actual Benign (1)               2 (FN)                   70 (TP)
```

| Metric Parameter | Value | Interpretation |
|---|:---:|---|
| **True Negatives (TN)** | **41** | Correctly identified malignant cases |
| **False Positives (FP)** | **1** | Benign patient incorrectly flagged |
| **False Negatives (FN)** | **2** | Malignant case missed |
| **True Positives (TP)** | **70** | Correctly identified benign cases |
| **Accuracy** | **97.37%** | 111 out of 114 correct clinical predictions |
| **Specificity** | **97.62%** | High true negative rate (crucial for oncology) |

The generated confusion matrix heatmap is saved under [`plots/confusion_matrix.png`](plots/confusion_matrix.png).

---

## Automated Testing & GitHub Actions CI

This repository contains full automated continuous integration via [GitHub Actions](.github/workflows/ci.yml):
- **Python Matrix Testing:** Automatically executes on Ubuntu with Python **3.9, 3.10, 3.11, and 3.12**.
- **Pytest Suite:** Tests all core SMTS operators, baselines, and benchmark functions.
- **ML Pipeline Verification:** Runs the diagnostic classification benchmark and verifies accuracy metrics.
- **GNU Octave / MATLAB Compatibility:** Validates MATLAB `.m` scripts in headless mode on every pull request and push.

Run all tests locally:
```bash
pytest tests/ -v
```

---

## Reproducibility & Execution Commands

Replicate all experimental tables, figures, and benchmarks using the provided scripts:

```bash
# 1. Run all unit tests
pytest tests/ -v

# 2. Run the full CEC Benchmark suite (30 runs across D=30, 50, 100)
python src/experiments/run_benchmarks.py --dims 30 50 100 --runs 30 --workers 10

# 3. Run the topological degree k ablation study
python src/experiments/ablation_study.py --dim 50 --runs 20 --workers 10

# 4. Run the machine learning classification benchmark
python src/experiments/ml_benchmark.py

# 5. Generate all publication figures (convergence curves, Friedman ranks)
python src/experiments/plot_summary.py

# 6. Run MATLAB/Octave tests
octave --no-gui --eval "addpath('matlab'); run('matlab/test_smts.m');"
```

---

## Citation

If you use SMTS in your research, please cite our paper:

```bibtex
@article{abualzait2026smts,
  title={Starling Murmuration Topological Search: Scale-Free Information Transfer and Dynamic Neighborhoods for High-Dimensional Optimization},
  author={Abu Alzait, Malik and Antigravity AI Research Team},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2026},
  note={Under Review. Code available at: https://github.com/mabualzait/SMTS}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

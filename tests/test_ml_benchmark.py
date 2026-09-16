"""
Unit test for ML diagnostic classification benchmark.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from experiments.ml_benchmark import create_loss_function, evaluate_classifier
from smts import SMTSOptimizer


def test_ml_optimization_smoke():
    # Synthetic dataset for fast test
    np.random.seed(42)
    N, D = 60, 5
    X = np.random.randn(N, D)
    w_true = np.random.randn(D)
    y = (np.dot(X, w_true) > 0).astype(int)

    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    loss_fn = create_loss_function(X_train, y_train, l2_reg=1e-4)

    opt = SMTSOptimizer(
        objective_func=loss_fn,
        dim=D + 1,
        bounds=(-3.0, 3.0),
        pop_size=20,
        max_evals=1000,
        seed=42,
    )
    res = opt.optimize()

    metrics = evaluate_classifier(res["best_x"], X_test, y_test)
    print(f"Smoke Test ML Accuracy: {metrics['accuracy'] * 100:.2f}%")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == len(y_test)
    print("test_ml_optimization_smoke passed.")


if __name__ == "__main__":
    test_ml_optimization_smoke()

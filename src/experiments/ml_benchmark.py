"""
Machine Learning Diagnostic Classification Benchmark for Metaheuristics.
Optimizes a high-dimensional regularized classifier on the Wisconsin Diagnostic
Breast Cancer (WDBC) benchmark dataset.

Computes:
1. Classification Accuracy
2. Precision, Recall (Sensitivity), Specificity, F1-Score
3. Complete Confusion Matrix (TP, TN, FP, FN)
4. Comparative performance vs Standard PSO and GWO
5. Generates publication-ready Confusion Matrix heatmaps.
"""

import sys
import os
import json
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_smts"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smts import SMTSOptimizer
from baselines import StandardPSOOptimizer, GWOOptimizer


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def create_loss_function(X_train: np.ndarray, y_train: np.ndarray, l2_reg: float = 1e-4):
    """
    Returns objective function f(theta) where theta = [w (D), b (1)].
    """
    N_samples, n_features = X_train.shape

    def objective(theta: np.ndarray) -> float:
        w = theta[:n_features]
        b = theta[n_features]
        logits = np.dot(X_train, w) + b
        p = sigmoid(logits)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        # Binary cross entropy
        bce = -np.mean(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))
        reg = l2_reg * np.sum(w ** 2)
        return float(bce + reg)

    return objective


def evaluate_classifier(theta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    n_features = X_test.shape[1]
    w = theta[:n_features]
    b = theta[n_features]
    probs = sigmoid(np.dot(X_test, w) + b)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "matrix": cm.tolist(),
        },
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Malignant (0)", "Benign (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")


def main():
    print("=" * 60)
    print("Machine Learning Benchmark: Diagnostic Classification (WDBC)")
    print("=" * 60)

    # 1. Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features (Classes: 0=Malignant, 1=Benign)")

    # 2. Train/Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Problem dimension: 30 weights + 1 bias = 31
    dim = X_train.shape[1] + 1
    loss_func = create_loss_function(X_train, y_train, l2_reg=1e-4)

    # 3. Optimize with SMTS, PSO, GWO
    pop_size = 40
    max_evals = 15000
    bounds = (-5.0, 5.0)

    results = {}

    optimizers = [
        ("SMTS", lambda: SMTSOptimizer(loss_func, dim=dim, bounds=bounds, pop_size=pop_size, max_evals=max_evals, seed=42)),
        ("PSO", lambda: StandardPSOOptimizer(loss_func, dim=dim, bounds=bounds, pop_size=pop_size, max_evals=max_evals, seed=42)),
        ("GWO", lambda: GWOOptimizer(loss_func, dim=dim, bounds=bounds, pop_size=pop_size, max_evals=max_evals, seed=42)),
    ]

    for name, opt_fn in optimizers:
        print(f"\nRunning {name}...")
        opt = opt_fn()
        res = opt.optimize()
        theta_best = res["best_x"]
        metrics = evaluate_classifier(theta_best, X_test, y_test)
        metrics["train_loss"] = float(res["best_f"])
        results[name] = metrics

        print(f"[{name}] Test Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"[{name}] Precision:        {metrics['precision']:.4f}")
        print(f"[{name}] Recall (Sens.):   {metrics['recall']:.4f}")
        print(f"[{name}] Specificity:      {metrics['specificity']:.4f}")
        print(f"[{name}] F1-Score:         {metrics['f1_score']:.4f}")
        print(f"[{name}] ROC-AUC:          {metrics['roc_auc']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"[{name}] Confusion Matrix: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    # 4. Save results and plots
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plots"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "classification_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ML benchmark metrics to {json_path}")

    # Plot SMTS Confusion Matrix
    cm_smts = np.array(results["SMTS"]["confusion_matrix"]["matrix"])
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm_smts,
        title=f"SMTS Confusion Matrix (Accuracy: {results['SMTS']['accuracy']*100:.2f}%)",
        save_path=cm_plot_path,
    )


if __name__ == "__main__":
    main()

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    N_FEATURES,
    SCALER_MIN,
    SCALER_MAX,
)


def load_everything():
    print("=" * 60)
    print("Loading trained model and data")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    weights = np.load(os.path.join(MODELS_DIR, "vqc_weights.npy"))
    with open(os.path.join(MODELS_DIR, "vqc_history.pkl"), "rb") as f:
        history = pickle.load(f)

    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_machine_learning.optimizers import COBYLA

    feature_map = ZZFeatureMap(
        feature_dimension=history["n_qubits"],
        reps=history["feature_map_reps"],
    )
    ansatz = RealAmplitudes(
        num_qubits=history["n_qubits"],
        reps=history["ansatz_reps"],
    )
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=1),
        sampler=StatevectorSampler(seed=RANDOM_SEED),
    )

    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    train_range = train_max - train_min
    train_range[train_range == 0] = 1.0
    X_dummy = (SCALER_MAX - SCALER_MIN) * (X_train[:2] - train_min) / train_range + SCALER_MIN
    vqc.fit(X_dummy, np.array([0, 1]))
    vqc._fit_result.x = weights

    print("  All loaded successfully")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "vqc": vqc,
        "history": history,
    }


def normalize_features(X_train, X_test):
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    train_range = train_max - train_min
    train_range[train_range == 0] = 1.0

    X_train_norm = (SCALER_MAX - SCALER_MIN) * (X_train - train_min) / train_range + SCALER_MIN
    X_test_norm = (SCALER_MAX - SCALER_MIN) * (X_test - train_min) / train_range + SCALER_MIN
    X_test_norm = np.clip(X_test_norm, SCALER_MIN, SCALER_MAX)

    return X_train_norm, X_test_norm


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None

    print(f"\n  Metrics:")
    for key, val in metrics.items():
        if val is not None:
            print(f"    {key:12s}: {val:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Examples"}, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix - Hybrid Quantum LLM", fontsize=13)

    save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Saved: {save_path}")


def plot_metrics_bar(metrics):
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        metric_names.append("roc_auc")

    values = [metrics[m] for m in metric_names]
    labels = [m.replace("_", " ").title() for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#1f77b4", alpha=0.85, edgecolor="black")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Hybrid Quantum LLM - Test Metrics", fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
               label="Random baseline (50%)")
    ax.legend(loc="upper right")

    save_path = os.path.join(FIGURES_DIR, "metrics_bar.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Saved: {save_path}")


def plot_decision_boundary_2d(vqc, X_train, X_test, y_train, y_test):
    if N_FEATURES != 2:
        return

    print("  Generating decision boundary plot (2D feature space)...")

    margin = 0.3
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - margin
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + margin
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - margin
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + margin

    grid_size = 30
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_predictions = vqc.predict(grid_points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(xx, yy, grid_predictions, alpha=0.3, cmap="RdYlBu")

    train_pos = X_train[y_train == 1]
    train_neg = X_train[y_train == 0]
    test_pos = X_test[y_test == 1]
    test_neg = X_test[y_test == 0]

    ax.scatter(train_neg[:, 0], train_neg[:, 1], c="blue", marker="o",
               s=40, alpha=0.6, label="Train: Negative", edgecolors="navy")
    ax.scatter(train_pos[:, 0], train_pos[:, 1], c="red", marker="o",
               s=40, alpha=0.6, label="Train: Positive", edgecolors="darkred")
    ax.scatter(test_neg[:, 0], test_neg[:, 1], c="blue", marker="^",
               s=60, alpha=0.9, label="Test: Negative", edgecolors="black", linewidths=1.5)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], c="red", marker="^",
               s=60, alpha=0.9, label="Test: Positive", edgecolors="black", linewidths=1.5)

    ax.set_xlabel("Feature 1 (LDA component, normalized to [-pi, pi])", fontsize=11)
    ax.set_ylabel("Feature 2 (PCA component, normalized to [-pi, pi])", fontsize=11)
    ax.set_title("VQC Decision Boundary in 2D Feature Space", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "decision_boundary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Saved: {save_path}")


def evaluate_quantum(vqc, X_test_norm, y_test):
    print("\n" + "=" * 60)
    print("Evaluating Hybrid Quantum LLM Model")
    print("=" * 60)

    y_pred = vqc.predict(X_test_norm)
    y_pred = np.asarray(y_pred).astype(int)

    try:
        y_proba = vqc.predict_proba(X_test_norm)[:, 1]
    except AttributeError:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_metrics_bar(metrics)

    return metrics


def main():
    data = load_everything()
    X_train_norm, X_test_norm = normalize_features(data["X_train"], data["X_test"])

    metrics = evaluate_quantum(data["vqc"], X_test_norm, data["y_test"])

    print("\n" + "=" * 60)
    print("Generating decision boundary visualization")
    print("=" * 60)
    plot_decision_boundary_2d(
        data["vqc"], X_train_norm, X_test_norm,
        data["y_train"], data["y_test"]
    )

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "hybrid_quantum_llm": metrics,
            "config": {
                "n_qubits": data["history"]["n_qubits"],
                "feature_map_reps": data["history"]["feature_map_reps"],
                "ansatz_reps": data["history"]["ansatz_reps"],
                "max_iter": data["history"]["max_iter"],
            },
            "training": {
                "train_accuracy": data["history"]["train_score"],
                "test_accuracy": data["history"]["test_score"],
                "final_loss": data["history"]["objective_values"][-1] if data["history"]["objective_values"] else None,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

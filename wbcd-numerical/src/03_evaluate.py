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
)


def load_everything():
    print("=" * 60)
    print("Loading trained model and data")
    print("=" * 60)

    X_train_scaled = np.load(os.path.join(MODELS_DIR, "X_train_used.npy"))
    X_test_scaled = np.load(os.path.join(MODELS_DIR, "X_test_used.npy"))
    y_train = np.load(os.path.join(MODELS_DIR, "y_train_used.npy"))
    y_test = np.load(os.path.join(MODELS_DIR, "y_test_used.npy"))

    with open(os.path.join(DATA_DIR, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)

    weights = np.load(os.path.join(MODELS_DIR, "vqc_weights.npy"))
    with open(os.path.join(MODELS_DIR, "vqc_history.pkl"), "rb") as f:
        history = pickle.load(f)

    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
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
    )

    vqc.fit(X_train_scaled[:2], np.array([0, 1]))
    vqc._fit_result.x = weights

    print("  All loaded successfully")

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "pca": pca,
        "vqc": vqc,
        "history": history,
    }


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
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        cbar_kws={"label": "Examples"}, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix - Hybrid Quantum Classifier (WBCD)", fontsize=13)

    save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Saved: {save_path}")


def plot_pca_explained_variance(pca):
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    components = range(1, len(explained) + 1)
    ax1.bar(components, explained * 100, alpha=0.7, color="steelblue",
            label="Individual variance")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained variance (%)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative * 100, "ro-", linewidth=2,
             markersize=8, label="Cumulative")
    ax2.set_ylabel("Cumulative variance (%)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim([0, 105])

    plt.title(f"PCA Explained Variance ({len(explained)} components, WBCD)", fontsize=13)
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "pca_explained_variance.png")
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
    ax.set_title("Hybrid Quantum Classifier - Test Metrics (WBCD)", fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
               label="Random baseline (50%)")
    ax.legend(loc="lower right")

    save_path = os.path.join(FIGURES_DIR, "metrics_bar.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Saved: {save_path}")


def evaluate_quantum(vqc, X_test, y_test):
    print("\n" + "=" * 60)
    print("Evaluating Hybrid Quantum Classifier")
    print("=" * 60)

    y_pred = vqc.predict(X_test)
    y_pred = np.asarray(y_pred).astype(int)

    try:
        y_proba = vqc.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_metrics_bar(metrics)

    return metrics


def main():
    data = load_everything()

    print("\n" + "=" * 60)
    print("Generating PCA explained variance plot")
    print("=" * 60)
    plot_pca_explained_variance(data["pca"])

    metrics = evaluate_quantum(data["vqc"], data["X_test"], data["y_test"])

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "hybrid_quantum_classifier_wbcd": metrics,
            "config": {
                "dataset": "Wisconsin Breast Cancer Diagnostic",
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

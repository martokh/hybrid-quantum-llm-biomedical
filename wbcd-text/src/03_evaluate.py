"""Stage 3 — rebuild the VQC from saved weights and produce evaluation artefacts.

Generates:
    results/figures/confusion_matrix.png
    results/figures/metrics_bar.png
    results/figures/pca_explained_variance.png
    results/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


CLASS_NAMES = ("malignant", "benign")  # sklearn convention: 0, 1


def rebuild_vqc() -> VQC:
    """Reconstruct an empty VQC with the same architecture used for training."""
    feature_map = ZZFeatureMap(
        feature_dimension=C.N_QUBITS, reps=C.FEATURE_MAP_REPS, entanglement="linear"
    )
    ansatz = RealAmplitudes(
        num_qubits=C.N_QUBITS, reps=C.ANSATZ_REPS, entanglement="linear"
    )
    optimizer = COBYLA(maxiter=1)  # never used at eval time
    return VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)


def apply_saved_scaler(X: np.ndarray, scaler_path: Path) -> np.ndarray:
    """Apply the MinMaxScaler that was fit on the training set."""
    p = np.load(scaler_path)
    data_min = p["data_min"]
    data_max = p["data_max"]
    target_min, target_max = 0.0, np.pi
    span = np.where(data_max > data_min, data_max - data_min, 1.0)
    return (X - data_min) / span * (target_max - target_min) + target_min


def warm_up_vqc(vqc: VQC, X_train_s: np.ndarray, y_train: np.ndarray) -> None:
    """VQC initialises its internal weight vector lazily on fit/predict.

    To inject our trained weights we trigger that initialisation by calling
    fit with a 1-iter optimizer, then overwrite the weights in place.
    """
    vqc.fit(X_train_s, y_train)


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics: dict, out_path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(keys, values, color="#4c72b0")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test-set performance — BioBERT + Quantum VQC")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_variance(out_path: Path) -> None:
    p = np.load(C.PCA_FILE)
    var = p["explained_variance_ratio"]
    cum = np.cumsum(var)
    idx = np.arange(1, len(var) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(idx, var, color="#55a868", alpha=0.85, label="per component")
    ax.plot(idx, cum, marker="o", color="#c44e52", label="cumulative")
    ax.set_xticks(idx)
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA on BioBERT [CLS] embeddings (768 -> 4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("Stage 3 / 3  -  Evaluate trained VQC")
    print("=" * 72)

    algorithm_globals.random_seed = C.RANDOM_SEED
    np.random.seed(C.RANDOM_SEED)

    # 1. Load everything
    print("\n[1/4] Loading data, scaler, and trained weights ...")
    X_train = np.load(C.X_TRAIN_FILE)
    X_test = np.load(C.X_TEST_FILE)
    y_train = np.load(C.Y_TRAIN_FILE)
    y_test = np.load(C.Y_TEST_FILE)
    weights = np.load(C.WEIGHTS_FILE)
    print(f"      X_test: {X_test.shape}  weights: {weights.shape}")

    X_train_s = apply_saved_scaler(X_train, C.SCALER_FILE)
    X_test_s = apply_saved_scaler(X_test, C.SCALER_FILE)

    # 2. Rebuild VQC and inject trained weights
    print("\n[2/4] Rebuilding VQC and loading trained weights ...")
    vqc = rebuild_vqc()
    warm_up_vqc(vqc, X_train_s, y_train)
    if vqc.weights.shape != weights.shape:
        raise RuntimeError(
            f"weight shape mismatch: rebuilt {vqc.weights.shape} vs "
            f"saved {weights.shape}"
        )
    # Overwrite the lazily-trained weights with the proper, COBYLA-fitted ones.
    vqc._fit_result.x = weights
    print("      weights injected successfully")

    # 3. Predict and compute metrics
    print("\n[3/4] Computing metrics on test set ...")
    y_pred = vqc.predict(X_test_s)
    try:
        proba = vqc.predict_proba(X_test_s)
        y_score = proba[:, 1]
    except Exception:
        y_score = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_qubits": C.N_QUBITS,
        "feature_map_reps": C.FEATURE_MAP_REPS,
        "ansatz_reps": C.ANSATZ_REPS,
        "max_iter": C.MAX_ITER,
        "biobert_model": C.BIOBERT_MODEL,
    }
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"      {k:<10s}: {metrics[k]:.4f}")

    C.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"      metrics.json -> {C.METRICS_FILE}")

    # 4. Figures
    print("\n[4/4] Generating figures ...")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion(cm, C.FIGURES_DIR / "confusion_matrix.png")
    plot_metrics_bar(metrics, C.FIGURES_DIR / "metrics_bar.png")
    plot_pca_variance(C.FIGURES_DIR / "pca_explained_variance.png")
    print(f"      figures -> {C.FIGURES_DIR}")

    print("\nDone. Compare this to the 67% baseline (without LLM).")


if __name__ == "__main__":
    main()

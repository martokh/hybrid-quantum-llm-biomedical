import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def rebuild_vqc_with_weights(weights: np.ndarray) -> VQC:
    feature_map = ZZFeatureMap(
        feature_dimension=config.N_QUBITS,
        reps=config.FEATURE_MAP_REPS,
    )
    ansatz = RealAmplitudes(num_qubits=config.N_QUBITS, reps=config.ANSATZ_REPS)
    sampler = StatevectorSampler(seed=config.RANDOM_SEED)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=1),
        sampler=sampler,
    )
    n_params = ansatz.num_parameters
    if weights.shape[0] != n_params:
        raise RuntimeError(
            f"Saved weights have shape {weights.shape}, expected ({n_params},)"
        )
    return vqc, weights


def predict_with_weights(vqc: VQC, weights: np.ndarray, X: np.ndarray):
    nn = vqc.neural_network
    raw = nn.forward(X, weights)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        proba_pos = raw
        proba = np.stack([1.0 - proba_pos, proba_pos], axis=1)
    else:
        if raw.shape[1] == 1:
            proba_pos = raw[:, 0]
            proba = np.stack([1.0 - proba_pos, proba_pos], axis=1)
        else:
            proba = raw / raw.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1).astype(np.int64)
    return y_pred, proba


def plot_confusion(cm: np.ndarray, target_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[f"not {target_label}", target_label],
        yticklabels=[f"not {target_label}", target_label],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics: dict, out_path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(keys, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(out_path: Path) -> None:
    pca = joblib.load(config.PCA_PATH)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = np.arange(1, len(evr) + 1)
    ax.bar(idx, evr, color="#1f77b4", label="per component")
    ax.plot(idx, cum, "-o", color="#d62728", label="cumulative")
    ax.set_xticks(idx)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    algorithm_globals.random_seed = config.RANDOM_SEED
    np.random.seed(config.RANDOM_SEED)

    print("[1/4] Loading arrays and trained weights ...")
    X_train = np.load(config.X_TRAIN_PATH)
    X_test = np.load(config.X_TEST_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)
    y_test = np.load(config.Y_TEST_PATH)
    weights = np.load(config.VQC_WEIGHTS_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    target_label = config.TARGET_LABEL_PATH.read_text(encoding="utf-8").strip()
    print(f"      weights shape: {weights.shape}")
    print(f"      target label : '{target_label}'")

    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_test_s = np.clip(X_test_s, config.SCALER_MIN, config.SCALER_MAX)

    print("[2/4] Rebuilding VQC and running inference ...")
    vqc, weights = rebuild_vqc_with_weights(weights)
    y_pred, proba = predict_with_weights(vqc, weights, X_test_s)
    train_pred, _ = predict_with_weights(vqc, weights, X_train_s)

    print("[3/4] Computing metrics ...")
    train_acc = float(accuracy_score(y_train, train_pred))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        roc = float(roc_auc_score(y_test, proba[:, 1]))
    except ValueError:
        roc = float("nan")

    metrics = {
        "target_label": target_label,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_accuracy": train_acc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
    }
    print(json.dumps(metrics, indent=2))

    print("[4/4] Writing figures and metrics.json ...")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion(cm, target_label, config.FIGURES_DIR / "confusion_matrix.png")
    plot_metrics_bar(metrics, config.FIGURES_DIR / "metrics_bar.png")
    plot_pca_variance(config.FIGURES_DIR / "pca_explained_variance.png")

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"      wrote: {config.METRICS_PATH}")
    print(f"      wrote: {config.FIGURES_DIR / 'confusion_matrix.png'}")
    print(f"      wrote: {config.FIGURES_DIR / 'metrics_bar.png'}")
    print(f"      wrote: {config.FIGURES_DIR / 'pca_explained_variance.png'}")
    print("Done.")


if __name__ == "__main__":
    main()

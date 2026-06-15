"""Stage 2 — train the Variational Quantum Classifier on BioBERT/PCA features.

Architecture:
    MinMaxScaler -> [0, pi]
    ZZFeatureMap(4 qubits, reps=2)
    RealAmplitudes(4 qubits, reps=2)
    COBYLA optimizer, maxiter=150

Persistence note:
    We avoid pickling the VQC object (it's brittle across qiskit versions).
    We save only the trained `weights` as a .npy file plus the scaler params,
    and rebuild the VQC from config.py during evaluation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# qiskit / qiskit-machine-learning
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


# --------------------------------------------------------------------------
# Loss-callback helper (closes over a list it can append to)
# --------------------------------------------------------------------------
def make_callback(history: list[tuple[int, float]], log_every: int = 5):
    state = {"i": 0}

    def cb(weights, obj_val):
        state["i"] += 1
        history.append((state["i"], float(obj_val)))
        if state["i"] % log_every == 0 or state["i"] == 1:
            print(f"      iter {state['i']:>3d}  loss = {obj_val:.6f}")

    return cb


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Stage 2 / 3  -  Train Variational Quantum Classifier")
    print("=" * 72)

    algorithm_globals.random_seed = C.RANDOM_SEED
    np.random.seed(C.RANDOM_SEED)

    # 1. Load PCA-reduced embeddings
    print("\n[1/5] Loading PCA-reduced BioBERT embeddings ...")
    X_train = np.load(C.X_TRAIN_FILE)
    X_test = np.load(C.X_TEST_FILE)
    y_train = np.load(C.Y_TRAIN_FILE)
    y_test = np.load(C.Y_TEST_FILE)
    print(f"      X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"      X_test : {X_test.shape}   y_test : {y_test.shape}")

    # 2. Scale to [0, pi] for the angle-encoded feature map
    print("\n[2/5] MinMax-scaling features to [0, pi] ...")
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    np.savez(C.SCALER_FILE, data_min=scaler.data_min_, data_max=scaler.data_max_)
    print(f"      scaler params saved to {C.SCALER_FILE}")

    # 3. Build the quantum circuit (feature map + ansatz)
    print(
        f"\n[3/5] Building quantum circuit "
        f"({C.N_QUBITS} qubits, fm_reps={C.FEATURE_MAP_REPS}, "
        f"ansatz_reps={C.ANSATZ_REPS}) ..."
    )
    feature_map = ZZFeatureMap(
        feature_dimension=C.N_QUBITS, reps=C.FEATURE_MAP_REPS, entanglement="linear"
    )
    ansatz = RealAmplitudes(
        num_qubits=C.N_QUBITS, reps=C.ANSATZ_REPS, entanglement="linear"
    )
    print(f"      trainable parameters in ansatz: {ansatz.num_parameters}")

    # Render the full circuit (feature_map then ansatz on the same wires)
    try:
        full = feature_map.compose(ansatz)
        fig = full.decompose().draw(output="mpl", fold=-1)
        fig.savefig(C.FIGURES_DIR / "quantum_circuit.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      diagram saved to {C.FIGURES_DIR / 'quantum_circuit.png'}")
    except Exception as exc:  # drawing is non-critical for training
        print(f"      (could not render circuit diagram: {exc})")

    # 4. Train VQC with COBYLA
    print(f"\n[4/5] Training VQC with COBYLA (maxiter={C.MAX_ITER}) ...")
    history: list[tuple[int, float]] = []
    optimizer = COBYLA(maxiter=C.MAX_ITER, rhobeg=C.COBYLA_RHOBEG)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=make_callback(history, log_every=5),
    )

    t0 = time.time()
    vqc.fit(X_train_s, y_train)
    dt = time.time() - t0
    print(f"      training done in {dt:.1f} s, {len(history)} iterations")

    train_acc = vqc.score(X_train_s, y_train)
    test_acc = vqc.score(X_test_s, y_test)
    print(f"      train accuracy: {train_acc:.4f}")
    print(f"      test  accuracy: {test_acc:.4f}")

    # 5. Persist weights and loss curve (no pickle of the VQC object)
    print("\n[5/5] Saving weights and loss curve ...")
    np.save(C.WEIGHTS_FILE, np.asarray(vqc.weights, dtype=float))
    print(f"      weights -> {C.WEIGHTS_FILE}  shape={vqc.weights.shape}")

    if history:
        it, loss = zip(*history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(it, loss, color="#1f77b4", lw=1.8)
        ax.set_xlabel("COBYLA iteration")
        ax.set_ylabel("Objective value")
        ax.set_title("VQC training loss (BioBERT + WBCD text)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "training_loss.png", dpi=150)
        plt.close(fig)
        print(f"      loss curve -> {C.FIGURES_DIR / 'training_loss.png'}")

    print("\nStage 2 done. Next: python src/03_evaluate.py")


if __name__ == "__main__":
    main()

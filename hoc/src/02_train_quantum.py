import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_arrays():
    X_train = np.load(config.X_TRAIN_PATH)
    X_test = np.load(config.X_TEST_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)
    y_test = np.load(config.Y_TEST_PATH)
    print(f"X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"y_train balance={np.bincount(y_train)}, y_test balance={np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


def scale_to_pi_range(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(config.SCALER_MIN, config.SCALER_MAX))
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_test_s = np.clip(X_test_s, config.SCALER_MIN, config.SCALER_MAX)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"scaler range = [{config.SCALER_MIN:.4f}, {config.SCALER_MAX:.4f}]")
    return X_train_s, X_test_s


def build_circuits():
    feature_map = ZZFeatureMap(
        feature_dimension=config.N_QUBITS,
        reps=config.FEATURE_MAP_REPS,
    )
    ansatz = RealAmplitudes(num_qubits=config.N_QUBITS, reps=config.ANSATZ_REPS)
    print(f"feature_map: ZZFeatureMap(qubits={config.N_QUBITS}, reps={config.FEATURE_MAP_REPS})")
    print(f"ansatz     : RealAmplitudes(qubits={config.N_QUBITS}, reps={config.ANSATZ_REPS})")
    print(f"trainable parameters: {ansatz.num_parameters}")
    return feature_map, ansatz


def draw_circuit(feature_map, ansatz, out_path: Path):
    full = feature_map.compose(ansatz).decompose()
    try:
        fig = full.draw(output="mpl", fold=40)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote: {out_path}")
        return
    except Exception as exc:
        print(f"      mpl circuit drawer failed ({exc!r}); falling back to text dump")

    try:
        text_repr = full.draw(output="text", fold=120)
        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text(str(text_repr), encoding="utf-8")
        print(f"wrote: {txt_path}")
    except Exception as exc:
        print(f"      text drawer also failed ({exc!r}); skipping circuit figure")


def main():
    algorithm_globals.random_seed = config.RANDOM_SEED
    np.random.seed(config.RANDOM_SEED)

    print("[1/5] Loading prepared arrays ...")
    X_train, X_test, y_train, y_test = load_arrays()

    print("[2/5] Scaling features to [0, pi] ...")
    X_train_s, X_test_s = scale_to_pi_range(X_train, X_test)

    print("[3/5] Building quantum circuits ...")
    feature_map, ansatz = build_circuits()
    draw_circuit(feature_map, ansatz, config.FIGURES_DIR / "quantum_circuit.png")

    print("[4/5] Training VQC with COBYLA ...")
    loss_history: list[float] = []

    def cobyla_callback(weights, obj_value):
        loss_history.append(float(obj_value))
        i = len(loss_history)
        if i == 1 or i % config.LOG_EVERY == 0:
            print(f"      iter {i:3d}  loss={obj_value:.5f}")

    optimizer = COBYLA(maxiter=config.MAX_ITER)
    sampler = StatevectorSampler(seed=config.RANDOM_SEED)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        callback=cobyla_callback,
    )
    vqc.fit(X_train_s, y_train)

    train_acc = vqc.score(X_train_s, y_train)
    test_acc = vqc.score(X_test_s, y_test)
    print(f"      train accuracy = {train_acc:.4f}")
    print(f"      test  accuracy = {test_acc:.4f}")

    print("[5/5] Saving weights and loss curve ...")
    weights = np.asarray(vqc.weights, dtype=np.float64)
    np.save(config.VQC_WEIGHTS_PATH, weights)
    np.save(config.LOSS_HISTORY_PATH, np.asarray(loss_history, dtype=np.float64))
    print(f"      wrote: {config.VQC_WEIGHTS_PATH}  (shape={weights.shape})")
    print(f"      wrote: {config.LOSS_HISTORY_PATH}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="#1f77b4")
    ax.set_xlabel("COBYLA iteration")
    ax.set_ylabel("Objective value (cross-entropy loss)")
    ax.set_title("VQC training loss")
    ax.grid(alpha=0.3)
    loss_path = config.FIGURES_DIR / "training_loss.png"
    fig.savefig(loss_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote: {loss_path}")
    print("Done.")


if __name__ == "__main__":
    main()

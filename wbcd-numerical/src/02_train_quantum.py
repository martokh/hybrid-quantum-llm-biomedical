import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    N_QUBITS,
    FEATURE_MAP_REPS,
    ANSATZ_REPS,
    OPTIMIZER_NAME,
    MAX_ITER,
)


def load_data():
    print("=" * 60)
    print("STEP 1/5: Loading data")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Labels: train={int(y_train.sum())}/{len(y_train)} positive, "
          f"test={int(y_test.sum())}/{len(y_test)} positive")

    assert X_train.shape[1] == N_QUBITS

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    print("\n" + "=" * 60)
    print("STEP 2/5: Feature scaling to [0, pi]")
    print("=" * 60)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_test_scaled = np.clip(X_test_scaled, 0, np.pi)

    print(f"  Train range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    print(f"  Test range:  [{X_test_scaled.min():.3f}, {X_test_scaled.max():.3f}]")

    return X_train_scaled, X_test_scaled, scaler


def build_circuit():
    print("\n" + "=" * 60)
    print("STEP 3/5: Building quantum circuit")
    print("=" * 60)

    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=FEATURE_MAP_REPS,
    )

    ansatz = RealAmplitudes(
        num_qubits=N_QUBITS,
        reps=ANSATZ_REPS,
    )

    print(f"  Qubits: {N_QUBITS}")
    print(f"  Feature map: ZZFeatureMap(reps={FEATURE_MAP_REPS})")
    print(f"  Ansatz: RealAmplitudes(reps={ANSATZ_REPS})")
    print(f"  Trainable parameters: {ansatz.num_parameters}")
    print(f"  Hilbert space: 2^{N_QUBITS} = {2**N_QUBITS}")

    return feature_map, ansatz


def visualize_circuit(feature_map, ansatz):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    full = feature_map.compose(ansatz).decompose()
    fig = full.draw(output="mpl", style="iqp", fold=20)
    save_path = os.path.join(FIGURES_DIR, "quantum_circuit.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Circuit saved: {save_path}")


def train_vqc(X_train, y_train, feature_map, ansatz):
    print("\n" + "=" * 60)
    print("STEP 4/5: Training VQC")
    print("=" * 60)

    objective_values = []

    def callback_fn(weights, obj_val):
        objective_values.append(obj_val)
        if len(objective_values) % 10 == 0:
            print(f"  Iteration {len(objective_values):3d}: loss = {obj_val:.4f}")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER),
        callback=callback_fn,
    )

    print(f"  Optimizer: COBYLA(maxiter={MAX_ITER})")
    print(f"  Training examples: {len(X_train)} (balanced 50/50)")
    print(f"\n  Starting training (loss every 10 iterations):\n")

    vqc.fit(X_train, y_train)

    final_loss = objective_values[-1] if objective_values else float("nan")
    print(f"\n  Training completed. Final loss: {final_loss:.4f}")
    print(f"  Total iterations recorded: {len(objective_values)}")

    return vqc, objective_values


def plot_training_curve(objective_values):
    if len(objective_values) == 0:
        print("  No iterations recorded - skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = range(1, len(objective_values) + 1)
    ax.plot(iterations, objective_values, "b-", linewidth=1.5)
    ax.scatter(iterations, objective_values, c="blue", s=10)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        f"VQC Training - Wisconsin Breast Cancer "
        f"({N_QUBITS} qubits, {OPTIMIZER_NAME})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "training_loss.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"  Curve saved: {save_path}")


def evaluate_quick(vqc, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 5/5: Quick evaluation")
    print("=" * 60)

    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)

    print(f"  Train accuracy: {train_score:.4f} ({train_score * 100:.2f}%)")
    print(f"  Test accuracy:  {test_score:.4f} ({test_score * 100:.2f}%)")

    return train_score, test_score


def save_model(vqc, objective_values, scaler, train_score, test_score,
               X_train, X_test, y_train, y_test):
    os.makedirs(MODELS_DIR, exist_ok=True)

    np.save(os.path.join(MODELS_DIR, "vqc_weights.npy"), vqc.weights)

    np.save(os.path.join(MODELS_DIR, "X_train_used.npy"), X_train)
    np.save(os.path.join(MODELS_DIR, "X_test_used.npy"), X_test)
    np.save(os.path.join(MODELS_DIR, "y_train_used.npy"), y_train)
    np.save(os.path.join(MODELS_DIR, "y_test_used.npy"), y_test)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, "vqc_history.pkl"), "wb") as f:
        pickle.dump({
            "objective_values": objective_values,
            "n_qubits": N_QUBITS,
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz_reps": ANSATZ_REPS,
            "optimizer": OPTIMIZER_NAME,
            "max_iter": MAX_ITER,
            "train_score": train_score,
            "test_score": test_score,
        }, f)
    print(f"\n  Model and data saved to {MODELS_DIR}")


def main():
    algorithm_globals.random_seed = RANDOM_SEED

    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    feature_map, ansatz = build_circuit()
    visualize_circuit(feature_map, ansatz)

    vqc, objective_values = train_vqc(X_train_scaled, y_train, feature_map, ansatz)
    plot_training_curve(objective_values)

    train_score, test_score = evaluate_quick(
        vqc, X_train_scaled, y_train, X_test_scaled, y_test
    )

    save_model(
        vqc, objective_values, scaler, train_score, test_score,
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETED")
    print("=" * 60)
    print("\nNext step: python src/03_evaluate.py")


if __name__ == "__main__":
    main()

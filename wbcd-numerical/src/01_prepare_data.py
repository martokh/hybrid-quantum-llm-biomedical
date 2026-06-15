import os
import sys
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Shared diplom-level helpers (one sampling function for both WBCD pipelines).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.sampling import balanced_train_test_split

from config import (
    N_FEATURES,
    RANDOM_SEED,
    N_TRAIN_PER_CLASS,
    N_TEST_PER_CLASS,
    DATA_DIR,
)


def load_wbcd():
    print("=" * 60)
    print("STEP 1/5: Loading Wisconsin Breast Cancer Diagnostic dataset")
    print("=" * 60)

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print(f"  Total examples: {len(X)}")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Classes: {target_names[0]} (label=0) vs {target_names[1]} (label=1)")
    print(f"  Distribution: {target_names[1]}={int(y.sum())}, "
          f"{target_names[0]}={len(y) - int(y.sum())}")
    print(f"  Class balance: {y.mean():.2%} positive")

    print(f"\n  Sample feature names (first 10):")
    for i in range(10):
        print(f"    {i + 1}. {feature_names[i]}")
    print(f"    ... and {len(feature_names) - 10} more clinical features")

    return X, y, feature_names, target_names


def split_and_standardize(X, y):
    print("\n" + "=" * 60)
    print("STEP 2/5: Balanced subsample and standardization")
    print("=" * 60)

    # Balanced, disjoint split: 50/class train + 15/class test -> 100/30.
    # Same helper the WBCD-text pipeline uses (common.sampling).
    X_train, X_test, y_train, y_test = balanced_train_test_split(
        X, y,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        seed=RANDOM_SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Train: {len(X_train)} examples "
          f"({int(y_train.sum())} positive, {len(y_train) - int(y_train.sum())} negative)")
    print(f"  Test:  {len(X_test)} examples "
          f"({int(y_test.sum())} positive, {len(y_test) - int(y_test.sum())} negative)")
    print(f"  Standardization: mean=0, std=1 (fit on train only)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def reduce_dimensions(X_train, X_test, n_features):
    print("\n" + "=" * 60)
    print(f"STEP 3/5: PCA reduction: 30 -> {n_features}")
    print("=" * 60)

    pca = PCA(n_components=n_features, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_var = pca.explained_variance_ratio_
    total_var = explained_var.sum()

    print(f"  Explained variance per component:")
    for i, var in enumerate(explained_var):
        print(f"    PC{i + 1}: {var:.4f} ({var * 100:.2f}%)")
    print(f"  Total explained variance: {total_var:.4f} ({total_var * 100:.2f}%)")

    return X_train_pca, X_test_pca, pca


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X, y, feature_names, target_names = load_wbcd()

    X_train, X_test, y_train, y_test, scaler = split_and_standardize(X, y)

    X_train_pca, X_test_pca, pca = reduce_dimensions(X_train, X_test, N_FEATURES)

    print("\n" + "=" * 60)
    print("STEP 4/5: Saving files")
    print("=" * 60)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_pca.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test_pca.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test.astype(np.int64))

    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(DATA_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"  All files saved to {DATA_DIR}")
    print(f"  X_train: {X_train_pca.shape}, X_test: {X_test_pca.shape}")

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETED")
    print("=" * 60)
    print("\nNext step: python src/02_train_quantum.py")


if __name__ == "__main__":
    main()

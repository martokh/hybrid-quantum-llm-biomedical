"""Stage 1 — turn WBCD numerical rows into BioBERT embeddings, reduce with PCA.

Pipeline:
    sklearn WBCD (30 floats per sample)
        -> to_clinical_text(...)
        -> BioBERT tokenizer + frozen model
        -> [CLS] token embedding (768-dim)
        -> PCA -> 4-dim
        -> save .npy for the quantum stage
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Shared diplom-level helpers (one sampling function for both WBCD pipelines).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.sampling import balanced_train_test_split  # noqa: E402
import config as C  # noqa: E402


# --------------------------------------------------------------------------
# Text generation
# --------------------------------------------------------------------------
def to_clinical_text(row: pd.Series) -> str:
    """Render a single WBCD row as a professional-sounding clinical paragraph.

    Only the ten *mean* features are used; this keeps the prompt short and
    well within BioBERT's 512-token limit.
    """
    return (
        "Breast tumor biopsy with the following morphological characteristics: "
        f"mean radius {row['mean radius']:.2f}, "
        f"mean texture {row['mean texture']:.2f}, "
        f"mean perimeter {row['mean perimeter']:.2f}, "
        f"mean area {row['mean area']:.2f}, "
        f"mean smoothness {row['mean smoothness']:.4f}, "
        f"mean compactness {row['mean compactness']:.4f}, "
        f"mean concavity {row['mean concavity']:.4f}, "
        f"mean concave points {row['mean concave points']:.4f}, "
        f"mean symmetry {row['mean symmetry']:.4f}, "
        f"mean fractal dimension {row['mean fractal dimension']:.4f}."
    )


# --------------------------------------------------------------------------
# BioBERT embeddings
# --------------------------------------------------------------------------
@torch.no_grad()
def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """Return [CLS] embeddings (N, 768) for a list of clinical descriptions."""
    embs = []
    for i in tqdm(range(0, len(texts), C.EMBED_BATCH_SIZE), desc="BioBERT batches"):
        batch = texts[i : i + C.EMBED_BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=C.MAX_TOKEN_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        embs.append(cls.cpu().numpy())
    return np.vstack(embs)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Stage 1 / 3  -  Prepare WBCD text embeddings for the quantum stage")
    print("=" * 72)

    # 1. Load WBCD
    print("\n[1/6] Loading Wisconsin Breast Cancer Diagnostic dataset ...")
    raw = load_breast_cancer(as_frame=True)
    X_all: pd.DataFrame = raw.data
    # sklearn convention: 0 = malignant, 1 = benign. We keep that convention.
    y_all: np.ndarray = raw.target.to_numpy()
    print(f"      total samples: {len(X_all)}  features: {X_all.shape[1]}")
    print(f"      class counts : malignant={np.sum(y_all == 0)}, "
          f"benign={np.sum(y_all == 1)}")

    # 2. Balanced, disjoint subsample: 50/class train + 15/class test -> 100/30.
    #    Same helper the numerical WBCD pipeline uses (common.sampling).
    print("\n[2/6] Balanced subsample (50/class train, 15/class test) ...")
    X_tr, X_te, y_tr, y_te = balanced_train_test_split(
        X_all,
        y_all,
        n_train_per_class=C.N_TRAIN_PER_CLASS,
        n_test_per_class=C.N_TEST_PER_CLASS,
        seed=C.RANDOM_SEED,
    )
    print(f"      train: {len(X_tr)}  (malignant={int(np.sum(y_tr == 0))}, "
          f"benign={int(np.sum(y_tr == 1))})")
    print(f"      test : {len(X_te)}  (malignant={int(np.sum(y_te == 0))}, "
          f"benign={int(np.sum(y_te == 1))})")

    # 3. Generate clinical texts
    print("\n[3/6] Generating clinical descriptions ...")
    train_texts = [to_clinical_text(r) for _, r in X_tr.iterrows()]
    test_texts = [to_clinical_text(r) for _, r in X_te.iterrows()]

    print("\n      Example clinical description (first train sample):")
    print(f"      label = {'benign' if y_tr[0] == 1 else 'malignant'}")
    print("      " + train_texts[0])

    # Persist 5 samples (mixed classes) for the thesis defense
    sample_lines = ["Sample clinical descriptions fed to BioBERT", "=" * 72, ""]
    for j in range(5):
        label = "benign" if y_tr[j] == 1 else "malignant"
        sample_lines.append(f"[{j + 1}] label = {label}")
        sample_lines.append(train_texts[j])
        sample_lines.append("")
    C.SAMPLE_TEXTS_FILE.write_text("\n".join(sample_lines), encoding="utf-8")
    print(f"      -> 5 examples written to {C.SAMPLE_TEXTS_FILE}")

    # 4. Load BioBERT (frozen)
    print(f"\n[4/6] Loading BioBERT '{C.BIOBERT_MODEL}' as frozen extractor ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(C.BIOBERT_MODEL)
    model = AutoModel.from_pretrained(C.BIOBERT_MODEL).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      BioBERT parameters: {n_params:,}  (all frozen)")

    # 5. Embed
    print("\n[5/6] Extracting [CLS] embeddings ...")
    print("      train:")
    emb_train = embed_texts(train_texts, tokenizer, model, device)
    print("      test :")
    emb_test = embed_texts(test_texts, tokenizer, model, device)
    print(f"      train embeddings shape: {emb_train.shape}")
    print(f"      test  embeddings shape: {emb_test.shape}")

    # 6. PCA 768 -> 4
    print(f"\n[6/6] PCA reduction {emb_train.shape[1]} -> {C.PCA_COMPONENTS} ...")
    pca = PCA(n_components=C.PCA_COMPONENTS, random_state=C.RANDOM_SEED)
    X_train_pca = pca.fit_transform(emb_train)
    X_test_pca = pca.transform(emb_test)
    var = pca.explained_variance_ratio_
    print(f"      explained variance ratio: {var}")
    print(f"      cumulative variance     : {var.sum():.4f}")

    # Save artefacts
    np.save(C.X_TRAIN_FILE, X_train_pca)
    np.save(C.X_TEST_FILE, X_test_pca)
    np.save(C.Y_TRAIN_FILE, y_tr)
    np.save(C.Y_TEST_FILE, y_te)
    # Persist PCA as plain arrays (no pickle, so it survives sklearn upgrades).
    np.savez(
        C.PCA_FILE,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    print("\nSaved artefacts:")
    for f in (
        C.X_TRAIN_FILE, C.X_TEST_FILE,
        C.Y_TRAIN_FILE, C.Y_TEST_FILE,
        C.PCA_FILE, C.SAMPLE_TEXTS_FILE,
    ):
        print(f"  - {f}")
    print("\nStage 1 done. Next: python src/02_train_quantum.py")


if __name__ == "__main__":
    main()

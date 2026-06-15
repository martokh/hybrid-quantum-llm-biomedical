import os
import sys
import pickle
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BIOBERT_MODEL_NAME,
    N_FEATURES,
    MAX_SEQUENCE_LENGTH,
    BATCH_SIZE,
    RANDOM_SEED,
    TRAIN_PER_CLASS,
    TEST_PER_CLASS,
    DATA_DIR,
)


HOC_LABEL_NAMES = {
    0: "activating_invasion_and_metastasis",
    1: "avoiding_immune_destruction",
    2: "cellular_energetics",
    3: "enabling_replicative_immortality",
    4: "evading_growth_suppressors",
    5: "genomic_instability_and_mutation",
    6: "inducing_angiogenesis",
    7: "resisting_cell_death",
    8: "sustaining_proliferative_signaling",
    9: "tumor_promoting_inflammation",
}


def load_and_prepare_hoc():
    print("=" * 60)
    print("STEP 1/6: Loading Hallmarks of Cancer dataset")
    print("=" * 60)

    dataset = load_dataset("qanastek/HoC", trust_remote_code=True)
    all_examples = list(dataset["train"]) + list(dataset["validation"]) + list(dataset["test"])
    print(f"  Combined examples: {len(all_examples)}")

    label_counter = Counter()
    for ex in all_examples:
        for lbl in ex["label"]:
            label_counter[lbl] += 1

    target_label_id = label_counter.most_common(1)[0][0]
    target_label_name = HOC_LABEL_NAMES.get(target_label_id)
    print(f"  Target label: [{target_label_id}] {target_label_name}")

    texts = []
    labels = []
    for ex in all_examples:
        has_label = 1 if target_label_id in ex["label"] else 0
        texts.append(ex["text"])
        labels.append(has_label)

    labels = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    need = TRAIN_PER_CLASS + TEST_PER_CLASS
    if len(pos_idx) < need or len(neg_idx) < need:
        raise RuntimeError(
            f"Not enough samples: pos={len(pos_idx)}, neg={len(neg_idx)}, "
            f"need {need} each (={TRAIN_PER_CLASS} train + {TEST_PER_CLASS} test)"
        )

    pos_sel = rng.choice(pos_idx, need, replace=False)
    neg_sel = rng.choice(neg_idx, need, replace=False)

    # Balanced, non-overlapping splits: first TRAIN_PER_CLASS per class -> train,
    # next TEST_PER_CLASS per class -> test (mirrors the HoC backbone).
    train_sel = np.concatenate([pos_sel[:TRAIN_PER_CLASS], neg_sel[:TRAIN_PER_CLASS]])
    test_sel = np.concatenate([pos_sel[TRAIN_PER_CLASS:], neg_sel[TRAIN_PER_CLASS:]])
    rng.shuffle(train_sel)
    rng.shuffle(test_sel)

    texts_train = [texts[i] for i in train_sel]
    texts_test = [texts[i] for i in test_sel]
    y_train = labels[train_sel]
    y_test = labels[test_sel]

    print(f"  Train: {len(texts_train)} examples, balance={np.bincount(y_train)}")
    print(f"  Test : {len(texts_test)} examples, balance={np.bincount(y_test)}")

    return texts_train, texts_test, y_train, y_test, target_label_name


def load_biobert():
    print("\n" + "=" * 60)
    print(f"STEP 2/6: Loading {BIOBERT_MODEL_NAME}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Loaded on device: {device}")

    return tokenizer, model, device


def extract_embeddings(texts, tokenizer, model, device, desc="Embeddings"):
    all_embeddings = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    progress = tqdm(range(0, len(texts), BATCH_SIZE), total=n_batches, desc=f"  {desc}")

    for i in progress:
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def reduce_dimensions(emb_train, emb_test, y_train, n_features):
    print("\n" + "=" * 60)
    print(f"STEP 5/6: Dimension reduction (pure PCA): 768 -> {n_features}")
    print("=" * 60)

    pca = PCA(n_components=n_features, random_state=RANDOM_SEED)
    X_train = pca.fit_transform(emb_train)
    X_test = pca.transform(emb_test)

    pca_var = pca.explained_variance_ratio_.sum()
    print(f"  Method: PCA({n_features} components)")
    print(f"  PCA explained variance: {pca_var * 100:.2f}%")
    print(f"  Final shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, pca


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    texts_train, texts_test, y_train, y_test, target_label = load_and_prepare_hoc()

    tokenizer, model, device = load_biobert()

    print("\n" + "=" * 60)
    print("STEP 4/6: Extracting BioBERT embeddings")
    print("=" * 60)

    emb_train = extract_embeddings(texts_train, tokenizer, model, device, "Train")
    emb_test = extract_embeddings(texts_test, tokenizer, model, device, "Test ")

    X_train, X_test, pca = reduce_dimensions(
        emb_train, emb_test, y_train, N_FEATURES
    )

    print("\n" + "=" * 60)
    print("STEP 6/6: Saving files")
    print("=" * 60)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    with open(os.path.join(DATA_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    with open(os.path.join(DATA_DIR, "target_label.txt"), "w", encoding="utf-8") as f:
        f.write(target_label)

    print(f"  All files saved to {DATA_DIR}")
    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETED")
    print("=" * 60)
    print(f"\n  Target label: {target_label}")
    print("\nNext step: python src/02_train_quantum.py")


if __name__ == "__main__":
    main()

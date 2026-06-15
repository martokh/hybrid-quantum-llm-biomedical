import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hoc_splits():
    print(f"[1/5] Loading dataset '{config.HOC_DATASET_NAME}' ...")
    ds = load_dataset(config.HOC_DATASET_NAME)
    print(f"      splits: {list(ds.keys())}")
    for split in ds.keys():
        print(f"      {split}: {len(ds[split])} rows; columns={ds[split].column_names}")
    return ds


def find_text_and_label_columns(dataset):
    cols = dataset.column_names
    text_col = next((c for c in cols if c.lower() in {"text", "abstract", "sentence", "document"}), None)
    label_col = next((c for c in cols if c.lower() in {"label", "labels", "label_text", "hallmark", "hallmarks"}), None)
    if text_col is None or label_col is None:
        raise RuntimeError(f"Could not auto-detect text/label columns in {cols}")
    return text_col, label_col


def coerce_labels_to_list(label_value):
    if label_value is None:
        return []
    if isinstance(label_value, (list, tuple, np.ndarray)):
        return [str(x) for x in label_value]
    return [str(label_value)]


def find_most_common_label(train_split, label_col):
    print("[2/5] Identifying the most common hallmark label ...")
    counter = Counter()
    for row in train_split:
        for lbl in coerce_labels_to_list(row[label_col]):
            counter[lbl] += 1
    if not counter:
        raise RuntimeError("No labels found in dataset.")
    target, freq = counter.most_common(1)[0]
    print(f"      Top-5: {counter.most_common(5)}")
    print(f"      Target label = '{target}' (count={freq})")
    return target


def balanced_subsample(dataset, text_col, label_col, target_label, n_per_class, rng):
    pos_texts, neg_texts = [], []
    for row in dataset:
        labels = coerce_labels_to_list(row[label_col])
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        if target_label in labels:
            pos_texts.append(text)
        else:
            neg_texts.append(text)

    rng.shuffle(pos_texts)
    rng.shuffle(neg_texts)
    if len(pos_texts) < n_per_class or len(neg_texts) < n_per_class:
        raise RuntimeError(
            f"Not enough samples: pos={len(pos_texts)}, neg={len(neg_texts)}, need {n_per_class} each"
        )
    pos_texts = pos_texts[:n_per_class]
    neg_texts = neg_texts[:n_per_class]
    texts = pos_texts + neg_texts
    labels = [1] * n_per_class + [0] * n_per_class
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return texts, np.array(labels, dtype=np.int64)


def build_biobert():
    print(f"[3/5] Loading BioBERT '{config.BIOBERT_MODEL_NAME}' (frozen) ...")
    tokenizer = AutoTokenizer.from_pretrained(config.BIOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BIOBERT_MODEL_NAME)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"      device={device}, params frozen=True")
    return tokenizer, model, device


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device):
    embeddings = []
    bs = config.EMBEDDING_BATCH_SIZE
    for start in tqdm(range(0, len(texts), bs), desc="Encoding"):
        batch = texts[start : start + bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        embeddings.append(cls)
    return np.vstack(embeddings)


def main():
    set_seed(config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)

    ds = load_hoc_splits()
    primary_split = "train" if "train" in ds else list(ds.keys())[0]
    text_col, label_col = find_text_and_label_columns(ds[primary_split])
    print(f"      using columns: text='{text_col}', label='{label_col}'")

    target_label = find_most_common_label(ds[primary_split], label_col)
    config.TARGET_LABEL_PATH.write_text(target_label, encoding="utf-8")

    train_source = ds[primary_split]
    test_source = ds["test"] if "test" in ds else (ds["validation"] if "validation" in ds else train_source)

    train_texts, y_train = balanced_subsample(
        train_source, text_col, label_col, target_label, config.TRAIN_PER_CLASS, rng
    )
    test_texts, y_test = balanced_subsample(
        test_source, text_col, label_col, target_label, config.TEST_PER_CLASS, rng
    )
    print(f"      train: {len(train_texts)} examples, class balance={np.bincount(y_train)}")
    print(f"      test : {len(test_texts)} examples, class balance={np.bincount(y_test)}")

    tokenizer, model, device = build_biobert()
    X_train_full = encode_texts(train_texts, tokenizer, model, device)
    X_test_full = encode_texts(test_texts, tokenizer, model, device)
    print(f"      train embeddings shape: {X_train_full.shape}")
    print(f"      test  embeddings shape: {X_test_full.shape}")

    print(f"[4/5] PCA reduction 768 -> {config.PCA_COMPONENTS} ...")
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_train = pca.fit_transform(X_train_full).astype(np.float32)
    X_test = pca.transform(X_test_full).astype(np.float32)
    print(f"      explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"      cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print("[5/5] Saving artifacts ...")
    np.save(config.X_TRAIN_PATH, X_train)
    np.save(config.X_TEST_PATH, X_test)
    np.save(config.Y_TRAIN_PATH, y_train)
    np.save(config.Y_TEST_PATH, y_test)
    joblib.dump(pca, config.PCA_PATH)
    print(f"      wrote: {config.X_TRAIN_PATH}")
    print(f"      wrote: {config.X_TEST_PATH}")
    print(f"      wrote: {config.Y_TRAIN_PATH}")
    print(f"      wrote: {config.Y_TEST_PATH}")
    print(f"      wrote: {config.PCA_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()

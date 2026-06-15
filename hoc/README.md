# Hybrid Quantum LLM for Biomedical Text Classification

Proof-of-concept system that combines a pre-trained biomedical language model (BioBERT) with a Variational Quantum Classifier (VQC) to classify PubMed abstracts from the Hallmarks of Cancer (HoC) dataset.

This repository accompanies the thesis **"Hybrid Quantum LLM System for Biomedical Big Data Analytics"**.

## Architecture

```
Biomedical text (HoC PubMed abstracts)
    -> BioBERT tokenizer + model (frozen, no fine-tuning)
    -> [CLS] token embeddings (768-dim float32 vectors)
    -> PCA reduction (768 -> 4 dimensions)
    -> MinMaxScaler to [0, pi] range
    -> ZZFeatureMap (4 qubits, reps=2)
    -> RealAmplitudes ansatz (4 qubits, reps=2)
    -> VQC.fit() with COBYLA optimizer
    -> Prediction (binary: hallmark present / absent)
```

## Repository layout

```
hybrid-quantum-llm-hoc/
├── config.py                # central hyperparameters and paths
├── requirements.txt         # pinned dependencies
├── src/
│   ├── 01_prepare_data.py   # BioBERT embeddings + PCA
│   ├── 02_train_quantum.py  # VQC training with COBYLA
│   └── 03_evaluate.py       # metrics and visualizations
├── data/                    # generated .npy arrays + target label
└── results/
    ├── figures/             # PNG visualizations
    ├── models/              # PCA, scaler, VQC weights, loss history
    └── metrics.json         # evaluation metrics
```

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- Approximately 2 GB of free disk space (BioBERT weights are cached by Hugging Face)

## Installation

```powershell
git clone https://github.com/martokh/hybrid_quantum_bio_llm_hoc.git
cd hybrid_quantum_bio_llm_hoc
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux / macOS replace the activation line with `source .venv/bin/activate`.

Optional dependency for a nicer quantum-circuit diagram:

```powershell
pip install pylatexenc
```

Without it the pipeline still runs and falls back to a text dump of the circuit.

## Running the pipeline

The three scripts are intended to be run in order:

```powershell
python src\01_prepare_data.py
python src\02_train_quantum.py
python src\03_evaluate.py
```

### Step 1 — Data preparation

Downloads the HoC dataset, identifies the most frequent hallmark label, balances the splits (50 positive + 50 negative for training, 15 + 15 for testing), encodes every abstract through frozen BioBERT, and applies PCA(768 -> 4). Outputs: `data/X_*.npy`, `data/y_*.npy`, `data/target_label.txt`, `results/models/pca.joblib`.

### Step 2 — Quantum training

Loads the prepared arrays, scales them to `[0, pi]`, builds a `ZZFeatureMap + RealAmplitudes` quantum circuit, trains the VQC with COBYLA for 150 iterations, and reports train / test accuracy. Outputs: `results/models/vqc_weights.npy`, `results/models/loss_history.npy`, `results/figures/training_loss.png`, `results/figures/quantum_circuit.png` (or `.txt` if `pylatexenc` is missing).

### Step 3 — Evaluation

Rebuilds the VQC from the saved weights and computes the test-set metrics: accuracy, precision, recall, F1, ROC-AUC. Outputs: `results/metrics.json`, `results/figures/confusion_matrix.png`, `results/figures/metrics_bar.png`, `results/figures/pca_explained_variance.png`.

## Key hyperparameters

All hyperparameters are centralized in `config.py`:

| Parameter | Value |
|-----------|-------|
| `RANDOM_SEED` | 42 |
| `BIOBERT_MODEL_NAME` | `dmis-lab/biobert-v1.1` |
| `HOC_DATASET_NAME` | `qanastek/HoC` |
| `TRAIN_PER_CLASS` | 50 |
| `TEST_PER_CLASS` | 15 |
| `PCA_COMPONENTS` | 4 |
| `SCALER_MIN`, `SCALER_MAX` | 0, pi |
| `N_QUBITS` | 4 |
| `FEATURE_MAP_REPS` | 2 |
| `ANSATZ_REPS` | 2 |
| `MAX_ITER` (COBYLA) | 150 |

## Expected results

Test accuracy lies in the range **60 % – 75 %**, which is consistent with reported results in the hybrid quantum-LLM literature for small training sets and four-qubit circuits.

## Design notes

- **BioBERT is frozen.** `param.requires_grad = False` on all parameters and `model.eval()` is set; no gradients are propagated through the language model.
- **VQC weights are persisted as `.npy`, not pickled.** The VQC object contains a local `parity` function that breaks pickling. We save `vqc.weights` only and reconstruct the network in step 3.
- **`matplotlib.use("Agg")` is set before importing `pyplot`.** This avoids the Tk backend errors observed on Windows in some environments.
- **Feature range `[0, pi]`.** Using `[-pi, pi]` causes information loss because of the trigonometric periodicity of the parameterized rotations inside the feature map.

## Dataset and model attributions

- **HoC**: Baker, S. et al. *Automatic semantic classification of scientific literature according to the hallmarks of cancer.* Bioinformatics, 2016. Hugging Face mirror: [`qanastek/HoC`](https://huggingface.co/datasets/qanastek/HoC).
- **BioBERT**: Lee, J. et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.* Bioinformatics, 2020. Weights: [`dmis-lab/biobert-v1.1`](https://huggingface.co/dmis-lab/biobert-v1.1).

## License

Released for academic use as part of the author's thesis. Underlying datasets and models retain their original licenses.

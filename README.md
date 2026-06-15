# Hybrid Quantum-LLM System for Biomedical Big Data Analytics

Source code for the bachelor thesis *"Development of a Hybrid Quantum-LLM
System for Biomedical Big Data Analytics"* — Technical University of Sofia,
Faculty of German Engineering and Industrial Management (FDIBA).

## Overview

This project implements a hybrid quantum-classical classifier for biomedical
data. A frozen **BioBERT** encoder feeds, via **PCA** reduction and **[0, π]**
scaling, a **4-qubit variational quantum classifier** (a ZZFeatureMap feature
map with a RealAmplitudes ansatz), trained with the **COBYLA** optimizer on a
statevector simulator.

The work has two goals:

1. **Feasibility (RQ1)** — show that such a hybrid system can be built and run
   end-to-end on real biomedical data. This is demonstrated by the primary
   **Hallmarks of Cancer (HoC)** experiment.
2. **The role of the LLM stage (RQ2)** — show that inserting a language model
   is *not* universally beneficial. This is demonstrated by the controlled
   **Wisconsin Breast Cancer (WBCD)** pair, which encodes identical structured
   data either directly or after passing it through BioBERT.

All four experiments share an **identical quantum backbone**; only the dataset
and the encoding regime differ.

## Repository structure

| Folder            | Experiment          | Encoding                          | Role             |
|-------------------|---------------------|-----------------------------------|------------------|
| `hoc/`            | Hallmarks of Cancer | BioBERT → PCA → angle             | Primary (RQ1)    |
| `pubmedqa/`       | PubMedQA            | BioBERT → PCA → angle             | Supporting       |
| `wbcd-numerical/` | WBCD (diagnostic)   | direct features → PCA → angle     | RQ2 (LLM-free)   |
| `wbcd-text/`      | WBCD (diagnostic)   | serialized text → BioBERT → PCA   | RQ2 (with LLM)   |

Each folder contains:
- `config.py` — all hyperparameters for that experiment
- `src/` — the three-stage pipeline:
  - `01_prepare_data.py` — load data, embed/encode, PCA, cache features
  - `02_train_quantum.py` — scale to [0, π], build and train the VQC (COBYLA)
  - `03_evaluate.py` — predict and write metrics + confusion matrix
- `results/` — generated artifacts (`metrics.json`, loss history, trained weights)

## Shared configuration

| Parameter            | Value                                |
|----------------------|--------------------------------------|
| Random seed          | 42                                   |
| Qubits               | 4                                    |
| Feature map          | ZZFeatureMap, reps = 2               |
| Ansatz               | RealAmplitudes, reps = 2 (12 params) |
| Optimizer            | COBYLA, max_iter = 150               |
| PCA target dimension | 4 (fitted on training set only)      |
| Scaling range        | [0, π] (fitted on train)             |
| Language model       | BioBERT (`dmis-lab/biobert-v1.1`), frozen |
| Max sequence length  | 256 tokens                           |
| Simulator            | Qiskit Aer statevector (exact)       |

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.10 or later.

## Running an experiment

```bash
cd hoc
python src/01_prepare_data.py
python src/02_train_quantum.py
python src/03_evaluate.py
```

The same three commands run any of the four experiments — just change the
folder (`pubmedqa`, `wbcd-numerical`, `wbcd-text`).

## Results summary

| Experiment       | Accuracy | F1    | ROC-AUC | Test set        |
|------------------|----------|-------|---------|-----------------|
| HoC              | 0.60     | 0.647 | 0.578   | 30 (balanced)   |
| PubMedQA         | 0.40     | 0.31  | 0.34    | 30 (balanced)   |
| WBCD-numerical   | 0.667    | 0.729 | 0.663   | 114 (natural)   |
| WBCD-text        | 0.533    | 0.500 | 0.48    | 30 (balanced)   |

The controlled WBCD comparison (numerical vs text, identical backbone) is the
core result: routing structured clinical data through BioBERT drops the
balance-independent ROC-AUC from 0.663 to 0.48 (at chance).

## Note on scope

All results come from **exact statevector simulation** (no hardware noise).
This work demonstrates **feasibility**, not quantum advantage; no configuration
is claimed to outperform a classical baseline.

## Citation

If you reference this work, please cite the thesis:

> Martin Hristov (2026). *Development of a Hybrid Quantum-LLM System for
> Biomedical Big Data Analytics.* Bachelor thesis, Technical University of
> Sofia, FDIBA.

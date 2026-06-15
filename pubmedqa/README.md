# Hybrid Quantum LLM System for Biomedical Big Data Analytics

Diploma thesis – Technical University of Sofia, FDIBA.

## Architecture

```
STAGE 1 (preprocessing, run once):
  PubMedQA text -> BioBERT -> 768-dim [CLS] embedding -> PCA -> 4-dim vector

STAGE 2 (quantum classifier):
  4-dim vector -> ZZFeatureMap -> RealAmplitudes ansatz -> measurement -> yes/no
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution

```bash
python src/01_prepare_data.py
python src/02_train_quantum.py
python src/03_train_classical.py
python src/04_evaluate.py
```

## Structure

```
quantum-llm-biomedical/
├── config.py
├── requirements.txt
├── data/
├── src/
└── results/
```

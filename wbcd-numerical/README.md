# Hybrid Quantum Classifier for Wisconsin Breast Cancer Diagnostic

## Architecture

```
WBCD (569 examples, 30 clinical features)
    -> StandardScaler (mean=0, std=1)
    -> PCA (30 -> 8 components)
    -> Normalize to [-pi, pi]
    -> Quantum encoding (ZZFeatureMap)
    -> Variational ansatz (RealAmplitudes)
    -> Measurement -> malignant/benign
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
python src/03_evaluate.py
```

## Structure

```
quantum-wbcd/
├── config.py
├── requirements.txt
├── data/             
├── src/              
└── results/          
```

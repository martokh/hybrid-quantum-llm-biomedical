import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Reproducibility (HoC backbone) -----------------------------------------
RANDOM_SEED = 42

# --- BioBERT (frozen feature extractor) -------------------------------------
BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 8

# --- Dataset (balanced sub-sampling, HoC scheme) ----------------------------
TRAIN_PER_CLASS = 50   # -> 100 train samples
TEST_PER_CLASS = 15    # -> 30  test samples

# --- Dimensionality reduction -----------------------------------------------
N_FEATURES = 4         # pure PCA -> 4 components

# --- Quantum classifier -----------------------------------------------------
N_QUBITS = N_FEATURES
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2

# --- Feature scaling to [0, pi] ---------------------------------------------
SCALER_MIN = 0.0
SCALER_MAX = float(np.pi)

OPTIMIZER_NAME = "COBYLA"
MAX_ITER = 150

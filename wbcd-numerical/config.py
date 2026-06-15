import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

RANDOM_SEED = 42

N_FEATURES = 4

# Balanced sub-sampling (same scheme as WBCD-text / HoC / PubMedQA)
N_TRAIN_PER_CLASS = 50   # -> 100 train samples (50/50)
N_TEST_PER_CLASS = 15    # -> 30  test samples (15/15)

N_QUBITS = N_FEATURES
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2

OPTIMIZER_NAME = "COBYLA"
MAX_ITER = 150

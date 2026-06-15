from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for _d in (DATA_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"
EMBEDDING_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

HOC_DATASET_NAME = "qanastek/HoC"

TRAIN_PER_CLASS = 50
TEST_PER_CLASS = 15

PCA_COMPONENTS = 4
SCALER_MIN = 0.0
SCALER_MAX = 3.141592653589793

N_QUBITS = 4
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2

MAX_ITER = 150
LOG_EVERY = 5

X_TRAIN_PATH = DATA_DIR / "X_train.npy"
X_TEST_PATH = DATA_DIR / "X_test.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"
PCA_PATH = MODELS_DIR / "pca.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
VQC_WEIGHTS_PATH = MODELS_DIR / "vqc_weights.npy"
LOSS_HISTORY_PATH = MODELS_DIR / "loss_history.npy"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TARGET_LABEL_PATH = DATA_DIR / "target_label.txt"

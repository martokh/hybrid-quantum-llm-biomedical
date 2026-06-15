"""Central configuration for the hybrid quantum LLM (BioBERT + WBCD) pipeline.

Keeping every tunable knob in one place makes the experiment reproducible and
easy to explain in the thesis defense.
"""

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# --- Dataset (balanced sub-sampling of WBCD) --------------------------------
N_TRAIN_PER_CLASS = 50   # -> 100 train samples
N_TEST_PER_CLASS = 15    # -> 30  test samples

# --- Text generation --------------------------------------------------------
# Only the first 10 "mean" features are used to keep descriptions focused and
# short. BioBERT has a 512-token limit; shorter prompts are cleaner.
MEAN_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]

# --- BioBERT (frozen feature extractor) -------------------------------------
BIOBERT_MODEL = "dmis-lab/biobert-v1.1"
MAX_TOKEN_LENGTH = 256
EMBED_BATCH_SIZE = 8

# --- PCA --------------------------------------------------------------------
PCA_COMPONENTS = 4

# --- Quantum classifier -----------------------------------------------------
N_QUBITS = 4
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2
MAX_ITER = 150            # COBYLA iterations
COBYLA_RHOBEG = 0.5       # initial trust-region size

# --- Artefact filenames -----------------------------------------------------
X_TRAIN_FILE = DATA_DIR / "X_train_embeddings.npy"
X_TEST_FILE = DATA_DIR / "X_test_embeddings.npy"
Y_TRAIN_FILE = DATA_DIR / "y_train.npy"
Y_TEST_FILE = DATA_DIR / "y_test.npy"
PCA_FILE = MODELS_DIR / "pca.npz"           # components + mean (no pickle)
SCALER_FILE = MODELS_DIR / "scaler.npz"      # min/scale arrays
WEIGHTS_FILE = MODELS_DIR / "vqc_weights.npy"
SAMPLE_TEXTS_FILE = DATA_DIR / "sample_texts.txt"
METRICS_FILE = RESULTS_DIR / "metrics.json"

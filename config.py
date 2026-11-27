"""Central settings for the Tennessee Eastman binary fault detector."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RUN_ID = None
DATASET_PATH = ROOT_DIR / "data" / "raw"  # Put the dataset CSV files here.
SAMPLE_SIZE = 20_000  # Maximum total rows sampled; None reads all rows (check RAM first).
CLASSIFICATION_TYPE = "binary"  # First version supports only NORMAL versus FAULTY.
NORMAL_LABEL = 0  # In TEP's faultNumber column, 0 denotes normal operation.
TEST_SIZE = 0.20  # Fraction of sampled rows held out for final evaluation.
RANDOM_STATE = 42  # Fixed seed makes sampling and splitting reproducible.
SVM_KERNEL = "rbf"  # SVC kernel: linear, rbf, or poly.
SVM_C = 1.0  # Larger C penalizes training mistakes more strongly.
SVM_GAMMA = "scale"  # RBF/poly kernel reach; can also be a positive number.
USE_ALL_FEATURES = True  # False uses only SELECTED_FEATURES below.
SELECTED_FEATURES = []  # Exact sensor columns; used only when USE_ALL_FEATURES=False.
NORMALIZE_DATA = True  # Standardize using training data only.
USE_CLASS_BALANCING = True  # Set SVC class_weight='balanced' to account for rare faults.
SAVE_MODEL = True
SAVE_PLOTS = True
SAVE_EXCEL_REPORTS = True
SAVE_CSV_REPORTS = True
SAVE_PREDICTIONS = True
SAVE_EDUCATIONAL_QP = True  # Run the tiny CVXOPT demonstration when CVXOPT is installed.
RUN_SVM_EXPERIMENTS = True  # Run bounded same-data SVM experiments after the main model.
RUN_PCA_EXPERIMENT = True  # Compare the primary SVC with and without PCA.
PCA_EXPLAINED_VARIANCE_TARGET = 0.95  # Retain at least this share of training variance.
EXPERIMENT_KERNELS = ["linear", "rbf"]  # Kernels included in the experiment table.
EXPERIMENT_C_VALUES = [0.1, 1.0, 10.0, 100.0]  # Required regularization experiment values.
EXPERIMENT_MAX_ROWS = 4_000  # Cap per kernel/C experiment so RBF fits remain manageable.
EXPERIMENT_SAMPLE_SIZES = [500, 1_000, 2_000, 4_000]  # Actual SVC fit-time curve sizes.
SCRATCH_COMPARE_ROWS = 1_000  # Small balanced set for scratch, QP, and library comparison.
SCRATCH_ITERATIONS = 1_000  # Number of mini-batch subgradient weight updates.
SCRATCH_LEARNING_RATE = 0.05  # Initial step size; scratch training decays it by data pass.
QP_MAX_ROWS = 1_000  # Dense CVXOPT matrices are quadratic; cap the educational subset.
MAX_CORRELATION_FEATURES = 20  # Keep the sensor correlation chart readable.
MAX_SVC_ROWS = 50_000  # Safety cap: kernel SVC runtime grows roughly quadratically.

# All results stay inside the project folder.
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LEGACY_MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR = LEGACY_MODEL_DIR  # Training runs point this into their unique run folder.
RESULTS_ROOT_DIR = ROOT_DIR / "results"
RESULTS_DIR = RESULTS_ROOT_DIR  # Repointed to a unique run folder by main.py.
PLOTS_DIR = RESULTS_DIR / "plots"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
EXCEL_DIR = RESULTS_DIR / "excel_reports"
CSV_DIR = RESULTS_DIR / "csv_reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SUMMARY_DIR = RESULTS_DIR / "summary"
LOG_DIR = ROOT_DIR / "logs"

def validate_config():
    """Stop early when a setting is unsupported or unsafe."""
    if CLASSIFICATION_TYPE != "binary":
        raise ValueError("CLASSIFICATION_TYPE must be 'binary' in this version.")
    if SVM_KERNEL not in {"linear", "rbf", "poly"}:
        raise ValueError("SVM_KERNEL must be 'linear', 'rbf', or 'poly'.")
    if SAMPLE_SIZE is not None and (not isinstance(SAMPLE_SIZE, int) or SAMPLE_SIZE < 2):
        raise ValueError("SAMPLE_SIZE must be None or an integer of at least 2.")
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be greater than 0 and less than 1.")
    if SVM_C <= 0:
        raise ValueError("SVM_C must be greater than 0.")
    if not (isinstance(SVM_GAMMA, str) and SVM_GAMMA in {"scale", "auto"}) and not (isinstance(SVM_GAMMA, (int, float)) and SVM_GAMMA > 0):
        raise ValueError("SVM_GAMMA must be 'scale', 'auto', or a positive number.")
    if not USE_ALL_FEATURES and not SELECTED_FEATURES:
        raise ValueError("Add columns to SELECTED_FEATURES when USE_ALL_FEATURES is False.")
    if any(kernel not in {"linear", "rbf"} for kernel in EXPERIMENT_KERNELS):
        raise ValueError("EXPERIMENT_KERNELS may contain only 'linear' and 'rbf'.")
    if any(value <= 0 for value in EXPERIMENT_C_VALUES):
        raise ValueError("Every EXPERIMENT_C_VALUES entry must be positive.")
    if EXPERIMENT_MAX_ROWS < 4 or SCRATCH_COMPARE_ROWS < 4 or QP_MAX_ROWS < 4:
        raise ValueError("Experiment row limits must be at least 4.")
    if len(EXPERIMENT_SAMPLE_SIZES) < 2 or any(size < 4 for size in EXPERIMENT_SAMPLE_SIZES):
        raise ValueError("EXPERIMENT_SAMPLE_SIZES must contain at least two sizes, each at least 4.")
    if SCRATCH_ITERATIONS < 1:
        raise ValueError("SCRATCH_ITERATIONS must be at least 1.")
    if SCRATCH_LEARNING_RATE <= 0:
        raise ValueError("SCRATCH_LEARNING_RATE must be positive.")
    if not 0 < PCA_EXPLAINED_VARIANCE_TARGET <= 1:
        raise ValueError("PCA_EXPLAINED_VARIANCE_TARGET must be greater than 0 and at most 1.")

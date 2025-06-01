"""
Configuration module

"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# Feature configuration path
FEATURE_CONFIG_PATH = CONFIG_DIR / "feature_config.yaml"

# Data
INPUT_DATA = DATA_DIR / "input_data.csv"
PROCESSED_DATA = DATA_DIR / "processed_visits_cleaned.csv"
FINAL_X_TRANFORMED = DATA_DIR / "final_X.npy"
FINAL_y = DATA_DIR / "final_y.npy"
FINAL_FEATURE_NAMES = DATA_DIR / "final_feature_names.npy"


# Models
CLASSIFICATION_MODEL = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "prediction_results.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
CLUSTER_METADATA_PATH = CONFIG_DIR / "cluster_metadata.yaml"

# Constants
SEED = 42
N_SPLITS = 5
N_CLUSTERS = 4


def ensure_directories():
    """
    Ensure that required project directories exist.
    Intended to be called at the top of any module that uses these paths.
    """
    for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

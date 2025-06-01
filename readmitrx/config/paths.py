"""
Configuration module

"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Feature configuration path
FEATURE_CONFIG_PATH = CONFIG_DIR / "feature_config.yaml"

# Data
INPUT_DATA = DATA_DIR / "input_data.csv"
PROCESSED_DATA = DATA_DIR / "processed_visits_cleaned.csv"
FINAL_X_TRANFORMED = DATA_DIR / "final_X.npy"
FINAL_y = DATA_DIR / "final_y.npy"

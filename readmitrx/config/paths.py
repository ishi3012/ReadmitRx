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

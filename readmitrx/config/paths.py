"""
Configuration module

"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent

# Feature configuration path
FEATURE_CONFIG_PATH = CONFIG_DIR / "feature_config.yaml"

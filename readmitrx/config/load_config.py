# readmitrx/config/load_config.py

"""
load_config.py — Utility to load model feature configuration from YAML.

Loads config from `readmitrx/config/feature_config.yaml` and parses feature groups
into a structured dictionary.

Returns:
    A dictionary with keys: 'num', 'cat', 'bin', 'text'

Example:
    >>> from readmitrx.config.load_config import load_feature_config
    >>> config = load_feature_config()
    >>> print(config["num"])
    ['age', 'risk_condition_count', 'sdoh_alcohol_risk']

Author: ReadmitRx Project Team (2025)
"""

import yaml
from pathlib import Path
from typing import Dict, List

from readmitrx.config.paths import FEATURE_CONFIG_PATH


def load_feature_config(
    config_path: Path = FEATURE_CONFIG_PATH,
) -> Dict[str, List[str]]:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        features = config.get("features", {})
        feature_dict = {
            "num": features.get("num__", []) or [],
            "cat": features.get("cat__", []) or [],
            "bin": features.get("bin__", []) or [],
            "text": features.get("text__", []) or [],
        }

    return feature_dict


if __name__ == "__main__":
    print(load_feature_config())

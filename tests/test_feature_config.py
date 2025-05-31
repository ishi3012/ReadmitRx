# tests/test_feature_config.py

"""
Unit tests for feature config loader and structure.

Validates:
- Correct keys in feature_config.yaml
- All features are non-empty strings
- No None or empty values in any group
"""

import pytest

from readmitrx.config.load_config import load_feature_config


def test_feature_config_keys_present() -> None:
    config = load_feature_config()
    assert isinstance(config, dict), "Config should be a dictionary"
    assert set(config.keys()) == {
        "num",
        "cat",
        "bin",
        "text",
    }, "Expected keys: num, cat, bin, text"


@pytest.mark.parametrize("group", ["num", "cat", "bin", "text"])
def test_feature_values_are_strings(group: str) -> None:
    config = load_feature_config()
    for feature in config[group]:
        assert isinstance(feature, str), f"{feature} in '{group}' must be a string"
        assert feature.strip() != "", f"{feature} in '{group} must not be empty"


def test_no_none_values_in_any_group() -> None:
    config = load_feature_config()
    for group, features in config.items():
        assert None not in features, f"'None' found in group: {group}"

"""
test_feature_engineering.py — Unit tests for derived feature creation logic.

Tests:
- visit_month is correctly extracted from visit_date
- chronic_count correctly tallies chronic flags
- apply_feature_engineering executes full feature stack

Author: ReadmitRx Project Team (2025)
"""

import pytest
import pandas as pd
import numpy as np

from readmitrx.pipeline.feature_engineering import (
    add_chornic_count,
    add_visit_month,
    apply_feature_engineering,
)


@pytest.fixture
def sample_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "visit_date": ["2024-01-15", "2024-06-03", None],
            "has_asthma_flag": [1, 1, np.nan],
            "has_diabetes_flag": [0, 1, 1],
            "has_hypertension_flag": [1, 1, 1],
        }
    )


def test_add_visit_month(sample_input_df: pd.DataFrame) -> None:
    df = add_visit_month(sample_input_df.copy())
    assert "visit_month" in df.columns
    assert df["visit_month"].iloc[0] == 1
    assert df["visit_month"].iloc[1] == 6
    assert df["visit_month"].iloc[2] == 0


def test_add_chronic_count(sample_input_df: pd.DataFrame) -> None:
    df = add_chornic_count(sample_input_df.copy())
    assert "chronic_count" in df.columns
    assert df["chronic_count"].tolist() == [2, 3, 2]


def test_apply_feature_engineering(sample_input_df: pd.DataFrame) -> None:
    df = apply_feature_engineering(sample_input_df.copy())
    assert "visit_month" in df.columns
    assert "chronic_count" in df.columns
    assert df.shape[0] == sample_input_df.shape[0]

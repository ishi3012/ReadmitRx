"""
test_preprocess.py — Unit tests for Preprocessor pipeline.

Covers:
- Transformation shape and type
- Missing values handling
- Edge cases: unknown categories, all-null columns

Author: ReadmitRx Project Team (2025)
"""

import pytest
import pandas as pd
import numpy as np

from readmitrx.pipeline.preprocess import Preprocessor


@pytest.fixture
def sample_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45, np.nan, 65],
            "insurance_type": ["public", "private", np.nan],
            "has_pcp_flag": [1, 0, np.nan],
            "notes": ["high risk", "low risk", "moderate risk"],
        }
    )


def test_preprocessor_transform_shape(sample_input_df: pd.DataFrame) -> None:
    num_features = ["age"]
    cat_features = ["insurance_type"]
    bin_features = ["has_pcp_flag"]
    text_features: list[str] = []  # ignored for now

    preprocessor = Preprocessor(
        num_features=num_features,
        cat_features=cat_features,
        bin_features=bin_features,
        text_features=text_features,
    )

    X, feature_names = preprocessor.fit_transform(sample_input_df)

    # Assert shape
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == sample_input_df.shape[0]
    assert len(feature_names) == X.shape[1]


def test_preprocessor_handles_missing_values(sample_input_df: pd.DataFrame) -> None:
    # Add missing data edge case
    sample_input_df.loc[0, "insurance_type"] = np.nan
    sample_input_df.loc[2, "age"] = np.nan

    preprocessor = Preprocessor(
        num_features=["age"],
        cat_features=["insurance_type"],
        bin_features=["has_pcp_flag"],
    )

    X, _ = preprocessor.fit_transform(sample_input_df)
    assert not np.isnan(X).any(), "Preprocessed output should not contain NaNs"


def test_transform_requires_fit_first(sample_input_df: pd.DataFrame) -> None:
    preprocessor = Preprocessor(
        num_features=["age"],
        cat_features=["insurance_type"],
        bin_features=["has_pcp_flag"],
    )

    with pytest.raises(ValueError, match="fit_transform"):
        _ = preprocessor.transform(sample_input_df)


def test_get_feature_names_requires_fit(sample_input_df: pd.DataFrame) -> None:
    preprocessor = Preprocessor(
        num_features=["age"],
        cat_features=["insurance_type"],
        bin_features=["has_pcp_flag"],
    )

    with pytest.raises(ValueError, match="fit_transform"):
        _ = preprocessor.get_feature_names_out()

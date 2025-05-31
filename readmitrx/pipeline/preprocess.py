"""
preprocess.py — Modular preprocessing pipeline for ReadmitRx.

This module builds a configurable, domain-agnostic sklearn pipeline that transforms
raw input features into model-ready format. Feature groups are passed explicitly or
derived from config.

Features:
- Supports numerical, categorical, binary, and text features
- Implements `ColumnTransformer` with prefix-based feature routing
- Returns NumPy arrays and preserves feature names for explainability

Intended Use:
- `build_column_transformer()`: returns sklearn ColumnTransformer based on feature groups
- `Preprocessor`: wrapper class for fit/transform and feature name extraction

Inputs:
- Pandas DataFrame with prefixed feature columns (e.g., num__age, cat__insurance_type)

Outputs:
- Transformed NumPy array + list of feature names

Functions:
- build_column_transformer(num, cat, bin, text) → ColumnTransformer
- Preprocessor.fit_transform(df) → (np.ndarray, List[str])
- Preprocessor.transform(df) → (np.ndarray, List[str])
- Preprocessor.get_feature_names_out() → List[str]

Example:
    >>> from readmitrx.pipeline.preprocess import Preprocessor
    >>> df = pd.read_csv("sinai_synthetic_data.csv")
    >>> preprocessor = Preprocessor()
    >>> X, feature_names = preprocessor.fit_transform(df)

Author: ReadmitRx Project Team (2025)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


from readmitrx.utils.logging import configure_logging

# Initialize logging
logger = configure_logging("preprocessor", "preprocessing.log")


def build_column_transformer(
    num_features: List[str],
    cat_features: List[str],
    bin_features: List[str],
    text_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for mixed-type feature processing.

    Args:
        num_features: List of numerical feature names (prefixed with 'num__')
        cat_features: List of categorical feature names (prefixed with 'cat__')
        bin_features: List of binary feature names (prefixed with 'bin__')
        text_features: Optional list of text feature names (prefixed with 'text__')

    Returns:
        sklearn.compose.ColumnTransformer
    """

    transformers = []

    if num_features:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("nums", num_pipeline, num_features))

    if cat_features:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_features))

    if bin_features:
        bin_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("passthrough", FunctionTransformer(validate=False)),
            ]
        )
        transformers.append(("bin", bin_pipeline, bin_features))

    if text_features:
        text_pipeline = Pipeline(
            [
                ("identity", FunctionTransformer(validate=False)),
            ]
        )
        transformers.append(("text", text_pipeline, text_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


class Preprocessor:
    """
    Preprocessor — Wrapper around sklearn ColumnTransformer with fit/transform support.

    Tracks transformed feature names for use in modeling and explainability (e.g., SHAP).
    """

    def __init__(
        self,
        num_features: List[str],
        cat_features: List[str],
        bin_features: List[str],
        text_features: Optional[List[str]] = None,
    ):
        logger.info("Initializing Preprocessor with features:")
        logger.info(f"  Numerical: {num_features}")
        logger.info(f"  Categorical: {cat_features}")
        logger.info(f"  Binary: {bin_features}")
        logger.info(f"  Text: {text_features if text_features else 'None'}")
        self.column_transformer = build_column_transformer(
            num_features, cat_features, bin_features, text_features
        )
        self._feature_names: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[NDArray[np.float64], List[str]]:
        """
        Fits the transformer and applies it to the input DataFrame.

        Args:
            df: Raw input DataFrame

        Returns:
            Tuple of (X_transformed, feature_names)
        """
        logger.info(f"Fitting and transforming data with shape: {df.shape}")
        X = self.column_transformer.fit_transform(df)
        try:
            self._feature_names = list(self.column_transformer.get_feature_names_out())
        except AttributeError:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            logger.warning(
                "Some transformers do not support get_feature_names_out(). "
                "Generated generic feature names."
            )

        logger.info(f"Output shape: {X.shape}")
        logger.info(f"First 5 transformed feature names: {self._feature_names[:5]}")
        return X, self._feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[NDArray[np.float64], List[str]]:
        """
        Applies transformation using previously fit ColumnTransformer.

        Args:
            df: Raw input DataFrame

        Returns:
            Tuple of (X_transformed, feature_names)
        """
        if self._feature_names is None:
            logger.error("transform() called before fit_transform().")
            raise ValueError("Must call fit_transform before transform.")
        logger.info(f"Transforming data with shape: {df.shape}")
        X = self.column_transformer.transform(df)
        logger.info(f"Output shape: {X.shape}")
        return X, self._feature_names

    def get_feature_names_out(self) -> List[str]:
        """
        Returns feature names from the latest transformation.

        Returns:
            List of transformed feature names
        """

        if self._feature_names is None:
            logger.error("get_feature_names_out() called before fit_transform().")
            raise ValueError("No features available — call fit_transform() first.")
        return self._feature_names

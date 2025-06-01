"""
preprocess.py — Modular preprocessing pipeline for ReadmitRx.

This module builds a configurable, domain-agnostic sklearn pipeline that transforms
raw input features into model-ready format. Feature groups are loaded dynamically
from a YAML config.

Features:
- Supports numerical, categorical, binary, and text features
- Implements `ColumnTransformer` with prefix-based routing
- Applies feature engineering before transformation
- Saves enriched DataFrame + transformed NumPy arrays
- Saves final feature names to `final_feature_names.npy`
- Logs all pipeline stages for reproducibility

Inputs:
- CSV: data/input_data.csv
- YAML config: config/feature_config.yaml

Outputs:
- Processed CSV: data/processed_visits.csv
- Transformed arrays: data/final_X.npy, data/final_y.npy
- Feature names: data/final_feature_names.npy

Author: ReadmitRx Project Team (2025)
"""

import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from readmitrx.utils.logging import configure_logging
from readmitrx.config.paths import (
    INPUT_DATA,
    PROCESSED_DATA,
    FEATURE_CONFIG_PATH,
    FINAL_X_TRANFORMED,
    FINAL_y,
)
from readmitrx.pipeline.feature_engineering import apply_feature_engineering

# Initialize logger
logger = configure_logging("preprocessor", "preprocessing.log")


def load_feature_config(
    config_path: str = str(FEATURE_CONFIG_PATH),
) -> Tuple[List[str], List[str], List[str], List[str]]:
    logger.info(f"Load features config file : {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features = config.get("features", {})
    num = features.get("num__", [])
    cat = features.get("cat__", [])
    bin_ = features.get("bin__", [])
    text = features.get("text__", [])
    logger.info(
        f"Loaded {len(num)} numerical, {len(cat)} categorical, {len(bin_)} binary features."
    )
    return num, cat, bin_, text


def build_column_transformer(
    num_features: List[str],
    cat_features: List[str],
    bin_features: List[str],
    text_features: Optional[List[str]] = None,
) -> ColumnTransformer:
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
        transformers=transformers, remainder="drop", verbose_feature_names_out=True
    )


def run_preprocessing_pipeline(
    input_csv: str = str(INPUT_DATA),
    processed_csv: str = str(PROCESSED_DATA),
    config_path: str = str(FEATURE_CONFIG_PATH),
) -> None:
    logger.info(f"Loading raw data from {input_csv}")
    df = pd.read_csv(input_csv)

    if "readmitted" not in df.columns:
        raise ValueError("Target column 'readmitted' not found in input.")

    logger.info(f"Initial shape: {df.shape}")
    df = apply_feature_engineering(df)
    df = clean_features(df)
    logger.info(f"Post feature-engineering shape: {df.shape}")

    X_df = df.drop(columns=["readmitted"])
    y = df["readmitted"].astype(int)

    num_features, cat_features, bin_features, text_features = load_feature_config(
        config_path
    )

    missing = [
        col
        for col in num_features + cat_features + bin_features
        if col not in X_df.columns
    ]
    if missing:
        raise ValueError(f"Missing required features in input data: {missing}")

    preprocessor = Preprocessor(num_features, cat_features, bin_features, text_features)
    X_transformed, feature_names = preprocessor.fit_transform(X_df)

    # Save outputs
    df.to_csv(processed_csv, index=False)
    np.save(FINAL_y, y.to_numpy())
    np.save(FINAL_X_TRANFORMED, X_transformed)
    np.save(
        FINAL_X_TRANFORMED.with_name("final_feature_names.npy"), np.array(feature_names)
    )

    logger.info(f"Saved enriched DataFrame to: {processed_csv}")
    logger.info(f"Saved transformed X to: {FINAL_X_TRANFORMED}")
    logger.info(f"Saved target y to: {FINAL_y}")
    logger.info(
        f"Saved feature names to: {FINAL_X_TRANFORMED.with_name('final_feature_names.npy')}"
    )
    logger.info("Preprocessing pipeline completed successfully.")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()]

    df.rename(
        columns={
            "healthedneeds_1.0": "healthneeds_1.0",
            "healthedneeds_0.0": "healthneeds_0.0",
            "healthedneeds_na": "healthneeds_na",
        },
        inplace=True,
    )

    contact_cols = [col for col in df.columns if col.startswith("sumContacts_")]
    if contact_cols:
        df["total_contacts"] = df[contact_cols].sum(axis=1)
        df["any_contacts_made"] = df["total_contacts"].gt(0).astype(int)
        df.drop(columns=contact_cols, inplace=True)

    numeric_df = df.select_dtypes(include=["number"])
    bin_cols = numeric_df.columns[numeric_df.max() <= 1]
    df[bin_cols] = df[bin_cols].astype(int)

    return df


class Preprocessor:
    def __init__(
        self,
        num_features: List[str],
        cat_features: List[str],
        bin_features: List[str],
        text_features: Optional[List[str]] = None,
    ) -> None:
        logger.info(
            f"Initializing Preprocessor with features: {num_features + cat_features + bin_features}"
        )
        self.column_transformer: ColumnTransformer = build_column_transformer(
            num_features, cat_features, bin_features, text_features
        )
        self._feature_names: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[NDArray[np.float64], List[str]]:
        logger.info(f"Fitting and transforming data with shape: {df.shape}")
        X = self.column_transformer.fit_transform(df)
        self._feature_names = []

        for name, transformer, cols in self.column_transformer.transformers:
            if name == "remainder":
                continue
            # Handle Pipeline
            if hasattr(transformer, "steps"):
                last_step = transformer.steps[-1][1]
                if hasattr(last_step, "get_feature_names_out"):
                    try:
                        names = last_step.get_feature_names_out(cols)
                    except Exception:
                        names = [f"{name}__{col}" for col in cols]
                else:
                    names = [f"{name}__{col}" for col in cols]
            elif hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                except Exception:
                    names = [f"{name}__{col}" for col in cols]
            else:
                names = [f"{name}__{col}" for col in cols]
            self._feature_names.extend(names)

        logger.info(f"Output shape: {X.shape}")
        logger.info(f"Transformed feature names: {self._feature_names}")
        return X, self._feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[NDArray[np.float64], List[str]]:
        if self._feature_names is None:
            raise ValueError("Must call fit_transform before transform.")
        X = self.column_transformer.transform(df)
        return X, self._feature_names

    def get_feature_names_out(self) -> List[str]:
        if self._feature_names is None:
            raise ValueError("No features available — call fit_transform() first.")
        return self._feature_names


if __name__ == "__main__":
    run_preprocessing_pipeline()

"""
feature_engineering.py — Derived feature creation pipeline for ReadmitRx.

This module contains reusable logic for deriving higher-level features from raw input data.
It supports both domain-agnostic and domain-informed transformation logic, allowing decoupling
from the preprocessing and model pipeline.

Features:
- Modular feature creation functions for date extraction and risk indicators
- Clean separation from sklearn preprocessing logic
- Logging-enabled for visibility during feature pipeline execution

Intended Use:
- Derive features such as:
    - `visit_month`: calendar month of referral
    - `has_chronic_condition`: flag if any chronic illness present
    - `chronic_count`: number of chronic conditions

Inputs:
- Raw Pandas DataFrame with source columns (e.g., `referral_date`, `has_diabetes_flag`)

Outputs:
- Enriched DataFrame with added derived features

Functions:
- add_visit_month(df)
- add_has_chronic_condition(df)
- add_chronic_count(df)
- apply_feature_engineering(df)

Example Usage:
    >>> df = pd.read_csv("sinai_synthetic_data.csv")
    >>> df = apply_feature_engineering(df)

Author: ReadmitRx Project Team (2025)
"""

import pandas as pd
from typing import List, Optional

from readmitrx.utils.logging import configure_logging

# Initialize logger
logger = configure_logging(
    log_name="feature_engineering", log_file="feature_engineering.log"
)


def add_visit_month(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Deriving feature: vist_month from visit_date")
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["visit_month"] = df["visit_date"].dt.month.fillna(0).astype(int)
    logger.debug(f"visit_month values: {df['visit_month'].unique().tolist()}")
    return df


def add_chornic_count(
    df: pd.DataFrame, chronic_fields: Optional[List[str]] = None
) -> pd.DataFrame:
    logger.info("Deriving feature: chronic_count")
    if chronic_fields is None:
        chronic_fields = [
            "has_asthma_flag",
            "has_diabetes_flag",
            "has_hypertension_flag",
        ]

        df["chronic_count"] = df[chronic_fields].fillna(0).astype(int).sum(axis=1)
        logger.debug(
            f"chronic_count stats: min={df['chronic_count'].min()}, max={df['chronic_count'].max()}"
        )
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying full feature engineering pipeline")
    df = add_visit_month(df)
    df = add_chornic_count(df)
    logger.info(
        f"Feature engineering complete. Columns now include : {list(df.columns)}"
    )
    return df

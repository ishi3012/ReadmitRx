"""
preprocess.py — Data cleaning and feature engineering for ReadmitRx pipeline.

This module transforms raw ED visit data into a cleaned format suitable
for ML model training and inference. It uses schema-aligned transformations
based on the `RawEDVisit` and `CleanedEDVisit` models.

Features:
- Parses dates and extracts visit month
- Derives chronic condition flags
- Creates binary target (`readmit_within_30`)
- Handles missing values and type conversions

Intended Use:
- Called in training and inference workflows before modeling

Inputs:
- DataFrame containing raw features from the de-identified ED dataset

Outputs:
- DataFrame compatible with `CleanedEDVisit` schema

Functions:
- preprocess_ed_visits(df: pd.DataFrame) -> pd.DataFrame

Example:
    >>> from readmitrx.pipeline.preprocess import preprocess_ed_visits
    >>> cleaned_df = preprocess_ed_visits(raw_df)

Author: ReadmitRx Project Team (2025)
"""

import logging
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_ed_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw ED visit data into a cleaned DataFrame.

    Args:
        df (pd.DataFrame): Raw ED dataset.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """

    df = df.copy()

    # Handle missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["sex_gender"] = df["sex_gender"].fillna("Unknown")
    df["language"] = df["language"].fillna("Unknown")

    # Processing chronic conditions
    chronic_cols = ["hypertension", "asthma", "diabetes"]
    for col in chronic_cols:
        df[col] = df[col].fillna(0).astype(bool)

    # Create has_chronic_condition flag
    df["has_chronic_condition"] = df[chronic_cols].any(axis=1)

    # Parse date and extract visit month
    df["referral_date"] = pd.to_datetime(df["referral_date"], errors="coerce")
    df["visit_month"] = df["referral_date"].dt.month

    # Derive binary readmission target
    df["readmit_within_30"] = df["day_readmit"].apply(
        lambda x: bool(x <= 30) if pd.notnull(x) else False
    )

    return df[
        [
            "record_id",
            "redcap_event_name",
            "new_patient",
            "age",
            "sex_gender",
            "latino",
            "language",
            "hypertension",
            "asthma",
            "diabetes",
            "referral_date",
            "day_readmit",
            "has_chronic_condition",
            "visit_month",
            "readmit_within_30",
        ]
    ]

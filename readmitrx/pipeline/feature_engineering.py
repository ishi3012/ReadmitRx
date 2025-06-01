"""
feature_engineering.py — Derived feature creation pipeline for ReadmitRx.

This module contains reusable logic for deriving higher-level features from raw input data.
It supports both domain-agnostic and domain-informed transformation logic, allowing decoupling
from the preprocessing and model pipeline.

Features:
- Modular feature creation functions for date extraction and risk indicators
- Clean separation from sklearn preprocessing logic
- Logging-enabled for visibility during feature pipeline execution

Author: ReadmitRx Project Team (2025)
"""

import pandas as pd
from typing import List, Optional, Dict
from readmitrx.utils.logging import configure_logging

logger = configure_logging("feature_engineering", "feature_engineering.log")


def add_visit_month(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Deriving feature: visit_month from visit_date")
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["visit_month"] = df["visit_date"].dt.month.fillna(0).astype(int)
    return df


def add_chronic_count(
    df: pd.DataFrame, chronic_fields: Optional[List[str]] = None
) -> pd.DataFrame:
    logger.info("Deriving feature: chronic_count")
    if chronic_fields is None:
        chronic_fields = ["asthma", "diabetes", "hypertension"]
    df["chronic_count"] = df[chronic_fields].fillna(0).astype(int).sum(axis=1)
    return df


def add_is_new_patient(df: pd.DataFrame) -> pd.DataFrame:
    required = ["new_patient_1", "new_patient_2", "new_patient_3"]
    if all(col in df.columns for col in required):
        df["is_new_patient"] = (df["new_patient_1"] == 1).astype(int)
        logger.info("Derived is_new_patient from new_patient_1")
    return df


def collapse_onehot_to_flag(
    df: pd.DataFrame,
    source_cols: List[str],
    new_col: str,
    positive_cols: List[str],
) -> pd.DataFrame:
    if all(col in df.columns for col in source_cols):
        if new_col in df.columns:
            logger.warning(f"Overwriting existing column: {new_col}")
        df[new_col] = df[positive_cols].fillna(0).astype(int).sum(axis=1).clip(upper=1)
        logger.info(f"Collapsed {source_cols} → {new_col}")
    else:
        missing = [col for col in source_cols if col not in df.columns]
        logger.warning(f"Skipping {new_col}; missing: {missing}")
    return df


def extract_risk_score(
    df: pd.DataFrame,
    source_cols: List[str],
    new_col: str,
    value_map: Dict[str, int],
) -> pd.DataFrame:
    if all(col in df.columns for col in source_cols):
        if new_col in df.columns:
            logger.warning(f"Overwriting existing column: {new_col}")
        df[new_col] = df[source_cols].fillna(0).astype(int).dot(pd.Series(value_map))
        logger.info(f"Extracted score: {new_col}")
    else:
        missing = [col for col in source_cols if col not in df.columns]
        logger.warning(f"Skipping {new_col}; missing: {missing}")
    return df


def add_socio_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    binary_flags: List[Dict[str, List[str] | str]] = [
        {
            "source_cols": ["sdoh_dv_0.0", "sdoh_dv_1.0", "sdoh_dv_na"],
            "positive_cols": ["sdoh_dv_1.0"],
            "new_col": "domestic_violence_risk",
        },
        {
            "source_cols": [
                "hiv_test_interest_0.0",
                "hiv_test_interest_1.0",
                "hiv_test_interest_na",
            ],
            "positive_cols": ["hiv_test_interest_1.0"],
            "new_col": "hiv_test_interest",
        },
        {
            "source_cols": [
                "covid_vax_signup_0.0",
                "covid_vax_signup_1.0",
                "covid_vax_signup_na",
            ],
            "positive_cols": ["covid_vax_signup_1.0"],
            "new_col": "covid_vaccine_signup",
        },
        {
            "source_cols": ["healthneeds_0.0", "healthneeds_1.0", "healthneeds_na"],
            "positive_cols": ["healthneeds_1.0"],
            "new_col": "health_education_needed",
        },
        {
            "source_cols": [
                "any_unmet_needs_0.0",
                "any_unmet_needs_1.0",
                "any_unmet_needs_na",
            ],
            "positive_cols": ["any_unmet_needs_1.0"],
            "new_col": "has_unmet_needs",
        },
        {
            "source_cols": [
                "referrals_yesno_0.0",
                "referrals_yesno_1.0",
                "referrals_yesno_na",
            ],
            "positive_cols": ["referrals_yesno_1.0"],
            "new_col": "referrals_made",
        },
        {
            "source_cols": [
                "healthedneeds_0.0",
                "healthedneeds_1.0",
                "healthedneeds_na",
            ],
            "positive_cols": ["healthedneeds_1.0"],
            "new_col": "health_education_needed",
        },
        {
            "source_cols": ["sdoh_pcp_0.0", "sdoh_pcp_1.0", "sdoh_pcp_na"],
            "positive_cols": ["sdoh_pcp_1.0"],
            "new_col": "has_pcp_flag",
        },
        {
            "source_cols": ["sdoh_ins_0.0", "sdoh_ins_1.0", "sdoh_ins_na"],
            "positive_cols": ["sdoh_ins_1.0"],
            "new_col": "has_insurance_flag",
        },
        {
            "source_cols": [
                "sdoh_housing_0.0",
                "sdoh_housing_1.0",
                "sdoh_housing_na",
                "sdoh_housing2_0.0",
                "sdoh_housing2_1.0",
                "sdoh_housing2_na",
            ],
            "positive_cols": ["sdoh_housing_1.0", "sdoh_housing2_1.0"],
            "new_col": "housing_insecure_flag",
        },
        {
            "source_cols": [
                "food_security_0.0",
                "food_security_1.0",
                "food_security_na",
            ],
            "positive_cols": ["food_security_1.0"],
            "new_col": "food_insecure_flag",
        },
        {
            "source_cols": ["sdoh_util_0.0", "sdoh_util_1.0", "sdoh_util_na"],
            "positive_cols": ["sdoh_util_1.0"],
            "new_col": "utility_insecure_flag",
        },
        {
            "source_cols": ["sdoh_trans_0.0", "sdoh_trans_1.0", "sdoh_trans_na"],
            "positive_cols": ["sdoh_trans_1.0"],
            "new_col": "transportation_insecure_flag",
        },
    ]

    for flag in binary_flags:
        df = collapse_onehot_to_flag(
            df,
            source_cols=flag["source_cols"],  # type: ignore[arg-type]
            new_col=flag["new_col"],  # type: ignore[arg-type]
            positive_cols=flag["positive_cols"],  # type: ignore[arg-type]
        )

    score_fields: List[Dict[str, object]] = [
        {
            "source_cols": ["sdoh_alc_1.0", "sdoh_alc_3.0"],
            "new_col": "sdoh_alcohol_risk",
            "value_map": {"sdoh_alc_1.0": 1, "sdoh_alc_3.0": 3},
        },
        {
            "source_cols": ["sdoh_substance_1.0", "sdoh_substance_2.0"],
            "new_col": "sdoh_substance_risk",
            "value_map": {"sdoh_substance_1.0": 1, "sdoh_substance_2.0": 2},
        },
        {
            "source_cols": ["sdoh_emotional_1.0", "sdoh_emotional_2.0"],
            "new_col": "sdoh_emotional_support_score",
            "value_map": {"sdoh_emotional_1.0": 1, "sdoh_emotional_2.0": 2},
        },
    ]

    for score in score_fields:
        df = extract_risk_score(
            df,
            source_cols=score["source_cols"],  # type: ignore[arg-type]
            new_col=score["new_col"],  # type: ignore[arg-type]
            value_map=score["value_map"],  # type: ignore[arg-type]
        )

    # Employment status
    if "sdoh_employ_1.0" in df.columns:
        df["is_employed_flag"] = df["sdoh_employ_1.0"].fillna(0).astype(int)
        logger.info("Derived is_employed_flag from sdoh_employ_1.0")

    return df


def add_is_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    risk_flags = [
        "health_education_needed",
        "has_unmet_needs",
        "referrals_made",
        "has_pcp_flag",
        "has_insurance_flag",
        "housing_insecure_flag",
        "food_insecure_flag",
        "utility_insecure_flag",
        "transportation_insecure_flag",
    ]

    for col in risk_flags:
        if col not in df.columns:
            logger.warning(f"Missing expected risk flag: {col}; defaulting to 0")
            df[col] = 0

    if "chronic_count" not in df.columns:
        logger.warning("Cannot compute is_high_risk: missing chronic_count")
        return df

    df["is_high_risk"] = (
        (df["chronic_count"] >= 2) | (df[risk_flags].sum(axis=1) > 0)
    ).astype(int)
    logger.info("Derived is_high_risk flag")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying full feature engineering pipeline")
    df = add_visit_month(df)
    df = add_chronic_count(df)
    df = add_is_new_patient(df)
    df = add_socio_behavioral_features(df)
    df = add_is_high_risk(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df

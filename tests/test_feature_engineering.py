"""
test_feature_engineering.py — Unit tests for derived feature creation logic.

Tests:
- visit_month extraction
- chronic_count calculation
- new patient flag
- one-hot to binary collapse
- ordinal score extraction
- is_high_risk derivation
- full apply_feature_engineering pipeline

Author: ReadmitRx Project Team (2025)
"""

import pandas as pd
import numpy as np

from readmitrx.pipeline.feature_engineering import (
    add_visit_month,
    add_chronic_count,
    add_is_new_patient,
    collapse_onehot_to_flag,
    extract_risk_score,
    add_socio_behavioral_features,
    add_is_high_risk,
    apply_feature_engineering,
)


def test_add_visit_month() -> None:
    df = pd.DataFrame({"visit_date": ["2024-01-01", "2024-06-15", None]})
    df = add_visit_month(df)
    assert "visit_month" in df.columns
    assert df["visit_month"].tolist() == [1, 6, 0]


def test_add_chronic_count() -> None:
    df = pd.DataFrame(
        {
            "asthma": [1, 0, np.nan],
            "diabetes": [1, 1, 0],
            "hypertension": [1, 0, 1],
        }
    )
    df = add_chronic_count(df)
    assert "chronic_count" in df.columns
    assert df["chronic_count"].tolist() == [3, 1, 1]


def test_add_is_new_patient() -> None:
    df = pd.DataFrame(
        {
            "new_patient_1": [1, 0],
            "new_patient_2": [0, 1],
            "new_patient_3": [0, 0],
        }
    )
    df = add_is_new_patient(df)
    assert "is_new_patient" in df.columns
    assert df["is_new_patient"].tolist() == [1, 0]


def test_collapse_onehot_to_flag() -> None:
    df = pd.DataFrame(
        {
            "a_0.0": [0, 1, 0],
            "a_1.0": [1, 0, 0],
            "a_na": [0, 0, 1],
        }
    )
    df = collapse_onehot_to_flag(
        df,
        source_cols=["a_0.0", "a_1.0", "a_na"],
        new_col="a_flag",
        positive_cols=["a_1.0"],
    )
    assert "a_flag" in df.columns
    assert df["a_flag"].tolist() == [1, 0, 0]


def test_extract_risk_score() -> None:
    df = pd.DataFrame(
        {
            "x_1.0": [1, 0, 0],
            "x_3.0": [0, 1, 0],
        }
    )
    df = extract_risk_score(
        df,
        source_cols=["x_1.0", "x_3.0"],
        new_col="x_score",
        value_map={"x_1.0": 1, "x_3.0": 3},
    )
    assert "x_score" in df.columns
    assert df["x_score"].tolist() == [1, 3, 0]


def test_add_socio_behavioral_features_binary() -> None:
    df = pd.DataFrame(
        {
            "sdoh_dv_1.0": [1],
            "sdoh_dv_0.0": [0],
            "sdoh_dv_na": [0],
            "healthedneeds_1.0": [1],
            "healthedneeds_0.0": [0],
            "healthedneeds_na": [0],
            "sdoh_pcp_1.0": [1],
            "sdoh_pcp_0.0": [0],
            "sdoh_pcp_na": [0],
            "sdoh_ins_1.0": [1],
            "sdoh_ins_0.0": [0],
            "sdoh_ins_na": [0],
            "food_security_1.0": [1],
            "food_security_0.0": [0],
            "food_security_na": [0],
            "sdoh_util_1.0": [1],
            "sdoh_util_0.0": [0],
            "sdoh_util_na": [0],
            "sdoh_trans_1.0": [1],
            "sdoh_trans_0.0": [0],
            "sdoh_trans_na": [0],
            "sdoh_housing_1.0": [1],
            "sdoh_housing2_1.0": [0],
            "sdoh_housing_0.0": [0],
            "sdoh_housing2_0.0": [0],
            "sdoh_housing_na": [0],
            "sdoh_housing2_na": [0],
            "any_unmet_needs_1.0": [1],
            "any_unmet_needs_0.0": [0],
            "any_unmet_needs_na": [0],
            "referrals_yesno_1.0": [1],
            "referrals_yesno_0.0": [0],
            "referrals_yesno_na": [0],
            "healthneeds_1.0": [1],
            "healthneeds_0.0": [0],
            "healthneeds_na": [0],
        }
    )
    df = add_socio_behavioral_features(df)
    expected_flags = [
        "domestic_violence_risk",
        "health_education_needed",
        "has_pcp_flag",
        "has_insurance_flag",
        "housing_insecure_flag",
        "food_insecure_flag",
        "utility_insecure_flag",
        "transportation_insecure_flag",
        "has_unmet_needs",
        "referrals_made",
    ]
    for col in expected_flags:
        assert col in df.columns
        assert df[col].iloc[0] == 1


def test_add_socio_behavioral_features_ordinal() -> None:
    df = pd.DataFrame(
        {
            "sdoh_alc_1.0": [1],
            "sdoh_alc_3.0": [0],
            "sdoh_substance_1.0": [0],
            "sdoh_substance_2.0": [1],
            "sdoh_emotional_1.0": [1],
            "sdoh_emotional_2.0": [0],
            "sdoh_employ_1.0": [1],
            "sdoh_employ_4.0": [0],
        }
    )
    df = add_socio_behavioral_features(df)
    assert df["sdoh_alcohol_risk"].iloc[0] == 1
    assert df["sdoh_substance_risk"].iloc[0] == 2
    assert df["sdoh_emotional_support_score"].iloc[0] == 1
    assert df["employment_status_code"].iloc[0] == 1


def test_add_is_high_risk() -> None:
    df = pd.DataFrame(
        {
            "chronic_count": [3],
            "health_education_needed": [0],
            "has_unmet_needs": [0],
            "referrals_made": [0],
            "has_pcp_flag": [0],
            "has_insurance_flag": [0],
            "housing_insecure_flag": [0],
            "food_insecure_flag": [0],
            "utility_insecure_flag": [0],
            "transportation_insecure_flag": [0],
        }
    )
    df = add_is_high_risk(df)
    assert "is_high_risk" in df.columns
    assert df["is_high_risk"].iloc[0] == 1


def test_apply_feature_engineering_pipeline_shape() -> None:
    df = pd.DataFrame(
        {
            "visit_date": ["2024-01-01"],
            "asthma": [1],
            "diabetes": [0],
            "hypertension": [1],
            "new_patient_1": [1],
            "new_patient_2": [0],
            "new_patient_3": [0],
            "sdoh_alc_1.0": [1],
            "sdoh_alc_3.0": [0],
            "sdoh_substance_1.0": [1],
            "sdoh_substance_2.0": [0],
            "sdoh_emotional_1.0": [1],
            "sdoh_emotional_2.0": [0],
            "sdoh_employ_1.0": [1],
            "sdoh_employ_4.0": [0],
            "healthedneeds_1.0": [1],
            "healthedneeds_0.0": [0],
            "healthedneeds_na": [0],
            "referrals_yesno_1.0": [1],
            "referrals_yesno_0.0": [0],
            "referrals_yesno_na": [0],
            "any_unmet_needs_1.0": [1],
            "any_unmet_needs_0.0": [0],
            "any_unmet_needs_na": [0],
            "sdoh_pcp_1.0": [1],
            "sdoh_pcp_0.0": [0],
            "sdoh_pcp_na": [0],
            "sdoh_ins_1.0": [1],
            "sdoh_ins_0.0": [0],
            "sdoh_ins_na": [0],
            "sdoh_housing_1.0": [1],
            "sdoh_housing2_1.0": [0],
            "sdoh_housing_0.0": [0],
            "sdoh_housing2_0.0": [0],
            "sdoh_housing_na": [0],
            "sdoh_housing2_na": [0],
            "food_security_1.0": [1],
            "food_security_0.0": [0],
            "food_security_na": [0],
            "sdoh_util_1.0": [1],
            "sdoh_util_0.0": [0],
            "sdoh_util_na": [0],
            "sdoh_trans_1.0": [1],
            "sdoh_trans_0.0": [0],
            "sdoh_trans_na": [0],
        }
    )

    df = apply_feature_engineering(df)
    assert "chronic_count" in df.columns
    assert "is_high_risk" in df.columns
    assert df.shape[0] == 1

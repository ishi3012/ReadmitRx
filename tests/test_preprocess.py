import pytest
import pandas as pd
from pathlib import Path
from readmitrx.pipeline.preprocess import (
    load_feature_config,
    clean_features,
    Preprocessor,
    run_preprocessing_pipeline,
)


def test_load_feature_config(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "features:\n  num__: [age]\n  cat__: [sex]\n  bin__: [has_pcp_flag]\n  text__: [notes]"
    )
    num, cat, bin_, text = load_feature_config(str(yaml_path))
    assert num == ["age"]
    assert cat == ["sex"]
    assert bin_ == ["has_pcp_flag"]
    assert text == ["notes"]


def test_clean_features_drops_and_renames() -> None:
    df = pd.DataFrame(
        {
            "healthedneeds_1.0": [1],
            "sumContacts_1": [1],
            "sumContacts_2": [2],
            "binary_col": [1.0],
        }
    )
    cleaned = clean_features(df)
    assert "healthneeds_1.0" in cleaned.columns
    assert "total_contacts" in cleaned.columns
    assert "any_contacts_made" in cleaned.columns
    assert cleaned["binary_col"].dtype == int


def test_transform_without_fit_raises_error() -> None:
    df = pd.DataFrame({"age": [30], "sex": ["female"], "has_pcp_flag": [1]})
    pre = Preprocessor(["age"], ["sex"], ["has_pcp_flag"])
    with pytest.raises(ValueError, match="fit_transform"):
        pre.transform(df)


def test_get_feature_names_out_without_fit_raises() -> None:
    pre = Preprocessor(["age"], ["sex"], ["has_pcp_flag"])
    with pytest.raises(ValueError, match="fit_transform"):
        pre.get_feature_names_out()


def test_run_preprocessing_pipeline_e2e(tmp_path: Path) -> None:
    input_csv = tmp_path / "input_data.csv"
    output_csv = tmp_path / "output_data.csv"
    config_yaml = tmp_path / "config.yaml"

    df = pd.DataFrame(
        {
            "visit_date": ["2024-01-01"],
            "asthma": [1],
            "diabetes": [1],
            "hypertension": [1],
            "new_patient_1": [1],
            "new_patient_2": [0],
            "new_patient_3": [0],
            "sdoh_pcp_1.0": [1],
            "sdoh_pcp_0.0": [0],
            "sdoh_pcp_na": [0],
            "readmitted": [1],
        }
    )
    df.to_csv(input_csv, index=False)
    config_yaml.write_text(
        "features:\n  num__: [chronic_count]\n  cat__: []\n  bin__: [has_pcp_flag]\n  text__: []"
    )

    run_preprocessing_pipeline(str(input_csv), str(output_csv), str(config_yaml))
    assert output_csv.exists()

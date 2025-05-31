"""
preprocess.py — Config-driven preprocessing pipeline for ReadmitRx.

Builds a modular, domain-agnostic sklearn pipeline that transforms raw input
features into model-ready format. Feature groups are loaded from YAML config
defined in `readmitrx/config/feature_config.yaml`.

Features:
- Handles numerical, categorical, binary, and text columns
- Uses ColumnTransformer and scikit-learn primitives
- Designed to be composed into larger training/inference pipelines

Inputs:
- Pandas DataFrame matching the RawVisit schema

Outputs:
- Unfitted sklearn Pipeline object with preprocessing logic

Example:
    >>> from readmitrx.pipeline.preprocess import build_preprocessing_pipeline
    >>> pipeline = build_preprocessing_pipeline()
    >>> X_transformed = pipeline.fit_transform(X_raw)

Author: ReadmitRx Project Team (2025)
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from readmitrx.config.load_config import load_feature_config


def build_preprocessing_pipeline() -> Pipeline:
    """
    Builds a ColumnTransformer-based sklearn pipeline using config-defined feature types.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline that transforms raw features into model-ready format
    """

    config = load_feature_config()

    num_features = config.get("num", [])
    cat_features = config.get("cat", [])
    bin_features = config.get("bin", [])
    text_features = config.get("text", [])

    transformers = []

    if num_features:
        num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="medium")), ("scalar", StandardScaler)]
        )
        transformers.append(("num", num_pipeline, num_features))

    if cat_features and cat_features[0] is not None:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_features))

    if bin_features:
        bin_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("identity", FunctionTransformer(validate=False)),
            ]
        )
        transformers.append(("bin", bin_pipeline, bin_features))

    if text_features and text_features[0] is not None:
        text_pipeline = Pipeline([("identity", FunctionTransformer(validate=False))])
        transformers.append(("text", text_pipeline, text_features))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])

    return pipeline


if __name__ == "__main__":
    print(build_preprocessing_pipeline())

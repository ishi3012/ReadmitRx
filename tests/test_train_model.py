import numpy as np
import pytest
from typing import Tuple, Dict, Type, Union
from numpy.typing import NDArray
from collections.abc import Generator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification

from readmitrx.pipeline.train_model import (
    train_model,
    compute_metrics,
    composite_score,
)


@pytest.fixture
def sample_data() -> (
    Generator[Tuple[NDArray[np.float64], NDArray[np.int_]], None, None]
):
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=42,
    )
    yield X, y


def test_compute_metrics_outputs_valid_keys(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.int_]],
) -> None:
    X, y = sample_data
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics: Dict[str, float] = compute_metrics(y, y_pred, y_prob)
    assert isinstance(metrics, dict)
    for key in ["roc_auc", "f1_score", "recall", "accuracy"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_composite_score_range() -> None:
    mock_metrics: Dict[str, float] = {
        "roc_auc": 0.6,
        "f1_score": 0.4,
        "recall": 0.5,
        "accuracy": 0.7,
    }
    score: float = composite_score(mock_metrics)
    assert 0.0 <= score <= 1.0
    assert round(score, 4) == round(0.4 * 0.6 + 0.3 * 0.4 + 0.2 * 0.5 + 0.1 * 0.7, 4)


@pytest.mark.parametrize(
    "model_cls", [LogisticRegression, RandomForestClassifier, CatBoostClassifier]
)
def test_train_model_pipeline_runs(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.int_]],
    model_cls: Type[ClassifierMixin],
) -> None:
    X, y = sample_data
    model: ClassifierMixin = (
        model_cls()
        if model_cls != LogisticRegression
        else model_cls(class_weight="balanced")
    )

    score: float
    metrics: Dict[str, float]
    trained_model: ClassifierMixin
    importances: Union[NDArray[np.float64], list[float]]

    score, metrics, trained_model, importances = train_model(
        X, y, model, name=model.__class__.__name__
    )

    assert isinstance(score, float)
    assert isinstance(metrics, dict)
    assert hasattr(trained_model, "fit")
    assert isinstance(importances, (list, np.ndarray))

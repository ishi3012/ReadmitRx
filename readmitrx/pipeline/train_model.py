"""
train_model.py — Model training pipeline for ReadmitRx.

This module trains multiple classifiers to predict 30-day ED readmission risk using
a preprocessed feature matrix and saves the best model by composite performance.

Features:
- Loads final_X.npy, final_y.npy, and final_feature_names.npy
- Trains LogisticRegression, RandomForest, and CatBoostClassifier
- Evaluates with Stratified K-Fold (default 5-fold)
- Logs ROC-AUC, F1, Recall, Accuracy
- Uses composite FAANG-style metric for ranking
- Saves best model, metrics, and feature importances

Inputs:
- data/final_X.npy
- data/final_y.npy
- data/final_feature_names.npy

Outputs:
- models/best_model.joblib
- metrics/model_metrics.json
- metrics/feature_importance.csv

Author: ReadmitRx Project Team (2025)
"""

import json
import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, cast
from numpy.typing import NDArray

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from sklearn.base import ClassifierMixin
from catboost import CatBoostClassifier  # type: ignore

from readmitrx.utils.logging import configure_logging
from readmitrx.config.paths import (
    FINAL_X_TRANFORMED,
    FINAL_y,
    FINAL_FEATURE_NAMES,
    CLASSIFICATION_MODEL,
    METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    SEED,
    N_SPLITS,
)

# Initialize logger
logger = configure_logging("training", "training.log")

# Ensure output directories exist
CLASSIFICATION_MODEL.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[NDArray[np.float64], NDArray[np.int_], List[str]]:
    """
    Load preprocessed feature matrix (X), target (y), and feature names.
    """
    X = np.load(FINAL_X_TRANFORMED)
    y = np.load(FINAL_y)
    features = np.load(FINAL_FEATURE_NAMES, allow_pickle=True).tolist()

    logger.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Loaded {len(features)} feature names")
    return X, y, features


def compute_metrics(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], y_prob: NDArray[np.float64]
) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1_score": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def composite_score(metrics: Dict[str, float]) -> float:
    return (
        0.4 * metrics["roc_auc"]
        + 0.3 * metrics["f1_score"]
        + 0.2 * metrics["recall"]
        + 0.1 * metrics["accuracy"]
    )


def train_model(
    X: NDArray[np.float64], y: NDArray[np.int_], model: ClassifierMixin, name: str
) -> Tuple[float, Dict[str, float], ClassifierMixin, NDArray[np.float64]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    importances = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        fold_metrics = compute_metrics(y_val, y_pred, y_prob)
        fold_score = composite_score(fold_metrics)
        scores.append((fold_score, fold_metrics))

        # Capture feature importances
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances.append(np.abs(model.coef_).flatten())

        logger.info(f"[{name}] Fold {i+1} composite score: {fold_score:.4f}")
        logger.info(f"[{name}] Fold {i+1} metrics: {fold_metrics}")

    avg_score = np.mean([s[0] for s in scores])
    avg_metrics = cast(
        Dict[str, float], pd.DataFrame([s[1] for s in scores]).mean().to_dict()
    )
    avg_importances = (
        np.mean(importances, axis=0).astype(np.float64)
        if importances
        else np.array([], dtype=np.float64)
    )
    return (
        float(avg_score),
        avg_metrics,
        model,
        cast(NDArray[np.float64], avg_importances),
    )


def main() -> None:
    all_model_results: dict[str, dict[str, float]] = {}
    X, y, feature_names = load_data()
    class_weight = {0: 1.0, 1: 5.0}
    models = {
        "logistic": LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight=class_weight
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=SEED, class_weight="balanced"
        ),
        "catboost": CatBoostClassifier(verbose=0, random_seed=SEED),
    }
    best_model: ClassifierMixin
    best_name: str
    best_score: float = -np.inf
    best_metrics: dict[str, float] = {}
    best_importances: NDArray[np.float64] = np.array([], dtype=np.float64)

    for name, model in models.items():
        logger.info(f"Training model: {name}")
        score, metrics, trained_model, importances = train_model(X, y, model, name)

        all_model_results[name] = {"composite_score": score, **metrics}

        if score > best_score:
            best_score = score
            best_model = trained_model
            best_metrics = metrics
            best_name = name
            best_importances = importances

    # Save model
    joblib.dump(best_model, CLASSIFICATION_MODEL)
    logger.info(f"Saved best model: {best_name} → {CLASSIFICATION_MODEL}")

    # Save all model metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(all_model_results, f, indent=2)
    logger.info(f"Saved all model metrics → {METRICS_PATH}")

    # Save feature importances
    if best_importances is not None and len(best_importances) > 0:
        if len(best_importances) != len(feature_names):
            logger.warning("Feature importance length mismatch. Skipping export.")
        else:
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": best_importances}
            ).sort_values("importance", ascending=False)
            importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
            logger.info(f"Saved feature importances → {FEATURE_IMPORTANCE_PATH}")

    logger.info(f"Best model: {best_name} | Composite Score: {best_score:.4f}")
    logger.info(f"Best model: {best_name} with metrics: {best_metrics}")


if __name__ == "__main__":
    main()

"""
train_model.py — Model training pipeline for binary readmission prediction.

This module loads preprocessed data, applies feature engineering and transformation,
trains a CatBoost classifier using 5-fold StratifiedKFold cross-validation, and saves
the best-performing model based on a composite score.

Features:
- Load CSV data and prepare X, y for modeling
- Apply feature engineering and column transformation
- Train CatBoostClassifier using StratifiedKFold
- Track and log ROC-AUC, F1-score, Recall, Accuracy per fold
- Save best model and final (X, y) for downstream use
- Designed to extend with LightGBM/XGBoost later

Intended Use:
- Called as a standalone script or imported from CLI/runner
- Used for first-pass evaluation of readmission risk models

Inputs:
- CSV file: `data/processed_visits.csv` (expects `readmitted` column)

Outputs:
- Saved model: `models/best_model.joblib`
- Final data: `data/final_X.npy`, `data/final_y.npy`
- Logged performance metrics and feature importances

Example:
    >>> from readmitrx.pipeline.train_model import train_and_save_model
    >>> train_and_save_model()

Author: ReadmitRx Project Team (2025)
"""

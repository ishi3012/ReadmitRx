"""
cluster.py — Modular clustering engine for ReadmitRx (KMeans + HDBSCAN support).

This module implements a pluggable unsupervised clustering framework that supports:
- KMeans for numerical feature spaces
- HDBSCAN for categorical or mixed encodings

Features:
- Dual-model training and model selection based on silhouette or cluster count
- Auto-encodes features for each model type using sklearn preprocessors
- Persists trained model to /models/cluster_model_{model_type}.joblib
- Adds a `cluster_id` column to the DataFrame
- Compatible with downstream routing and UMAP projection

Intended Use:
- Called during training (from `train_model.py`)
- Used for segmentation, explainability, or policy routing

Inputs:
- Cleaned and feature-engineered DataFrame (after preprocessing)

Outputs:
- Modified DataFrame with `.cluster_id` column
- Saved model artifact in /models

Functions:
- train_and_select_cluster_model(df, model_type)
- assign_clusters(df)
- generate_umap(df)

Author: ReadmitRx Project Team (2025)
"""

import os
import joblib
import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt

from typing import Literal
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from umap import UMAP

from readmitrx.utils.logging import configure_logging
from readmitrx.config.paths import (
    MODEL_DIR,
    DATA_DIR,
    ensure_directories,
    N_CLUSTERS,
    SEED,
    RESULTS_DIR,
)

# Initialize logger
logger = configure_logging("clustering", "clustering.log")

# Initialize models folder
ensure_directories()


def preprocess_for_kmeans(df: pd.DataFrame) -> np.ndarray:
    """
    Selects and scales numeric + boolean features for KMeans.

    Args:
        df (pd.DataFrame): Input DataFrame with mixed types.

    Returns:
        np.ndarray: Scaled numeric feature matrix.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64", "bool"]).copy()
    numeric_df = numeric_df.fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(numeric_df)


def preprocess_for_hdbscan(df: pd.DataFrame) -> np.ndarray:
    """
    Encodes categorical + boolean features for HDBSCAN using one-hot.

    Args:
        df (pd.DataFrame): Input DataFrame with mixed types.

    Returns:
        np.ndarray: One-hot encoded categorical feature matrix.
    """
    cat_df = df.select_dtypes(include=["object", "category", "bool"]).copy()
    cat_df = cat_df.fillna("missing").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return encoder.fit_transform(cat_df)


def train_select_cluster_model(
    df: pd.DataFrame, model_type: Literal["auto", "kmeans", "hdbscan"] = "auto"
) -> pd.DataFrame:
    """
    Trains one or both clustering models, selects best, saves it, and adds cluster_id to input.

    Args:
        df (pd.DataFrame): Cleaned and feature-engineered DataFrame.
        model_type (str): "auto", "kmeans", or "hdbscan".

    Returns:
        pd.DataFrame: Original DataFrame with appended `cluster_id` column.
    """
    # os.makedirs(MODELS_DIR, exist_ok=True)
    results: dict = {}

    # KMeans branch
    if model_type in ("auto", "kmeans"):
        X_kmeans = preprocess_for_kmeans(df)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init="auto")
        kmeans_labels = kmeans.fit_predict(X_kmeans)
        silhouette_kmeans = silhouette_score(X_kmeans, kmeans_labels)
        results["kmeans"] = {
            "model": kmeans,
            "labels": kmeans_labels,
            "score": silhouette_kmeans,
        }
        logger.info(f"KMeans: Silhouette Score = {silhouette_kmeans:.3f}")

    # HDBSCAN branch
    if model_type in ("auto", "hdbscan"):
        X_hdbscan = preprocess_for_hdbscan(df)
        hdb = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
        hdb_labels = hdb.fit_predict(X_hdbscan)
        cluster_count = len(np.unique(hdb_labels[hdb_labels >= 0]))
        results["hdbscan"] = {
            "model": hdb,
            "labels": hdb_labels,
            "score": cluster_count,
        }
        logger.info(f"HDBSCAN: Clusters found = {cluster_count}")

    # Model selection
    if model_type == "kmeans":
        best_key = "kmeans"
    elif model_type == "hdbscan":
        best_key = "hdbscan"
    else:
        # Auto mode: prefer KMeans if score is equal or better
        best_key = max(results, key=lambda k: results[k]["score"])

    best_model = results[best_key]["model"]
    best_labels = results[best_key]["labels"]

    model_path = os.path.join(MODEL_DIR, f"cluster_model_{best_key}.joblib")
    joblib.dump(best_model, model_path)

    logger.info(f"Saved {best_key.upper()} model to {model_path}")
    logger.info(f"Final clusters: {len(np.unique(best_labels))} — Model: {best_key}")

    df["cluster_id"] = best_labels
    return df


def assign_clusters(df: pd.DataFrame) -> np.ndarray:
    """
    Assigns cluster labels to new records using the saved model.

    Args:
        df (pd.DataFrame): New batch of cleaned feature data.

    Returns:
        np.ndarray: Array of cluster IDs (int or -1 if unassigned).
    """
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("cluster_model")]
    if not model_files:
        raise FileNotFoundError("No saved cluster model found in /models")

    model_file = model_files[0]
    model_path = os.path.join(MODEL_DIR, model_file)
    model = joblib.load(model_path)

    if "kmeans" in model_file:
        X = preprocess_for_kmeans(df)
        return model.predict(X)

    if "hdbscan" in model_file:
        X = preprocess_for_hdbscan(df)
        return model.labels_

    raise ValueError(f"Unrecognized model file: {model_file}")


def generate_umap(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save_path: Path = Path(DATA_DIR / "umap_projection.csv"),
) -> pd.DataFrame:
    """
    Projects high-dimensional data into 2D space for visualization using UMAP,
    and saves the projection + cluster_id to a CSV.

    Args:
        df (pd.DataFrame): Cleaned DataFrame with cluster_id.
        n_neighbors (int): UMAP neighbor count.
        min_dist (float): UMAP minimum distance.
        save_path (str): Output CSV path to save projection.

    Returns:
        pd.DataFrame: DataFrame with ['umap_x', 'umap_y'] columns added.
    """
    X: np.ndarray = preprocess_for_kmeans(df)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)
    umap_df = pd.DataFrame(embedding, columns=["umap_x", "umap_y"])
    df_out = pd.concat([df.reset_index(drop=True), umap_df], axis=1)

    # Save to disk
    df_out.to_csv(save_path, index=False)

    logger.info(f" Saved UMAP projection to {save_path}")
    return df_out


def plot_and_save_umap(
    df: pd.DataFrame, save_path: Path = Path(RESULTS_DIR / "umap_plot.png")
) -> None:

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        raise ValueError(
            "UMAP columns not found. Did you forget to call generate_umap()?"
        )

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["umap_x"], df["umap_y"], c=df["cluster_id"], cmap="tab10", alpha=0.7
    )
    plt.title("UMAP Projection by Cluster ID")
    plt.xlabel("UMAP X")
    plt.ylabel("UMAP Y")
    plt.colorbar(scatter, label="Cluster ID")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"UMAP plot saved to {save_path}")

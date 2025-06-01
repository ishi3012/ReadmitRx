import os
import pandas as pd
import numpy as np
import pytest

from readmitrx.cluster.cluster import (
    train_select_cluster_model,
    assign_clusters,
    generate_umap,
    MODEL_DIR,
)

TEST_DATA_PATH = "tests/test_data/sample_visits.csv"


@pytest.fixture
def test_df() -> pd.DataFrame:
    """Loads test data from a small cleaned CSV sample."""
    df = pd.read_csv(TEST_DATA_PATH)
    return df.sample(n=20, random_state=42) if len(df) > 20 else df


def test_train_model_saves_file(test_df) -> None:
    """Test that the model is saved after training."""
    df_with_clusters = train_select_cluster_model(test_df.copy())
    saved = any(f.startswith("cluster_model") for f in os.listdir(MODEL_DIR))
    assert saved, "Cluster model was not saved."
    assert "cluster_id" in df_with_clusters.columns


def test_assign_clusters_matches_shape(test_df) -> None:
    """Test cluster assignment shape matches input size."""
    labels = assign_clusters(test_df.copy())
    assert isinstance(labels, np.ndarray)
    assert labels.shape[0] == test_df.shape[0]


def test_umap_output_shape(test_df) -> None:
    """Test UMAP output adds exactly two columns."""
    df_umap = generate_umap(test_df.copy())
    assert "umap_x" in df_umap.columns
    assert "umap_y" in df_umap.columns
    assert df_umap.shape[0] == test_df.shape[0]


def test_cluster_ids_are_integers(test_df) -> None:
    """Ensure assigned cluster IDs are integers."""
    df_clustered = train_select_cluster_model(test_df.copy())
    assert pd.api.types.is_integer_dtype(df_clustered["cluster_id"])


def test_cluster_distribution_not_single_class(test_df) -> None:
    """Ensure clustering did not collapse into a single cluster."""
    df_clustered = train_select_cluster_model(test_df.copy())
    unique_clusters = df_clustered["cluster_id"].nunique()
    assert unique_clusters > 1, "All points assigned to the same cluster."

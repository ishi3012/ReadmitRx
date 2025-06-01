"""
labels.py — Cluster metadata loader for human-readable routing.

This module loads versioned metadata from cluster_metadata.yaml and
returns mappings from cluster IDs to routing labels or descriptions.

Author: ReadmitRx Project Team (2025)
"""

import yaml
from typing import Dict

from readmitrx.config.paths import CLUSTER_METADATA_PATH


def load_cluster_labels(version: str = "cluster_labels_v1") -> Dict[int, str]:
    """
    Load cluster_id → label mapping.

    Args:
        version (str): Which cluster label set to load.

    Returns:
        dict: Mapping {cluster_id: label_str}
    """
    with open(CLUSTER_METADATA_PATH, "r") as f:
        metadata = yaml.safe_load(f)

    return {int(cid): entry["label"] for cid, entry in metadata[version].items()}


def load_cluster_descriptions(version: str = "cluster_labels_v1") -> Dict[int, str]:
    """
    Load cluster_id → explanation mapping.

    Returns:
        dict: Mapping {cluster_id: description}
    """
    with open(CLUSTER_METADATA_PATH, "r") as f:
        metadata = yaml.safe_load(f)

    return {int(cid): entry["description"] for cid, entry in metadata[version].items()}


if __name__ == "__main__":
    labels = load_cluster_labels()
    print(labels)

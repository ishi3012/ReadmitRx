"""
policy.py — Metadata-driven routing policy using cluster labels.

Uses cluster ID + config metadata to return routing buckets like:
- "high_risk_patient_team"
- "behavioral_insecurity_team"
- "monitor_only"
- "low_priority"

Author: ReadmitRx Project Team (2025)
"""

from readmitrx.core.schemas import ClusteredVisit
from readmitrx.utils.labels import load_cluster_labels

CLUSTER_LABELS = load_cluster_labels(version="cluster_labels_v1")


def route_record(record: ClusteredVisit, risk_score: float) -> str:
    """
    Routes a patient to the CHW team or triage strategy based on cluster label.

    Args:
        record (ClusteredVisit): A clustered patient record
        risk_score (float): Predicted risk score (0–1)

    Returns:
        str: Routing label from cluster metadata (e.g. "high_risk_patient_team")
    """
    cluster_id = record.cluster_id if record.cluster_id is not None else -1
    return CLUSTER_LABELS.get(cluster_id, "low_priority")

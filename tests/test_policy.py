import pytest
from readmitrx.core.schemas import ClusteredVisit
from readmitrx.routing.policy import route_record
from readmitrx.utils.labels import load_cluster_labels

CLUSTER_LABELS = load_cluster_labels(version="cluster_labels_v1")


@pytest.mark.parametrize(
    "cluster_id,expected_label",
    [
        (0, "high_risk_patient_team"),
        (1, "low_priority"),
        (2, "behavioral_insecurity_team"),
        (3, "monitor_only"),
        (999, "low_priority"),  # unknown cluster fallback
    ],
)
def test_route_record_returns_expected_label(cluster_id, expected_label) -> None:
    record = ClusteredVisit(cluster_id=cluster_id)
    risk_score = 0.8
    result = route_record(record, risk_score)
    assert result == expected_label

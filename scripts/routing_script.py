import pandas as pd

from readmitrx.core.schemas import ClusteredVisit, PredictionResult
from readmitrx.routing.policy import route_record
from readmitrx.utils.labels import (
    load_cluster_labels,
    load_cluster_descriptions,
)

# === Load clustered input ===
df = pd.read_csv("data/clustered_visits.csv")

# === Load label metadata ===
CLUSTER_LABELS = load_cluster_labels()
CLUSTER_DESCRIPTIONS = load_cluster_descriptions()

# === Simulate risk_score column if missing ===
if "risk_score" not in df.columns:
    df["risk_score"] = 0.75  # default for testing

# === Generate routing decisions ===
results = []

for _, row in df.iterrows():
    record = ClusteredVisit(**row.to_dict())
    risk = float(row["risk_score"])
    label = route_record(record, risk)

    cluster_id = record.cluster_id if record.cluster_id is not None else -1
    cluster_label = CLUSTER_LABELS.get(cluster_id, "unlabeled")
    cluster_description = CLUSTER_DESCRIPTIONS.get(cluster_id, "N/A")

    result = PredictionResult(
        record_id=int(row.get("record_id", _)),
        risk_score=risk,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        cluster_description=cluster_description,
        resource_action_plan=label,
        followup_required=label != "low_priority",
        resource_notes=f"Routed to {label} (cluster {cluster_id})",
    )
    results.append(result.dict())


# === Save output ===
df_out = pd.DataFrame(results)
df_out.to_csv("data/routed_predictions.csv", index=False)
print("Routed predictions saved to data/routed_predictions.csv")

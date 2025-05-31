# tests/test_schemas.py

from readmitrx.core.schemas import (
    RawVisit,
    CleanedVisit,
    ClusteredVisit,
    PredictionResult,
)


def test_raw_visit_instantiation() -> None:
    visit = RawVisit(
        age=45,
        sex_female=True,
        race_black=False,
        language_non_english=False,
        hypertension=True,
        diabetes=False,
        asthma=True,
        total_contact_attempts=2,
        is_new_patient=True,
        housing_insecure_flag=True,
    )
    assert visit.age == 45
    assert visit.is_new_patient is True


def test_cleaned_visit_extends_raw() -> None:
    visit = CleanedVisit(
        age=60,
        hypertension=True,
        is_new_patient=False,
        total_contact_attempts=3,
        readmit_within_30=True,
        total_flags=2,
        is_high_risk=True,
    )
    assert visit.readmit_within_30
    assert visit.total_flags == 2


def test_clustered_visit_extends_cleaned() -> None:
    visit = ClusteredVisit(
        age=30,
        asthma=True,
        readmit_within_30=False,
        cluster_id=1,
    )
    assert visit.cluster_id == 1
    assert not visit.readmit_within_30


def test_prediction_result_schema() -> None:
    result = PredictionResult(
        record_id=123,
        risk_score=0.92,
        cluster_id=2,
        resource_action_plan="Follow up in 7 days",
        followup_required=True,
        resource_notes="Patient missed last appointment.",
    )
    assert result.risk_score > 0.9
    assert result.followup_required

"""
schemas.py — Domain-agnostic, typed data models for predictive ML pipelines.

This module defines reusable, Pydantic-based schemas for structured data exchange
across the ReadmitRx system. It supports input validation, feature engineering,
clustering, and prediction results — while remaining portable across verticals
such as healthcare, customer churn, and behavioral analytics.

Features:
- General-purpose names: risk_score, resource_action_plan, record_id
- Designed for ED readmission but extensible to churn or re-engagement domains
- Modular schema layers for raw ingestion, feature generation, clustering, and prediction
- Compatible with `mypy`, `pydantic`, and FastAPI validation

Intended Use:
- `RawVisit`: structured representation of raw input data
- `CleanedVisit`: after preprocessing and feature engineering
- `ClusteredVisit`: adds unsupervised cluster ID
- `PredictionResult`: model prediction + action plan fields

Example:
    >>> from readmitrx.core.schemas import RawVisit, PredictionResult
    >>> visit = RawVisit(age=54, race_black=True, hypertension=True)
    >>> result = PredictionResult(record_id=1, risk_score=0.87)

Author: ReadmitRx Project Team (2025)
"""

from pydantic import BaseModel
from typing import Optional


class RawVisit(BaseModel):
    """
    RawVisit — Domain-agnostic input schema representing raw visit data.

    This schema captures the raw input features from the source dataset
    in a generalized, reusable format suitable for both HealthTech and
    other industry applications (e.g., churn modeling, customer retention).

    Example:
        RawVisit(age=34, hypertension=True, total_contact_attempts=3)
    """

    # Demographics
    age: Optional[int] = None
    sex_female: Optional[bool] = None
    race_black: Optional[bool] = None
    race_hispanic: Optional[bool] = None
    race_other: Optional[bool] = None
    language_non_english: Optional[bool] = None

    # Clinical history
    has_comorbidity: Optional[bool] = None
    risk_condition_count: Optional[int] = None
    hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    asthma: Optional[bool] = None
    sdoh_alcohol_risk: Optional[int] = None
    sdoh_substance_risk: Optional[int] = None
    sdoh_emotional_support_score: Optional[int] = None

    # Visit metadata
    is_new_patient: Optional[bool] = None
    visit_date: Optional[str] = None

    # Contact & engagement
    total_contact_attempts: Optional[int] = None
    spoke_to_subject: Optional[bool] = None
    engaged: Optional[bool] = None
    referral_type_ed: Optional[bool] = None
    referral_type_high_risk: Optional[bool] = None
    referral_type_other: Optional[bool] = None
    time_spent_minutes: Optional[float] = None
    tme_spent_total: Optional[float] = None

    # Insurance and PCP
    insurance_na: Optional[bool] = None
    insurance_public: Optional[bool] = None
    insurance_uninsured: Optional[bool] = None
    has_pcp_flag: Optional[bool] = None
    has_insurance_flag: Optional[bool] = None

    # SDoH needs
    housing_insecure_flag: Optional[bool] = None
    housing_insecure_secondary: Optional[bool] = None
    fod_insecure_flag: Optional[bool] = None
    utility_insecure_flag: Optional[bool] = None
    transportation_barrier_flag: Optional[bool] = None
    employment_status_code: Optional[int] = None
    domestic_violence_risk: Optional[bool] = None
    hiv_test_interest: Optional[bool] = None
    covid_vaccine_signup: Optional[bool] = None
    health_education_needed: Optional[bool] = None
    sdoh_responded: Optional[bool] = None
    has_unmet_needs: Optional[bool] = None
    referrals_made: Optional[bool] = None


class CleanedVisit(RawVisit):
    """
    CleanedVisit — Schema after preprocessing and feature engineering.

    Extends:
        RawVisit

    Adds:
    - is_high_risk: Composite rule-based flag (optional)
    - total_flags: Count of triggered risk indicators (SDoH, comorbidities, etc.)
    """

    is_high_risk: Optional[bool] = None
    total_flags: Optional[int] = None


class ClusteredVisit(CleanedVisit):
    """
    ClusteredVisit — CleanedVisit schema plus clustering metadata.

    Extends:
        CleanedVisit

    Adds:
    - cluster_id: Segment/group assigned via unsupervised clustering
    """

    cluster_id: Optional[int] = None


class PredictionResult(BaseModel):
    """
    PredictionResult — Output schema for model inference + routing.

    Fields:
    - record_id: Unique identifier for the user or visit
    - risk_score: Predicted risk (e.g., of readmission, churn) between 0–1
    - cluster_id: (Optional) Cluster membership
    - resource_action_plan: Recommended intervention or action
    - followup_required: Whether follow-up is needed
    - resource_notes: Optional explanatory text for intervention

    Example:
        PredictionResult(record_id=101, risk_score=0.87, followup_required=True)
    """

    record_id: int
    risk_score: float
    cluster_id: Optional[int] = None
    resource_action_plan: Optional[str] = None
    followup_required: Optional[bool] = None
    resource_notes: Optional[str] = None

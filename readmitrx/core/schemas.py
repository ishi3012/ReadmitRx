"""
schemas.py — Typed data models for ED readmission prediction pipeline.

This module defines standardized data contracts using Pydantic models
for use across the preprocessing, modeling, clustering, and prediction
components of the ReadmitRx pipeline.

Features:
- Validates raw clinical and CHW fields from de-identified ED data
- Enables clear I/O structure for transformations and model input
- Supports future extension with derived features (e.g., cluster_id)

Intended Use:
- `RawEDVisit`: direct mapping from CSV input
- `CleanedEDVisit`: after preprocessing & feature engineering
- `ClusteredEDVisit`: after unsupervised grouping (e.g. KMeans)
- `PredictionResult`: model prediction + routing outcome

Inputs:
- Raw ED visit data (from Sinai de-identified Excel export)

Outputs:
- Typed Python objects used in downstream pipeline steps

Functions/Classes:
- RawEDVisit: minimally cleaned ED row
- CleanedEDVisit: engineered features for training/inference
- ClusteredEDVisit: adds unsupervised cluster ID
- PredictionResult: structured model prediction output

Args:
    record_id (int): Unique visit or patient identifier
    redcap_event_name (str): Longitudinal visit name (e.g., round_1_arm_1)
    new_patient (bool): Whether this was the patient's first encounter
    age (int): Patient's age
    sex_gender (str): Self-reported sex/gender field
    latino (int): Latino ethnicity flag (0 or 1)
    language (str): Primary language
    hypertension, asthma, diabetes (bool): Chronic condition flags
    referral_date (date): Referral intake date
    day_readmit (int): Days until readmission (target)

Example Usage:
    >>> from readmitrx.core.schemas import RawEDVisit
    >>> visit = RawEDVisit(record_id=1, age=54, diabetes=True)

Author: ReadmitRx Project Team (2025)
"""

from pydantic import BaseModel
from typing import Optional
from datetime import date


class RawEDVisit(BaseModel):
    record_id: int
    redcap_event_name: Optional[str]
    new_patient: Optional[bool]
    age: Optional[int]
    sex_gender: Optional[str]
    latino: Optional[str]
    language: Optional[str]
    hypertension: Optional[str]
    asthma: Optional[str]
    diabetes: Optional[str]
    referral_date: Optional[date]
    day_readmit: Optional[int]


class CleanedEDVisit(RawEDVisit):
    has_chronic_condition: Optional[bool]
    visit_month: Optional[int]
    readmit_within_30: Optional[bool]


class ClusteredEDVisist(CleanedEDVisit):
    cluster_id: Optional[int]


class PredictionResult(BaseModel):
    record_id: int
    readmit_probability: float
    cluster_id: Optional[int]
    chw_action: Optional[str]

from typing import Optional, List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_file: str
    decision_threshold: float
    selection_rule: str
    validation_metrics_at_threshold: dict
    test_metrics_at_frozen_threshold: dict
    feature_columns: List[str]


class RiskScoreResponse(BaseModel):
    transaction_id: str
    model_version: str
    fraud_probability: float
    risk_score: int
    risk_category: str


class ContributionItem(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str
    magnitude: float


class RiskExplainResponse(BaseModel):
    transaction_id: str
    model_version: str
    fraud_probability: float
    additivity_check_passed: bool
    header: str
    reasons: List[str]
    contributions: List[ContributionItem]


class RiskEvaluateResponse(BaseModel):
    request_id: str
    transaction_id: str
    model_version: str
    fraud_probability: float
    threshold: float
    risk_score: int
    risk_category: str
    action: str
    policy_rule_id: str
    policy_reason: str
    explanation_header: str
    reasons: List[str]
    timestamp: str
    audit_persisted: bool
    audit_error: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None

"""
Risk service: the ONLY place that wires together feature engineering -> model
inference -> risk scoring -> SHAP explanation -> decision engine for the API.

This module deliberately contains NO business/policy logic of its own. It:
  - builds a raw transaction DataFrame from validated Pydantic input (pure data
    marshalling, not a decision),
  - calls the EXISTING build_features() (Phase 1, unchanged),
  - calls the EXISTING RiskExplainer.explain() (Phase 6, unchanged),
  - calls the EXISTING build_explanation_text() (Phase 6, unchanged),
  - calls the EXISTING make_decision() (Phase 7, unchanged).
Every threshold, rule, and scoring formula is owned by those modules, imported
here, never re-implemented.
"""

import uuid
import pandas as pd

from ml.features.build_features import build_features, FEATURE_COLUMNS
from ml.evaluation.explainability import build_explanation_text
from ml.evaluation.decision_engine import make_decision, InvalidTransactionError
from ml.evaluation.policy import DECISION_THRESHOLD

from backend.services.model_loader import ModelBundle
from backend.schemas.transaction import RiskRequest


class FeatureGenerationError(RuntimeError):
    """Raised when the raw transaction context cannot be turned into features."""


def _build_feature_row(request: RiskRequest) -> pd.Series:
    """
    Converts the validated request (current transaction + optional prior history
    for the same customer) into the single engineered feature row for the
    transaction being scored, using the existing, unmodified feature pipeline.
    """
    all_txns = list(request.prior_transactions) + [request.transaction]
    raw_records = [t.model_dump() for t in all_txns]
    raw_df = pd.DataFrame(raw_records)

    try:
        features_df = build_features(raw_df)
    except Exception as e:
        raise FeatureGenerationError(f"Feature engineering failed: {e}") from e

    # the transaction being scored is the one with the matching transaction_id
    # (also guaranteed to be the latest timestamp for this customer, enforced by
    # RiskRequest validation)
    match = features_df[features_df["transaction_id"] == request.transaction.transaction_id]
    if len(match) != 1:
        raise FeatureGenerationError(
            f"Expected exactly 1 row for transaction_id="
            f"{request.transaction.transaction_id!r} after feature engineering, got {len(match)}"
        )
    return match.iloc[0]


def score_only(bundle: ModelBundle, request: RiskRequest) -> dict:
    """Probability + risk score/category. No SHAP, no decision, no audit write."""
    row = _build_feature_row(request)
    X = pd.DataFrame([row[FEATURE_COLUMNS].astype(float).values], columns=FEATURE_COLUMNS)
    probability = float(bundle.model.predict_proba(X)[0, 1])
    from ml.evaluation.risk_scoring import score_transaction
    risk = score_transaction(probability)
    return dict(
        transaction_id=request.transaction.transaction_id,
        model_version=bundle.metadata.get("model_name", "lgbm_v1"),
        **risk,
    )


def explain_only(bundle: ModelBundle, request: RiskRequest) -> dict:
    """Probability + full SHAP explanation. No decision, no audit write."""
    row = _build_feature_row(request)
    result = bundle.explainer.explain(row)
    explanation = build_explanation_text(result, decision_threshold=DECISION_THRESHOLD)
    return dict(
        transaction_id=request.transaction.transaction_id,
        model_version=bundle.metadata.get("model_name", "lgbm_v1"),
        fraud_probability=result["fraud_probability"],
        additivity_check_passed=result["additivity_check_passed"],
        header=explanation["header"],
        reasons=explanation["reasons"],
        contributions=result["contributions"],
    )


def evaluate_full(bundle: ModelBundle, request: RiskRequest):
    """
    Full pipeline: features -> inference -> SHAP -> decision engine.
    Returns (DecisionRecord, explanation_dict, request_id). Does NOT persist --
    persistence is the caller's (API route's) responsibility, so failure there
    can be handled without losing the already-computed decision.
    """
    request_id = str(uuid.uuid4())
    row = _build_feature_row(request)
    result = bundle.explainer.explain(row)
    explanation = build_explanation_text(result, decision_threshold=DECISION_THRESHOLD)

    decision = make_decision(
        transaction_id=request.transaction.transaction_id,
        model_probability=result["fraud_probability"],
        amount=request.transaction.amount,
        model_explanation=explanation["reasons"],
    )
    return decision, explanation, request_id

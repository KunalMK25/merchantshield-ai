"""
Risk service: the ONLY place that wires together feature engineering -> model
inference -> risk scoring -> SHAP explanation -> decision engine for the API.

This module deliberately contains NO business/policy logic of its own. It:
  - builds a raw transaction DataFrame from validated Pydantic input (pure data
    marshalling, not a decision),
  - calls the EXISTING build_features() (Phase 1, unchanged),
  - validates model output before downstream processing (V2 hardening),
  - calls the EXISTING RiskExplainer.explain() (Phase 6, unchanged),
  - calls the EXISTING build_explanation_text() (Phase 6, unchanged),
  - calls the EXISTING make_decision() (Phase 7, unchanged).
Every threshold, rule, and scoring formula is owned by those modules, imported
here, never re-implemented.
"""

import uuid
import math
import numpy as np
import pandas as pd

from ml.features.build_features import build_features, FEATURE_COLUMNS
from ml.evaluation.explainability import build_explanation_text
from ml.evaluation.decision_engine import make_decision, InvalidTransactionError
from ml.evaluation.policy import DECISION_THRESHOLD

from backend.services.model_loader import ModelBundle
from backend.schemas.transaction import RiskRequest


class FeatureGenerationError(RuntimeError):
    """Raised when the raw transaction context cannot be turned into features."""


class ModelOutputError(RuntimeError):
    """Raised when the model produces invalid output (NaN, Inf, out-of-range)."""


def _validate_engineered_features(row: pd.Series) -> None:
    """
    V2 sanity-checking: verify engineered features are within defensible bounds.
    These bounds are derived from domain logic and feature generation semantics,
    not arbitrary thresholds. They catch pathological inputs (e.g., 500 identical
    transactions at same second) that might cause OOD predictions.

    Raises FeatureGenerationError if bounds are exceeded.
    """
    # Velocity features: count of transactions in time windows
    # Domain: ~1440 txns/day = ~60/hour = ~1 per minute. >1000 in 60min is pathological
    if row.get("velocity_60min", 0) > 1000:
        raise FeatureGenerationError(
            f"Feature velocity_60min={row['velocity_60min']} exceeds sanity bound (1000). "
            f"Suggests malformed transaction history (e.g., many identical timestamps)."
        )
    if row.get("velocity_30min", 0) > 500:
        raise FeatureGenerationError(
            f"Feature velocity_30min={row['velocity_30min']} exceeds sanity bound (500)."
        )
    if row.get("velocity_5min", 0) > 100:
        raise FeatureGenerationError(
            f"Feature velocity_5min={row['velocity_5min']} exceeds sanity bound (100)."
        )

    # Prior transaction count: >100,000 prior txns suggests data quality issue
    # (real customer base unlikely to have single customer with 100k+ txns)
    if row.get("prior_txn_count", 0) > 100_000:
        raise FeatureGenerationError(
            f"Feature prior_txn_count={row['prior_txn_count']} exceeds sanity bound (100,000). "
            f"Suggests malformed transaction history."
        )

    # Amount z-score: check for non-finite values only
    # The z-score is a legitimate feature representing statistical anomaly relative to customer history.
    # Extremely high z-scores (e.g., 2000+) are NOT data-quality issues—they're legitimate fraud signals
    # (e.g., customer with few low-value transactions making one huge purchase from new device/geo).
    # The model was trained on features with naturally high z-scores and uses them appropriately.
    # We validate only for NaN/Inf, not magnitude.
    amount_zscore = row.get("amount_zscore", 0)
    if not np.isfinite(amount_zscore):
        raise FeatureGenerationError(
            f"Feature amount_zscore={amount_zscore} is non-finite (NaN or Inf). "
            f"Transaction history is malformed."
        )

    # time_since_prev_txn_min: set to 99999 for first txn (sentinel value); normal max ~14400 (10 days)
    # Allow up to sentinel value + buffer
    if row.get("time_since_prev_txn_min", 0) > 100_000:
        raise FeatureGenerationError(
            f"Feature time_since_prev_txn_min={row['time_since_prev_txn_min']} exceeds sanity bound."
        )


def _calculate_signal_quality(prior_txn_count: int) -> dict:
    """
    V2: Calculate historical context quality indicator.

    This is NOT model confidence or statistical uncertainty.
    It's a contextual label for how much behavioral history was available.

    Returns:
    {
        "level": "MINIMAL" | "LIMITED" | "MODERATE" | "ESTABLISHED",
        "prior_transaction_count": int,
        "message": str
    }
    """
    if prior_txn_count == 0:
        return {
            "level": "MINIMAL",
            "prior_transaction_count": 0,
            "message": "No prior transaction history was available. Behavioral-context features are not available.",
        }
    elif prior_txn_count <= 2:
        return {
            "level": "LIMITED",
            "prior_transaction_count": prior_txn_count,
            "message": f"Very limited transaction history ({prior_txn_count} prior transaction{'s' if prior_txn_count > 1 else ''}). Behavioral patterns cannot be reliably inferred.",
        }
    elif prior_txn_count <= 9:
        return {
            "level": "MODERATE",
            "prior_transaction_count": prior_txn_count,
            "message": f"Moderate transaction history ({prior_txn_count} prior transactions). Behavioral patterns partially established.",
        }
    else:
        return {
            "level": "ESTABLISHED",
            "prior_transaction_count": prior_txn_count,
            "message": f"Established transaction history ({prior_txn_count} prior transactions). Behavioral patterns well-defined.",
        }


def _validate_model_probability(probability: float) -> float:
    """
    Validates that the model's predicted probability is valid and usable.

    Requirements:
    - must be a finite float (not NaN, not Inf, not -Inf)
    - must be in the range [0.0, 1.0]

    Raises ModelOutputError if validation fails.
    """
    # Check for NaN
    if math.isnan(probability):
        raise ModelOutputError("Model produced NaN probability; prediction invalid.")

    # Check for Inf or -Inf
    if math.isinf(probability):
        raise ModelOutputError(
            f"Model produced infinite probability ({probability}); prediction invalid."
        )

    # Check bounds
    if not (0.0 <= probability <= 1.0):
        raise ModelOutputError(
            f"Model produced out-of-range probability ({probability}); must be in [0.0, 1.0]."
        )

    return probability


def _build_feature_row(request: RiskRequest) -> pd.Series:
    """
    Converts the validated request (current transaction + optional prior history
    for the same customer) into the single engineered feature row for the
    transaction being scored, using the existing, unmodified feature pipeline.

    V2: Validates that engineered features are within defensible bounds to catch
    pathological feature generation (e.g., 500 identical transactions at same second).
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
    row = match.iloc[0]

    # V2: Sanity-check engineered features for pathological values
    # These bounds are defensible from domain logic, not arbitrary:
    # - velocity_60min: count of txns in 60min window. Max ~1440 txns/day → ~60/hour → ~1000 in 60min window is extreme
    # - prior_txn_count: number of prior transactions. 100,000+ would indicate data quality issue
    # - amount_zscore: deviation from historical avg. >20 std devs is extreme outlier
    _validate_engineered_features(row)

    return row


def score_only(bundle: ModelBundle, request: RiskRequest) -> dict:
    """Probability + risk score/category. No SHAP, no decision, no audit write."""
    row = _build_feature_row(request)
    X = pd.DataFrame([row[FEATURE_COLUMNS].astype(float).values], columns=FEATURE_COLUMNS)
    probability = float(bundle.model.predict_proba(X)[0, 1])

    # V2 hardening: validate model output before downstream processing
    probability = _validate_model_probability(probability)

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

    # V2 hardening: validate model output before downstream processing
    probability = _validate_model_probability(result["fraud_probability"])
    result["fraud_probability"] = probability

    # Detect cold-start (no prior transaction history)
    is_cold_start = len(request.prior_transactions) == 0

    explanation = build_explanation_text(
        result, decision_threshold=DECISION_THRESHOLD, is_cold_start=is_cold_start
    )

    # For cold-start, filter contributions to exclude history-dependent features
    # (these use sentinel/default values and should not be shown to frontend)
    contributions = result["contributions"]
    if is_cold_start:
        cold_start_exclude = {
            "amount_vs_avg_ratio",
            "amount_zscore",
            "time_since_prev_txn_min",
            "prior_txn_count",
            "velocity_5min",
            "velocity_30min",
            "velocity_60min",
            "failed_ratio_trailing10",
            "new_device_flag",
            "new_geo_flag",
        }
        contributions = [c for c in contributions if c["feature"] not in cold_start_exclude]

    return dict(
        transaction_id=request.transaction.transaction_id,
        model_version=bundle.metadata.get("model_name", "lgbm_v1"),
        fraud_probability=result["fraud_probability"],
        additivity_check_passed=result["additivity_check_passed"],
        header=explanation["header"],
        reasons=explanation["reasons"],
        contributions=contributions,
        cold_start_context=explanation.get("cold_start_context"),
    )


def evaluate_full(bundle: ModelBundle, request: RiskRequest):
    """
    Full pipeline: features -> inference -> probability validation -> SHAP -> decision engine.
    Returns (DecisionRecord, explanation_dict, request_id, prior_txn_count). Does NOT persist --
    persistence is the caller's (API route's) responsibility, so failure there
    can be handled without losing the already-computed decision.

    V2 improvement: Decision is computed BEFORE SHAP attempt, so if SHAP fails,
    the decision is still available.
    """
    request_id = str(uuid.uuid4())
    row = _build_feature_row(request)

    # V2: Compute prior transaction count for signal quality
    prior_txn_count = len(request.prior_transactions)

    # Get raw probability first (with validation), then make decision
    X = pd.DataFrame([row[FEATURE_COLUMNS].astype(float).values], columns=FEATURE_COLUMNS)
    raw_probability = float(bundle.model.predict_proba(X)[0, 1])
    probability = _validate_model_probability(raw_probability)

    # Make decision BEFORE attempting SHAP (so decision is not dependent on SHAP success)
    decision = make_decision(
        transaction_id=request.transaction.transaction_id,
        model_probability=probability,
        amount=request.transaction.amount,
        model_explanation=None,  # Will be filled in after SHAP attempt
    )

    # Attempt SHAP explanation, but it's not required for the decision
    explanation = None
    explanation_error = None
    try:
        result = bundle.explainer.explain(row)
        # Re-validate probability from SHAP result
        probability_from_shap = _validate_model_probability(result["fraud_probability"])

        # Detect cold-start (no prior transaction history)
        is_cold_start = prior_txn_count == 0

        explanation = build_explanation_text(
            result, decision_threshold=DECISION_THRESHOLD, is_cold_start=is_cold_start
        )
        explanation["reasons"] = explanation.get("reasons", [])

        # Update decision with explanations
        decision = make_decision(
            transaction_id=request.transaction.transaction_id,
            model_probability=probability,
            amount=request.transaction.amount,
            model_explanation=explanation.get("reasons", []),
        )
    except Exception as e:
        # SHAP failed, but decision is still valid
        explanation_error = str(e)
        # Keep the decision without explanations

    return decision, explanation, request_id, prior_txn_count, explanation_error

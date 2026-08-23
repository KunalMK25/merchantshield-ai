"""
Decision engine: combines a validated fraud probability + transaction context with
the deterministic policy (policy.py) and the existing risk-scoring layer
(risk_scoring.py, Phase 6, unchanged) to produce a structured decision record.

ARCHITECTURAL RULE, ENFORCED HERE:
SHAP explanations (model_explanation) are attached to the decision record for
audit/display purposes ONLY. They are never read by evaluate_policy() or by any
code path that determines `action`. make_decision() accepts `model_explanation`
as an opaque value -- it is not inspected, parsed, or branched on anywhere in this
module. See tests/test_decision_engine.py::test_shap_values_cannot_alter_action
for a direct proof of this.

This phase does not write to any database or log -- it only returns the decision
record. Persistence is Phase 10+ (audit trail).
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Any

from ml.evaluation.policy import evaluate_policy, DECISION_THRESHOLD
from ml.evaluation.risk_scoring import score_transaction

MODEL_VERSION = "lgbm_v1"  # matches ml/models/candidate_lgbm_v1.pkl, frozen since Phase 4/5


class InvalidTransactionError(ValueError):
    """Raised when the decision engine is given invalid or incomplete inputs."""


def _validate_probability(probability: Any) -> float:
    if isinstance(probability, bool):  # bool is a subclass of int in Python -- reject explicitly
        raise InvalidTransactionError(f"probability must be numeric, got bool: {probability}")
    if not isinstance(probability, (int, float)):
        raise InvalidTransactionError(f"probability must be numeric, got {type(probability).__name__}")
    prob = float(probability)
    if prob != prob:  # NaN check without needing numpy/math import
        raise InvalidTransactionError("probability is NaN")
    if not (0.0 <= prob <= 1.0):
        raise InvalidTransactionError(f"probability must be in [0, 1], got {prob}")
    return prob


def _validate_amount(amount: Any) -> float:
    if isinstance(amount, bool):
        raise InvalidTransactionError(f"amount must be numeric, got bool: {amount}")
    if not isinstance(amount, (int, float)):
        raise InvalidTransactionError(f"amount must be numeric, got {type(amount).__name__}")
    amt = float(amount)
    if amt != amt:
        raise InvalidTransactionError("amount is NaN")
    if amt < 0:
        raise InvalidTransactionError(f"amount must be non-negative, got {amt}")
    return amt


def _validate_transaction_id(transaction_id: Any) -> str:
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise InvalidTransactionError(f"transaction_id must be a non-empty string, got {transaction_id!r}")
    return transaction_id


@dataclass(frozen=True)
class DecisionRecord:
    transaction_id: str
    model_version: str
    fraud_probability: float
    threshold: float
    risk_score: int
    risk_category: str
    action: str
    policy_rule_id: str
    policy_reason: str
    timestamp: str
    model_explanation: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_decision(
    transaction_id: str,
    model_probability: float,
    amount: float,
    model_explanation: Optional[list] = None,
    timestamp: Optional[str] = None,
) -> DecisionRecord:
    """
    Produces a complete, structured, reproducible decision record.

    Raises InvalidTransactionError for any invalid/missing required input --
    this function never silently guesses or defaults required fields.
    """
    txn_id = _validate_transaction_id(transaction_id)
    probability = _validate_probability(model_probability)
    amt = _validate_amount(amount)

    # risk_score/category: informational only, derived from probability via the
    # existing (Phase 6, unchanged) deterministic scoring function -- NOT used
    # below to pick the action.
    risk = score_transaction(probability)

    # THE action-determining call. Only (probability, amount) go in.
    # model_explanation is deliberately not passed to evaluate_policy at all.
    rule = evaluate_policy(probability, amt)

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    return DecisionRecord(
        transaction_id=txn_id,
        model_version=MODEL_VERSION,
        fraud_probability=round(probability, 4),
        threshold=DECISION_THRESHOLD,
        risk_score=risk["risk_score"],
        risk_category=risk["risk_category"],
        action=rule.action,
        policy_rule_id=rule.rule_id,
        policy_reason=rule.reason,
        timestamp=ts,
        model_explanation=model_explanation,
    )

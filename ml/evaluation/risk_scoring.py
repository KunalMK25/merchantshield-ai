"""
Risk scoring layer.

DELIBERATELY NOT ML: this module takes a probability (from the LightGBM model) and
applies a fixed, documented, auditable transform to get a 0-100 risk score and a
category label. Nothing here is learned or fit to data -- it's business logic that
a risk analyst can read, question, and change without retraining anything.

NOTE ON RELATIONSHIP TO THE DECISION THRESHOLD (0.40, from Phase 5):
The 0-100 score/category shown here is a continuous, human-readable risk indicator.
The binary "flag vs allow" decision used elsewhere in the system uses the frozen
cost-optimal probability threshold (0.40) directly, NOT the score buckets below.
A transaction can show as MEDIUM risk (score 40-60) and still be on either side of
the 0.40 probability decision boundary -- the score buckets are for human readability
in the dashboard/audit trail, the decision threshold is for the actual bounded action.
This separation is intentional and documented further in docs/explainability.md.
"""

from dataclasses import dataclass, field


@dataclass
class RiskScoreConfig:
    low_max: int = 30
    medium_max: int = 60
    high_max: int = 80
    # anything above high_max is CRITICAL


def probability_to_score(probability: float) -> int:
    """Linear 0-1 -> 0-100 mapping, rounded. Simple and auditable on purpose."""
    probability = max(0.0, min(1.0, float(probability)))
    return int(round(probability * 100))


def score_to_category(score: int, config: RiskScoreConfig = RiskScoreConfig()) -> str:
    if score <= config.low_max:
        return "LOW"
    elif score <= config.medium_max:
        return "MEDIUM"
    elif score <= config.high_max:
        return "HIGH"
    else:
        return "CRITICAL"


def score_transaction(probability: float, config: RiskScoreConfig = RiskScoreConfig()) -> dict:
    score = probability_to_score(probability)
    category = score_to_category(score, config)
    return dict(fraud_probability=round(float(probability), 4), risk_score=score, risk_category=category)

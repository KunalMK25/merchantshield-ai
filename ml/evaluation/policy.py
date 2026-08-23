"""
Policy configuration for the MerchantShield decision engine.

SOURCE OF TRUTH: raw fraud_probability from the LightGBM model, plus minimal
transaction context (amount). The 0-100 risk_score / LOW-MEDIUM-HIGH-CRITICAL
category from risk_scoring.py (Phase 6) is computed and attached to every decision
record for human readability, but rules below are keyed on PROBABILITY, never on
the score bucket -- this is deliberate (see docs/decision_engine.md) so that
changing score-bucket display boundaries can never silently change what action is
taken.

DECISION_THRESHOLD (0.40) is the frozen, cost-optimized boundary from Phase 5. It is
NOT re-derived here and must not be changed without rerunning that analysis.

CRITICAL_MIN (0.80) and the amount-escalation cutoff are additional POLICY
assumptions layered on top of the frozen threshold, for splitting the "friction"
region into "step up verification" vs "block outright". CRITICAL_MIN is chosen
using evidence already collected in Phase 5's threshold sweep: at probability 0.80,
validation precision was 0.941 (see ml/models/threshold_sweep_validation.csv) --
i.e. when the model is this confident, the overwhelming majority of what it flags
really is fraud, which is the bar we want before recommending an outright block
rather than a lower-friction step-up check. This is a policy assumption, not a
cost-model re-optimization, and is labeled as such.
"""

from dataclasses import dataclass
from typing import Callable

# Frozen from Phase 5 -- do not change without rerunning threshold_analysis.py
DECISION_THRESHOLD = 0.40

# Policy assumptions, layered on top of the frozen threshold (see module docstring)
LOW_MAX = 0.15                 # below this, no friction of any kind
CRITICAL_MIN = 0.80            # at/above this, recommend BLOCK instead of step-up
LARGE_AMOUNT_CUTOFF = 25_000.0  # INR; escalates medium-probability transactions

ALLOWED_ACTIONS = {"ALLOW", "ALLOW_WITH_MONITORING", "STEP_UP_VERIFICATION", "BLOCK"}


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    action: str
    reason: str
    condition: Callable[[float, float], bool]  # (probability, amount) -> bool


# Rules are evaluated IN ORDER; the first matching rule wins. This list is the
# entire policy -- nothing about action selection lives anywhere else.
POLICY_RULES = [
    PolicyRule(
        rule_id="CRITICAL_BLOCK",
        action="BLOCK",
        reason=(
            f"Fraud probability is at or above the configured blocking boundary "
            f"({CRITICAL_MIN}), where validation precision is high enough that most "
            f"flagged transactions at this confidence level are genuinely fraudulent."
        ),
        condition=lambda p, amt: p >= CRITICAL_MIN,
    ),
    PolicyRule(
        rule_id="HIGH_STEP_UP",
        action="STEP_UP_VERIFICATION",
        reason=(
            f"Fraud probability is at or above the frozen cost-optimal decision "
            f"threshold ({DECISION_THRESHOLD}), warranting additional verification "
            f"before the transaction proceeds."
        ),
        condition=lambda p, amt: p >= DECISION_THRESHOLD,
    ),
    PolicyRule(
        rule_id="MEDIUM_AMOUNT_ESCALATION",
        action="STEP_UP_VERIFICATION",
        reason=(
            f"Fraud probability is below the decision threshold but the transaction "
            f"amount (>= Rs{LARGE_AMOUNT_CUTOFF:,.0f}) is large enough that the "
            f"potential downside justifies additional verification despite the "
            f"lower model confidence."
        ),
        condition=lambda p, amt: p >= LOW_MAX and amt >= LARGE_AMOUNT_CUTOFF,
    ),
    PolicyRule(
        rule_id="MEDIUM_MONITOR",
        action="ALLOW_WITH_MONITORING",
        reason=(
            f"Fraud probability is elevated above baseline ({LOW_MAX}) but below "
            f"the decision threshold ({DECISION_THRESHOLD}); transaction is allowed "
            f"but flagged for monitoring rather than immediate friction."
        ),
        condition=lambda p, amt: p >= LOW_MAX,
    ),
    PolicyRule(
        rule_id="LOW_ALLOW",
        action="ALLOW",
        reason=f"Fraud probability is below {LOW_MAX}; no additional friction is warranted.",
        condition=lambda p, amt: True,  # catch-all -- always matches if nothing else did
    ),
]


def evaluate_policy(probability: float, amount: float) -> PolicyRule:
    """
    Returns the first matching PolicyRule for the given (probability, amount).
    Deterministic: same inputs always produce the same rule. Fails safe if, due to
    a future bug, no rule matches or an unrecognized action is produced.
    """
    for rule in POLICY_RULES:
        if rule.condition(probability, amount):
            if rule.action not in ALLOWED_ACTIONS:
                raise RuntimeError(
                    f"Policy rule '{rule.rule_id}' produced an unrecognized action "
                    f"'{rule.action}' -- failing safe rather than taking an undefined action."
                )
            return rule
    # LOW_ALLOW is a catch-all and should always match; reaching here means a bug.
    raise RuntimeError(
        f"No policy rule matched probability={probability}, amount={amount}. "
        f"Failing safe -- no action can be determined."
    )

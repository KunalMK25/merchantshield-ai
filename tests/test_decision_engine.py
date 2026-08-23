import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json

from ml.evaluation.policy import evaluate_policy, DECISION_THRESHOLD, LOW_MAX, CRITICAL_MIN, LARGE_AMOUNT_CUTOFF
from ml.evaluation.decision_engine import make_decision, InvalidTransactionError, DecisionRecord, MODEL_VERSION

SMALL_AMOUNT = 500.0
LARGE_AMOUNT = 30_000.0  # above LARGE_AMOUNT_CUTOFF


# ---------------------------------------------------------------------------
# 1-4: One representative case per category
# ---------------------------------------------------------------------------

def test_low_probability_allows():
    d = make_decision("txn_low", 0.05, SMALL_AMOUNT)
    assert d.action == "ALLOW"
    assert d.policy_rule_id == "LOW_ALLOW"
    assert d.risk_category == "LOW"


def test_medium_probability_allows_with_monitoring():
    # p=0.35 chosen so BOTH the policy rule (MEDIUM_MONITOR, since 0.15<=p<0.40)
    # AND the independent risk_score bucket (31-60=MEDIUM) happen to agree here --
    # see test_risk_category_and_policy_action_are_independent for a case where
    # they deliberately do NOT agree, which is expected behavior, not a bug.
    d = make_decision("txn_med", 0.35, SMALL_AMOUNT)
    assert d.action == "ALLOW_WITH_MONITORING"
    assert d.policy_rule_id == "MEDIUM_MONITOR"
    assert d.risk_category == "MEDIUM"


def test_high_probability_steps_up_verification():
    d = make_decision("txn_high", 0.65, SMALL_AMOUNT)
    assert d.action == "STEP_UP_VERIFICATION"
    assert d.policy_rule_id == "HIGH_STEP_UP"
    assert d.risk_category == "HIGH"


def test_critical_probability_blocks():
    d = make_decision("txn_critical", 0.95, SMALL_AMOUNT)
    assert d.action == "BLOCK"
    assert d.policy_rule_id == "CRITICAL_BLOCK"
    assert d.risk_category == "CRITICAL"


# ---------------------------------------------------------------------------
# 5-6: Threshold boundary behavior
# ---------------------------------------------------------------------------

def test_exactly_at_decision_threshold_is_step_up():
    # >= semantics: exactly 0.40 must match HIGH_STEP_UP, not MEDIUM
    d = make_decision("txn_boundary", DECISION_THRESHOLD, SMALL_AMOUNT)
    assert d.action == "STEP_UP_VERIFICATION"
    assert d.policy_rule_id == "HIGH_STEP_UP"


def test_just_below_decision_threshold_is_medium():
    d = make_decision("txn_below", DECISION_THRESHOLD - 0.001, SMALL_AMOUNT)
    assert d.action == "ALLOW_WITH_MONITORING"
    assert d.policy_rule_id == "MEDIUM_MONITOR"


def test_just_above_decision_threshold_is_step_up():
    d = make_decision("txn_above", DECISION_THRESHOLD + 0.001, SMALL_AMOUNT)
    assert d.action == "STEP_UP_VERIFICATION"
    assert d.policy_rule_id == "HIGH_STEP_UP"


def test_exactly_at_critical_min_is_block():
    d = make_decision("txn_crit_boundary", CRITICAL_MIN, SMALL_AMOUNT)
    assert d.action == "BLOCK"
    assert d.policy_rule_id == "CRITICAL_BLOCK"


def test_just_below_critical_min_is_step_up_not_block():
    d = make_decision("txn_crit_below", CRITICAL_MIN - 0.001, SMALL_AMOUNT)
    assert d.action == "STEP_UP_VERIFICATION"
    assert d.policy_rule_id == "HIGH_STEP_UP"


def test_exactly_at_low_max_is_medium_monitor():
    d = make_decision("txn_low_boundary", LOW_MAX, SMALL_AMOUNT)
    assert d.action == "ALLOW_WITH_MONITORING"
    assert d.policy_rule_id == "MEDIUM_MONITOR"


def test_just_below_low_max_is_allow():
    d = make_decision("txn_low_below", LOW_MAX - 0.001, SMALL_AMOUNT)
    assert d.action == "ALLOW"
    assert d.policy_rule_id == "LOW_ALLOW"


def test_probability_exactly_zero_and_one():
    d0 = make_decision("txn_zero", 0.0, SMALL_AMOUNT)
    assert d0.action == "ALLOW"
    d1 = make_decision("txn_one", 1.0, SMALL_AMOUNT)
    assert d1.action == "BLOCK"


# ---------------------------------------------------------------------------
# Amount-escalation rule (medium probability + large amount)
# ---------------------------------------------------------------------------

def test_medium_probability_with_large_amount_escalates():
    # probability alone would be MEDIUM_MONITOR, but a large amount escalates it
    d = make_decision("txn_escalate", 0.20, LARGE_AMOUNT)
    assert d.action == "STEP_UP_VERIFICATION"
    assert d.policy_rule_id == "MEDIUM_AMOUNT_ESCALATION"


def test_low_probability_with_large_amount_does_not_escalate():
    # below LOW_MAX -- amount escalation rule requires p >= LOW_MAX, so a very
    # confident-legitimate transaction is NOT escalated just because it's large
    # (avoids unnecessary customer friction, per project requirement)
    d = make_decision("txn_large_but_safe", 0.05, LARGE_AMOUNT)
    assert d.action == "ALLOW"
    assert d.policy_rule_id == "LOW_ALLOW"


def test_amount_exactly_at_escalation_cutoff():
    d = make_decision("txn_cutoff", 0.20, LARGE_AMOUNT_CUTOFF)
    assert d.policy_rule_id == "MEDIUM_AMOUNT_ESCALATION"


def test_amount_just_below_escalation_cutoff():
    d = make_decision("txn_cutoff_below", 0.20, LARGE_AMOUNT_CUTOFF - 0.01)
    assert d.policy_rule_id == "MEDIUM_MONITOR"


# ---------------------------------------------------------------------------
# 7: Invalid probabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_prob", [-0.01, 1.01, -5, 100, float("inf"), float("-inf")])
def test_invalid_probability_out_of_range_rejected(bad_prob):
    with pytest.raises(InvalidTransactionError):
        make_decision("txn_bad", bad_prob, SMALL_AMOUNT)


def test_nan_probability_rejected():
    with pytest.raises(InvalidTransactionError, match="NaN"):
        make_decision("txn_nan", float("nan"), SMALL_AMOUNT)


@pytest.mark.parametrize("bad_prob", ["0.5", None, [0.5], {"p": 0.5}, True, False])
def test_non_numeric_or_bool_probability_rejected(bad_prob):
    with pytest.raises(InvalidTransactionError):
        make_decision("txn_bad_type", bad_prob, SMALL_AMOUNT)


def test_invalid_amount_rejected():
    with pytest.raises(InvalidTransactionError):
        make_decision("txn_bad_amount", 0.5, -100.0)
    with pytest.raises(InvalidTransactionError):
        make_decision("txn_bad_amount2", 0.5, float("nan"))
    with pytest.raises(InvalidTransactionError):
        make_decision("txn_bad_amount3", 0.5, "not a number")


# ---------------------------------------------------------------------------
# 8: Missing transaction fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_id", ["", "   ", None, 12345])
def test_missing_or_invalid_transaction_id_rejected(bad_id):
    with pytest.raises(InvalidTransactionError):
        make_decision(bad_id, 0.5, SMALL_AMOUNT)


def test_make_decision_requires_probability_and_amount_arguments():
    with pytest.raises(TypeError):
        make_decision("txn_missing_args")  # missing required positional args


# ---------------------------------------------------------------------------
# 9: Deterministic repeated decisions
# ---------------------------------------------------------------------------

def test_repeated_calls_produce_identical_decision_fields():
    d1 = make_decision("txn_det", 0.55, SMALL_AMOUNT, timestamp="2026-08-21T00:00:00Z")
    d2 = make_decision("txn_det", 0.55, SMALL_AMOUNT, timestamp="2026-08-21T00:00:00Z")
    assert d1.action == d2.action
    assert d1.policy_rule_id == d2.policy_rule_id
    assert d1.risk_score == d2.risk_score
    assert d1.risk_category == d2.risk_category
    assert d1.fraud_probability == d2.fraud_probability
    assert d1.to_dict() == d2.to_dict()


def test_evaluate_policy_is_pure_and_deterministic():
    r1 = evaluate_policy(0.42, 1000.0)
    r2 = evaluate_policy(0.42, 1000.0)
    assert r1.rule_id == r2.rule_id
    assert r1.action == r2.action


# ---------------------------------------------------------------------------
# 10: Policy-rule traceability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probability,amount,expected_rule", [
    (0.02, SMALL_AMOUNT, "LOW_ALLOW"),
    (0.20, SMALL_AMOUNT, "MEDIUM_MONITOR"),
    (0.20, LARGE_AMOUNT, "MEDIUM_AMOUNT_ESCALATION"),
    (0.55, SMALL_AMOUNT, "HIGH_STEP_UP"),
    (0.90, SMALL_AMOUNT, "CRITICAL_BLOCK"),
])
def test_every_decision_traces_to_the_correct_rule(probability, amount, expected_rule):
    d = make_decision("txn_trace", probability, amount)
    assert d.policy_rule_id == expected_rule
    assert d.policy_reason is not None and len(d.policy_reason) > 0
    # reason text should reference the mechanism actually responsible for the decision
    if expected_rule == "MEDIUM_AMOUNT_ESCALATION":
        assert "amount" in d.policy_reason.lower()


def test_decision_record_includes_model_version_and_threshold():
    d = make_decision("txn_meta", 0.5, SMALL_AMOUNT)
    assert d.model_version == MODEL_VERSION
    assert d.threshold == DECISION_THRESHOLD


def test_model_version_matches_frozen_metadata_file():
    # Regression guard (Phase 9.5 remediation): decision_engine.MODEL_VERSION is the
    # single source of truth for the model's public label. The frozen metadata file
    # (used by /health, /model/info, /risk/score, /risk/explain) must carry the exact
    # same value, or different API responses would show different "model versions"
    # for what is actually the same frozen artifact.
    import json
    with open("ml/models/lgbm_v1_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == MODEL_VERSION, (
        f"ml/models/lgbm_v1_metadata.json model_name ({metadata['model_name']!r}) "
        f"does not match ml.evaluation.decision_engine.MODEL_VERSION ({MODEL_VERSION!r})"
    )


def test_risk_category_and_policy_action_are_independent():
    # Architectural guarantee from Phase 6/7: the 0-100 risk_score bucket is
    # informational only and is NOT the source of truth for the action.
    # p=0.25 -> risk_score=25 -> bucket is LOW (<=30) per risk_scoring.py,
    # but policy fires MEDIUM_MONITOR because 0.25 >= LOW_MAX(0.15). If a future
    # change ever made these two implicitly agree in all cases, that would hide
    # a real coupling bug -- this test exists specifically to keep them decoupled.
    d = make_decision("txn_decoupled", 0.25, SMALL_AMOUNT)
    assert d.risk_category == "LOW"
    assert d.action == "ALLOW_WITH_MONITORING"
    assert d.policy_rule_id == "MEDIUM_MONITOR"


# ---------------------------------------------------------------------------
# 11: SHAP values cannot alter the selected action
# ---------------------------------------------------------------------------

def test_shap_values_cannot_alter_action():
    probability, amount = 0.55, SMALL_AMOUNT

    explanation_a = [{"feature": "amount_zscore", "value": 5.0, "shap_value": 3.0, "direction": "increases_risk"}]
    explanation_b = [{"feature": "new_device_flag", "value": 0.0, "shap_value": -3.0, "direction": "decreases_risk"}]
    explanation_contradictory = [{"feature": "everything", "value": -999, "shap_value": -999, "direction": "decreases_risk"}] * 20

    d_none = make_decision("txn_shap", probability, amount, model_explanation=None)
    d_a = make_decision("txn_shap", probability, amount, model_explanation=explanation_a)
    d_b = make_decision("txn_shap", probability, amount, model_explanation=explanation_b)
    d_contra = make_decision("txn_shap", probability, amount, model_explanation=explanation_contradictory)

    assert d_none.action == d_a.action == d_b.action == d_contra.action == "STEP_UP_VERIFICATION"
    assert d_none.policy_rule_id == d_a.policy_rule_id == d_b.policy_rule_id == d_contra.policy_rule_id


def test_evaluate_policy_signature_does_not_accept_explanation():
    # structural guarantee: evaluate_policy only takes (probability, amount) --
    # there is no parameter through which an explanation could even be passed in.
    import inspect
    sig = inspect.signature(evaluate_policy)
    assert list(sig.parameters.keys()) == ["probability", "amount"]


# ---------------------------------------------------------------------------
# 12: Serialization of a complete decision record
# ---------------------------------------------------------------------------

def test_decision_record_serializes_to_json_with_all_required_fields():
    explanation = [{"feature": "velocity_5min", "value": 6.0, "shap_value": 1.2, "direction": "increases_risk"}]
    d = make_decision("txn_serialize", 0.9998, 21523.0, model_explanation=explanation,
                       timestamp="2026-08-21T10:00:00Z")
    record = d.to_dict()

    required_fields = {
        "transaction_id", "model_version", "fraud_probability", "threshold",
        "risk_score", "risk_category", "action", "policy_rule_id", "policy_reason",
        "timestamp", "model_explanation",
    }
    assert required_fields.issubset(record.keys())

    # must be JSON-serializable as-is
    serialized = json.dumps(record)
    reloaded = json.loads(serialized)
    assert reloaded["transaction_id"] == "txn_serialize"
    assert reloaded["action"] == "BLOCK"
    assert reloaded["policy_rule_id"] == "CRITICAL_BLOCK"
    assert reloaded["model_explanation"] == explanation


def test_decision_record_is_immutable():
    d = make_decision("txn_immutable", 0.5, SMALL_AMOUNT)
    with pytest.raises(Exception):
        d.action = "ALLOW"  # frozen dataclass should reject mutation


# ---------------------------------------------------------------------------
# Fail-safe behavior for unknown actions/rules
# ---------------------------------------------------------------------------

def test_unknown_action_from_a_hypothetical_rule_fails_safe(monkeypatch):
    import ml.evaluation.policy as policy_module

    bad_rule = policy_module.PolicyRule(
        rule_id="BROKEN_RULE",
        action="DO_SOMETHING_UNDEFINED",
        reason="test-only broken rule",
        condition=lambda p, amt: True,
    )
    monkeypatch.setattr(policy_module, "POLICY_RULES", [bad_rule])
    with pytest.raises(RuntimeError, match="unrecognized action"):
        policy_module.evaluate_policy(0.5, 100.0)


def test_no_matching_rule_fails_safe(monkeypatch):
    import ml.evaluation.policy as policy_module
    monkeypatch.setattr(policy_module, "POLICY_RULES", [])  # no catch-all at all
    with pytest.raises(RuntimeError, match="No policy rule matched"):
        policy_module.evaluate_policy(0.5, 100.0)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "pytest", __file__, "-v"]))

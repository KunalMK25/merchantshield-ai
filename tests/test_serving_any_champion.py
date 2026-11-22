"""
Phase 4 tests: everything needed to SERVE a promoted champion other than lgbm_v1.

- the SHAP explainer must work for every promotable model family
- the decision policy must be able to use the champion's own threshold, while
  staying byte-for-byte identical to the frozen policy at the frozen threshold
"""

import numpy as np
import pandas as pd
import pytest

from ml.contest.runner import run_contest
from ml.evaluation import policy
from ml.evaluation.decision_engine import make_decision, MODEL_VERSION
from ml.evaluation.explainability import RiskExplainer
from ml.features.build_features import FEATURE_COLUMNS


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["amount"] = np.exp(rng.normal(6, 1, size=n))
    X["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-(1.8 * X["amount_zscore"] - 2.5)))).astype(int)
    return X


@pytest.fixture(scope="module")
def contest():
    entries = run_contest(_frame(3000, 1), _frame(1500, 2), None,
                          names=["random_forest", "lightgbm", "xgboost", "catboost"])
    return {e["name"]: e for e in entries}, _frame(50, 9)


@pytest.mark.parametrize("name", ["random_forest", "lightgbm", "xgboost", "catboost"])
def test_explainer_works_for_every_promotable_family(contest, name):
    entries, rows = contest
    explainer = RiskExplainer(entries[name]["model"])
    for i in range(5):
        result = explainer.explain(rows.iloc[i])
        assert result["additivity_check_passed"] is True, f"{name}: SHAP does not add up"
        assert len(result["contributions"]) == len(FEATURE_COLUMNS)
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["top_reasons"]


# ------------------------------------------------------------------ policy

PROBS = [0.0, 0.1, 0.15, 0.2, 0.3, 0.39, 0.4, 0.41, 0.6, 0.79, 0.8, 0.95, 1.0]
AMOUNTS = [100.0, 24_999.0, 25_000.0, 90_000.0]


def test_threshold_aware_policy_matches_frozen_policy_at_frozen_threshold():
    for p in PROBS:
        for amt in AMOUNTS:
            a = policy.evaluate_policy(p, amt)
            b = policy.evaluate_policy_at(p, amt, policy.DECISION_THRESHOLD)
            assert (a.rule_id, a.action, a.reason) == (b.rule_id, b.action, b.reason)


def test_threshold_aware_policy_moves_the_flag_boundary():
    # frozen policy: 0.35 is only monitored; a champion tuned to 0.30 must step up
    assert policy.evaluate_policy(0.35, 100.0).action == "ALLOW_WITH_MONITORING"
    assert policy.evaluate_policy_at(0.35, 100.0, 0.30).action == "STEP_UP_VERIFICATION"
    assert policy.evaluate_policy_at(0.29, 100.0, 0.30).action == "ALLOW_WITH_MONITORING"
    # the policy text must quote the threshold actually in force
    assert "0.3" in policy.evaluate_policy_at(0.35, 100.0, 0.30).reason


def test_threshold_aware_policy_still_blocks_at_critical_boundary_and_never_unblocks():
    assert policy.evaluate_policy_at(0.95, 100.0, 0.30).action == "BLOCK"
    assert policy.evaluate_policy_at(0.95, 100.0, 0.90).action == "BLOCK"  # threshold above block line


def test_threshold_aware_policy_rejects_nonsense_thresholds():
    for bad in [0.0, 1.0, -0.1, 1.5, float("nan")]:
        with pytest.raises(ValueError):
            policy.evaluate_policy_at(0.5, 100.0, bad)


def test_make_decision_defaults_are_unchanged():
    d = make_decision("t1", 0.35, 100.0, timestamp="2026-01-01T00:00:00Z")
    assert d.model_version == MODEL_VERSION and d.threshold == policy.DECISION_THRESHOLD
    assert d.action == "ALLOW_WITH_MONITORING"


def test_make_decision_uses_champion_threshold_and_version():
    d = make_decision("t2", 0.35, 100.0, timestamp="2026-01-01T00:00:00Z",
                      decision_threshold=0.30, model_version="xgboost@run_x")
    assert d.threshold == 0.30 and d.model_version == "xgboost@run_x"
    assert d.action == "STEP_UP_VERIFICATION"

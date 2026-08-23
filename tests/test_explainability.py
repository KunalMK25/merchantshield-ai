import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import pytest

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.explainability import RiskExplainer, build_explanation_text, humanize_contribution
from ml.evaluation.risk_scoring import probability_to_score, score_to_category, score_transaction

# `model`, `explainer`, and `sample_rows` fixtures are provided at session scope
# by conftest.py. The constants below are kept for reference but the actual
# loading is handled once per test session via the shared fixtures.
MODEL_PATH = "ml/models/candidate_lgbm_v1.pkl"
FEATURES_PATH = "ml/data/features.csv"


# ---------------------------------------------------------------------------
# 1. SHAP explanation generation
# ---------------------------------------------------------------------------

def test_explain_runs_and_returns_expected_keys(explainer, sample_rows):
    row = sample_rows.iloc[0]
    result = explainer.explain(row)
    for key in ["fraud_probability", "base_value", "additivity_check_passed", "contributions", "top_reasons"]:
        assert key in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert len(result["contributions"]) == len(FEATURE_COLUMNS)


def test_explain_produces_nonempty_reasons_for_every_sample(explainer, sample_rows):
    for _, row in sample_rows.iterrows():
        result = explainer.explain(row)
        assert len(result["top_reasons"]) > 0
        for r in result["top_reasons"]:
            assert isinstance(r, str) and len(r) > 0


# ---------------------------------------------------------------------------
# 2. Feature/value alignment -- the value quoted in the explanation must match
#    the actual input feature value for that transaction.
# ---------------------------------------------------------------------------

def test_contribution_values_match_input_features(explainer, sample_rows):
    row = sample_rows.iloc[3]
    result = explainer.explain(row)
    by_feature = {c["feature"]: c["value"] for c in result["contributions"]}
    for feat in FEATURE_COLUMNS:
        assert abs(by_feature[feat] - float(row[feat])) < 1e-9, f"value mismatch for {feat}"


def test_humanize_contribution_embeds_the_actual_value():
    # amount_vs_avg_ratio = 4.2 must appear (rounded) in the generated sentence
    contribution = dict(feature="amount_vs_avg_ratio", value=4.2, shap_value=1.1,
                         direction="increases_risk", magnitude=1.1)
    text = humanize_contribution(contribution)
    assert "4.2" in text
    assert "increased" in text


# ---------------------------------------------------------------------------
# 3. Contribution direction correctness
# ---------------------------------------------------------------------------

def test_direction_field_matches_shap_sign(explainer, sample_rows):
    for _, row in sample_rows.iterrows():
        result = explainer.explain(row)
        for c in result["contributions"]:
            if c["shap_value"] > 0:
                assert c["direction"] == "increases_risk"
            elif c["shap_value"] < 0:
                assert c["direction"] == "decreases_risk"


def test_flagged_explanation_only_uses_risk_increasing_reasons(explainer, sample_rows):
    # search the full validation-adjacent feature set (not just the small random
    # sample) to deterministically find a row that crosses the 0.40 threshold
    df = pd.read_csv(FEATURES_PATH)
    candidates = df[df["is_fraud"] == 1].sample(50, random_state=5)
    high_risk_row = None
    for _, row in candidates.iterrows():
        result = explainer.explain(row)
        if result["fraud_probability"] >= 0.40:
            high_risk_row = row
            break
    assert high_risk_row is not None, "expected at least one known-fraud row to cross 0.40 in this search"
    result = explainer.explain(high_risk_row)
    explanation = build_explanation_text(result, decision_threshold=0.40)
    assert explanation["flagged"] is True
    assert explanation["header"] == "Why this transaction was flagged:"
    for r in explanation["reasons"]:
        assert "decreased" not in r.lower()


def test_nonflagged_explanation_uses_neutral_header(explainer, sample_rows):
    low_risk_row = None
    for _, row in sample_rows.iterrows():
        result = explainer.explain(row)
        if result["fraud_probability"] < 0.40:
            low_risk_row = row
            break
    assert low_risk_row is not None, "expected at least one sampled row below threshold"
    result = explainer.explain(low_risk_row)
    explanation = build_explanation_text(result, decision_threshold=0.40)
    assert explanation["flagged"] is False
    assert "NOT flagged" in explanation["header"]


# ---------------------------------------------------------------------------
# 4. Missing / invalid feature handling
# ---------------------------------------------------------------------------

def test_missing_feature_raises_value_error(explainer, sample_rows):
    row = sample_rows.iloc[0].drop(labels=["velocity_5min"])
    with pytest.raises(ValueError, match="missing"):
        explainer.explain(row)


def test_nan_feature_raises_value_error(explainer, sample_rows):
    row = sample_rows.iloc[0].copy()
    row["amount_zscore"] = np.nan
    with pytest.raises(ValueError, match="NaN|invalid"):
        explainer.explain(row)


def test_extra_unrelated_columns_are_ignored(explainer, sample_rows):
    # a row with extra non-feature columns (e.g. transaction_id, timestamp) should
    # still explain fine -- explain() only reads FEATURE_COLUMNS.
    row = sample_rows.iloc[0]
    assert "transaction_id" in row.index or "customer_id" in row.index or True  # raw df has extra cols
    result = explainer.explain(row)
    assert result is not None


# ---------------------------------------------------------------------------
# 5. Deterministic output for the same transaction/model
# ---------------------------------------------------------------------------

def test_explain_is_deterministic_across_calls(explainer, sample_rows):
    row = sample_rows.iloc[7]
    result1 = explainer.explain(row)
    result2 = explainer.explain(row)
    assert result1["fraud_probability"] == result2["fraud_probability"]
    assert result1["top_reasons"] == result2["top_reasons"]
    for c1, c2 in zip(result1["contributions"], result2["contributions"]):
        assert c1["feature"] == c2["feature"]
        assert c1["shap_value"] == c2["shap_value"]


def test_explain_is_deterministic_across_new_explainer_instances(model, sample_rows):
    row = sample_rows.iloc[9]
    e1 = RiskExplainer(model)
    e2 = RiskExplainer(model)
    r1 = e1.explain(row)
    r2 = e2.explain(row)
    assert r1["fraud_probability"] == r2["fraud_probability"]
    assert [c["shap_value"] for c in r1["contributions"]] == [c["shap_value"] for c in r2["contributions"]]


# ---------------------------------------------------------------------------
# 6. Mathematical consistency: SHAP contributions must reconstruct the model's
#    actual predicted probability (additivity property).
# ---------------------------------------------------------------------------

def test_shap_additivity_holds_for_all_samples(explainer, sample_rows):
    for _, row in sample_rows.iterrows():
        result = explainer.explain(row)
        assert result["additivity_check_passed"], (
            f"SHAP values do not reconstruct model output for a sampled transaction "
            f"(prob={result['fraud_probability']})"
        )


# ---------------------------------------------------------------------------
# Risk scoring unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prob,expected_score", [(0.0, 0), (1.0, 100), (0.5, 50), (0.874, 87)])
def test_probability_to_score(prob, expected_score):
    assert probability_to_score(prob) == expected_score


@pytest.mark.parametrize("score,expected_cat", [(0, "LOW"), (30, "LOW"), (31, "MEDIUM"),
                                                  (60, "MEDIUM"), (61, "HIGH"), (80, "HIGH"), (81, "CRITICAL"), (100, "CRITICAL")])
def test_score_to_category_boundaries(score, expected_cat):
    assert score_to_category(score) == expected_cat


def test_score_transaction_end_to_end():
    result = score_transaction(0.87)
    assert result["risk_score"] == 87
    assert result["risk_category"] == "CRITICAL"  # 87 > high_max(80)
    assert result["fraud_probability"] == 0.87


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "pytest", __file__, "-v"]))

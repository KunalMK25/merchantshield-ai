"""
Account alerts from the supervised model's scores.

An account-day's risk is the highest fraud probability among that day's payments. Two tier thresholds
are chosen on validation days only (the model's probabilities are inflated by class weighting, so
useful thresholds sit near 0.99, and they differ between model types, so they are re-chosen per model).
"""

import numpy as np
import pandas as pd
import pytest

from ml.cases.store import CaseStore
from ml.monitoring.anomaly import OUTPUT_COLUMNS
from ml.monitoring.supervised_alerts import (
    TIER_GRID, account_day_scores, calibrate_thresholds, detect_supervised,
)


def make_rows(seed=0, n_customers=40, days=20, flip=False):
    """Transaction rows with a model score: fraud rows score high, normal rows low, with overlap."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        for d in range(days):
            for k in range(rng.integers(1, 4)):
                fraud = int(rng.random() < 0.02)
                p = np.clip(rng.beta(8, 2) if fraud else rng.beta(1, 30), 0, 0.99999)
                rows.append(dict(customer_id=f"c{c}", timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d, hours=k + 1),
                                 amount=float(10 + 90 * rng.random()), is_fraud=(1 - fraud) if flip else fraud, p=float(p)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rows():
    return make_rows()


def scores_of(df, origin=None):
    return account_day_scores(df, df["p"].to_numpy(), origin=origin)


# ------------------------------------------------------------ account-day scores

def test_an_account_days_risk_is_its_highest_payment_score():
    df = pd.DataFrame({
        "customer_id": ["a", "a", "a", "b"],
        "timestamp": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 15:00", "2026-01-02 09:00", "2026-01-01 10:00"]),
        "amount": [10.0, 40.0, 5.0, 7.0], "is_fraud": [0, 1, 0, 0]})
    s = account_day_scores(df, np.array([0.1, 0.9, 0.3, 0.2]))
    a1 = s[(s.customer_id == "a") & (s.day == 0)].iloc[0]
    assert a1.max_prob == 0.9 and a1.n_txns == 2 and a1.total == 50.0 and a1.fraud == 1
    assert s[(s.customer_id == "a") & (s.day == 1)].iloc[0].max_prob == 0.3
    assert len(s) == 3


def test_day_numbers_follow_the_given_origin_so_they_match_the_other_detectors():
    df = pd.DataFrame({"customer_id": ["a"], "timestamp": pd.to_datetime(["2026-01-11 09:00"]),
                       "amount": [1.0], "is_fraud": [0]})
    s = account_day_scores(df, np.array([0.5]), origin=pd.Timestamp("2026-01-01"))
    assert s.day.iloc[0] == 10


def test_scores_and_rows_must_line_up():
    df = pd.DataFrame({"customer_id": ["a"], "timestamp": pd.to_datetime(["2026-01-01"]), "amount": [1.0], "is_fraud": [0]})
    with pytest.raises(ValueError, match="one score per row"):
        account_day_scores(df, np.array([0.1, 0.2]))


# --------------------------------------------------------------- thresholds

def test_thresholds_make_two_distinct_tiers_and_maximise_the_mean_f1(rows):
    val = scores_of(rows)
    cal = calibrate_thresholds(val)
    assert cal["critical"] > cal["high"] and cal["critical"] in TIER_GRID and cal["high"] in TIER_GRID
    assert cal["score"] == pytest.approx(max(g["score"] for g in cal["grid"]))
    assert all(g["critical"] > g["high"] for g in cal["grid"])
    assert "validation" in cal["objective"].lower()
    chosen = next(g for g in cal["grid"] if g["critical"] == cal["critical"] and g["high"] == cal["high"])
    assert chosen["score"] == pytest.approx((chosen["critical_tier"]["f1"] + chosen["critical_plus_high"]["f1"]) / 2)


def test_the_grid_reaches_far_enough_up_for_inflated_probabilities():
    assert max(TIER_GRID) >= 0.999 and min(TIER_GRID) <= 0.5 and list(TIER_GRID) == sorted(TIER_GRID)


def test_thresholds_need_positives_to_learn_from(rows):
    with pytest.raises(ValueError, match="compromised"):
        calibrate_thresholds(scores_of(rows).assign(fraud=0))


def test_thresholds_depend_only_on_the_validation_scores_given(rows):
    val = scores_of(rows)
    a = calibrate_thresholds(val)
    b = calibrate_thresholds(val.copy())
    assert (a["critical"], a["high"], a["score"]) == (b["critical"], b["high"], b["score"])


# --------------------------------------------------------------------- alerts

def test_alerts_are_tiered_by_the_two_thresholds(rows):
    s = scores_of(rows)
    flags = detect_supervised(s, critical=0.9, high=0.5)
    assert set(flags.severity) <= {"critical", "high"}
    crit = s[s.max_prob >= 0.9]
    high = s[(s.max_prob >= 0.5) & (s.max_prob < 0.9)]
    assert (flags.severity == "critical").sum() == len(crit) and (flags.severity == "high").sum() == len(high)
    assert flags["iso_score"].is_monotonic_decreasing or flags.severity.iloc[0] == "critical"


def test_alert_records_match_the_shape_the_queue_and_pages_expect(rows):
    flags = detect_supervised(scores_of(rows), critical=0.9, high=0.5)
    assert list(flags.columns) == OUTPUT_COLUMNS
    r = flags.iloc[0]
    assert r.signals[0]["name"] == "model_score" and "%" in r.signals[0]["detail"]
    assert r.recommended_action.startswith("Recommend")
    assert "block" not in r.recommended_action.lower()
    assert r.points in (2, 3) and bool(r.iso_flag) is False


def test_no_alerts_gives_an_empty_frame_with_the_same_columns(rows):
    empty = detect_supervised(scores_of(rows), critical=1.1, high=1.05)
    assert empty.empty and list(empty.columns) == OUTPUT_COLUMNS


def test_the_alerts_flow_into_the_analyst_queue(rows, tmp_path):
    from ml.monitoring.anomaly import to_records
    flags = detect_supervised(scores_of(rows), critical=0.9, high=0.5)
    n = CaseStore(tmp_path / "cases.db").ingest(to_records(flags), job_id="j")
    assert n == len(flags) > 0

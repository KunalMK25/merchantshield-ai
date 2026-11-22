"""Phase 5 tests: drift statistics (PSI, KS) and the drift report."""

import os

import numpy as np
import pandas as pd
import pytest

from ml.data.daily_batch import DriftConfig, fit_population, initial_state, simulate_day, warm_up
from ml.features.build_features import FEATURE_COLUMNS, build_features
from ml.monitoring.drift import (
    DRIFT_FEATURES, MATURING_FEATURES, drift_report, ks_test, psi,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")


# --------------------------------------------------------------------- PSI

def test_psi_of_identical_samples_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_matches_the_hand_computed_value_for_a_binary_feature():
    ref = np.array([0] * 500 + [1] * 500)
    cur = np.array([0] * 800 + [1] * 200)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, cur) == pytest.approx(expected, rel=1e-6)


def test_psi_separates_a_real_shift_from_sampling_noise():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    assert psi(ref, rng.normal(0, 1, 3000)) < 0.05
    assert psi(ref, rng.normal(1, 1, 3000)) > 0.25


def test_psi_is_finite_when_a_bucket_empties_or_the_reference_is_constant():
    assert np.isfinite(psi(np.zeros(1000), np.ones(1000)))
    assert np.isfinite(psi(np.arange(1000), np.arange(5000, 6000)))


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        psi([], [1, 2, 3])
    with pytest.raises(ValueError):
        psi([1, 2, 3], [])


# ---------------------------------------------------------------------- KS

def test_ks_flags_a_shift_and_not_identical_samples():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 4000)
    same = ks_test(a, rng.normal(0, 1, 4000))
    shifted = ks_test(a, rng.normal(0.5, 1, 4000))
    assert same["p_value"] > 0.01 and shifted["p_value"] < 1e-6
    assert shifted["statistic"] > same["statistic"]


# ------------------------------------------------------------------ report

def frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in DRIFT_FEATURES})
    df["amount"] = df["amount"] + shift
    return df


def test_report_marks_only_the_shifted_feature():
    ref, base, cur = frame(8000, 1), frame(8000, 2), frame(2000, 3, shift=2.0)
    rep = drift_report(ref, base, cur)
    assert rep["status"] == "drifted"
    assert [f["feature"] for f in rep["features"] if f["severity"] == "significant"] == ["amount"]
    assert rep["max_psi"] == max(f["psi"] for f in rep["features"])
    assert rep["n_reference"] == 8000 and rep["n_current"] == 2000


def test_report_is_stable_when_nothing_changed():
    rep = drift_report(frame(8000, 1), frame(8000, 2), frame(2000, 3))
    assert rep["status"] == "stable"
    assert all(f["severity"] == "stable" for f in rep["features"])


def test_maturing_features_are_only_compared_with_the_recent_window():
    recent, base, cur = frame(6000, 1), frame(6000, 2), frame(2000, 3)
    base["prior_txn_count"] = base["prior_txn_count"] - 5.0   # history naturally accumulates
    base["account_age_days"] = base["account_age_days"] - 5.0
    rep = drift_report(recent, base, cur)
    by = {f["feature"]: f for f in rep["features"]}
    for name in MATURING_FEATURES:
        assert by[name]["psi_baseline"] is None and by[name]["severity"] == "stable"
    assert rep["status"] == "stable"


def test_calendar_only_features_are_not_monitored():
    assert "day_of_week" not in DRIFT_FEATURES
    assert set(DRIFT_FEATURES) <= set(FEATURE_COLUMNS)


def test_score_drift_is_reported_and_counts_toward_status():
    rng = np.random.default_rng(4)
    rep = drift_report(frame(6000, 1), frame(6000, 2), frame(2000, 3),
                       score_reference=rng.beta(1, 30, 6000), score_current=rng.beta(4, 10, 2000))
    assert rep["score"]["severity"] == "significant" and rep["status"] == "drifted"


def test_too_little_current_data_is_not_judged():
    rep = drift_report(frame(6000, 1), frame(6000, 2), frame(50, 3, shift=5.0))
    assert rep["status"] == "insufficient_data"


# --------------------------------------------- against the real demo data

@pytest.fixture(scope="module")
def world():
    base = pd.read_csv(BASE)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    profiles, merchants = fit_population(base)
    state = warm_up(profiles, merchants, initial_state(base))
    days, s = [], state
    for d in range(60, 64):
        df, s = simulate_day(profiles, merchants, d, s, seed=5)
        days.append(df)
    return base, profiles, merchants, pd.concat(days), s


def _report(world, drift):
    base, profiles, merchants, history, state = world
    today, _ = simulate_day(profiles, merchants, 64, state, drift=drift, seed=5)
    feats = build_features(pd.concat([base, history, today], ignore_index=True))
    day = (feats["timestamp"] - feats["timestamp"].min().normalize()).dt.days
    cur = feats[feats["transaction_id"].isin(today["transaction_id"])]
    return drift_report(feats[(day >= 50) & (day < 64)], feats[day < 40], cur)


def test_an_ordinary_new_day_is_not_called_drift(world):
    rep = _report(world, None)
    assert rep["status"] == "stable", [f for f in rep["features"] if f["severity"] != "stable"]


def test_a_spending_shift_is_caught_and_attributed_to_amounts(world):
    rep = _report(world, DriftConfig(amount_scale=2.5))
    assert rep["status"] == "drifted"
    flagged = {f["feature"] for f in rep["features"] if f["severity"] == "significant"}
    assert {"amount", "amount_zscore", "amount_vs_avg_ratio"} <= flagged

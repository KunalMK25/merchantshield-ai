"""
Slow drift: a change too gradual for the daily check, which judges each day against the last few weeks and so
absorbs it. Two 28-day windows two months apart are compared; each spans four full weeks, so the weekly cycle
cancels out. At tens of thousands of rows per window the sampling noise in PSI is negligible, so 0.10 is a
real shift, not a fluctuation.
"""

import numpy as np
import pandas as pd
import pytest

from ml.monitoring.drift import (
    CREEP_PSI, MATURING_FEATURES, DRIFT_FEATURES, creep_report, creep_windows, drift_report,
)


def history(days=120, rows_per_day=700, seed=0, amount_ramp=0.0, ramp_start=None, maturing_growth=0.0):
    """Daily rows of Gaussian features; `amount` creeps up linearly from `ramp_start` by `amount_ramp` (in sigmas)."""
    rng = np.random.default_rng(seed)
    parts = []
    for d in range(days):
        df = pd.DataFrame({f: rng.normal(size=rows_per_day) for f in DRIFT_FEATURES})
        if amount_ramp and ramp_start is not None and d >= ramp_start:
            df["amount"] += amount_ramp * min(1.0, (d - ramp_start) / (days - 1 - ramp_start))
        for m in MATURING_FEATURES:
            df[m] += maturing_growth * d
        df["day"] = d
        parts.append(df)
    feats = pd.concat(parts, ignore_index=True)
    return feats, feats["day"]


def test_the_windows_are_four_full_weeks_two_months_apart():
    (cs, ce), (es, ee) = creep_windows(100)
    assert (cs, ce, es, ee) == (73, 100, 17, 44)
    assert ce - cs + 1 == 28 and ee - es + 1 == 28 and cs - ee - 1 == 28     # a 28-day gap between them


def test_an_unchanging_process_is_stable():
    feats, day = history()
    r = creep_report(feats, day, today=119)
    assert r["status"] == "stable" and r["max_psi"] < 0.02 and r["flagged_features"] == []


def test_a_gradual_shift_is_caught_that_the_daily_check_misses():
    feats, day = history(amount_ramp=0.6, ramp_start=60)        # amount creeps up by 0.6 sigma over the last 60 days
    r = creep_report(feats, day, today=119)
    assert r["status"] == "creeping" and r["flagged_features"] == ["amount"] and r["max_psi"] >= CREEP_PSI

    # the daily verdict compares with the same weekday in the last four weeks, so it sees almost nothing
    cur = feats[day == 119]
    ref = feats[day.isin([112, 105, 98, 91])]
    assert drift_report(ref, ref, cur)["status"] == "stable"


def test_features_that_grow_by_construction_do_not_count():
    feats, day = history(maturing_growth=0.5)
    r = creep_report(feats, day, today=119)
    assert r["status"] == "stable"
    assert {f["feature"] for f in r["features"]}.isdisjoint(MATURING_FEATURES)


def test_too_little_history_or_data_is_not_judged():
    feats, day = history(days=60)
    assert creep_report(feats, day, today=59)["status"] == "insufficient_data"
    sparse, sday = history(days=120, rows_per_day=5)
    assert creep_report(sparse, sday, today=119)["status"] == "insufficient_data"


def test_the_report_names_what_it_compared():
    feats, day = history()
    r = creep_report(feats, day, today=119)
    assert r["window_days"] == 28 and r["gap_days"] == 28
    assert r["n_current"] == 28 * 700 and r["n_earlier"] == 28 * 700
    assert all({"feature", "psi", "severity"} <= set(f) for f in r["features"])


def test_the_daily_job_records_the_slow_drift_status(tmp_path):
    """Through the real job on a prepared dataset with enough history; a short history says so instead of guessing."""
    from ml.datasets.prepare import prepare
    from ml.datasets.workspace import Workspace
    from ml.retrain.job import RetrainConfig, run_daily_job
    from tests.dataset_fixtures import write_sparkov_like

    src = tmp_path / "src"
    src.mkdir()
    write_sparkov_like(src / "fraudTrain.csv", customers=120, days=130)
    prepare("sparkov", src, tmp_path / "root", stream_days=15)
    c = RetrainConfig.from_workspace(Workspace("sparkov", tmp_path / "root"), models=["random_forest"])
    c.data_dir, c.registry_dir = str(tmp_path / "d"), str(tmp_path / "r")
    m = run_daily_job(c)["monitoring"]
    assert m["creep_status"] in ("stable", "creeping")
    assert isinstance(m["creep_flagged_features"], list)

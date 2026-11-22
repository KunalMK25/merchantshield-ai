"""Phase 5 tests: drift + anomaly monitoring inside the daily retraining job."""

import json
import os

import pandas as pd
import pytest

from ml.data.daily_batch import DriftConfig
from ml.retrain.job import RetrainConfig, run_daily_job

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")


@pytest.fixture(scope="module")
def mini_base(tmp_path_factory):
    df = pd.read_csv(BASE)
    keep = sorted(df["customer_id"].unique())[:400]
    path = tmp_path_factory.mktemp("base") / "raw.csv"
    df[df["customer_id"].isin(keep)].to_csv(path, index=False)
    return str(path)


def cfg(tmp_path, mini_base, **kw):
    return RetrainConfig(base_raw_path=mini_base, data_dir=str(tmp_path / "data"),
                         registry_dir=str(tmp_path / "registry"), models=["random_forest"],
                         bootstrap=False, **kw)


def test_a_normal_run_records_drift_and_flagged_accounts(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    report = run_daily_job(c)

    m = report["monitoring"]
    assert m["drift_status"] in ("stable", "watch")
    assert m["accounts_checked"] > 0 and "critical" in m and "flagged_features" in m
    assert m["score_psi"] is None          # no champion existed yet, so no model to score with

    lines = [json.loads(l) for l in (tmp_path / "data" / "drift_log.jsonl").read_text().splitlines()]
    assert lines[-1]["job_id"] == report["job_id"]
    assert lines[-1]["drift"]["features"] and lines[-1]["anomalies"]["accounts_checked"] > 0

    day_file = tmp_path / "data" / "anomalies" / "day_0060.json"
    rec = json.loads(day_file.read_text())
    assert rec["day"] == 60 and isinstance(rec["records"], list)
    for r in rec["records"]:
        assert r["severity"] in ("critical", "high", "watch")
        assert r["signals"] is not None


def test_a_spending_shift_is_reported_as_drift(tmp_path, mini_base):
    report = run_daily_job(cfg(tmp_path, mini_base, drift=DriftConfig(amount_scale=3.0)))
    m = report["monitoring"]
    assert m["drift_status"] == "drifted"
    assert "amount" in m["flagged_features"]


def test_the_serving_model_s_scores_are_watched_once_a_champion_exists(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    run_daily_job(c)                                   # promotes the first champion
    second = run_daily_job(c)
    assert isinstance(second["monitoring"]["score_psi"], float)


def test_a_fraud_surge_raises_more_flags_than_a_normal_day(tmp_path, mini_base):
    plain = run_daily_job(cfg(tmp_path / "a", mini_base))
    surge = run_daily_job(cfg(tmp_path / "b", mini_base, drift=DriftConfig(fraud_rate_multiplier=60.0)))
    flagged = lambda r: r["monitoring"]["critical"] + r["monitoring"]["high"]
    assert flagged(surge) > flagged(plain)


def test_a_monitoring_failure_does_not_stop_retraining(tmp_path, mini_base, monkeypatch):
    import ml.retrain.job as job

    def boom(*a, **k):
        raise RuntimeError("monitor broke")

    monkeypatch.setattr(job, "drift_report", boom)
    report = run_daily_job(cfg(tmp_path, mini_base))
    assert report["status"] == "ok" and report["decision"]["promoted"] is True
    assert "monitor broke" in report["monitoring"]["error"]


# ------------------------------------------------ comparing like with like (weekday)

def weekly_frame(days=45, rows_per_day=900, weekend_shift=2.0, seed=0):
    """Rows with a weekly cycle: weekend days (day % 7 in 5, 6) have larger amounts."""
    import numpy as np
    from ml.monitoring.drift import DRIFT_FEATURES
    rng = np.random.default_rng(seed)
    parts = []
    for d in range(days):
        df = pd.DataFrame({f: rng.normal(size=rows_per_day) for f in DRIFT_FEATURES})
        if d % 7 in (5, 6):
            df["amount"] = df["amount"] + weekend_shift
        df["day"] = d
        parts.append(df)
    feats = pd.concat(parts, ignore_index=True)
    return feats, feats["day"]


def test_a_single_day_is_compared_with_the_same_weekday_in_earlier_weeks():
    from ml.retrain.job import reference_window
    feats, day = weekly_frame()
    ref, label = reference_window(feats, day, [40])
    assert sorted(ref["day"].unique()) == [12, 19, 26, 33] and "weekday" in label


def test_a_multi_day_run_or_a_thin_history_falls_back_to_the_trailing_window():
    from ml.retrain.job import reference_window
    feats, day = weekly_frame()
    ref, label = reference_window(feats, day, [40, 41])
    assert sorted(ref["day"].unique()) == list(range(26, 40)) and "trailing" in label
    ref2, label2 = reference_window(feats, day, [10])           # fewer than 4 weeks of history exist
    assert "trailing" in label2 and ref2["day"].max() == 9


def test_weekly_seasonality_is_not_mistaken_for_drift():
    """The reason for the rule: a Saturday looks 'drifted' next to a mixed 14-day window, and normal next to Saturdays."""
    from ml.monitoring.drift import drift_report
    from ml.retrain.job import RECENT_DAYS, reference_window
    feats, day = weekly_frame()
    cur = feats[day == 40]                                      # day 40 is a Saturday in this frame (40 % 7 == 5)
    trailing = feats[(day >= 40 - RECENT_DAYS) & (day < 40)]
    same, _ = reference_window(feats, day, [40])
    assert drift_report(trailing, trailing, cur)["status"] == "drifted"
    assert drift_report(same, same, cur)["status"] == "stable"

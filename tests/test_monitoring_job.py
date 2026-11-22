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

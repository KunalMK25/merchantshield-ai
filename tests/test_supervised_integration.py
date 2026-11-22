"""Supervised account alerts inside evaluate and the daily job."""

import json
import shutil

import pandas as pd
import pytest

from ml.datasets.evaluate import evaluate_workspace
from ml.datasets.prepare import prepare
from ml.datasets.workspace import Workspace
from ml.retrain.job import RetrainConfig, run_daily_job
from tests.dataset_fixtures import write_sparkov_like

pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


@pytest.fixture
def ws(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_sparkov_like(src / "fraudTrain.csv", customers=90, days=80)
    prepare("sparkov", src, tmp_path / "root", stream_days=15)
    return Workspace("sparkov", tmp_path / "root")


def test_evaluate_scores_supervised_alerts_beside_the_unsupervised_detector(ws):
    r = evaluate_workspace(ws, models=["logistic_regression", "random_forest"])
    s = r["anomaly_supervised"]
    assert s["config"] == "supervised" and s["model"] == r["winner"]
    assert s["thresholds"]["critical"] > s["thresholds"]["high"]
    assert set(s["tiers"]) == {"critical", "critical_plus_high"} and set(s["validation"]) == {"critical", "critical_plus_high"}
    # same sealed test window and the same notion of "compromised account-day" as the other detector
    assert s["compromised_account_days"] == r["anomaly"]["compromised_account_days"]
    assert s["accounts_checked"] == r["anomaly"]["accounts_checked"]
    assert "validation" in s["calibrated_on"].lower()


def test_flipping_every_test_label_cannot_change_the_chosen_thresholds(ws, tmp_path):
    first = evaluate_workspace(ws, models=["logistic_regression"])["anomaly_supervised"]

    root2 = tmp_path / "root2"
    shutil.copytree(ws.path, root2 / "sparkov")
    ws2 = Workspace("sparkov", root2)
    day0 = pd.Timestamp(json.loads(ws2.config_path.read_text())["day0"])
    stream = pd.read_csv(ws2.stream_path, parse_dates=["timestamp"])
    test_days = max(3, 80 // 10)
    in_test = (stream["timestamp"].dt.normalize() - day0).dt.days >= 80 - test_days
    assert in_test.any()
    stream.loc[in_test, "is_fraud"] = 1 - stream.loc[in_test, "is_fraud"]
    stream.to_csv(ws2.stream_path, index=False)

    second = evaluate_workspace(ws2, models=["logistic_regression"])["anomaly_supervised"]
    assert second["thresholds"] == first["thresholds"] and second["validation"] == first["validation"]


def cfg(ws, tmp_path, **kw):
    c = RetrainConfig.from_workspace(ws, models=["random_forest"], **kw)
    c.data_dir, c.registry_dir = str(tmp_path / "d"), str(tmp_path / "r")
    return c


def test_the_job_defaults_to_the_unsupervised_detector(ws, tmp_path):
    c = cfg(ws, tmp_path)
    assert c.alert_source == "unsupervised"
    assert run_daily_job(c)["monitoring"]["alert_source"] == "unsupervised"


def test_supervised_alerts_need_a_champion_so_the_first_run_falls_back(ws, tmp_path):
    c = cfg(ws, tmp_path, alert_source="supervised")
    first = run_daily_job(c)["monitoring"]
    assert first["alert_source"] == "unsupervised" and "champion" in first["alert_fallback_reason"]
    second = run_daily_job(c)["monitoring"]
    assert second["alert_source"] == "supervised" and second["alert_fallback_reason"] is None
    assert second["accounts_checked"] > 0


def test_supervised_records_reach_the_day_files_with_their_own_signal(ws, tmp_path):
    c = cfg(ws, tmp_path, alert_source="supervised")
    run_daily_job(c)
    run_daily_job(c)
    records = []
    for f in (tmp_path / "d" / "anomalies").glob("day_*.json"):
        records += json.loads(f.read_text())["records"]
    supervised = [r for r in records if r["signals"] and r["signals"][0]["name"] == "model_score"]
    assert supervised, "the second run should have produced model_score alerts"
    assert all(r["severity"] in ("critical", "high") for r in supervised)


def test_an_unknown_alert_source_is_rejected_before_anything_runs(ws, tmp_path):
    with pytest.raises(ValueError, match="alert_source"):
        run_daily_job(cfg(ws, tmp_path, alert_source="magic"))
    assert not (tmp_path / "d" / "daily").exists() or not list((tmp_path / "d" / "daily").glob("*.csv"))

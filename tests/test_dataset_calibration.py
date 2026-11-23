"""Calibrating the detector on a prepared dataset, and the guarantees around the test days."""

import json
import shutil

import pandas as pd
import pytest

from ml.datasets.calibrate import calibrate_workspace, main
from ml.datasets.evaluate import evaluate_workspace
from ml.datasets.prepare import prepare
from ml.datasets.workspace import Workspace
from ml.monitoring.anomaly import AnomalyConfig
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


def test_calibration_is_written_with_its_provenance(ws):
    out = calibrate_workspace(ws)
    saved = json.loads((ws.path / "anomaly_config.json").read_text())
    assert saved == json.loads(json.dumps(out))
    assert saved["dataset"] == "sparkov" and len(saved["grid"]) == 15
    assert 0.9 < saved["quantile"] < 1 and 0.9 < saved["iso_quantile"] < 1
    assert saved["val_start"] + saved["val_days"] == saved["test_start"]
    assert saved["calibrated_on"].startswith("days ") and "validation" in saved["objective"].lower()


def test_flipping_every_test_day_label_cannot_change_the_calibration(ws, tmp_path):
    first = calibrate_workspace(ws)

    root2 = tmp_path / "root2"
    shutil.copytree(ws.path, root2 / "sparkov")
    ws2 = Workspace("sparkov", root2)
    day0 = pd.Timestamp(json.loads(ws2.config_path.read_text())["day0"])
    stream = pd.read_csv(ws2.stream_path, parse_dates=["timestamp"])
    in_test = (stream["timestamp"].dt.normalize() - day0).dt.days >= first["test_start"]
    assert in_test.any(), "fixture must have rows in the test window"
    stream.loc[in_test, "is_fraud"] = 1 - stream.loc[in_test, "is_fraud"]
    stream.to_csv(ws2.stream_path, index=False)

    second = calibrate_workspace(ws2)
    assert (second["quantile"], second["iso_quantile"], second["score"]) == (first["quantile"], first["iso_quantile"], first["score"])
    assert second["grid"] == first["grid"]


def test_evaluate_reports_calibrated_and_default_detectors_side_by_side(ws):
    plain = evaluate_workspace(ws, models=["logistic_regression"])
    assert plain["anomaly"]["config"] == "default" and "anomaly_default" not in plain

    calibrate_workspace(ws)
    r = evaluate_workspace(ws, models=["logistic_regression"])
    assert r["anomaly"]["config"] == "calibrated" and r["anomaly_default"]["config"] == "default"
    assert r["calibration"]["quantile"] > 0.9 and "calibrated_on" in r["calibration"]
    for key in ("anomaly", "anomaly_default"):
        assert set(r[key]["tiers"]) == {"critical", "critical_plus_high", "everything_flagged"}


def test_evaluate_refuses_a_test_window_that_overlaps_the_calibration_days(ws):
    cal = calibrate_workspace(ws)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_workspace(ws, models=["logistic_regression"], test_days=cal["test_days"] + 4)
    ok = evaluate_workspace(ws, models=["logistic_regression"], test_days=max(2, cal["test_days"] - 3))
    assert ok["anomaly"]["config"] == "calibrated"


def test_the_daily_job_uses_the_workspaces_calibration(ws, tmp_path):
    plain = RetrainConfig.from_workspace(ws, models=["random_forest"])
    assert plain.anomaly is None

    cal = calibrate_workspace(ws)
    c = RetrainConfig.from_workspace(ws, models=["random_forest"])
    assert c.anomaly == AnomalyConfig(quantile=cal["quantile"], iso_quantile=cal["iso_quantile"])
    c.data_dir, c.registry_dir = str(tmp_path / "d"), str(tmp_path / "r")
    report = run_daily_job(c)
    assert report["monitoring"]["anomaly_quantile"] == cal["quantile"]
    assert report["monitoring"]["accounts_checked"] > 0


def test_calibration_needs_days_left_for_training(ws):
    with pytest.raises(ValueError, match="spans only"):
        calibrate_workspace(ws, val_days=60, test_days=60)


def test_the_command_summarises_the_search_and_handles_a_missing_workspace(ws, capsys, tmp_path):
    main(["sparkov", "--workspace-root", str(ws.root)])
    out = capsys.readouterr().out
    assert "quantile" in out.lower() and "validation" in out.lower() and "anomaly_config.json" in out
    with pytest.raises(SystemExit):
        main(["sparkov", "--workspace-root", str(tmp_path / "nope")])
    assert "prepare" in capsys.readouterr().out

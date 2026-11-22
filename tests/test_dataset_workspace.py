"""Workspaces and real-day replay: running the daily lifecycle on a prepared public dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from ml.datasets.prepare import prepare
from tests.dataset_fixtures import write_sparkov_like
from ml.datasets.workspace import Workspace
from ml.retrain.job import RetrainConfig, run_daily_job

# Tiny fixtures can leave a validation window with no fraud rows; sklearn warns, the code handles it.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    src = tmp_path_factory.mktemp("src")
    write_sparkov_like(src / "fraudTrain.csv")
    prepare("sparkov", src, root, stream_days=20)
    return Workspace("sparkov", root)


def cfg(prepared, tmp_path, **kw):
    """The prepared workspace's inputs, but with outputs redirected so each test starts clean."""
    c = RetrainConfig.from_workspace(prepared, models=["random_forest"], **kw)
    c.data_dir, c.registry_dir = str(tmp_path / "data"), str(tmp_path / "registry")
    return c


def test_a_config_from_a_workspace_points_at_its_files_and_scaled_costs(prepared):
    c = RetrainConfig.from_workspace(prepared)
    meta = json.loads(prepared.config_path.read_text())
    assert c.base_raw_path == str(prepared.raw_path) and c.stream_path == str(prepared.stream_path)
    assert c.data_dir == str(prepared.data_dir) and c.registry_dir == str(prepared.registry_dir)
    assert c.cost.fp_cost == meta["fp_cost"] and c.cost.fn_cost_fraction == meta["fn_cost_fraction"]
    assert c.bootstrap is False          # the original synthetic champion has no place in a dataset workspace


def test_an_unprepared_workspace_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ml.datasets.prepare"):
        RetrainConfig.from_workspace(Workspace("sparkov", tmp_path))


def test_the_first_run_replays_the_first_held_back_real_day(prepared, tmp_path):
    stream = pd.read_csv(prepared.stream_path, parse_dates=["timestamp"])
    day0 = pd.Timestamp(json.loads(prepared.config_path.read_text())["day0"])
    stream["_day"] = (stream["timestamp"].dt.normalize() - day0).dt.days
    first = stream[stream["_day"] == stream["_day"].min()]

    report = run_daily_job(cfg(prepared, tmp_path))

    assert report["source"] == "replay" and report["sim_days"] == [int(first["_day"].iloc[0])]
    assert report["new_rows"] == len(first) and report["new_fraud_rows"] == int(first["is_fraud"].sum())
    saved = pd.read_csv(next((tmp_path / "data" / "daily").glob("day_*.csv")))
    assert set(saved["transaction_id"]) == set(first["transaction_id"])       # the real rows, labels intact
    assert report["cost"]["fp_cost"] == json.loads(prepared.config_path.read_text())["fp_cost"]


def test_each_run_replays_the_next_day_and_never_the_same_one_twice(prepared, tmp_path):
    c = cfg(prepared, tmp_path)
    days = [run_daily_job(c)["sim_days"][0] for _ in range(3)]
    assert days == sorted(set(days)) and len(days) == 3


def test_after_the_stream_runs_out_the_job_continues_with_synthetic_days(prepared, tmp_path):
    c = cfg(prepared, tmp_path)
    reports = []
    while True:
        r = run_daily_job(c)
        reports.append(r)
        if r["source"] != "replay":
            break
        assert len(reports) <= 25, "stream should be exhausted within its 20 days"
    last = reports[-1]
    assert last["source"] == "synthetic" and last["sim_days"][0] > max(x["sim_days"][0] for x in reports[:-1])
    # synthetic ids must not collide with replayed real ids
    ids = pd.concat([pd.read_csv(f, usecols=["transaction_id"]) for f in (tmp_path / "data" / "daily").glob("day_*.csv")])
    assert ids["transaction_id"].is_unique


def test_a_multi_day_run_can_mix_real_and_synthetic_days(prepared, tmp_path):
    c = cfg(prepared, tmp_path, days=40)           # more than the 20 held-back days
    r = run_daily_job(c)
    assert r["source"] == "mixed" and len(r["sim_days"]) == 40


def test_without_a_stream_the_job_is_unchanged_and_reports_synthetic(prepared, tmp_path):
    c = cfg(prepared, tmp_path)
    c.stream_path = None
    assert run_daily_job(c)["source"] == "synthetic"

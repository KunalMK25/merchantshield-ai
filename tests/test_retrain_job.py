"""Phase 4 tests: the daily retraining job (generate a day, retrain, maybe promote)."""

import json
import os

import pandas as pd
import pytest

from ml.data.daily_batch import DailyStore, DriftConfig
from ml.registry.store import ModelRegistry
from ml.retrain.job import JobAlreadyRunning, RetrainConfig, rolling_split, run_daily_job

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")
FAST_MODELS = ["logistic_regression", "random_forest"]


@pytest.fixture(scope="module")
def mini_base(tmp_path_factory):
    """A slice of the real demo history (first 250 customers) so jobs run in seconds."""
    df = pd.read_csv(BASE)
    keep = sorted(df["customer_id"].unique())[:250]
    path = tmp_path_factory.mktemp("base") / "raw.csv"
    df[df["customer_id"].isin(keep)].to_csv(path, index=False)
    return str(path)


def cfg(tmp_path, mini_base, **kw):
    return RetrainConfig(base_raw_path=mini_base, data_dir=str(tmp_path / "data"),
                         registry_dir=str(tmp_path / "registry"), models=FAST_MODELS,
                         bootstrap=False, **kw)


def test_rolling_split_matches_the_projects_original_boundaries():
    ts = pd.date_range("2026-04-01", periods=60 * 4, freq="6h")
    df = pd.DataFrame({"timestamp": ts, "is_fraud": 0})
    train, val, test = rolling_split(df, val_days=10, test_days=10)
    days = lambda d: ((d["timestamp"] - pd.Timestamp("2026-04-01")).dt.days)
    assert days(train).max() == 39 and days(val).min() == 40 and days(val).max() == 49
    assert days(test).min() == 50
    # one more day shifts every boundary forward
    df2 = pd.DataFrame({"timestamp": pd.date_range("2026-04-01", periods=61 * 4, freq="6h"), "is_fraud": 0})
    _, v2, t2 = rolling_split(df2, val_days=10, test_days=10)
    assert days(v2).min() == 41 and days(t2).min() == 51


def test_first_run_generates_a_day_trains_and_promotes(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    report = run_daily_job(c)

    assert report["status"] == "ok"
    assert report["sim_days"] == [60]
    assert report["new_rows"] > 0 and report["total_rows"] > report["new_rows"]
    assert report["split"]["val"] > 0 and report["split"]["test"] > 0
    assert report["winner"] in FAST_MODELS

    store = DailyStore(c.data_dir)
    assert store.state()["next_day"] == 61
    assert len(list((tmp_path / "data" / "daily").glob("day_*.csv"))) == 1

    reg = ModelRegistry(c.registry_dir)
    assert len(reg.list_runs()) == 1
    assert report["decision"]["promoted"] is True          # nothing to beat yet
    assert reg.champion()["run_id"] == report["run_id"]

    log = [json.loads(l) for l in (tmp_path / "data" / "retrain_log.jsonl").read_text().splitlines()]
    assert [e["job_id"] for e in log] == [report["job_id"]]


def test_each_run_continues_with_the_next_day(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    first = run_daily_job(c)
    second = run_daily_job(c)
    assert (first["sim_days"], second["sim_days"]) == ([60], [61])
    assert second["total_rows"] == first["total_rows"] + second["new_rows"]
    assert len(ModelRegistry(c.registry_dir).list_runs()) == 2
    assert "promoted" in second["decision"] and second["decision"]["reason"]


def test_several_days_in_one_run(tmp_path, mini_base):
    report = run_daily_job(cfg(tmp_path, mini_base, days=3))
    assert report["sim_days"] == [60, 61, 62]
    assert len(list((tmp_path / "data" / "daily").glob("day_*.csv"))) == 3


def test_bad_model_name_fails_fast_without_touching_state(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    c.models = ["no_such_model"]
    with pytest.raises(ValueError):
        run_daily_job(c)
    assert not (tmp_path / "data" / "daily").exists() or not list((tmp_path / "data" / "daily").glob("*.csv"))
    assert not DailyStore(c.data_dir).has_state()


def test_overlapping_runs_are_refused(tmp_path, mini_base):
    import fcntl
    c = cfg(tmp_path, mini_base)
    os.makedirs(c.data_dir, exist_ok=True)
    with open(os.path.join(c.data_dir, "job.lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(JobAlreadyRunning):
            run_daily_job(c)
    run_daily_job(c)  # lock released: runs normally


def test_drift_is_applied_and_recorded(tmp_path, mini_base):
    plain = run_daily_job(cfg(tmp_path / "a", mini_base))
    drifted = run_daily_job(cfg(tmp_path / "b", mini_base, drift=DriftConfig(amount_scale=3.0)))
    assert drifted["drift"]["amount_scale"] == 3.0 and plain["drift"]["amount_scale"] == 1.0
    day = lambda root: pd.read_csv(next((root / "data" / "daily").glob("day_*.csv")))["amount"].median()
    assert day(tmp_path / "b") > 2 * day(tmp_path / "a")

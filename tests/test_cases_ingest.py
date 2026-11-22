"""Phase 6 tests: flags reach the alert queue from the daily job and from a backfill."""

import json
import os

import pandas as pd
import pytest

from ml.cases.ingest import ingest_directory, main as ingest_main
from ml.cases.store import CaseStore
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


def test_the_daily_job_queues_serious_flags_and_never_acts_on_accounts(tmp_path, mini_base):
    c = cfg(tmp_path, mini_base)
    report = run_daily_job(c)
    m = report["monitoring"]

    store = CaseStore(os.path.join(c.data_dir, "cases.db"))
    alerts = store.list_alerts()
    assert len(alerts) == m["critical"] + m["high"] == m["alerts_created"]
    assert all(a["status"] == "open" and a["account_status"] == "active" for a in alerts)
    assert store.list_accounts() == [] and store.summary()["held_accounts"] == 0


def test_a_failure_while_queueing_does_not_lose_the_monitoring_result(tmp_path, mini_base, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("queue down")

    monkeypatch.setattr(CaseStore, "ingest", boom)
    report = run_daily_job(cfg(tmp_path, mini_base))
    m = report["monitoring"]
    assert m["drift_status"] and m["alerts_created"] is None and "queue down" in m["alerts_error"]
    assert (tmp_path / "data" / "drift_log.jsonl").exists()


def write_day(data_dir, day, records):
    folder = os.path.join(data_dir, "anomalies")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"day_{day:04d}.json"), "w") as f:
        json.dump({"day": day, "job_id": f"job{day}", "records": records}, f)


def flag(customer, day, severity):
    return {"customer_id": customer, "day": day, "severity": severity, "points": 3, "iso_score": 0.6,
            "n_txns": 4, "total_amount": 900.0, "signals": [], "recommended_action": "Monitor"}


def test_backfill_ingests_every_day_file_once(tmp_path):
    write_day(tmp_path, 60, [flag("a", 60, "critical"), flag("b", 60, "watch")])
    write_day(tmp_path, 61, [flag("c", 61, "high")])
    assert ingest_directory(str(tmp_path)) == 2
    assert ingest_directory(str(tmp_path)) == 0                       # idempotent
    assert {a["customer_id"] for a in CaseStore(tmp_path / "cases.db").list_alerts()} == {"a", "c"}


def test_backfill_tolerates_missing_or_corrupt_files(tmp_path):
    assert ingest_directory(str(tmp_path)) == 0
    write_day(tmp_path, 60, [flag("a", 60, "critical")])
    (tmp_path / "anomalies" / "day_0061.json").write_text("not json")
    assert ingest_directory(str(tmp_path)) == 1


def test_backfill_command_reports_what_it_did(tmp_path, capsys):
    write_day(tmp_path, 60, [flag("a", 60, "critical")])
    ingest_main(["--data-dir", str(tmp_path)])
    assert "1 new alert" in capsys.readouterr().out

"""Phase 4 tests: the in-app scheduler and the read-only retraining status API."""

import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.retrain_routes import router
from ml.retrain.job import JobAlreadyRunning
from ml.retrain.scheduler import RetrainScheduler


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_scheduler_runs_repeatedly_then_stops():
    calls = []
    s = RetrainScheduler(lambda: calls.append(time.time()) or {"status": "ok"}, interval_seconds=0.05)
    s.start()
    assert wait_for(lambda: len(calls) >= 3)
    s.stop()
    n = len(calls)
    time.sleep(0.2)
    assert len(calls) == n, "no runs after stop()"
    assert s.status()["enabled"] is False          # stopped
    assert s.status()["last_result"]["status"] == "ok"


def test_scheduler_never_overlaps_runs():
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow():
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.08)
        with lock:
            active[0] -= 1
        return {"status": "ok"}

    s = RetrainScheduler(slow, interval_seconds=0.01)
    s.start()
    time.sleep(0.5)
    s.stop()
    assert peak[0] == 1


def test_a_failing_run_does_not_kill_the_scheduler():
    outcomes = iter([RuntimeError("boom"), {"status": "ok"}, {"status": "ok"}])
    seen = []

    def job():
        o = next(outcomes)
        seen.append(o)
        if isinstance(o, Exception):
            raise o
        return o

    s = RetrainScheduler(job, interval_seconds=0.03)
    s.start()
    assert wait_for(lambda: len(seen) >= 3)
    s.stop()
    assert s.status()["last_result"]["status"] == "ok"
    assert s.status()["last_error"] is None or "boom" in s.status()["last_error"]


def test_another_process_holding_the_lock_is_a_skip_not_an_error():
    def job():
        raise JobAlreadyRunning("busy")

    s = RetrainScheduler(job, interval_seconds=0.02)
    s.start()
    assert wait_for(lambda: s.status()["skipped_runs"] >= 2)
    s.stop()
    assert s.status()["last_error"] is None


def test_status_reports_the_next_run_time():
    s = RetrainScheduler(lambda: {"status": "ok"}, interval_seconds=3600)
    st = s.status()
    assert st["enabled"] is False and st["next_run_at"] is None
    s.start()
    try:
        st = s.status()
        assert st["enabled"] is True and st["interval_hours"] == pytest.approx(1.0)
        assert st["next_run_at"] is not None
    finally:
        s.stop()


# ---------------------------------------------------------------------- API

def make_client(tmp_path, scheduler=None):
    app = FastAPI()
    app.state.retrain_data_dir = str(tmp_path)
    app.state.retrain_scheduler = scheduler
    app.include_router(router)
    return TestClient(app)


def test_status_without_any_history(tmp_path):
    body = make_client(tmp_path).get("/retrain/status").json()
    assert body["scheduler"]["enabled"] is False
    assert body["last_run"] is None and body["next_simulated_day"] is None
    assert make_client(tmp_path).get("/retrain/log").json() == {"runs": []}


def test_log_and_status_read_the_job_log(tmp_path):
    entries = [{"job_id": f"j{i}", "status": "ok", "sim_days": [60 + i], "finished_at": f"2026-11-2{i}T00:00:00Z"}
               for i in range(3)]
    (tmp_path / "retrain_log.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    (tmp_path / "state.json").write_text(json.dumps({"next_day": 63}))
    c = make_client(tmp_path)

    runs = c.get("/retrain/log").json()["runs"]
    assert [r["job_id"] for r in runs] == ["j2", "j1", "j0"]             # newest first
    assert [r["job_id"] for r in c.get("/retrain/log?limit=2").json()["runs"]] == ["j2", "j1"]
    st = c.get("/retrain/status").json()
    assert st["last_run"]["job_id"] == "j2" and st["next_simulated_day"] == 63


def test_a_corrupt_log_line_is_skipped(tmp_path):
    (tmp_path / "retrain_log.jsonl").write_text('{"job_id": "ok1", "status": "ok"}\nnot json\n')
    assert [r["job_id"] for r in make_client(tmp_path).get("/retrain/log").json()["runs"]] == ["ok1"]

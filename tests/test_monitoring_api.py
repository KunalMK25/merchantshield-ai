"""Phase 5 tests: read-only monitoring endpoints."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.monitoring_routes import router


def client(tmp_path):
    app = FastAPI()
    app.state.retrain_data_dir = str(tmp_path)
    app.include_router(router)
    return TestClient(app)


def drift_line(job, status, psi, days):
    return {"job_id": job, "at": f"2026-11-2{days[0] - 59}T00:00:00Z", "sim_days": days,
            "drift": {"status": status, "max_psi": psi, "n_reference": 5000, "n_current": 3500,
                      "features": [{"feature": "amount", "psi": psi, "severity": "significant" if psi > .25 else "stable"}],
                      "score": {"psi": 0.02, "severity": "stable"}},
            "anomalies": {"flagged": 2, "accounts_checked": 900, "by_severity": {"critical": 1, "high": 0, "watch": 1}}}


def test_everything_is_empty_before_any_run(tmp_path):
    c = client(tmp_path)
    assert c.get("/monitoring/drift/latest").json() == {"drift": None}
    assert c.get("/monitoring/drift/history").json() == {"history": []}
    assert c.get("/monitoring/anomalies/latest").json() == {"day": None, "summary": None, "records": []}


def test_drift_latest_and_history(tmp_path):
    lines = [drift_line("j1", "stable", 0.02, [60]), drift_line("j2", "drifted", 0.8, [61])]
    (tmp_path / "drift_log.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    c = client(tmp_path)

    latest = c.get("/monitoring/drift/latest").json()
    assert latest["job_id"] == "j2" and latest["drift"]["status"] == "drifted"

    hist = c.get("/monitoring/drift/history").json()["history"]
    assert [h["job_id"] for h in hist] == ["j2", "j1"]                          # newest first, bad line skipped
    assert hist[0]["status"] == "drifted" and hist[0]["flagged_features"] == ["amount"]
    assert hist[0]["accounts_flagged"] == 2 and hist[0]["score_psi"] == 0.02
    assert [h["job_id"] for h in c.get("/monitoring/drift/history?limit=1").json()["history"]] == ["j2"]


def test_latest_anomalies_pick_the_newest_day_and_can_filter(tmp_path):
    (tmp_path / "anomalies").mkdir()
    recs = [{"customer_id": "a", "severity": "critical"}, {"customer_id": "b", "severity": "watch"},
            {"customer_id": "c", "severity": "high"}]
    for day in (60, 61):
        (tmp_path / "anomalies" / f"day_{day:04d}.json").write_text(
            json.dumps({"day": day, "summary": {"flagged": 3}, "records": recs if day == 61 else []}))
    c = client(tmp_path)

    body = c.get("/monitoring/anomalies/latest").json()
    assert body["day"] == 61 and len(body["records"]) == 3
    serious = c.get("/monitoring/anomalies/latest?severity=critical&severity=high").json()
    assert [r["customer_id"] for r in serious["records"]] == ["a", "c"]
    assert len(c.get("/monitoring/anomalies/latest?limit=1").json()["records"]) == 1
    assert c.get("/monitoring/anomalies/latest?day=60").json()["day"] == 60
    assert c.get("/monitoring/anomalies/latest?day=99").json()["records"] == []

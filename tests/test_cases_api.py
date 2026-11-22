"""Phase 6 tests: the case-management API and its analyst guard."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.case_routes import router
from ml.cases.store import CaseStore


def rec(customer, day=60, severity="critical"):
    return {"customer_id": customer, "day": day, "severity": severity, "points": 3, "iso_score": 0.7,
            "n_txns": 5, "total_amount": 1234.0, "signals": [{"name": "takeover", "label": "x", "detail": "d"}],
            "recommended_action": "Recommend temporary account hold pending analyst review"}


def make(tmp_path, token=None, with_store=True):
    app = FastAPI()
    app.state.cases_store = CaseStore(tmp_path / "cases.db") if with_store else None
    app.state.analyst_token = token
    app.include_router(router)
    return TestClient(app), app.state.cases_store


def body(name="Priya Nair", note="checked"):
    return {"decided_by": name, "note": note}


def test_store_unavailable_gives_503(tmp_path):
    c, _ = make(tmp_path, with_store=False)
    assert c.get("/cases/alerts").status_code == 503
    assert c.post("/cases/alerts/1/hold", json=body()).status_code == 503


def test_listing_filtering_summary_and_detail(tmp_path):
    c, store = make(tmp_path)
    store.ingest([rec("a"), rec("b", severity="high")])
    alerts = c.get("/cases/alerts").json()["alerts"]
    assert [a["customer_id"] for a in alerts] == ["a", "b"]
    assert [a["customer_id"] for a in c.get("/cases/alerts?severity=high").json()["alerts"]] == ["b"]
    assert c.get("/cases/summary").json()["open_alerts"] == 2
    detail = c.get(f"/cases/alerts/{alerts[0]['id']}").json()
    assert detail["customer_id"] == "a" and detail["notices"] == [] and detail["events"]
    assert c.get("/cases/alerts/9999").status_code == 404


def test_hold_flow_over_http(tmp_path):
    c, store = make(tmp_path)
    store.ingest([rec("a")])
    aid = c.get("/cases/alerts").json()["alerts"][0]["id"]

    r = c.post(f"/cases/alerts/{aid}/hold", json=body())
    assert r.status_code == 200
    out = r.json()
    assert out["account"]["status"] == "held" and out["notice"]["kind"] == "hold"
    assert c.post(f"/cases/alerts/{aid}/hold", json=body()).status_code == 409   # already decided
    assert [x["customer_id"] for x in c.get("/cases/accounts?status=held").json()["accounts"]] == ["a"]
    assert c.get("/cases/alerts?status=open").json()["alerts"] == []
    assert any(e["kind"] == "hold_placed" for e in c.get("/cases/events").json()["events"])


def test_dismiss_release_and_close_over_http(tmp_path):
    c, store = make(tmp_path)
    store.ingest([rec("a"), rec("b")])
    ids = {a["customer_id"]: a["id"] for a in c.get("/cases/alerts").json()["alerts"]}

    assert c.post(f"/cases/alerts/{ids['b']}/dismiss", json=body()).json()["alert"]["status"] == "dismissed"
    c.post(f"/cases/alerts/{ids['a']}/hold", json=body())
    assert c.post("/cases/accounts/a/release", json=body(note="")).status_code == 422     # reason required
    assert c.post("/cases/accounts/a/release", json=body(note="verified")).json()["account"]["status"] == "active"
    assert c.post("/cases/accounts/a/close", json=body(note="x" * 5)).status_code == 409   # not held any more


def test_bad_requests_are_rejected_cleanly(tmp_path):
    c, store = make(tmp_path)
    store.ingest([rec("a")])
    aid = c.get("/cases/alerts").json()["alerts"][0]["id"]
    assert c.post(f"/cases/alerts/{aid}/hold", json={"note": "no analyst"}).status_code == 422
    assert c.post(f"/cases/alerts/{aid}/hold", json=body(name="x")).status_code == 422
    assert c.post(f"/cases/alerts/{aid}/hold", json=body(note="n" * 600)).status_code == 422
    assert c.post("/cases/alerts/9999/hold", json=body()).status_code == 404
    assert c.get("/cases/alerts/%s" % aid).json()["status"] == "open"


def test_when_a_token_is_configured_writes_need_it_but_reads_do_not(tmp_path):
    c, store = make(tmp_path, token="s3cret")
    store.ingest([rec("a")])
    aid = c.get("/cases/alerts").json()["alerts"][0]["id"]                 # reads stay open
    assert c.get("/cases/summary").status_code == 200

    assert c.post(f"/cases/alerts/{aid}/hold", json=body()).status_code == 401
    assert c.post(f"/cases/alerts/{aid}/hold", json=body(), headers={"X-Analyst-Token": "nope"}).status_code == 401
    assert c.get(f"/cases/alerts/{aid}").json()["status"] == "open"        # nothing changed
    ok = c.post(f"/cases/alerts/{aid}/hold", json=body(), headers={"X-Analyst-Token": "s3cret"})
    assert ok.status_code == 200


def test_without_a_token_writes_are_allowed_for_the_demo(tmp_path):
    c, store = make(tmp_path)
    store.ingest([rec("a")])
    aid = c.get("/cases/alerts").json()["alerts"][0]["id"]
    assert c.post(f"/cases/alerts/{aid}/dismiss", json=body()).status_code == 200


def test_an_empty_token_setting_means_demo_mode_not_a_guard(tmp_path):
    c, store = make(tmp_path, token="")        # MS_ANALYST_TOKEN="" is treated as unset
    store.ingest([rec("a")])
    aid = c.get("/cases/alerts").json()["alerts"][0]["id"]
    assert c.post(f"/cases/alerts/{aid}/hold", json=body()).status_code == 200

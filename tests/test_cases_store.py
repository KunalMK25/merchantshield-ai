"""
Phase 6 tests: the alert queue, simulated account holds, customer notices and audit trail.

The rule these tests protect: the system only RECOMMENDS. An account changes state only
through an explicit, attributed human action, and every change leaves an audit event.
"""

import pytest

from ml.cases.store import CaseError, CaseStore, Conflict, NotFound


def rec(customer="cust_001", day=60, severity="critical", **kw):
    return {"customer_id": customer, "day": day, "severity": severity, "points": 3 if severity == "critical" else 2,
            "iso_score": 0.71, "iso_flag": True, "n_txns": 9, "total_amount": 48000.0,
            "signals": [{"name": "takeover", "label": "New device and new region",
                         "detail": "Used 1 new device(s) from 1 new region(s) in one day"}],
            "recommended_action": "Recommend temporary account hold pending analyst review", **kw}


@pytest.fixture
def store(tmp_path):
    return CaseStore(tmp_path / "cases.db")


def first_alert(store, **kw):
    store.ingest([rec(**kw)], job_id="job1")
    return store.list_alerts()[0]


# ------------------------------------------------------------------ ingest

def test_only_critical_and_high_flags_become_alerts(store):
    n = store.ingest([rec("a", severity="critical"), rec("b", severity="high"), rec("c", severity="watch")])
    assert n == 2
    assert {a["customer_id"] for a in store.list_alerts()} == {"a", "b"}


def test_ingest_is_idempotent_and_never_reopens_a_decided_alert(store):
    store.ingest([rec()])
    alert = store.list_alerts()[0]
    store.dismiss(alert["id"], "Priya", "customer travelling")
    assert store.ingest([rec()]) == 0
    assert len(store.list_alerts()) == 1
    assert store.get_alert(alert["id"])["status"] == "dismissed"


def test_ingest_alone_never_changes_an_account(store):
    store.ingest([rec(severity="critical")])
    assert store.account("cust_001")["status"] == "active"
    assert store.list_accounts() == []


def test_alerts_keep_their_evidence(store):
    a = first_alert(store)
    assert a["severity"] == "critical" and a["status"] == "open" and a["n_txns"] == 9
    assert a["signals"][0]["name"] == "takeover"
    assert a["account_status"] == "active"


def test_queue_is_ordered_critical_first_then_newest(store):
    store.ingest([rec("a", 60, "high"), rec("b", 59, "critical"), rec("c", 61, "critical"), rec("d", 61, "high")])
    assert [a["customer_id"] for a in store.list_alerts()] == ["c", "b", "d", "a"]


# ------------------------------------------------------------------- hold

def test_a_hold_is_a_human_decision_with_a_notice_and_an_audit_trail(store):
    a = first_alert(store)
    out = store.hold(a["id"], "Priya Nair", "Matches customer's compromised-device pattern")

    assert out["alert"]["status"] == "actioned" and out["alert"]["decided_by"] == "Priya Nair"
    assert out["account"]["status"] == "held"
    assert out["notice"]["kind"] == "hold" and "temporarily" in out["notice"]["message"]
    assert "not sent" in out["notice"]["delivery"].lower()

    kinds = [e["kind"] for e in store.events()]
    assert {"alert_created", "hold_placed", "notice_created"} <= set(kinds)
    placed = next(e for e in store.events() if e["kind"] == "hold_placed")
    assert placed["actor"] == "Priya Nair" and placed["customer_id"] == "cust_001"


def test_notices_never_reveal_how_fraud_is_detected(store):
    a = first_alert(store)
    msg = store.hold(a["id"], "Priya", "x")["notice"]["message"].lower()
    for leak in ("signal", "isolation", "anomaly", "device", "region", "threshold", "model"):
        assert leak not in msg


def test_an_alert_can_only_be_decided_once(store):
    a = first_alert(store)
    store.hold(a["id"], "Priya", "x")
    with pytest.raises(Conflict):
        store.hold(a["id"], "Ravi", "again")
    with pytest.raises(Conflict):
        store.dismiss(a["id"], "Ravi", "again")


def test_a_second_alert_on_a_held_account_confirms_without_a_second_notice(store):
    store.ingest([rec(day=60), rec(day=61)])
    older, newer = sorted(store.list_alerts(), key=lambda a: a["day"])
    store.hold(older["id"], "Priya", "first")
    out = store.hold(newer["id"], "Priya", "second")
    assert out["alert"]["status"] == "actioned" and out["notice"] is None
    assert sum(1 for e in store.events() if e["kind"] == "notice_created") == 1


def test_unknown_alert_and_blank_analyst_are_rejected(store):
    a = first_alert(store)
    with pytest.raises(NotFound):
        store.hold(9999, "Priya", "")
    for bad in ("", "  ", "x", "y" * 200):
        with pytest.raises(ValueError):
            store.hold(a["id"], bad, "")
    assert store.get_alert(a["id"])["status"] == "open"


# ---------------------------------------------------------------- dismiss

def test_dismissing_leaves_the_account_alone_and_sends_nothing(store):
    a = first_alert(store)
    out = store.dismiss(a["id"], "Priya", "customer confirmed the purchases")
    assert out["alert"]["status"] == "dismissed" and out["alert"]["decision_note"].startswith("customer confirmed")
    assert store.account("cust_001")["status"] == "active"
    assert "notice_created" not in [e["kind"] for e in store.events()]


# ---------------------------------------------------- release and closing

def held(store):
    a = first_alert(store)
    store.hold(a["id"], "Priya", "x")
    return a


def test_release_restores_the_account_and_needs_a_reason(store):
    held(store)
    with pytest.raises(ValueError):
        store.release("cust_001", "Priya", "")
    out = store.release("cust_001", "Priya", "Customer verified by phone")
    assert out["account"]["status"] == "active" and out["notice"]["kind"] == "release"


def test_release_requires_a_held_account(store):
    first_alert(store)
    with pytest.raises(Conflict):
        store.release("cust_001", "Priya", "not held")
    with pytest.raises(Conflict):
        store.release("nobody", "Priya", "unknown account")


def test_closing_is_a_second_explicit_step_and_final(store):
    a = first_alert(store)
    with pytest.raises(Conflict):
        store.close("cust_001", "Priya", "straight to close")      # must be held first
    store.hold(a["id"], "Priya", "x")
    with pytest.raises(ValueError):
        store.close("cust_001", "Priya", "")
    out = store.close("cust_001", "Ravi", "Fraud confirmed with customer; reissue required")
    assert out["account"]["status"] == "closed" and out["notice"]["kind"] == "close"
    for action in (lambda: store.release("cust_001", "Ravi", "undo"),
                   lambda: store.close("cust_001", "Ravi", "again")):
        with pytest.raises(Conflict):
            action()


def test_a_closed_account_cannot_be_held_again(store):
    store.ingest([rec(day=60), rec(day=61)])
    older, newer = sorted(store.list_alerts(), key=lambda a: a["day"])
    store.hold(older["id"], "Priya", "x")
    store.close("cust_001", "Priya", "confirmed fraud")
    with pytest.raises(Conflict):
        store.hold(newer["id"], "Priya", "too late")
    store.dismiss(newer["id"], "Priya", "account already closed")   # still resolvable


# --------------------------------------------------------- integrity, views

def test_a_failure_midway_leaves_no_partial_change(store, monkeypatch):
    a = first_alert(store)

    def boom(*args, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(CaseStore, "_insert_notice", boom)
    with pytest.raises(RuntimeError):
        store.hold(a["id"], "Priya", "x")
    assert store.get_alert(a["id"])["status"] == "open"
    assert store.account("cust_001")["status"] == "active"
    assert "hold_placed" not in [e["kind"] for e in store.events()]


def test_get_alert_bundles_account_notices_and_history(store):
    a = first_alert(store)
    store.hold(a["id"], "Priya", "x")
    full = store.get_alert(a["id"])
    assert full["account_status"] == "held"
    assert [n["kind"] for n in full["notices"]] == ["hold"]
    assert {"alert_created", "hold_placed"} <= {e["kind"] for e in full["events"]}


def test_filters_and_summary(store):
    store.ingest([rec("a", severity="critical"), rec("b", severity="high"), rec("c", 61, "critical")])
    alerts = {a["customer_id"]: a for a in store.list_alerts()}
    store.hold(alerts["a"]["id"], "Priya", "x")
    store.dismiss(alerts["b"]["id"], "Priya", "fine")

    assert [a["customer_id"] for a in store.list_alerts(status="open")] == ["c"]
    assert {a["customer_id"] for a in store.list_alerts(severities=["critical"])} == {"a", "c"}
    s = store.summary()
    assert (s["open_alerts"], s["open_critical"], s["open_high"]) == (1, 1, 0)
    assert (s["held_accounts"], s["closed_accounts"], s["dismissed_alerts"]) == (1, 0, 1)
    assert [x["customer_id"] for x in store.list_accounts(status="held")] == ["a"]


def test_state_survives_reopening_the_database(tmp_path):
    path = tmp_path / "cases.db"
    s1 = CaseStore(path)
    s1.ingest([rec()])
    s1.hold(s1.list_alerts()[0]["id"], "Priya", "x")
    s2 = CaseStore(path)
    assert s2.account("cust_001")["status"] == "held" and len(s2.events()) >= 3

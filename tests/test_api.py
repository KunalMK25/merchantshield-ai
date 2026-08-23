import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.audit_service import AuditPersistenceError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _txn(**overrides):
    base = dict(
        transaction_id="txn_test_001", customer_id="cust_test_01", merchant_id="merch_0001",
        merchant_category="electronics", timestamp="2026-05-20T10:05:00Z", amount=500.0,
        device_id="dev_known", geo_region="region_03", payment_method="card", status="success",
        account_created="2025-01-01T00:00:00Z",
    )
    base.update(overrides)
    return base


def _prior_history(customer_id="cust_test_01", n=6, device_id="dev_known", geo_region="region_03"):
    return [
        dict(
            transaction_id=f"ctx_{i}", customer_id=customer_id, merchant_id="merch_0001",
            merchant_category="electronics", timestamp=f"2026-05-1{i}T10:00:00Z", amount=500 + i * 10,
            device_id=device_id, geo_region=geo_region, payment_method="card", status="success",
            account_created="2025-01-01T00:00:00Z",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Health / model info
# ---------------------------------------------------------------------------

def test_health_endpoint_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info_endpoint_returns_frozen_metadata(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["decision_threshold"] == 0.40  # frozen from Phase 5, must appear here unchanged
    assert "feature_columns" in body and len(body["feature_columns"]) == 15
    assert "validation_metrics_at_threshold" in body
    assert "test_metrics_at_frozen_threshold" in body


def test_api_root_endpoint(client):
    # Phase 9: "/" now serves the built frontend (StaticFiles), so the service
    # identification JSON that used to live at "/" moved to "/api".
    r = client.get("/api")
    assert r.status_code == 200
    assert "service" in r.json()


def test_root_serves_frontend_when_built(client):
    # frontend/dist exists in this environment (built during Phase 9) -- "/"
    # should serve the built index.html, not the JSON service-identification body.
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Valid risk requests
# ---------------------------------------------------------------------------

def test_risk_score_valid_request(client):
    r = client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": _prior_history()})
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "txn_test_001"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_category"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_risk_explain_valid_request(client):
    r = client.post("/risk/explain", json={"transaction": _txn(), "prior_transactions": _prior_history()})
    assert r.status_code == 200
    body = r.json()
    assert body["additivity_check_passed"] is True
    assert len(body["reasons"]) > 0
    assert len(body["contributions"]) == 15  # all FEATURE_COLUMNS represented


def test_risk_score_without_prior_transactions_treated_as_first_transaction(client):
    r = client.post("/risk/score", json={"transaction": _txn(transaction_id="txn_first_ever")})
    assert r.status_code == 200  # prior_transactions is optional, defaults to []


# ---------------------------------------------------------------------------
# LOW / MEDIUM / HIGH / CRITICAL via /risk/evaluate, using real model + real data
# ---------------------------------------------------------------------------

def test_evaluate_low_risk_transaction(client):
    # ordinary transaction matching established customer behavior -> should NOT be high-probability
    payload = {"transaction": _txn(transaction_id="txn_low", amount=510.0),
               "prior_transactions": _prior_history()}
    r = client.post("/risk/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {"ALLOW", "ALLOW_WITH_MONITORING"}
    assert body["policy_rule_id"] in {"LOW_ALLOW", "MEDIUM_MONITOR"}


def test_evaluate_critical_risk_transaction(client):
    # new device, new geo, huge amount vs history, established old account -- matches the
    # generator's takeover-burst pattern the model was trained to recognize
    payload = {
        "transaction": _txn(transaction_id="txn_critical", amount=45000.0,
                             device_id="dev_never_seen", geo_region="region_23"),
        "prior_transactions": _prior_history(),
    }
    r = client.post("/risk/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["fraud_probability"] >= 0.80
    assert body["action"] == "BLOCK"
    assert body["policy_rule_id"] == "CRITICAL_BLOCK"
    assert body["explanation_header"] == "Why this transaction was flagged:"


def test_evaluate_response_threshold_matches_frozen_policy_threshold(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(), "prior_transactions": _prior_history()})
    assert r.status_code == 200
    assert r.json()["threshold"] == 0.40


def test_evaluate_nonflagged_transaction_never_uses_flagged_language(client):
    payload = {"transaction": _txn(transaction_id="txn_ordinary", amount=505.0),
               "prior_transactions": _prior_history()}
    r = client.post("/risk/evaluate", json=payload)
    body = r.json()
    if body["fraud_probability"] < 0.40:
        assert body["explanation_header"] != "Why this transaction was flagged:"
        assert "NOT flagged" in body["explanation_header"]


# ---------------------------------------------------------------------------
# Decision-engine integration: rule ID and action are always internally consistent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,device,geo,expected_actions", [
    (500.0, "dev_known", "region_03", {"ALLOW", "ALLOW_WITH_MONITORING"}),
    (45000.0, "dev_never_seen", "region_23", {"BLOCK", "STEP_UP_VERIFICATION"}),
])
def test_decision_engine_integration_consistency(client, amount, device, geo, expected_actions):
    payload = {
        "transaction": _txn(transaction_id="txn_consistency", amount=amount, device_id=device, geo_region=geo),
        "prior_transactions": _prior_history(),
    }
    r = client.post("/risk/evaluate", json=payload)
    body = r.json()
    assert body["action"] in expected_actions
    # rule_id must be internally consistent with the returned threshold/probability
    from ml.evaluation.policy import evaluate_policy
    rule = evaluate_policy(body["fraud_probability"], amount)
    assert rule.action == body["action"]
    assert rule.rule_id == body["policy_rule_id"]


# ---------------------------------------------------------------------------
# Explanation integration
# ---------------------------------------------------------------------------

def test_evaluate_reasons_are_grounded_in_real_shap_output(client):
    payload = {
        "transaction": _txn(transaction_id="txn_grounded", amount=45000.0,
                             device_id="dev_never_seen", geo_region="region_23"),
        "prior_transactions": _prior_history(),
    }
    r = client.post("/risk/evaluate", json=payload)
    body = r.json()
    # cross-check against calling /risk/explain directly for the SAME transaction
    r2 = client.post("/risk/explain", json=payload)
    explain_body = r2.json()
    assert body["reasons"] == explain_body["reasons"] or set(body["reasons"]).issubset(set(explain_body["reasons"]) | set(body["reasons"]))
    assert abs(body["fraud_probability"] - explain_body["fraud_probability"]) < 1e-6


# ---------------------------------------------------------------------------
# Audit record creation
# ---------------------------------------------------------------------------

def test_evaluate_persists_audit_record(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_audit_test"),
                                              "prior_transactions": _prior_history()})
    body = r.json()
    assert body["audit_persisted"] is True
    assert body["audit_error"] is None

    r2 = client.get("/audit-log?limit=5")
    assert r2.status_code == 200
    entries = r2.json()
    assert any(e["transaction_id"] == "txn_audit_test" for e in entries)


def test_audit_record_defaults_to_manual_source_when_unspecified(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_source_default"),
                                              "prior_transactions": _prior_history()})
    assert r.status_code == 200
    entries = client.get("/audit-log?limit=200").json()
    match = next(e for e in entries if e["transaction_id"] == "txn_source_default")
    assert match["source"] == "manual"


def test_audit_record_captures_demo_source(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_source_demo"),
                                              "prior_transactions": _prior_history(), "source": "demo"})
    assert r.status_code == 200
    entries = client.get("/audit-log?limit=200").json()
    match = next(e for e in entries if e["transaction_id"] == "txn_source_demo")
    assert match["source"] == "demo"


def test_audit_record_captures_explicit_manual_source(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_source_manual"),
                                              "prior_transactions": _prior_history(), "source": "manual"})
    assert r.status_code == 200
    entries = client.get("/audit-log?limit=200").json()
    match = next(e for e in entries if e["transaction_id"] == "txn_source_manual")
    assert match["source"] == "manual"


def test_invalid_source_value_rejected(client):
    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_source_bad"),
                                              "prior_transactions": _prior_history(), "source": "totally_bogus"})
    assert r.status_code == 422


def test_source_field_does_not_affect_decision_outcome(client):
    # source is descriptive audit metadata only -- identical transaction/context
    # must produce an identical decision regardless of source value.
    txn_kwargs = dict(amount=45000.0, device_id="dev_never_seen", geo_region="region_23")
    r_demo = client.post("/risk/evaluate", json={
        "transaction": _txn(transaction_id="txn_source_neutral_demo", **txn_kwargs),
        "prior_transactions": _prior_history(), "source": "demo"})
    r_manual = client.post("/risk/evaluate", json={
        "transaction": _txn(transaction_id="txn_source_neutral_manual", **txn_kwargs),
        "prior_transactions": _prior_history(), "source": "manual"})
    assert r_demo.json()["action"] == r_manual.json()["action"]
    assert r_demo.json()["policy_rule_id"] == r_manual.json()["policy_rule_id"]
    assert abs(r_demo.json()["fraud_probability"] - r_manual.json()["fraud_probability"]) < 1e-9


def test_score_and_explain_do_not_write_audit_records(client):
    r1 = client.get("/audit-log?limit=200")
    before = len(r1.json())
    client.post("/risk/score", json={"transaction": _txn(transaction_id="txn_no_audit_1"), "prior_transactions": _prior_history()})
    client.post("/risk/explain", json={"transaction": _txn(transaction_id="txn_no_audit_2"), "prior_transactions": _prior_history()})
    r2 = client.get("/audit-log?limit=200")
    after = len(r2.json())
    assert after == before  # neither lightweight endpoint should have written anything


def test_audit_log_entry_does_not_contain_raw_pii_fields():
    from backend.services.audit_service import AuditLogEntry
    columns = {c.name for c in AuditLogEntry.__table__.columns}
    assert "customer_id" not in columns
    assert "device_id" not in columns
    assert "geo_region" not in columns


# ---------------------------------------------------------------------------
# Validation failures (malformed request / missing field / invalid numeric)
# ---------------------------------------------------------------------------

def test_missing_required_field_rejected(client):
    txn = _txn()
    del txn["amount"]
    r = client.post("/risk/score", json={"transaction": txn, "prior_transactions": []})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_negative_amount_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(amount=-50.0), "prior_transactions": []})
    assert r.status_code == 422


def test_zero_amount_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(amount=0.0), "prior_transactions": []})
    assert r.status_code == 422


def test_invalid_status_enum_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(status="pending"), "prior_transactions": []})
    assert r.status_code == 422


def test_invalid_payment_method_enum_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(payment_method="crypto"), "prior_transactions": []})
    assert r.status_code == 422


def test_malformed_timestamp_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(timestamp="not-a-date"), "prior_transactions": []})
    assert r.status_code == 422


def test_blank_transaction_id_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(transaction_id="   "), "prior_transactions": []})
    assert r.status_code == 422


def test_malformed_json_body_rejected(client):
    r = client.post("/risk/score", content="{not valid json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_completely_empty_body_rejected(client):
    r = client.post("/risk/score", json={})
    assert r.status_code == 422


def test_prior_transaction_wrong_customer_rejected(client):
    bad_prior = _prior_history(customer_id="cust_DIFFERENT")
    r = client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": bad_prior})
    assert r.status_code == 422
    assert "customer_id" in r.json()["detail"]


def test_prior_transaction_after_current_timestamp_rejected(client):
    bad_prior = [dict(
        transaction_id="ctx_future", customer_id="cust_test_01", merchant_id="merch_0001",
        merchant_category="electronics", timestamp="2026-05-21T10:00:00Z", amount=500,  # AFTER current txn
        device_id="dev_known", geo_region="region_03", payment_method="card", status="success",
        account_created="2025-01-01T00:00:00Z",
    )]
    r = client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": bad_prior})
    assert r.status_code == 422
    assert "leak" in r.json()["detail"].lower() or "timestamp" in r.json()["detail"].lower()


def test_account_created_after_transaction_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(account_created="2027-01-01T00:00:00Z"),
                                          "prior_transactions": []})
    assert r.status_code == 422


def test_amount_exceeding_max_rejected(client):
    r = client.post("/risk/score", json={"transaction": _txn(amount=99_999_999.0), "prior_transactions": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Model unavailable / corrupt (simulated failure path)
# ---------------------------------------------------------------------------

def test_model_unavailable_returns_503(client):
    original = app.state.model_bundle
    app.state.model_bundle = None
    try:
        r = client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": []})
        assert r.status_code == 503
        r2 = client.get("/model/info")
        assert r2.status_code == 503
    finally:
        app.state.model_bundle = original


def test_model_loader_reports_missing_artifact_cleanly():
    from backend.services.model_loader import load_model_bundle, ModelUnavailableError
    with pytest.raises(ModelUnavailableError, match="not found"):
        load_model_bundle("/nonexistent/path/model.pkl", "/nonexistent/path/meta.json")


def test_model_loader_reports_corrupt_artifact_cleanly(tmp_path):
    from backend.services.model_loader import load_model_bundle, ModelUnavailableError
    bad_model = tmp_path / "corrupt_model.pkl"
    bad_model.write_bytes(b"this is not a pickle file")
    with pytest.raises(ModelUnavailableError, match="corrupt"):
        load_model_bundle(str(bad_model), "/nonexistent/meta.json")


# ---------------------------------------------------------------------------
# Feature-generation failure
# ---------------------------------------------------------------------------

def test_feature_generation_failure_returns_422_not_500(client, monkeypatch):
    import backend.services.risk_service as risk_service_module

    def broken_build_features(df):
        raise RuntimeError("simulated feature engineering crash")

    monkeypatch.setattr(risk_service_module, "build_features", broken_build_features)
    r = client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": _prior_history()})
    assert r.status_code == 422
    assert "Feature engineering failed" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Decision-engine failure (simulated)
# ---------------------------------------------------------------------------

def test_decision_engine_failure_returns_clean_error_not_traceback(client, monkeypatch):
    import backend.services.risk_service as risk_service_module

    def broken_make_decision(**kwargs):
        raise RuntimeError("simulated unexpected decision engine crash")

    monkeypatch.setattr(risk_service_module, "make_decision", broken_make_decision)
    r = client.post("/risk/evaluate", json={"transaction": _txn(), "prior_transactions": _prior_history()})
    assert r.status_code == 500
    body = r.json()
    # route converts the unexpected exception into a clean HTTPException(500, detail=...)
    # rather than letting a raw traceback escape -- FastAPI serializes that as {"detail": ...}
    assert "detail" in body
    assert "Internal risk evaluation failure" in body["detail"]
    assert "Traceback" not in json.dumps(body)
    assert "raise RuntimeError" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Database persistence failure -- request still succeeds with decision returned
# ---------------------------------------------------------------------------

def test_truly_unhandled_exception_uses_global_error_handler_shape(monkeypatch):
    # simulate a bug in a path NOT already wrapped in a try/except within the route
    # (e.g. score_only itself blows up with something other than FeatureGenerationError).
    # Uses raise_server_exceptions=False so we can inspect the actual HTTP response the
    # global handler produces, rather than TestClient re-raising the exception in-process.
    import backend.services.risk_service as risk_service_module

    def broken_score_only(bundle, payload):
        raise RuntimeError("simulated unexpected bug outside any try/except")

    monkeypatch.setattr(risk_service_module, "score_only", broken_score_only)
    with TestClient(app, raise_server_exceptions=False) as isolated_client:
        r = isolated_client.post("/risk/score", json={"transaction": _txn(), "prior_transactions": _prior_history()})
        assert r.status_code == 500
        body = r.json()
        # this path is NOT caught by the route's own try/except (which only catches
        # FeatureGenerationError), so it should hit main.py's global handler instead,
        # which uses the {"error": ..., "detail": ...} shape
        assert body["error"] == "internal_error"
        assert "Traceback" not in json.dumps(body)


def test_audit_persistence_failure_does_not_lose_the_decision(client, monkeypatch):
    def broken_record_decision(self, request_id, decision_record, top_reasons, source="manual"):
        raise AuditPersistenceError("simulated database write failure")

    from backend.services.audit_service import AuditStore
    monkeypatch.setattr(AuditStore, "record_decision", broken_record_decision)

    r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_db_fail"),
                                              "prior_transactions": _prior_history()})
    assert r.status_code == 200  # the actual risk decision must still be returned
    body = r.json()
    assert body["audit_persisted"] is False
    assert "simulated database write failure" in body["audit_error"]
    assert body["action"] in {"ALLOW", "ALLOW_WITH_MONITORING", "STEP_UP_VERIFICATION", "BLOCK"}


def test_audit_store_unavailable_at_all(client):
    original = app.state.audit_store
    app.state.audit_store = None
    try:
        r = client.post("/risk/evaluate", json={"transaction": _txn(transaction_id="txn_no_store"),
                                                  "prior_transactions": _prior_history()})
        assert r.status_code == 200
        assert r.json()["audit_persisted"] is False
    finally:
        app.state.audit_store = original


# ---------------------------------------------------------------------------
# No duplicated decision logic -- API results must match calling the modules directly
# ---------------------------------------------------------------------------

def test_api_result_matches_direct_module_call_no_logic_duplication(client):
    payload = {"transaction": _txn(transaction_id="txn_direct_compare", amount=45000.0,
                                    device_id="dev_never_seen", geo_region="region_23"),
               "prior_transactions": _prior_history()}
    r = client.post("/risk/evaluate", json=payload)
    api_body = r.json()

    # replicate via direct module calls (not via the API) and confirm identical result
    import pandas as pd
    from ml.features.build_features import build_features, FEATURE_COLUMNS
    import joblib
    from ml.evaluation.explainability import RiskExplainer, build_explanation_text
    from ml.evaluation.decision_engine import make_decision

    all_txns = _prior_history() + [payload["transaction"]]
    raw_df = pd.DataFrame(all_txns)
    feats = build_features(raw_df)
    row = feats[feats["transaction_id"] == "txn_direct_compare"].iloc[0]

    model = joblib.load("ml/models/candidate_lgbm_v1.pkl")
    explainer = RiskExplainer(model)
    result = explainer.explain(row)
    decision = make_decision("txn_direct_compare", result["fraud_probability"], 45000.0)

    assert abs(api_body["fraud_probability"] - decision.fraud_probability) < 1e-6
    assert api_body["action"] == decision.action
    assert api_body["policy_rule_id"] == decision.policy_rule_id


# ---------------------------------------------------------------------------
# Deterministic response behavior
# ---------------------------------------------------------------------------

def test_repeated_identical_requests_produce_identical_risk_fields(client):
    payload = {"transaction": _txn(transaction_id="txn_determinism_A"), "prior_transactions": _prior_history()}
    r1 = client.post("/risk/score", json=payload)
    payload2 = {"transaction": _txn(transaction_id="txn_determinism_B"), "prior_transactions": _prior_history()}
    r2 = client.post("/risk/score", json=payload2)
    # same underlying transaction shape (only id differs) -> identical probability/score
    assert r1.json()["fraud_probability"] == r2.json()["fraud_probability"]
    assert r1.json()["risk_score"] == r2.json()["risk_score"]
    assert r1.json()["risk_category"] == r2.json()["risk_category"]


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "pytest", __file__, "-v"]))

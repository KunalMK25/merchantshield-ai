"""Phase 4 tests: serving whichever model the registry says is champion, with hot reload."""

import os

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings
from backend.services.model_provider import ModelProvider, build_champion_bundle
from ml.contest.runner import run_contest
from ml.features.build_features import FEATURE_COLUMNS
from ml.registry.store import ModelRegistry


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["amount"] = np.exp(rng.normal(6, 1, size=n))
    X["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-(1.8 * X["amount_zscore"] - 2.5)))).astype(int)
    return X


@pytest.fixture(scope="module")
def entries():
    return run_contest(_frame(3000, 1), _frame(1500, 2), _frame(1500, 3),
                       names=["random_forest", "lightgbm", "xgboost"])


def registry_with(tmp_path, entries, promote="lightgbm"):
    reg = ModelRegistry(tmp_path)
    run_id = reg.register_run(entries)
    reg.promote(run_id, promote, reason="test")
    return reg, run_id


# ------------------------------------------------------------ bundle build

def test_champion_bundle_carries_the_champions_own_threshold_and_label(tmp_path, entries):
    reg, run_id = registry_with(tmp_path, entries, "xgboost")
    champ = reg.champion()
    bundle = build_champion_bundle(reg, champ)
    assert bundle.decision_threshold == champ["threshold"]
    assert bundle.metadata["model_name"] == f"xgboost@{run_id}"
    assert bundle.metadata["selected_threshold"] == champ["threshold"]
    assert hasattr(bundle.model, "predict_proba") and bundle.explainer is not None


def test_imported_champion_keeps_its_original_metadata(tmp_path):
    from ml.contest.run import bootstrap_champion
    reg = ModelRegistry(tmp_path)
    bootstrap_champion(reg)
    bundle = build_champion_bundle(reg, reg.champion())
    assert bundle.metadata["model_name"] == "lgbm_v1"
    assert bundle.decision_threshold == 0.4


# ---------------------------------------------------------------- provider

def make_app(reg, now):
    app = FastAPI()
    app.state.model_bundle = None
    provider = ModelProvider(reg, check_every=5.0, clock=lambda: now[0])
    app.state.model_provider = provider
    return app, provider


def test_provider_reloads_when_champion_changes_and_throttles_checks(tmp_path, entries):
    reg, run_id = registry_with(tmp_path, entries, "lightgbm")
    now = [100.0]
    app, provider = make_app(reg, now)

    provider.refresh(app)
    assert app.state.model_bundle.metadata["model_name"] == f"lightgbm@{run_id}"
    first = app.state.model_bundle

    reg.promote(run_id, "xgboost", reason="switch")
    now[0] += 1.0                       # inside the throttle window: no re-read yet
    provider.refresh(app)
    assert app.state.model_bundle is first

    now[0] += 10.0                      # window passed: picks up the new champion
    provider.refresh(app)
    assert app.state.model_bundle.metadata["model_name"] == f"xgboost@{run_id}"


def test_provider_keeps_serving_if_the_new_artifact_is_broken(tmp_path, entries):
    reg, run_id = registry_with(tmp_path, entries, "lightgbm")
    now = [0.0]
    app, provider = make_app(reg, now)
    provider.refresh(app)
    good = app.state.model_bundle

    reg.promote(run_id, "xgboost", reason="switch")
    (tmp_path / "runs" / run_id / "xgboost.pkl").write_bytes(b"not a model")
    now[0] += 10
    provider.refresh(app)               # must not raise
    assert app.state.model_bundle is good


def test_provider_does_not_resurrect_a_deliberately_degraded_bundle(tmp_path, entries):
    reg, _ = registry_with(tmp_path, entries, "lightgbm")
    now = [0.0]
    app, provider = make_app(reg, now)
    provider.refresh(app)
    app.state.model_bundle = None       # what the API tests do to simulate a failure
    now[0] += 10
    provider.refresh(app)               # champion unchanged -> nothing to reload
    assert app.state.model_bundle is None


def test_provider_with_no_champion_changes_nothing(tmp_path):
    app, provider = make_app(ModelRegistry(tmp_path), [0.0])
    sentinel = object()
    app.state.model_bundle = sentinel
    provider.refresh(app)
    assert app.state.model_bundle is sentinel


# ------------------------------------------------------- through the real API

def test_api_serves_and_hot_swaps_the_champion(tmp_path, entries, monkeypatch):
    from backend.main import app
    reg, run_id = registry_with(tmp_path, entries, "lightgbm")
    monkeypatch.setattr(settings, "REGISTRY_DIR", str(tmp_path))

    body = {"transaction": {"transaction_id": "txn_swap_1", "customer_id": "c1", "merchant_id": "m1", "amount": 1200.0,
                            "timestamp": "2026-06-01T10:00:00Z", "device_id": "d1", "geo_region": "region_01",
                            "merchant_category": "fashion", "payment_method": "upi", "status": "success",
                            "account_created": "2026-01-01T00:00:00Z"},
            "prior_transactions": []}
    with TestClient(app) as client:
        info = client.get("/model/info").json()
        assert info["model_version"] == f"lightgbm@{run_id}"
        assert info["decision_threshold"] == reg.champion()["threshold"]

        reg.promote(run_id, "xgboost", reason="switch")
        app.state.model_provider.check_every = 0
        assert client.get("/health").json()["model_version"] == f"xgboost@{run_id}"

        r = client.post("/risk/evaluate", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        thr = reg.champion()["threshold"]
        from ml.evaluation.policy import evaluate_policy_at
        assert d["model_version"] == f"xgboost@{run_id}"
        assert d["threshold"] == thr
        assert d["action"] == evaluate_policy_at(d["fraud_probability"], 1200.0, thr).action

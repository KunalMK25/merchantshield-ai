"""API tests for the read-only model registry endpoints (Phase 3)."""

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.registry_routes import router
from ml.contest.runner import run_contest
from ml.features.build_features import FEATURE_COLUMNS
from ml.registry.store import ModelRegistry


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["amount"] = np.exp(rng.normal(6, 1, size=n))
    X["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-(1.8 * X["amount_zscore"] - 3)))).astype(int)
    return X


def make_client(registry):
    app = FastAPI()
    app.state.registry = registry
    app.include_router(router)
    return TestClient(app)


def test_empty_registry(tmp_path):
    c = make_client(ModelRegistry(tmp_path))
    assert c.get("/registry/champion").json() == {"champion": None}
    assert c.get("/registry/runs").json() == {"runs": []}
    assert c.get("/registry/runs/nope").status_code == 404


def test_registry_unavailable_returns_503():
    app = FastAPI()
    app.include_router(router)
    assert TestClient(app).get("/registry/runs").status_code == 503


def test_runs_champion_and_events(tmp_path):
    reg = ModelRegistry(tmp_path)
    entries = run_contest(_frame(3000, 1), _frame(1500, 2), _frame(1500, 3), names=["random_forest"])
    run_id = reg.register_run(entries, data_note="unit")
    reg.promote(run_id, "random_forest", reason="test")
    c = make_client(reg)

    champ = c.get("/registry/champion").json()["champion"]
    assert champ["run_id"] == run_id and champ["model_name"] == "random_forest"

    runs = c.get("/registry/runs").json()["runs"]
    assert runs[0]["run_id"] == run_id and runs[0]["candidates"][0]["name"] == "random_forest"
    assert "model" not in runs[0]["candidates"][0]

    assert c.get(f"/registry/runs/{run_id}").json()["run_id"] == run_id
    kinds = [e["event"] for e in c.get("/registry/events").json()["events"]]
    assert "promoted" in kinds and "run_registered" in kinds

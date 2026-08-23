"""
Shared pytest configuration and fixtures for the MerchantShield test suite.

SESSION-SCOPED FIXTURES: the model artifact, SHAP explainer, and feature CSV
are expensive to load (~5-10 s each). Scoping them to the session means they
load once per `pytest tests/` invocation rather than once per test file or
per test function. The FastAPI TestClient is also session-scoped so the
backend app's lifespan (which loads the model at startup) runs once.

AUDIT ISOLATION: the session-scoped client reuses a single in-memory SQLite
database created when the app starts. Tests that check the audit log always
look up their specific transaction_id (never assume a count-from-zero), so
they are safe under a shared client. Tests that mutate app.state (model_bundle
or audit_store to simulate failures) always restore the original value in a
finally block, which is verified by the test_api.py tests themselves.

PATH SETUP: sys.path manipulation here is redundant alongside the __init__.py
packages now present in ml/ and tests/, but it remains harmless and ensures
the project root is importable even when pytest is invoked from outside the
project root (e.g. `pytest merchantshield-ai/tests/`).
"""

import os
import sys

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure the project root (parent of tests/) is on sys.path so that
# `from ml.x import y` and `from backend.x import y` resolve correctly
# regardless of how pytest was invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Paths to artifacts -- must match backend/config/settings.py exactly
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
_FEATURES_PATH = os.path.join(_PROJECT_ROOT, "ml", "data", "features.csv")


# ---------------------------------------------------------------------------
# Model + explainer (session-scoped: loaded once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    """Frozen LightGBM model, loaded once for the entire test session."""
    return joblib.load(_MODEL_PATH)


@pytest.fixture(scope="session")
def explainer(model):
    """SHAP RiskExplainer wrapping the session-scoped model."""
    from ml.evaluation.explainability import RiskExplainer
    return RiskExplainer(model)


@pytest.fixture(scope="session")
def sample_rows():
    """
    20 randomly-sampled feature rows from the full features CSV, used by
    explainability tests. Fixed random_state=11 matches the original fixture
    in test_explainability.py for identical sampling behaviour.
    """
    df = pd.read_csv(_FEATURES_PATH)
    return df.sample(20, random_state=11).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FastAPI test client (session-scoped: app lifespan runs once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient, session-scoped so the backend's lifespan hook (which
    loads the LightGBM model and opens the SQLite connection) runs exactly once
    per test session rather than once per test function.
    """
    from backend.main import app
    with TestClient(app) as c:
        yield c

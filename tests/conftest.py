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

MARKS:
  integration -- marks tests that require the locally downloaded IEEE-CIS CSV
                 files (train_transaction.csv, train_identity.csv) and take
                 several minutes to run. Excluded from the default test run.
                 Run explicitly with: pytest tests/ -m integration
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
# Custom mark registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom marks to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that load full external datasets "
        "(slow, ~10-15 min, requires local IEEE-CIS data files)",
    )


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


@pytest.fixture(scope="session", autouse=True)
def ensure_raw_transactions():
    """
    Ensures ml/data/raw_transactions.csv exists for tests that depend on it.
    Generated deterministically from the project's synthetic pipeline if missing.
    Session-scoped and autouse=True so it runs exactly once at session start,
    before any tests that might need it.
    """
    import os

    raw_path = os.path.join(_PROJECT_ROOT, "ml", "data", "raw_transactions.csv")

    if not os.path.exists(raw_path):
        from ml.data.generate_synthetic import generate

        # Generate raw transactions deterministically
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        generate(output_path=raw_path)


@pytest.fixture(scope="session")
def sample_rows():
    """
    20 randomly-sampled feature rows from the full features CSV, used by
    explainability tests. Fixed random_state=11 matches the original fixture
    in test_explainability.py for identical sampling behaviour.

    If features.csv does not exist, generates it deterministically from the
    project's synthetic data pipeline to ensure reproducible testing
    without committing 450+ MB of generated data.
    """
    import os

    if not os.path.exists(_FEATURES_PATH):
        # Regenerate features.csv using the project's deterministic pipeline
        from ml.features.build_features import build_features

        # Ensure raw_transactions.csv exists first (ensure_raw_transactions fixture handles this)
        # Then build features from it
        raw = pd.read_csv(os.path.join(_PROJECT_ROOT, "ml", "data", "raw_transactions.csv"))
        df = build_features(raw)
        os.makedirs(os.path.dirname(_FEATURES_PATH), exist_ok=True)
        df.to_csv(_FEATURES_PATH, index=False)

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

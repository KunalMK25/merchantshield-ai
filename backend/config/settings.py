"""
Backend configuration.

IMPORTANT: this module does NOT define 0.40, 0.80, or any other policy number.
Those live exactly once, in ml/evaluation/policy.py, and are imported wherever
needed (see services/risk_service.py). Duplicating them here would violate the
Phase 8 instruction against re-implementing decision logic in the API layer.
"""

import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # merchantshield-ai/backend/
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)  # merchantshield-ai/

MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
MODEL_METADATA_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "lgbm_v1_metadata.json")

AUDIT_DB_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
AUDIT_DB_PATH = os.path.join(AUDIT_DB_DIR, "audit.db")
AUDIT_DB_URL = f"sqlite:///{AUDIT_DB_PATH}"

API_TITLE = "MerchantShield AI — Risk Manager API"
API_DESCRIPTION = (
    "Defensive fraud-risk scoring API for the Razorpay AI Buildathon 2026 "
    "(Track 02 — AI Risk Manager). Scores transactions for fraud risk, explains "
    "the score with SHAP, and returns a bounded, rule-based recommended action. "
    "Strictly detect/flag/recommend — never performs an irreversible financial action."
)
API_VERSION = "1.0.0"

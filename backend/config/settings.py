"""
Backend configuration.

IMPORTANT: this module does NOT define 0.40, 0.80, or any other policy number.
Those live exactly once, in ml/evaluation/policy.py, and are imported wherever
needed (see services/risk_service.py). Duplicating them here would violate the
Phase 8 instruction against re-implementing decision logic in the API layer.
"""

import json
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # merchantshield-ai/backend/
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)  # merchantshield-ai/

MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
MODEL_METADATA_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "lgbm_v1_metadata.json")

REGISTRY_DIR = os.environ.get("MS_REGISTRY_DIR", os.path.join(PROJECT_ROOT, "ml", "registry_store"))
DATA_STORE_DIR = os.environ.get("MS_DATA_STORE_DIR", os.path.join(PROJECT_ROOT, "ml", "data_store"))
# Optional shared secret for analyst actions (hold/dismiss/release/close). Unset = demo mode, no token.
ANALYST_TOKEN = os.environ.get("MS_ANALYST_TOKEN") or None

# Hours between automatic retraining runs inside the API process; unset/0 = disabled (use cron instead).
RETRAIN_INTERVAL_HOURS = float(os.environ.get("MS_RETRAIN_INTERVAL_HOURS", "0") or 0)

AUDIT_DB_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
AUDIT_DB_PATH = os.path.join(AUDIT_DB_DIR, "audit.db")
AUDIT_DB_URL = f"sqlite:///{AUDIT_DB_PATH}"

API_TITLE = "MerchantShield AI — Risk Manager API"
API_DESCRIPTION = (
    "Defensive fraud-risk scoring API as a final-year project "
    "Scores transactions for fraud risk, explains "
    "the score with SHAP, and returns a bounded, rule-based recommended action. "
    "Strictly detect/flag/recommend — never performs an irreversible financial action."
)
API_VERSION = "1.0.0"


_CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def resolve_currency_symbol(explicit, data_store_dir) -> str:
    """
    Symbol the dashboard shows next to amounts. An explicit MS_CURRENCY_SYMBOL wins; otherwise a dataset
    workspace reports its own currency (recorded when the dataset was prepared); otherwise rupees, the
    currency of the original synthetic data. An unreadable config never breaks startup.
    """
    if explicit:
        return explicit
    try:
        with open(os.path.join(os.path.dirname(os.path.normpath(str(data_store_dir))), "config.json")) as f:
            return _CURRENCY_SYMBOLS.get(json.load(f).get("currency"), "₹")
    except (OSError, ValueError, AttributeError):
        return "₹"


CURRENCY_SYMBOL = resolve_currency_symbol(os.environ.get("MS_CURRENCY_SYMBOL"), DATA_STORE_DIR)

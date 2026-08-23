"""
Model loading.

Loads the frozen LightGBM artifact and its metadata exactly ONCE, at application
startup (see backend/main.py lifespan). Every request reuses the same in-memory
model and RiskExplainer -- there is no per-request joblib.load or shap.TreeExplainer
construction, both of which are relatively expensive.

Does NOT retrain, does NOT modify the model. If the artifact is missing or corrupt,
startup fails loudly (the app will not come up half-broken and silently 500 on
every request).
"""

import json
import os
import joblib

from ml.evaluation.explainability import RiskExplainer


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifact or metadata cannot be loaded."""


class ModelBundle:
    """Holds the loaded model, explainer, and metadata for the lifetime of the app."""

    def __init__(self, model, explainer: RiskExplainer, metadata: dict, model_path: str):
        self.model = model
        self.explainer = explainer
        self.metadata = metadata
        self.model_path = model_path


def load_model_bundle(model_path: str, metadata_path: str) -> ModelBundle:
    if not os.path.exists(model_path):
        raise ModelUnavailableError(f"Model artifact not found at {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ModelUnavailableError(f"Model artifact at {model_path} could not be loaded (corrupt?): {e}") from e

    if not hasattr(model, "predict_proba"):
        raise ModelUnavailableError(
            f"Loaded object from {model_path} does not look like a classifier "
            f"(no predict_proba) -- refusing to serve it."
        )

    if not os.path.exists(metadata_path):
        raise ModelUnavailableError(f"Model metadata not found at {metadata_path}")

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        raise ModelUnavailableError(f"Model metadata at {metadata_path} could not be parsed: {e}") from e

    try:
        explainer = RiskExplainer(model)
    except Exception as e:
        raise ModelUnavailableError(f"Failed to construct SHAP explainer for the loaded model: {e}") from e

    return ModelBundle(model=model, explainer=explainer, metadata=metadata, model_path=model_path)

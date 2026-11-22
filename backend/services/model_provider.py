"""
Serves whichever model the registry names champion, and follows promotions.

The API process loads the champion at startup (falling back to the original lgbm_v1
files if the registry is empty). After that, `refresh` runs cheaply before requests:
at most once per `check_every` seconds it reads the small champion pointer, and only if
that changed does it load the new model and swap it in. A cron-driven retrain in another
process therefore takes effect without restarting the API.

Safety: a champion whose artifact cannot be loaded is skipped (the current model keeps
serving and the failure is logged), and a deliberately degraded app (model_bundle None)
is never "healed" unless the champion actually changed.
"""

import logging
import threading
import time

from backend.services.model_loader import ModelBundle
from ml.evaluation.explainability import RiskExplainer
from ml.registry.store import ModelRegistry

logger = logging.getLogger("merchantshield")

RETRY_AFTER_FAILURE = 60.0


def champion_key(champ):
    return None if champ is None else (champ["run_id"], champ["model_name"], champ["promoted_at"])


def _selection_rule(champ):
    return ("minimize total expected cost on validation subject to recall >= 80% "
            f"(registry champion, promoted {champ['promoted_at']})")


def _as_serving_metrics(m):
    if not m:
        return {}
    keep = {k: m[k] for k in ("precision", "recall", "f1", "fp", "fn", "tp", "tn", "roc_auc", "pr_auc") if k in m}
    keep["total_cost"] = m.get("cost")
    return keep


def build_champion_bundle(registry: ModelRegistry, champ: dict) -> ModelBundle:
    run = registry.get_run(champ["run_id"])
    cand = next(c for c in run["candidates"] if c["name"] == champ["model_name"])
    model = registry.load_model(champ["run_id"], champ["model_name"])
    if not hasattr(model, "predict_proba"):
        raise ValueError("champion artifact is not a classifier")

    metadata = dict(cand.get("metadata") or {
        "model_name": f"{champ['model_name']}@{champ['run_id']}",
        "model_file": f"registry/runs/{champ['run_id']}/{champ['model_name']}.pkl",
        "selection_rule": _selection_rule(champ),
        "validation_metrics_at_threshold": _as_serving_metrics(cand["val"]),
        "test_metrics_at_frozen_threshold": _as_serving_metrics(cand.get("test")),
    })
    metadata["selected_threshold"] = champ["threshold"]   # the threshold currently in force
    return ModelBundle(model=model, explainer=RiskExplainer(model), metadata=metadata,
                       model_path=str(registry.root / "runs" / champ["run_id"]))


class ModelProvider:
    def __init__(self, registry: ModelRegistry, check_every: float = 2.0, clock=time.monotonic):
        self.registry = registry
        self.check_every = check_every
        self._clock = clock
        self._loaded_key = None
        self._last_check = None
        self._failed = None            # (key, when) of a champion that would not load
        self._lock = threading.Lock()

    def initial_bundle(self):
        """Champion bundle for startup, or None if there is no usable champion."""
        champ = self.registry.champion()
        self._last_check = self._clock()
        if champ is None:
            return None
        try:
            bundle = build_champion_bundle(self.registry, champ)
        except Exception as e:
            logger.error(f"Champion {champ['model_name']}@{champ['run_id']} failed to load: {e}")
            return None
        self._loaded_key = champion_key(champ)
        return bundle

    def refresh(self, app) -> None:
        now = self._clock()
        if self._last_check is not None and now - self._last_check < self.check_every:
            return
        with self._lock:
            self._last_check = now
            try:
                champ = self.registry.champion()
            except Exception as e:
                logger.error(f"Could not read champion pointer: {e}")
                return
            key = champion_key(champ)
            if champ is None or key == self._loaded_key:
                return
            if self._failed and self._failed[0] == key and now - self._failed[1] < RETRY_AFTER_FAILURE:
                return
            try:
                bundle = build_champion_bundle(self.registry, champ)
            except Exception as e:
                logger.error(f"Promoted champion {champ['model_name']}@{champ['run_id']} could not be "
                             f"loaded; keeping the current model: {e}")
                self._failed = (key, now)
                return
            app.state.model_bundle = bundle          # one reference assignment: requests never see a half-swap
            self._loaded_key = key
            self._failed = None
            logger.info(f"Now serving champion {bundle.metadata['model_name']}")

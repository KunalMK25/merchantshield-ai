"""
File-based model registry.

Layout under `root`:
  runs/<run_id>/run.json        metrics of every candidate in that contest run
  runs/<run_id>/<name>.pkl      the fitted models
  champion.json                 the model currently designated for serving
  history.json                  stack of past champions (used by rollback)
  events.jsonl                  append-only audit trail of promotions and rollbacks

Promotion is guarded: only tree-based models (SHAP-explainable) can serve, and a
challenger must beat the champion's validation cost by a relative margin when both
are scored on the same, current validation data.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

import joblib

from ml.contest.selection import pick_threshold, select_winner
from ml.features.build_features import FEATURE_COLUMNS


class RegistryError(RuntimeError):
    pass


def _clean(obj):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- io
    def _read(self, name, default):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def _write(self, name, data):
        tmp = self.root / (name + ".tmp")
        tmp.write_text(json.dumps(_clean(data), indent=2))
        os.replace(tmp, self.root / name)  # atomic: readers never see a half-written file

    def _event(self, kind, **fields):
        with open(self.root / "events.jsonl", "a") as f:
            f.write(json.dumps(_clean({"at": _now(), "event": kind, **fields})) + "\n")

    # --------------------------------------------------------------- runs
    def register_run(self, entries, data_note: str = "") -> str:
        n = len(self.list_runs()) + 1
        run_id = f"run_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}_{n:04d}"
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True)

        candidates = []
        for e in entries:
            joblib.dump(e["model"], run_dir / f"{e['name']}.pkl")
            candidates.append({k: v for k, v in e.items() if k != "model"})
        winner = select_winner(entries)
        record = {
            "run_id": run_id, "created_at": _now(), "data_note": data_note,
            "winner": winner["name"] if winner else None, "candidates": candidates,
        }
        (run_dir / "run.json").write_text(json.dumps(_clean(record), indent=2))
        self._event("run_registered", run_id=run_id, winner=record["winner"])
        return run_id

    def list_runs(self):
        runs = []
        for d in sorted((self.root / "runs").iterdir()):
            f = d / "run.json"
            if f.exists():
                runs.append(json.loads(f.read_text()))
        return runs

    def get_run(self, run_id):
        f = self.root / "runs" / run_id / "run.json"
        if not f.exists():
            raise RegistryError(f"Unknown run: {run_id}")
        return json.loads(f.read_text())

    def load_model(self, run_id, name):
        f = self.root / "runs" / run_id / f"{name}.pkl"
        if not f.exists():
            raise RegistryError(f"No model {name!r} in run {run_id!r}")
        return joblib.load(f)

    def import_existing(self, model_path, name, tree_based, threshold, val_metrics, note=""):
        """Register an already-trained artifact (e.g. the original lgbm_v1) as a run."""
        n = len(self.list_runs()) + 1
        run_id = f"run_import_{n:04d}"
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True)
        shutil.copy(model_path, run_dir / f"{name}.pkl")
        entry = {"name": name, "tree_based": tree_based, "train_seconds": None,
                 "val": {**val_metrics, "threshold": threshold, "recall_constraint_met": True},
                 "test": None}
        record = {"run_id": run_id, "created_at": _now(), "data_note": note,
                  "winner": name, "candidates": [entry]}
        (run_dir / "run.json").write_text(json.dumps(_clean(record), indent=2))
        self._event("run_imported", run_id=run_id, model=name)
        return run_id

    # ----------------------------------------------------------- champion
    def champion(self):
        return self._read("champion.json", None)

    def history(self):
        return self._read("history.json", [])

    def events(self):
        p = self.root / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def promote(self, run_id, model_name, reason, threshold=None, val_cost=None):
        run = self.get_run(run_id)
        cand = next((c for c in run["candidates"] if c["name"] == model_name), None)
        if cand is None:
            raise RegistryError(f"No candidate {model_name!r} in run {run_id!r}")
        if not cand["tree_based"]:
            raise RegistryError(f"{model_name!r} is not tree-based; only tree models can serve "
                                f"because decisions are explained with SHAP.")
        record = {
            "run_id": run_id, "model_name": model_name,
            "threshold": threshold if threshold is not None else cand["val"]["threshold"],
            "val_cost": val_cost if val_cost is not None else cand["val"]["cost"],
            "promoted_at": _now(), "reason": reason,
        }
        history = self.history()
        history.append(record)
        self._write("history.json", history)
        self._write("champion.json", record)
        self._event("promoted", **record)
        return record

    def rollback(self):
        history = self.history()
        if len(history) < 2:
            raise RegistryError("Nothing to roll back to: no earlier champion.")
        undone = history.pop()
        previous = history[-1]
        self._write("history.json", history)
        self._write("champion.json", previous)
        self._event("rolled_back", from_run=undone["run_id"], to_run=previous["run_id"])
        return previous


def maybe_promote(registry: ModelRegistry, run_id, val, cost, margin=0.01, min_recall=0.8):
    """
    Decide whether the best tree-based candidate of `run_id` should replace the champion.
    Both are scored on the SAME current validation frame with the same threshold rule,
    so a model trained on older data can't win or lose just because the data changed.
    """
    run = registry.get_run(run_id)
    qualified = [c for c in run["candidates"]
                 if c["tree_based"] and c["val"].get("recall_constraint_met")]
    if not qualified:
        return {"promoted": False, "run_id": run_id,
                "reason": "No tree-based candidate met the recall floor, so nothing can serve."}
    best = min(qualified, key=lambda c: (c["val"]["cost"], c["name"]))

    X, y, amounts = val[FEATURE_COLUMNS], val["is_fraud"].values, val["amount"].values
    champ = registry.champion()
    if champ is None:
        rec = registry.promote(run_id, best["name"], reason="First champion: no incumbent to beat.")
        return {"promoted": True, "run_id": run_id, "model_name": best["name"], "reason": rec["reason"]}

    cand_model = registry.load_model(run_id, best["name"])
    cand_thr, cand_m = pick_threshold(y, cand_model.predict_proba(X)[:, 1], amounts, cost, min_recall)
    champ_model = registry.load_model(champ["run_id"], champ["model_name"])
    _, champ_m = pick_threshold(y, champ_model.predict_proba(X)[:, 1], amounts, cost, min_recall)

    needed = champ_m["cost"] * (1 - margin)
    detail = {"challenger_cost": cand_m["cost"], "champion_cost": champ_m["cost"], "margin": margin}
    if cand_m["recall_constraint_met"] and cand_m["cost"] < needed:
        rec = registry.promote(
            run_id, best["name"], threshold=cand_thr, val_cost=cand_m["cost"],
            reason=f"Beat champion on validation cost: {cand_m['cost']:.0f} vs {champ_m['cost']:.0f} "
                   f"(required at least {margin:.0%} lower).")
        return {"promoted": True, "run_id": run_id, "model_name": best["name"],
                "reason": rec["reason"], **detail}
    reason = (f"Challenger {best['name']} did not beat the champion by {margin:.0%}: "
              f"{cand_m['cost']:.0f} vs {champ_m['cost']:.0f}.")
    registry._event("promotion_declined", run_id=run_id, reason=reason)
    return {"promoted": False, "run_id": run_id, "reason": reason, **detail}

"""
Phase 3 tests: model contest, threshold selection, and the model registry.

Everything here uses small synthetic frames so the whole file runs in seconds
and never touches ml/data/features.csv or the served model.
"""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.cost_model import CostAssumptions
from ml.features.build_features import FEATURE_COLUMNS
from ml.contest.selection import pick_threshold, evaluate_at, select_winner
from ml.contest.runner import run_contest
from ml.registry.store import ModelRegistry, RegistryError, maybe_promote

COST = CostAssumptions()


def make_frame(n, seed, flip=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["amount"] = np.exp(rng.normal(6, 1, size=n))
    logit = 1.8 * X["amount_zscore"] + 1.2 * X["new_device_flag"] - 3.2
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    if flip:
        y = 1 - y
    X["is_fraud"] = y
    return X


@pytest.fixture(scope="module")
def frames():
    return make_frame(4000, 1), make_frame(2000, 2), make_frame(2000, 3)


# --------------------------------------------------------------- selection

def test_pick_threshold_minimises_cost_under_recall_floor():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    prob = np.clip(y * 0.5 + rng.random(3000) * 0.5, 0, 1)
    amounts = np.full(3000, 400.0)

    thr, m = pick_threshold(y, prob, amounts, COST, min_recall=0.8)

    assert m["recall"] >= 0.8 and m["recall_constraint_met"] is True
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    feasible = [evaluate_at(y, prob, amounts, COST, t) for t in grid]
    feasible = [f for f in feasible if f["recall"] >= 0.8]
    assert m["cost"] == pytest.approx(min(f["cost"] for f in feasible))
    assert thr in grid


def test_pick_threshold_flags_when_recall_floor_unreachable():
    y = np.array([1, 0] * 500)
    prob = np.full(1000, 0.5)  # uninformative: only recall 0 or 1 at the edges
    prob[::2] = 0.01  # all fraud scored very low
    thr, m = pick_threshold(y, prob, np.full(1000, 100.0), COST, min_recall=0.8)
    assert m["recall_constraint_met"] is False
    assert 0.05 <= thr <= 0.95


def test_select_winner_ignores_recall_failures_and_is_deterministic():
    entries = [
        {"name": "b", "val": {"cost": 100.0, "recall_constraint_met": True}},
        {"name": "a", "val": {"cost": 100.0, "recall_constraint_met": True}},
        {"name": "cheap_but_bad", "val": {"cost": 1.0, "recall_constraint_met": False}},
    ]
    assert select_winner(entries)["name"] == "a"  # tie broken by name
    assert select_winner([entries[2]]) is None


# ------------------------------------------------------------------ contest

def test_contest_returns_scored_entries(frames):
    train, val, test = frames
    entries = run_contest(train, val, test, names=["logistic_regression", "random_forest"])
    assert {e["name"] for e in entries} == {"logistic_regression", "random_forest"}
    for e in entries:
        assert {"precision", "recall", "cost", "threshold"} <= set(e["val"])
        assert {"precision", "recall", "cost"} <= set(e["test"])
        assert e["train_seconds"] >= 0
        assert hasattr(e["model"], "predict_proba")
    assert next(e for e in entries if e["name"] == "random_forest")["tree_based"] is True
    assert next(e for e in entries if e["name"] == "logistic_regression")["tree_based"] is False


def test_selection_never_looks_at_test_labels(frames):
    train, val, test = frames
    names = ["logistic_regression", "random_forest"]
    a = run_contest(train, val, test, names=names)
    b = run_contest(train, val, make_frame(2000, 3, flip=True), names=names)
    assert select_winner(a)["name"] == select_winner(b)["name"]
    for ea, eb in zip(a, b):
        assert ea["val"]["threshold"] == eb["val"]["threshold"]


# ----------------------------------------------------------------- registry

def entries_for(frames, names):
    train, val, test = frames
    return run_contest(train, val, test, names=names)


def test_register_run_persists_metrics_and_models(tmp_path, frames):
    reg = ModelRegistry(tmp_path)
    entries = entries_for(frames, ["logistic_regression", "random_forest"])
    run_id = reg.register_run(entries, data_note="synthetic unit-test data")

    runs = reg.list_runs()
    assert [r["run_id"] for r in runs] == [run_id]
    assert {c["name"] for c in runs[0]["candidates"]} == {"logistic_regression", "random_forest"}
    assert "model" not in runs[0]["candidates"][0]
    assert runs[0]["winner"] in {"logistic_regression", "random_forest"}
    assert reg.load_model(run_id, "random_forest").predict_proba(
        frames[0][FEATURE_COLUMNS].head(3)).shape == (3, 2)


def test_promote_rollback_cycle(tmp_path, frames):
    reg = ModelRegistry(tmp_path)
    assert reg.champion() is None
    r1 = reg.register_run(entries_for(frames, ["random_forest"]))
    r2 = reg.register_run(entries_for(frames, ["random_forest"]))

    reg.promote(r1, "random_forest", reason="first")
    reg.promote(r2, "random_forest", reason="second")
    assert reg.champion()["run_id"] == r2
    assert [h["run_id"] for h in reg.history()] == [r1, r2]

    restored = reg.rollback()
    assert restored["run_id"] == r1
    assert reg.champion()["run_id"] == r1
    with pytest.raises(RegistryError):
        reg.rollback()  # nothing earlier left


def test_promote_rejects_non_tree_models_and_unknown_runs(tmp_path, frames):
    reg = ModelRegistry(tmp_path)
    run_id = reg.register_run(entries_for(frames, ["logistic_regression"]))
    with pytest.raises(RegistryError, match="tree"):
        reg.promote(run_id, "logistic_regression", reason="x")
    with pytest.raises(RegistryError):
        reg.promote("nope", "random_forest", reason="x")


def test_maybe_promote_first_champion_and_margin(tmp_path, frames):
    train, val, test = frames
    reg = ModelRegistry(tmp_path)

    run1 = reg.register_run(entries_for(frames, ["random_forest"]))
    d1 = maybe_promote(reg, run1, val, COST, margin=0.03)
    assert d1["promoted"] is True and "first" in d1["reason"].lower()

    # Same model re-registered: no improvement, so the champion must stay.
    run2 = reg.register_run(entries_for(frames, ["random_forest"]))
    d2 = maybe_promote(reg, run2, val, COST, margin=0.03)
    assert d2["promoted"] is False
    assert reg.champion()["run_id"] == run1

    # An unreachable margin keeps even a better challenger out.
    d3 = maybe_promote(reg, run2, val, COST, margin=0.99)
    assert d3["promoted"] is False


def test_maybe_promote_never_picks_non_tree_winner(tmp_path, frames):
    reg = ModelRegistry(tmp_path)
    run = reg.register_run(entries_for(frames, ["logistic_regression"]))
    d = maybe_promote(reg, run, frames[1], COST, margin=0.0)
    assert d["promoted"] is False
    assert "tree" in d["reason"].lower()
    assert reg.champion() is None

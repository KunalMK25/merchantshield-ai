"""
Champion churn: a promoted model must stay for a minimum number of daily runs.

Measured on equivalent models (differing only by random seed), a single validation window shows a relative
'saving' with a heavy right tail (95th percentile +15.7%), so no margin on one window separates small real
improvements from noise: a 3% margin promoted an equivalent model on 15.7% of days. Statistical tests on the
saving (a paired bootstrap, a sign-flip permutation test) were both tried and still promoted equivalent models
on 9-10% of windows at a nominal 1%, so they are not used. A cooldown bounds the swap rate directly.
"""

import numpy as np
import pandas as pd
import pytest

from ml.contest.runner import run_contest
from ml.evaluation.cost_model import CostAssumptions
from ml.features.build_features import FEATURE_COLUMNS
from ml.registry.store import ModelRegistry, maybe_promote
from ml.retrain.job import RetrainConfig

COST = CostAssumptions()


def frame(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["amount"] = np.exp(rng.normal(6, 1, size=n))
    X["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-(1.4 * X["amount_zscore"] + 0.8 * X["velocity_5min"] - 3.8)))).astype(int)
    return X


def run(reg, rows, seed=1):
    return reg.register_run(run_contest(frame(rows, seed + 10), frame(1500, seed + 20), None,
                                        names=["random_forest"], seed=seed))


@pytest.fixture
def setup(tmp_path):
    reg = ModelRegistry(tmp_path)
    weak = run(reg, 150, seed=1)
    reg.promote(weak, "random_forest", reason="incumbent")
    return reg, frame(4000, 99)


def test_runs_since_champion_counts_daily_runs(setup):
    reg, _ = setup
    assert reg.runs_since_champion() == 0
    run(reg, 300, seed=2)
    run(reg, 300, seed=3)
    assert reg.runs_since_champion() == 2


def test_the_count_is_right_when_the_champion_is_an_imported_run(tmp_path):
    """Imported runs are named differently and sort differently; creation order must still decide."""
    from ml.contest.run import bootstrap_champion
    reg = ModelRegistry(tmp_path)
    bootstrap_champion(reg)
    first = run(reg, 300, seed=2)
    run(reg, 300, seed=3)
    assert reg.runs_since_champion(first) == 1 and reg.runs_since_champion() == 2


def test_no_champion_means_no_count(tmp_path):
    assert ModelRegistry(tmp_path).runs_since_champion() is None


def test_a_better_challenger_waits_out_the_cooldown(setup):
    reg, val = setup
    strong = run(reg, 6000, seed=5)                                   # one run after the incumbent
    d = maybe_promote(reg, strong, val, COST, margin=0.03, cooldown_runs=3)
    assert d["promoted"] is False and d["cooldown"] == {"age": 1, "required": 3}
    assert "cooldown" in d["reason"].lower() or "waiting" in d["reason"].lower()
    assert reg.champion()["run_id"] != strong
    assert any(e["event"] == "promotion_declined" and "waiting" in e["reason"] for e in reg.events())

    run(reg, 300, seed=6)
    run(reg, 300, seed=7)
    strong2 = run(reg, 6000, seed=8)                                  # now four runs after the incumbent
    d2 = maybe_promote(reg, strong2, val, COST, margin=0.03, cooldown_runs=3)
    assert d2["promoted"] is True and reg.champion()["run_id"] == strong2


def test_a_cooldown_of_zero_is_the_old_behaviour(setup):
    reg, val = setup
    strong = run(reg, 6000, seed=5)
    assert maybe_promote(reg, strong, val, COST, margin=0.03, cooldown_runs=0)["promoted"] is True


def test_the_first_champion_is_never_delayed(tmp_path):
    reg = ModelRegistry(tmp_path)
    first = run(reg, 1500, seed=1)
    assert maybe_promote(reg, first, frame(2000, 9), COST, cooldown_runs=7)["promoted"] is True


def test_the_cooldown_applies_even_when_the_margin_is_cleared_by_a_wide_gap(setup):
    reg, val = setup
    strong = run(reg, 6000, seed=5)
    d = maybe_promote(reg, strong, val, COST, margin=0.0, cooldown_runs=2)
    assert d["promoted"] is False and "challenger_cost" not in d      # declined before any scoring


def test_the_job_defaults_are_sized_to_the_measured_noise():
    c = RetrainConfig()
    assert c.margin == pytest.approx(0.10) and c.cooldown_runs == 7

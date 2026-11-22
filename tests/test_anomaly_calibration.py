"""
Per-dataset calibration of the account anomaly detector.

The fixed thresholds were tuned on synthetic data; on real data they flooded the alert tiers with
ordinary customers. Thresholds can now be derived from a dataset's own reference days, and the one
tunable quantile is chosen on validation days only.
"""

import numpy as np
import pandas as pd
import pytest

from ml.monitoring.anomaly import (
    AnomalyConfig, AnomalyDetector, account_day_table, calibrate, signal_masks, signals_for,
)


def make_table(n_customers=120, days=60, seed=0, fraud_from=30):
    """Account-day table with heavy-tailed spending, like real data, and fraud days that spend 10x."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = 20 + 40 * rng.random()
        for d in range(days):
            n = 1 + rng.poisson(1.5)
            amounts = base * rng.lognormal(0, 0.9, n)
            fraud = int(d >= fraud_from and rng.random() < 0.03)
            if fraud:
                amounts = np.append(amounts, base * 12)
            for i, a in enumerate(amounts):
                rows.append(dict(customer_id=f"c{c}", timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d, hours=1 + i),
                                 amount=float(a), amount_vs_avg_ratio=1.0, amount_zscore=0.0, new_device_flag=0,
                                 new_geo_flag=0, status="success", is_night=0, is_fraud=int(fraud and i == len(amounts) - 1)))
    f = pd.DataFrame(rows)
    # a plausible ratio feature: amount over the customer's running mean
    g = f.groupby("customer_id")["amount"]
    f["amount_vs_avg_ratio"] = f["amount"] / g.transform(lambda s: s.shift(1).expanding(min_periods=2).mean()).fillna(f["amount"])
    return account_day_table(f)


@pytest.fixture(scope="module")
def table():
    return make_table()


# ----------------------------------------------------------------- config

def test_default_config_keeps_the_original_thresholds():
    c = AnomalyConfig()
    assert (c.amount_ratio, c.spend_ratio, c.burst_ratio, c.burst_min_n, c.failed_min, c.iso_quantile) == (4.0, 6.0, 4.0, 5, 3, 0.995)
    assert c.quantile is None


def test_a_custom_threshold_changes_what_fires():
    row = pd.Series(dict(new_dev=0, new_geo=0, max_ratio=6.0, failed=0, n_vs_usual=1.0, n=2, spend_vs_usual=1.0))
    assert {s["name"] for s in signals_for(row)} == {"amount_spike"}
    assert signals_for(row, AnomalyConfig(amount_ratio=10.0)) == []


def test_resolving_a_quantile_config_uses_the_reference_distribution(table):
    ref = table[table.day < 45]
    c = AnomalyConfig(quantile=0.99).resolved(ref)
    assert c.amount_ratio == pytest.approx(ref["max_ratio"].quantile(0.99))
    assert c.spend_ratio == pytest.approx(ref["spend_vs_usual"].quantile(0.99))
    assert c.burst_ratio == pytest.approx(ref["n_vs_usual"].quantile(0.99))
    assert (c.failed_min, c.burst_min_n, c.iso_quantile) == (3, 5, 0.995)       # count thresholds stay as set
    plain = AnomalyConfig()
    assert plain.resolved(ref) is plain                                         # no quantile: nothing to resolve


def test_quantile_thresholds_fire_on_about_that_share_of_reference_days(table):
    ref = table[table.day < 45]
    cfg = AnomalyConfig(quantile=0.99).resolved(ref)
    m = signal_masks(ref, cfg)
    assert 0.004 < m["amount_spike"].mean() < 0.016
    assert 0.004 < m["spend_surge"].mean() < 0.016


# ------------------------------------------------- vectorised detect == per-row

def test_detect_matches_the_per_row_definition(table):
    ref, cur = table[table.day < 45], table[table.day >= 45]
    det = AnomalyDetector(seed=0, config=AnomalyConfig()).fit(ref)
    flags = det.detect(cur)

    scores = -det._forest.score_samples(det._matrix(cur))
    expected = {}
    for (_, row), s in zip(cur.iterrows(), scores):
        pts = int(s > det._threshold) + len(signals_for(row, det.config))
        if pts:
            expected[(row["customer_id"], int(row["day"]))] = pts
    got = {(r.customer_id, r.day): r.points for r in flags.itertuples()}
    assert got == expected


# ------------------------------------------------------------- calibrate()

def test_calibrate_searches_the_whole_grid_on_validation_only(table):
    ref, val = table[table.day < 35], table[(table.day >= 35) & (table.day < 45)]
    cal = calibrate(ref, val, quantiles=(0.98, 0.99, 0.995), iso_quantiles=(0.99, 0.995))
    assert len(cal["grid"]) == 6
    assert cal["quantile"] in (0.98, 0.99, 0.995) and cal["iso_quantile"] in (0.99, 0.995)
    best = max(g["score"] for g in cal["grid"])
    assert cal["score"] == pytest.approx(best)
    chosen = next(g for g in cal["grid"] if g["quantile"] == cal["quantile"] and g["iso_quantile"] == cal["iso_quantile"])
    assert chosen["score"] == pytest.approx(best)
    assert chosen["score"] == pytest.approx((chosen["critical"]["f1"] + chosen["critical_plus_high"]["f1"]) / 2)
    assert {"critical", "critical_plus_high"} <= set(chosen)
    assert "validation" in cal["objective"].lower()


def test_calibrate_is_deterministic(table):
    ref, val = table[table.day < 35], table[(table.day >= 35) & (table.day < 45)]
    a = calibrate(ref, val, quantiles=(0.98, 0.99), iso_quantiles=(0.99,))
    b = calibrate(ref, val, quantiles=(0.98, 0.99), iso_quantiles=(0.99,))
    assert a["quantile"] == b["quantile"] and a["score"] == pytest.approx(b["score"])


def test_calibrate_needs_something_to_learn_from(table):
    ref, val = table[table.day < 35], table[(table.day >= 35) & (table.day < 45)].assign(fraud=0)
    with pytest.raises(ValueError, match="compromised"):
        calibrate(ref, val)


def test_calibration_beats_the_defaults_on_heavy_tailed_data(table):
    """The point of the exercise: defaults flood the tiers on heavy-tailed spending, calibrated thresholds do not."""
    ref, val = table[table.day < 35], table[(table.day >= 35) & (table.day < 45)]
    test = table[table.day >= 45]
    cal = calibrate(ref, val)
    truth = set(map(tuple, test.loc[test.fraud == 1, ["customer_id", "day"]].to_numpy()))

    def tier(config, fit_on):
        flags = AnomalyDetector(seed=0, config=config).fit(fit_on).detect(test)
        sel = flags[flags.severity.isin(["critical", "high"])]
        hits = set(map(tuple, sel[["customer_id", "day"]].to_numpy())) & truth
        return (len(hits) / len(sel) if len(sel) else 0.0), len(hits) / len(truth)

    fit_on = table[table.day < 45]
    p_def, r_def = tier(AnomalyConfig(), fit_on)
    p_cal, r_cal = tier(AnomalyConfig(quantile=cal["quantile"], iso_quantile=cal["iso_quantile"]), fit_on)
    assert p_cal >= p_def and r_cal > r_def, f"defaults {p_def:.0%}/{r_def:.0%}, calibrated {p_cal:.0%}/{r_cal:.0%}"

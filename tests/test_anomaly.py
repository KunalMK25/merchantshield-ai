"""Phase 5 tests: per-account anomaly detection (Isolation Forest + explainable signals)."""

import os

import numpy as np
import pandas as pd
import pytest

from ml.data.daily_batch import DriftConfig, fit_population, initial_state, simulate_day, warm_up
from ml.features.build_features import build_features
from ml.monitoring.anomaly import (
    AnomalyDetector, SEVERITIES, account_day_table, signals_for, summarize,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "ml", "data", "features.csv")
BASE = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")


def txn_rows(customer, day, n, amount=100.0, ratio=1.0, new_dev=0, new_geo=0, failed=0):
    rows = []
    for i in range(n):
        rows.append(dict(customer_id=customer,
                         timestamp=pd.Timestamp("2026-04-01") + pd.Timedelta(days=day, hours=i + 1),
                         amount=amount, amount_vs_avg_ratio=ratio, amount_zscore=0.0,
                         new_device_flag=int(i < new_dev), new_geo_flag=int(i < new_geo),
                         status="failed" if i < failed else "success", is_night=0, is_fraud=0))
    return pd.DataFrame(rows)


def history(customer="c1", days=8, per_day=2):
    return pd.concat([txn_rows(customer, d, per_day) for d in range(days)], ignore_index=True)


# ------------------------------------------------------------------ table

def test_table_aggregates_each_customer_day():
    t = account_day_table(pd.concat([txn_rows("c1", 0, 3, amount=50.0, failed=1),
                                     txn_rows("c2", 0, 1, amount=700.0)], ignore_index=True))
    c1 = t[t.customer_id == "c1"].iloc[0]
    assert (c1.n, c1.total, c1.failed) == (3, 150.0, 1)
    assert len(t) == 2


def test_usual_levels_use_only_earlier_days_and_stay_neutral_without_history():
    base = history(days=8, per_day=2)
    t = account_day_table(base).sort_values("day")
    assert (t.head(3)["n_vs_usual"] == 1.0).all() and (t.head(3)["spend_vs_usual"] == 1.0).all()

    spike = pd.concat([base, txn_rows("c1", 8, 10, amount=500.0)], ignore_index=True)
    t2 = account_day_table(spike)
    assert t2[t2.day == 8].iloc[0].n_vs_usual == pytest.approx(5.0)       # 10 vs usual 2
    # changing a later day must not alter any earlier day's relative values
    earlier = lambda d: d[d.day < 8].sort_values("day")[["n_vs_usual", "spend_vs_usual"]].to_numpy()
    np.testing.assert_allclose(earlier(t), earlier(t2))


# ---------------------------------------------------------------- signals

def one_day(**kw):
    t = account_day_table(pd.concat([history(), txn_rows("c1", 8, kw.pop("n", 2), **kw)], ignore_index=True))
    return t[t.day == 8].iloc[0]


def names(row):
    return {s["name"] for s in signals_for(row)}


def test_a_normal_day_raises_no_signal():
    assert names(one_day()) == set()


def test_each_signal_fires_on_its_pattern():
    assert "takeover" in names(one_day(new_dev=1, new_geo=1))
    assert "amount_spike" in names(one_day(ratio=6.0))
    assert "card_testing" in names(one_day(n=4, failed=3))
    assert "activity_burst" in names(one_day(n=10))
    assert "spend_surge" in names(one_day(n=2, amount=1500.0))
    assert "takeover" not in names(one_day(new_dev=1))      # a new device alone is not a takeover


def test_signals_explain_themselves_in_words():
    sigs = signals_for(one_day(ratio=6.0))
    assert sigs and all(isinstance(s["detail"], str) and s["detail"] for s in sigs)
    assert "6.0" in next(s for s in sigs if s["name"] == "amount_spike")["detail"]


# --------------------------------------------------------- real-data checks

@pytest.fixture(scope="module")
def table():
    feats = pd.read_csv(FEATURES)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    return account_day_table(feats)


@pytest.fixture(scope="module")
def detector(table):
    return AnomalyDetector(seed=0).fit(table[table.day < 45])


def test_flags_known_takeovers_on_days_it_never_trained_on(table, detector):
    test_days = table[table.day >= 45]
    flags = detector.detect(test_days)
    serious = flags[flags.severity.isin(["critical", "high"])]
    truth = set(map(tuple, test_days.loc[test_days.fraud == 1, ["customer_id", "day"]].to_numpy()))
    hits = set(map(tuple, serious[["customer_id", "day"]].to_numpy())) & truth
    precision, recall = len(hits) / len(serious), len(hits) / len(truth)
    assert precision >= 0.85, f"precision {precision:.2f}"
    assert recall >= 0.5, f"recall {recall:.2f}"


def test_most_customers_are_left_alone(table, detector):
    flags = detector.detect(table[table.day >= 45])
    assert len(flags) / len(table[table.day >= 45]) < 0.02


def test_records_are_recommendations_only_and_ranked(table, detector):
    flags = detector.detect(table[table.day >= 45])
    assert set(flags.severity) <= set(SEVERITIES)
    assert flags["recommended_action"].str.contains("ecommend|onitor").all()
    assert not flags["recommended_action"].str.contains("block|close|freeze", case=False).any()
    order = flags.severity.map({s: i for i, s in enumerate(SEVERITIES)})
    assert order.is_monotonic_increasing or order.tolist() == sorted(order.tolist())
    assert flags["signals"].map(lambda s: isinstance(s, list)).all()


def test_empty_input_gives_an_empty_frame_with_the_same_columns(table, detector):
    full = detector.detect(table[table.day >= 45])
    empty = detector.detect(table.iloc[0:0])
    assert empty.empty and list(empty.columns) == list(full.columns)


def test_fitting_requires_enough_reference_data():
    with pytest.raises(ValueError):
        AnomalyDetector().fit(pd.DataFrame())


def test_a_fraud_surge_raises_the_flagged_count():
    base = pd.read_csv(BASE)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    profiles, merchants = fit_population(base)
    state = warm_up(profiles, merchants, initial_state(base))
    plain, _ = simulate_day(profiles, merchants, 60, state, seed=9)
    surge, _ = simulate_day(profiles, merchants, 60, state, seed=9, drift=DriftConfig(fraud_rate_multiplier=40.0))

    def flagged(extra):
        feats = build_features(pd.concat([base, extra], ignore_index=True))
        t = account_day_table(feats)
        det = AnomalyDetector(seed=0).fit(t[t.day < 55])
        return det.detect(t[t.day == 60])

    assert len(flagged(surge)) > len(flagged(plain)) + 5


def test_summary_reports_counts_and_the_rate_against_expectation(table, detector):
    today = table[table.day == 50]
    s = summarize(detector.detect(today), n_accounts=len(today), reference_rate=detector.reference_rate)
    assert s["accounts_checked"] == len(today)
    assert s["flagged"] == sum(s["by_severity"].values())
    assert set(s["by_severity"]) == set(SEVERITIES)
    assert s["rate_vs_expected"] >= 0

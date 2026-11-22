"""Phase 4 tests: the daily synthetic batch generator and its on-disk store."""

import os

import numpy as np
import pandas as pd
import pytest

from ml.data.daily_batch import (
    DailyStore, DriftConfig, fit_population, initial_state, simulate_day, warm_up,
)
from ml.features.build_features import build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")
RAW_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "merchant_category", "timestamp",
               "amount", "device_id", "geo_region", "payment_method", "status",
               "account_created", "is_fraud"]


def tiny_raw():
    rows = []
    for c, geo, dev in [("c1", "region_01", "dev_a"), ("c2", "region_02", "dev_b")]:
        for i in range(12):
            rows.append(dict(
                transaction_id=f"txn_{len(rows)+1:07d}", customer_id=c, merchant_id="m1",
                merchant_category="fashion", timestamp=pd.Timestamp("2026-04-01") + pd.Timedelta(hours=7 * i),
                amount=100.0 + 10 * i, device_id=dev, geo_region=geo, payment_method="upi",
                status="success", account_created=pd.Timestamp("2026-01-01"), is_fraud=0))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def base():
    df = pd.read_csv(BASE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture(scope="module")
def population(base):
    profiles, merchants = fit_population(base)
    return profiles, warm_up(profiles, merchants, initial_state(base))


# ---------------------------------------------------------------- fitting

def test_fit_population_learns_each_customer_from_history():
    profiles, merchants = fit_population(tiny_raw())
    p = {x["customer_id"]: x for x in profiles}
    assert p["c1"]["home_geo"] == "region_01" and p["c1"]["legit_devices"] == ["dev_a"]
    assert p["c2"]["home_geo"] == "region_02"
    assert p["c1"]["base_amount_mean"] == pytest.approx(155.0)
    assert p["c1"]["activity_rate_per_day"] > 0
    assert merchants and merchants[0]["merchant_id"] == "m1"


def test_fit_population_ignores_fraud_rows():
    raw = tiny_raw()
    raw.loc[0, ["is_fraud", "device_id", "geo_region", "amount"]] = [1, "dev_evil", "region_09", 99999.0]
    profiles, _ = fit_population(raw)
    c1 = next(x for x in profiles if x["customer_id"] == "c1")
    assert "dev_evil" not in c1["legit_devices"] and c1["home_geo"] == "region_01"
    assert c1["base_amount_mean"] < 1000


def test_warm_up_seeds_episodes_without_consuming_days_or_ids(base):
    profiles, merchants = fit_population(base)
    cold = initial_state(base)
    warm = warm_up(profiles, merchants, cold)
    assert cold["episodes"] == {} and len(warm["episodes"]) > 0
    assert warm["next_day"] == cold["next_day"] and warm["txn_counter"] == cold["txn_counter"]


def test_initial_state_continues_after_history(base):
    s = initial_state(base)
    assert s["next_day"] == 60 and s["episodes"] == {}
    assert s["txn_counter"] == max(int(t.split("_")[1]) for t in base["transaction_id"])


# ------------------------------------------------------------- simulation

def test_simulate_day_is_deterministic_and_day_specific(population):
    profiles, state = population
    merchants = fit_population(pd.read_csv(BASE, nrows=5000))[1]  # any merchant list
    a, _ = simulate_day(profiles, merchants, 60, state, seed=7)
    b, _ = simulate_day(profiles, merchants, 60, state, seed=7)
    c, _ = simulate_day(profiles, merchants, 61, state, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_simulated_day_has_raw_schema_and_unique_continuing_ids(population, base):
    profiles, state = population
    merchants = fit_population(base)[1]
    df, new_state = simulate_day(profiles, merchants, 60, state, seed=1)
    assert list(df.columns) == RAW_COLUMNS
    assert df["transaction_id"].is_unique
    assert not set(df["transaction_id"]) & set(base["transaction_id"])
    day0 = pd.Timestamp(state["day0"])
    assert df["timestamp"].min() >= day0 + pd.Timedelta(days=60)
    assert df["timestamp"].max() < day0 + pd.Timedelta(days=61)
    assert new_state["next_day"] == 61
    assert new_state["txn_counter"] == state["txn_counter"] + len(df)
    assert state["next_day"] == 60, "input state must not be mutated"


def test_active_episode_produces_fraud_for_that_customer(population, base):
    profiles, state = population
    merchants = fit_population(base)[1]
    cust = profiles[0]["customer_id"]
    st = {**state, "episodes": {cust: {"onset_day": 59, "pattern": "card_testing", "peak_intensity": 0.95}}}
    df, new_state = simulate_day(profiles, merchants, 60, st, seed=3)
    mine = df[df["customer_id"] == cust]
    assert len(mine) >= 6 and mine["is_fraud"].all()
    # episodes expire after their window
    st_old = {**state, "episodes": {cust: {"onset_day": 40, "pattern": "card_testing", "peak_intensity": 0.95}}}
    _, after = simulate_day(profiles, merchants, 60, st_old, seed=3)
    assert cust not in after["episodes"]


def test_generated_day_mirrors_the_demo_data(population, base):
    """A new day must look like the history it was fitted to, not like a different world."""
    profiles, state = population
    merchants = fit_population(base)[1]
    df, _ = simulate_day(profiles, merchants, 60, state, seed=11)

    per_day = len(base) / 60
    assert 0.6 * per_day < len(df) < 1.4 * per_day
    assert 0.5 * base["is_fraud"].mean() < df["is_fraud"].mean() < 2.0 * base["is_fraud"].mean()

    # if device/geo identities were not carried over, EVERY row would look "new"
    combined = build_features(pd.concat([base, df], ignore_index=True))
    new_rows = combined[combined["transaction_id"].isin(df["transaction_id"]) & (combined["is_fraud"] == 0)]
    old_rows = combined[combined["transaction_id"].isin(base["transaction_id"]) & (combined["is_fraud"] == 0)]
    assert new_rows["new_device_flag"].mean() < 0.1
    assert new_rows["new_geo_flag"].mean() < max(0.1, 3 * old_rows["new_geo_flag"].mean())


def test_drift_knobs_change_the_data(population, base):
    profiles, state = population
    merchants = fit_population(base)[1]
    plain, _ = simulate_day(profiles, merchants, 60, state, seed=5)
    bigger, _ = simulate_day(profiles, merchants, 60, state, seed=5, drift=DriftConfig(amount_scale=2.0))
    ratio = bigger["amount"].median() / plain["amount"].median()
    assert 1.7 < ratio < 2.3
    more_fraud, _ = simulate_day(profiles, merchants, 60, state, seed=5,
                                 drift=DriftConfig(fraud_rate_multiplier=40.0))
    assert more_fraud["is_fraud"].mean() > plain["is_fraud"].mean()


# ------------------------------------------------------------------ store

def test_store_round_trip_and_progression(tmp_path, population, base):
    profiles, state = population
    merchants = fit_population(base)[1]
    store = DailyStore(tmp_path)
    assert store.state(base) == initial_state(base)
    assert not store.has_state()
    assert store.load_daily().empty

    d1, s1 = simulate_day(profiles, merchants, 60, state, seed=1)
    store.append_day(60, d1, s1)
    assert store.has_state()
    d2, s2 = simulate_day(profiles, merchants, 61, store.state(base), seed=1)
    store.append_day(61, d2, s2)

    assert store.state(base)["next_day"] == 62
    loaded = store.load_daily()
    assert len(loaded) == len(d1) + len(d2)
    assert loaded["transaction_id"].is_unique
    assert list(loaded.columns) == RAW_COLUMNS

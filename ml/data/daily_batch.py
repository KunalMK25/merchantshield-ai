"""
Daily synthetic batch generator.

Extends the demo dataset one simulated day at a time so the model can be retrained on
fresh data. Customers and merchants are NOT invented: they are fitted from the existing
raw history (home region, devices, spend level, activity rate), so a new day continues
the same population and the engineered features (device/region novelty, z-scores, ...)
mean the same thing they did in the original data.

Fraud follows the same episode mechanism as ml/data/generate_synthetic.py: a customer
is compromised for a few days, with one of four patterns and a decaying intensity.
Episodes span several days, so their state is carried between batches.

`DriftConfig` lets a run deliberately shift the world (bigger amounts, more fraud,
more legitimate device changes) so the drift monitoring planned for a later phase has
something real to detect. The default is no drift: a pure mirror of the history.
"""

import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "merchant_category", "timestamp",
               "amount", "device_id", "geo_region", "payment_method", "status",
               "account_created", "is_fraud"]

PATTERNS = ["takeover_burst", "card_testing", "quiet_single", "synthetic_new_account"]
PATTERN_P = [0.35, 0.30, 0.20, 0.15]
EPISODE_DAYS = 4                   # an episode lasts onset .. onset + 4
EPISODE_RATE = 0.045 / 60          # chance per customer-day of starting one (matches the original 4.5% over 60 days)
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
PAYMENT_P = [0.5, 0.35, 0.1, 0.05]


@dataclass(frozen=True)
class DriftConfig:
    amount_scale: float = 1.0           # multiplies every transaction amount
    fraud_rate_multiplier: float = 1.0  # multiplies the chance a compromise episode starts
    new_device_multiplier: float = 1.0  # multiplies legitimate travel / second-device behaviour


# ----------------------------------------------------------------- fitting

def fit_population(raw: pd.DataFrame):
    """Customer profiles and merchants learned from legitimate rows of the history."""
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["account_created"] = pd.to_datetime(raw["account_created"])
    legit = raw[raw["is_fraud"] == 0]
    hist_start = raw["timestamp"].min().normalize()
    hist_end = raw["timestamp"].max().normalize()

    profiles = []
    for cid, g in legit.groupby("customer_id", sort=True):
        created = g["account_created"].iloc[0]
        first_day = max(created.normalize(), hist_start)
        days = max((hist_end - first_day).days + 1, 1)
        home_geo = g["geo_region"].value_counts().index[0]
        std = float(g["amount"].std(ddof=0))
        profiles.append({
            "customer_id": cid,
            "account_created": created,
            "home_geo": home_geo,
            "legit_devices": list(g["device_id"].value_counts().index),
            "base_amount_mean": float(g["amount"].mean()),
            "base_amount_std": max(std if np.isfinite(std) else 0.0, 10.0),
            "activity_rate_per_day": max(len(g) / days, 0.05),
            "travels": bool((g["geo_region"] != home_geo).any()),
        })

    merchants = []
    for mid, g in legit.groupby("merchant_id", sort=True):
        merchants.append({"merchant_id": mid, "category": g["merchant_category"].iloc[0],
                          "avg_ticket": float(g["amount"].mean())})
    return profiles, merchants


def initial_state(raw: pd.DataFrame) -> dict:
    ts = pd.to_datetime(raw["timestamp"])
    day0 = ts.min().normalize()
    return {
        "day0": day0.isoformat(),
        "next_day": int((ts.max().normalize() - day0).days) + 1,
        "txn_counter": max(int(t.split("_")[1]) for t in raw["transaction_id"]),
        "episodes": {},
    }


# -------------------------------------------------------------- simulation

def _device(rng) -> str:
    return f"dev_{int(rng.integers(0, 16 ** 10)):010x}"


def simulate_day(profiles, merchants, day_index: int, state: dict, drift: DriftConfig = None, seed: int = 42):
    """
    One simulated day. Returns (dataframe in the raw schema, new state). The input state is
    never mutated, and the same (profiles, day, state, seed) always gives the same rows.
    """
    drift = drift or DriftConfig()
    rng = np.random.default_rng([seed, day_index])
    day_start = pd.Timestamp(state["day0"]) + timedelta(days=day_index)
    episodes = {k: dict(v) for k, v in state["episodes"].items()}
    counter = state["txn_counter"]
    regions = sorted({p["home_geo"] for p in profiles})
    rows = []

    for prof in profiles:
        cid = prof["customer_id"]
        if day_start < prof["account_created"]:
            continue

        comp = episodes.get(cid)
        if comp is not None and day_index > comp["onset_day"] + EPISODE_DAYS:
            del episodes[cid]
            comp = None
        if comp is None and rng.random() < EPISODE_RATE * drift.fraud_rate_multiplier:
            comp = {"onset_day": day_index, "pattern": str(rng.choice(PATTERNS, p=PATTERN_P)),
                    "peak_intensity": float(rng.uniform(0.5, 1.0))}
            episodes[cid] = comp

        n_txns = int(rng.poisson(prof["activity_rate_per_day"]))
        intensity = 0.0
        pattern = None
        if comp is not None:
            pattern = comp["pattern"]
            intensity = comp["peak_intensity"] * np.exp(-0.4 * (day_index - comp["onset_day"]))
            if pattern == "takeover_burst":
                n_txns += int(rng.integers(4, 9))
            elif pattern == "card_testing":
                n_txns += int(rng.integers(6, 14))
            else:
                n_txns += 1

        travel_p = 0.02 * drift.new_device_multiplier
        second_dev_p = 0.03 * drift.new_device_multiplier
        for _ in range(n_txns):
            merch = merchants[int(rng.integers(0, len(merchants)))]
            ts = day_start + timedelta(seconds=int(rng.integers(0, 86400)))
            is_fraud = False

            if pattern in ("takeover_burst", "synthetic_new_account") and rng.random() < 0.8 * intensity + 0.15:
                device_id = _device(rng)
                geo = str(rng.choice([g for g in regions if g != prof["home_geo"]]))
                is_fraud = True
            elif prof["travels"] and rng.random() < travel_p:
                device_id = str(rng.choice(prof["legit_devices"]))
                geo = str(rng.choice(regions))
            elif len(prof["legit_devices"]) > 1 and rng.random() < second_dev_p:
                device_id = str(rng.choice(prof["legit_devices"]))
                geo = prof["home_geo"]
            else:
                device_id = prof["legit_devices"][0]
                geo = prof["home_geo"]

            mean, std = prof["base_amount_mean"], prof["base_amount_std"]
            if pattern == "takeover_burst" and rng.random() < 0.7:
                amount = max(mean * rng.uniform(2.5, 6.0), merch["avg_ticket"] * rng.uniform(1.2, 2.5))
                is_fraud = True
            elif pattern == "card_testing":
                amount = rng.uniform(1, 50)
                is_fraud = True
            elif pattern == "quiet_single":
                amount = max(rng.normal(mean, std), 20)
                is_fraud = True
            elif pattern == "synthetic_new_account":
                amount = merch["avg_ticket"] * rng.uniform(1.5, 3.0)
                is_fraud = True
            else:
                amount = max(rng.normal(mean, std), 10)
            amount = round(float(amount) * drift.amount_scale, 2)

            if pattern == "card_testing":
                status = str(rng.choice(["failed", "success"], p=[0.75, 0.25]))
            else:
                status = str(rng.choice(["failed", "success"], p=[0.06, 0.94]))

            counter += 1
            rows.append({
                "transaction_id": f"txn_{counter:07d}", "customer_id": cid,
                "merchant_id": merch["merchant_id"], "merchant_category": merch["category"],
                "timestamp": ts, "amount": amount, "device_id": device_id, "geo_region": geo,
                "payment_method": str(rng.choice(PAYMENT_METHODS, p=PAYMENT_P)), "status": status,
                "account_created": prof["account_created"].strftime("%Y-%m-%d"),
                "is_fraud": int(is_fraud),
            })

    df = pd.DataFrame(rows, columns=RAW_COLUMNS).sort_values("timestamp").reset_index(drop=True)
    new_state = {**state, "next_day": day_index + 1, "txn_counter": counter, "episodes": episodes}
    return df, new_state


def warm_up(profiles, merchants, state: dict, days: int = EPISODE_DAYS + 1, seed: int = 42) -> dict:
    """
    A brand-new state has no compromise episodes running, so the first real day would
    under-produce fraud (episodes last several days). Simulate a few unrecorded days
    just before `next_day` and keep only their episodes, so the first batch starts from
    steady state. Rows are discarded; day counter and transaction ids are untouched.
    """
    cur = {**state, "episodes": {}}
    for d in range(state["next_day"] - days, state["next_day"]):
        _, cur = simulate_day(profiles, merchants, d, cur, seed=seed + 1_000_003)
    return {**state, "episodes": cur["episodes"]}


# ------------------------------------------------------------------- store

class DailyStore:
    """Daily raw batches plus the generator state, kept under one directory."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "daily").mkdir(parents=True, exist_ok=True)

    def has_state(self) -> bool:
        return (self.root / "state.json").exists()

    def state(self, base_raw: pd.DataFrame = None) -> dict:
        path = self.root / "state.json"
        if path.exists():
            return json.loads(path.read_text())
        if base_raw is None:
            raise FileNotFoundError("No generator state yet; pass the base history to create one.")
        return initial_state(base_raw)

    def append_day(self, day_index: int, df: pd.DataFrame, new_state: dict):
        df.to_csv(self.root / "daily" / f"day_{day_index:04d}.csv", index=False)
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(new_state))
        os.replace(tmp, self.root / "state.json")  # state only advances once the day is on disk

    def load_daily(self) -> pd.DataFrame:
        files = sorted((self.root / "daily").glob("day_*.csv"))
        if not files:
            return pd.DataFrame(columns=RAW_COLUMNS)
        out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

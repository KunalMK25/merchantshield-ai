"""
Feature engineering for MerchantShield AI.

LEAKAGE RULE (enforced and tested — see tests/test_no_leakage.py):
For a transaction at time T, every feature value must depend ONLY on transactions
strictly before T for that customer (or global/merchant stats computed the same way).
No feature may look at the current transaction's own fraud label, or at any
transaction that happens at or after T.

This module has NO knowledge of the generator in ml/data/generate_synthetic.py.
It only reads the raw observable columns of the event log.
"""

import numpy as np
import pandas as pd


RAW_COLUMNS_USED = [
    "transaction_id", "customer_id", "merchant_id", "merchant_category",
    "timestamp", "amount", "device_id", "geo_region", "payment_method",
    "status", "account_created",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_created"] = pd.to_datetime(df["account_created"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    out_frames = []
    for cid, g in df.groupby("customer_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)

        # --- running (strictly prior) statistics ---
        prior_count = np.arange(n)  # number of prior transactions (0 for first)
        amounts = g["amount"].values
        # expanding mean/std computed on values BEFORE current index
        cum_sum = np.concatenate(([0.0], np.cumsum(amounts)[:-1]))
        cum_sum_sq = np.concatenate(([0.0], np.cumsum(amounts ** 2)[:-1]))
        prior_mean = np.divide(cum_sum, prior_count, out=np.zeros(n), where=prior_count > 0)
        prior_var = np.divide(cum_sum_sq, prior_count, out=np.zeros(n), where=prior_count > 0) - prior_mean ** 2
        prior_var = np.clip(prior_var, a_min=0, a_max=None)
        prior_std = np.sqrt(prior_var)

        amount_zscore = np.where(prior_count >= 3, (amounts - prior_mean) / np.where(prior_std > 1e-6, prior_std, 1.0), 0.0)
        amount_vs_avg_ratio = np.where(prior_mean > 1e-6, amounts / np.where(prior_mean > 1e-6, prior_mean, 1.0), 1.0)

        # --- time since previous transaction (minutes) ---
        prev_ts = g["timestamp"].shift(1)
        time_since_prev_min = (g["timestamp"] - prev_ts).dt.total_seconds() / 60.0
        time_since_prev_min = time_since_prev_min.fillna(99999).values  # first txn: large "unknown" sentinel

        # --- velocity: count of prior transactions within trailing windows ---
        ts_vals = g["timestamp"].values.astype("datetime64[s]").astype(np.int64)
        vel_5min = np.zeros(n, dtype=int)
        vel_30min = np.zeros(n, dtype=int)
        vel_60min = np.zeros(n, dtype=int)
        j5 = j30 = j60 = 0
        for i in range(n):
            t = ts_vals[i]
            while j5 < i and ts_vals[j5] < t - 5 * 60:
                j5 += 1
            while j30 < i and ts_vals[j30] < t - 30 * 60:
                j30 += 1
            while j60 < i and ts_vals[j60] < t - 60 * 60:
                j60 += 1
            vel_5min[i] = i - j5
            vel_30min[i] = i - j30
            vel_60min[i] = i - j60

        # --- device / geo novelty (has this value been seen in PRIOR transactions?) ---
        seen_devices = set()
        seen_geos = set()
        new_device_flag = np.zeros(n, dtype=int)
        new_geo_flag = np.zeros(n, dtype=int)
        devices = g["device_id"].values
        geos = g["geo_region"].values
        for i in range(n):
            new_device_flag[i] = 0 if devices[i] in seen_devices else 1
            new_geo_flag[i] = 0 if geos[i] in seen_geos else 1
            seen_devices.add(devices[i])
            seen_geos.add(geos[i])
        # first-ever transaction: not meaningfully "new" (no history to compare against)
        if n > 0:
            new_device_flag[0] = 0
            new_geo_flag[0] = 0

        # --- failed-transaction ratio in trailing 10 prior transactions ---
        is_failed = (g["status"] == "failed").astype(int).values
        failed_ratio_trailing10 = np.zeros(n)
        for i in range(n):
            lo = max(0, i - 10)
            window = is_failed[lo:i]
            failed_ratio_trailing10[i] = window.mean() if len(window) > 0 else 0.0

        # --- account age at transaction time (days) ---
        account_age_days = (g["timestamp"] - g["account_created"]).dt.total_seconds() / 86400.0

        g = g.assign(
            prior_txn_count=prior_count,
            amount_zscore=amount_zscore,
            amount_vs_avg_ratio=amount_vs_avg_ratio,
            time_since_prev_txn_min=time_since_prev_min,
            velocity_5min=vel_5min,
            velocity_30min=vel_30min,
            velocity_60min=vel_60min,
            new_device_flag=new_device_flag,
            new_geo_flag=new_geo_flag,
            failed_ratio_trailing10=failed_ratio_trailing10,
            account_age_days=account_age_days,
        )
        out_frames.append(g)

    result = pd.concat(out_frames, axis=0).sort_values("timestamp").reset_index(drop=True)

    # hour-of-day / day-of-week (cyclical, no leakage risk - purely calendar)
    result["hour_of_day"] = result["timestamp"].dt.hour
    result["is_night"] = ((result["hour_of_day"] >= 0) & (result["hour_of_day"] < 5)).astype(int)
    result["day_of_week"] = result["timestamp"].dt.dayofweek

    return result


FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "amount_vs_avg_ratio",
    "prior_txn_count",
    "time_since_prev_txn_min",
    "velocity_5min",
    "velocity_30min",
    "velocity_60min",
    "new_device_flag",
    "new_geo_flag",
    "failed_ratio_trailing10",
    "account_age_days",
    "hour_of_day",
    "is_night",
    "day_of_week",
]


if __name__ == "__main__":
    raw = pd.read_csv("ml/data/raw_transactions.csv")
    feats = build_features(raw)
    feats.to_csv("ml/data/features.csv", index=False)
    print(f"Built features for {len(feats):,} transactions")
    print(feats[FEATURE_COLUMNS + ["is_fraud"]].describe().T)

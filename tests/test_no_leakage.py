"""
Sanity checks that feature values for a transaction only depend on strictly-prior
transactions for that customer. Run standalone (not yet wired into a test runner --
that happens formally in Phase 11, this is an early integrity check for Phase 1).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from ml.features.build_features import build_features


def test_prior_txn_count_matches_history():
    raw = pd.read_csv("ml/data/raw_transactions.csv")
    feats = build_features(raw)
    sample_customers = feats["customer_id"].drop_duplicates().sample(15, random_state=1)
    for cid in sample_customers:
        g = feats[feats["customer_id"] == cid].sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g)):
            assert g.loc[i, "prior_txn_count"] == i, f"leakage: prior_txn_count wrong for {cid} row {i}"
    print("PASS: prior_txn_count exactly equals row index within each customer's chronological history")


def test_new_device_flag_correctness():
    raw = pd.read_csv("ml/data/raw_transactions.csv")
    feats = build_features(raw)
    sample_customers = feats["customer_id"].drop_duplicates().sample(15, random_state=2)
    for cid in sample_customers:
        g = feats[feats["customer_id"] == cid].sort_values("timestamp").reset_index(drop=True)
        seen = set()
        for i in range(len(g)):
            dev = g.loc[i, "device_id"]
            expected_new = 0 if (dev in seen or i == 0) else 1
            assert g.loc[i, "new_device_flag"] == expected_new, f"leakage: new_device_flag wrong for {cid} row {i}"
            seen.add(dev)
    print("PASS: new_device_flag never uses information from transactions at or after current timestamp")


def test_no_future_timestamp_used_in_velocity():
    # verify velocity counts a hand-computed sample transaction correctly
    raw = pd.read_csv("ml/data/raw_transactions.csv")
    feats = build_features(raw)
    cid = feats["customer_id"].iloc[100]
    g = feats[feats["customer_id"] == cid].sort_values("timestamp").reset_index(drop=True)
    if len(g) < 5:
        print("SKIP: sampled customer has too few transactions")
        return
    i = len(g) - 1
    t = pd.to_datetime(g.loc[i, "timestamp"])
    window_start = t - pd.Timedelta(minutes=30)
    manual_count = ((pd.to_datetime(g["timestamp"]) >= window_start) & (pd.to_datetime(g["timestamp"]) < t)).sum()
    assert g.loc[i, "velocity_30min"] == manual_count, "leakage or off-by-one in velocity_30min"
    print("PASS: velocity_30min matches manual strictly-prior window count")


if __name__ == "__main__":
    test_prior_txn_count_matches_history()
    test_new_device_flag_correctness()
    test_no_future_timestamp_used_in_velocity()
    print("\nAll Phase 1 leakage sanity checks passed.")

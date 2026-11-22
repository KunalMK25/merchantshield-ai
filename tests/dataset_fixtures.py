"""Shared builders for dataset-integration tests."""

import numpy as np
import pandas as pd


def write_sparkov_like(path, customers=60, days=70, seed=0):
    """A larger Sparkov-layout file: enough history, and a few percent fraud on a few customers."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(customers):
        base = 30 + 40 * rng.random()
        for d in range(days):
            for k in range(2):
                fraud = int(c % 6 == 0 and d % 11 == 0 and k == 1 and d > 5)
                ts = pd.Timestamp("2019-01-01") + pd.Timedelta(days=d, hours=8 + 5 * k, minutes=c)
                rows.append({"": len(rows), "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                             "cc_num": 4000000000000000 + c, "merchant": f"fraud_Shop {c % 7}",
                             "category": ["grocery_pos", "gas_transport"][k],
                             "amt": round(base * (1 + rng.random()) * (6 if fraud else 1), 2),
                             "trans_num": f"{c:03d}{d:03d}{k}x", "merch_lat": 40.0 + c % 5 + rng.random() * 0.2,
                             "merch_long": -75.0 - c % 5 - rng.random() * 0.2, "is_fraud": fraud})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

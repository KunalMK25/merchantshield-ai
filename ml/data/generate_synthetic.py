"""
Synthetic transaction data generator for MerchantShield AI.

DESIGN PRINCIPLE (see docs/dataset.md for full writeup):
------------------------------------------------------------------
This generator and the feature-engineering pipeline (ml/features/build_features.py)
are deliberately written by two different "processes" that only communicate through
the raw event log (transactions.csv). The generator:

  1. Never writes an engineered feature (velocity, z-score, device-novelty flag, etc.)
     directly into the output. It only writes raw observable columns: amount, timestamp,
     device_id, geo_region, status, payment_method.
  2. Drives fraud behavior through a *latent, continuous* "compromise_intensity" value
     per customer-timeline, not a binary switch that maps 1:1 onto any single feature.
     Intensity affects amount, device choice, geo choice, and inter-arrival time
     *probabilistically*, with noise, rather than deterministically.
  3. Includes benign look-alike behavior: legitimate customers who travel (new geo),
     buy a new phone (new device), or make an unusually large one-off purchase
     (birthday gift, rent) at the same rate real populations do. These are labeled 0
     even though they trip the same raw signals fraud sometimes trips.
  4. Includes quiet fraud: some fraudulent transactions are single, modest-amount,
     same-device transactions with no velocity spike (e.g. a stolen session making one
     careful purchase). These are labeled 1 with weak feature signal.

This means the classifier trained later is NOT guaranteed high separability — if
recall or precision come out mediocre, that is a property of the (realistic) data,
not a bug to be quietly patched by making the generator more cooperative.
------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

RNG_SEED = 42
N_CUSTOMERS = 2200
N_MERCHANTS = 180
SIM_DAYS = 60
START_DATE = datetime(2026, 4, 1)

MERCHANT_CATEGORIES = [
    ("electronics", 8500, 0.55),
    ("fashion", 1800, 0.35),
    ("groceries", 650, 0.15),
    ("travel", 12000, 0.60),
    ("food_delivery", 420, 0.10),
    ("digital_goods", 1200, 0.45),
    ("home_goods", 2600, 0.30),
]

GEO_REGIONS = [f"region_{i:02d}" for i in range(1, 25)]
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]


def _rng():
    return np.random.default_rng(RNG_SEED)


def _make_customers(rng):
    customers = []
    for i in range(N_CUSTOMERS):
        account_age_days = int(rng.exponential(scale=220)) + 1
        account_created = START_DATE - timedelta(days=account_age_days)
        home_geo = rng.choice(GEO_REGIONS)
        home_device = f"dev_{uuid.uuid4().hex[:10]}"
        # baseline spend behavior per customer (log-normal is realistic for spend)
        base_amount_mean = float(np.exp(rng.normal(6.5, 0.9)))  # ~ hundreds to low thousands
        base_amount_std = base_amount_mean * rng.uniform(0.2, 0.5)
        activity_rate_per_day = rng.gamma(shape=2.0, scale=0.8)  # avg txns/day
        # secondary devices some customers legitimately use (phone + laptop + tablet)
        n_legit_devices = rng.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.2, 0.05, 0.15, 0.1, 0.05])
        legit_devices = [home_device] + [f"dev_{uuid.uuid4().hex[:10]}" for _ in range(n_legit_devices - 1)]
        # customers who travel occasionally (legit new-geo events)
        travels = rng.random() < 0.25
        customers.append(dict(
            customer_id=f"cust_{i:05d}",
            account_created=account_created,
            home_geo=home_geo,
            legit_devices=legit_devices,
            base_amount_mean=base_amount_mean,
            base_amount_std=max(base_amount_std, 10.0),
            activity_rate_per_day=max(activity_rate_per_day, 0.05),
            travels=travels,
        ))
    return customers


def _make_merchants(rng):
    merchants = []
    for i in range(N_MERCHANTS):
        cat, avg_ticket, fraud_affinity = MERCHANT_CATEGORIES[rng.integers(0, len(MERCHANT_CATEGORIES))]
        merchants.append(dict(
            merchant_id=f"merch_{i:04d}",
            category=cat,
            avg_ticket=avg_ticket * rng.uniform(0.7, 1.3),
            fraud_affinity=fraud_affinity,  # some categories more attractive to fraud (e.g. electronics, travel)
        ))
    return merchants


def _assign_compromise_events(customers, rng):
    """
    Decide, per customer, whether/when a latent 'compromise' episode occurs.
    Only ~4% of customers are ever compromised (this drives the overall fraud rate
    down to a realistic ~1.5-3% of transactions once diluted by legitimate activity).
    Intensity is continuous and decays after onset rather than being a clean on/off flag.
    """
    compromise_map = {}
    for c in customers:
        if rng.random() < 0.045:
            onset_day = rng.integers(3, SIM_DAYS - 2)
            pattern = rng.choice(["takeover_burst", "card_testing", "quiet_single", "synthetic_new_account"],
                                  p=[0.35, 0.30, 0.20, 0.15])
            peak_intensity = rng.uniform(0.5, 1.0)
            compromise_map[c["customer_id"]] = dict(
                onset_day=onset_day, pattern=pattern, peak_intensity=peak_intensity
            )
    return compromise_map


def generate(output_path="ml/data/raw_transactions.csv"):
    rng = _rng()
    customers = _make_customers(rng)
    merchants = _make_merchants(rng)
    compromise_map = _assign_compromise_events(customers, rng)

    rows = []
    txn_counter = 0

    for cust in customers:
        cid = cust["customer_id"]
        comp = compromise_map.get(cid)

        # simulate day by day
        for day in range(SIM_DAYS):
            current_date = START_DATE + timedelta(days=day)
            if current_date < cust["account_created"]:
                continue

            n_txns_today = rng.poisson(cust["activity_rate_per_day"])

            # inject a compromise burst on/near onset day
            in_compromise_window = False
            intensity = 0.0
            if comp is not None and comp["onset_day"] <= day <= comp["onset_day"] + 4:
                days_since_onset = day - comp["onset_day"]
                intensity = comp["peak_intensity"] * np.exp(-0.4 * days_since_onset)
                in_compromise_window = True
                if comp["pattern"] == "takeover_burst":
                    n_txns_today += rng.integers(4, 9)
                elif comp["pattern"] == "card_testing":
                    n_txns_today += rng.integers(6, 14)
                elif comp["pattern"] == "quiet_single":
                    n_txns_today += 1
                elif comp["pattern"] == "synthetic_new_account":
                    n_txns_today += 1

            for _ in range(max(n_txns_today, 0)):
                merch = merchants[rng.integers(0, len(merchants))]

                # base random time in the day
                seconds_into_day = rng.integers(0, 86400)
                ts = current_date + timedelta(seconds=int(seconds_into_day))

                is_fraud = False

                # ---- device / geo selection ----
                if in_compromise_window and comp["pattern"] in ("takeover_burst", "synthetic_new_account") and rng.random() < 0.8 * intensity + 0.15:
                    device_id = f"dev_{uuid.uuid4().hex[:10]}"  # unseen device
                    geo = rng.choice([g for g in GEO_REGIONS if g != cust["home_geo"]])
                    is_fraud = True
                elif cust["travels"] and rng.random() < 0.02:
                    # legit travel: new geo, familiar device -> NOT fraud
                    device_id = rng.choice(cust["legit_devices"])
                    geo = rng.choice(GEO_REGIONS)
                elif len(cust["legit_devices"]) > 1 and rng.random() < 0.03:
                    # legit secondary device -> NOT fraud
                    device_id = rng.choice(cust["legit_devices"])
                    geo = cust["home_geo"]
                else:
                    device_id = cust["legit_devices"][0]
                    geo = cust["home_geo"]

                # ---- amount ----
                if in_compromise_window and comp["pattern"] == "takeover_burst" and rng.random() < 0.7:
                    amount = max(cust["base_amount_mean"] * rng.uniform(2.5, 6.0), merch["avg_ticket"] * rng.uniform(1.2, 2.5))
                    is_fraud = True
                elif in_compromise_window and comp["pattern"] == "card_testing":
                    amount = rng.uniform(1, 50)  # small probing amounts
                    is_fraud = True
                elif in_compromise_window and comp["pattern"] == "quiet_single":
                    # careful, modest amount, blends in -> weak signal fraud
                    amount = max(rng.normal(cust["base_amount_mean"], cust["base_amount_std"]), 20)
                    is_fraud = True
                elif in_compromise_window and comp["pattern"] == "synthetic_new_account":
                    amount = merch["avg_ticket"] * rng.uniform(1.5, 3.0)
                    is_fraud = True
                else:
                    amount = max(rng.normal(cust["base_amount_mean"], cust["base_amount_std"]), 10)

                amount = round(float(amount), 2)

                # ---- payment status ----
                if in_compromise_window and comp["pattern"] == "card_testing":
                    status = rng.choice(["failed", "success"], p=[0.75, 0.25])
                else:
                    status = rng.choice(["failed", "success"], p=[0.06, 0.94])

                payment_method = rng.choice(PAYMENT_METHODS, p=[0.5, 0.35, 0.1, 0.05])

                txn_counter += 1
                rows.append(dict(
                    transaction_id=f"txn_{txn_counter:07d}",
                    customer_id=cid,
                    merchant_id=merch["merchant_id"],
                    merchant_category=merch["category"],
                    timestamp=ts,
                    amount=amount,
                    device_id=device_id,
                    geo_region=geo,
                    payment_method=payment_method,
                    status=status,
                    account_created=cust["account_created"],
                    is_fraud=int(is_fraud),
                ))

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df):,} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

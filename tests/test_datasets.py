"""
Dataset integration tests.

The real public datasets (Sparkov, TabFormer, PaySim, IEEE-CIS) are large and need a Kaggle login, so
these tests write small files in each dataset's published column layout and check the mapping into
MerchantShield's raw schema. They prove the mapping logic; they cannot prove the layouts match a file
you download today, which is why every loader reports exactly which column is missing.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.data.daily_batch import RAW_COLUMNS
from ml.datasets.catalog import DATASETS, NOT_INTEGRATED, find_files
from ml.datasets.loaders import load_ieee, load_paysim, load_sparkov, load_tabformer
from ml.datasets.prepare import prepare
from ml.datasets.schema import DatasetError, profile, validate_raw


# ---------------------------------------------------------------- fixtures

def write_sparkov(path, customers=6, days=40, fraud_days=(), start="2019-01-01"):
    rng = np.random.default_rng(0)
    rows = []
    for c in range(customers):
        for d in range(days):
            for k in range(2):
                ts = pd.Timestamp(start) + pd.Timedelta(days=d, hours=8 + 5 * k, minutes=c)
                fraud = int(d in fraud_days and c == 0 and k == 1)
                rows.append({
                    "": len(rows), "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "cc_num": 4000000000000000 + c, "merchant": f"fraud_Shop {c % 3}",
                    "category": ["grocery_pos", "gas_transport"][k], "amt": round(20 + rng.random() * 80 + 500 * fraud, 2),
                    "first": "Ann", "last": "Lee", "gender": "F", "street": "1 Main St", "city": "Town",
                    "state": "PA", "zip": 15000, "lat": 40.0 + c, "long": -75.0 - c, "city_pop": 9000,
                    "job": "Nurse", "dob": "1980-01-01", "trans_num": f"{c:03d}{d:03d}{k}abcdef",
                    "unix_time": int(ts.timestamp()), "merch_lat": 40.0 + c + rng.random() * 0.3,
                    "merch_long": -75.0 - c - rng.random() * 0.3, "is_fraud": fraud})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_tabformer(path, users=4, days=30, errors=(3,)):
    rows = []
    for u in range(users):
        for d in range(days):
            online = d % 5 == 0
            rows.append({
                "User": u, "Card": 0, "Year": 2015, "Month": 1 + d // 28, "Day": 1 + d % 28,
                "Time": f"{8 + d % 10:02d}:{d % 60:02d}", "Amount": f"${10 + u + d:.2f}",
                "Use Chip": "Online Transaction" if online else "Chip Transaction",
                "Merchant Name": 1000 + d % 7, "Merchant City": "ONLINE" if online else "Austin",
                "Merchant State": "" if online else "TX", "Zip": 78701.0, "MCC": 5411,
                "Errors?": "Insufficient Balance," if d in errors else "",
                "Is Fraud?": "Yes" if (u == 0 and d == 25) else "No"})
    rows.append({**rows[0], "Day": 2, "Amount": "$-15.00"})              # a refund: must be dropped
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_paysim(path, n=300):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        fraud = int(i % 50 == 0)
        rows.append({"step": 1 + i // 10, "type": ["PAYMENT", "TRANSFER", "CASH_OUT"][i % 3],
                     "amount": round(float(rng.random() * 5000) + 1, 2), "nameOrig": f"C{1000 + i}",
                     "oldbalanceOrg": 5000.0, "newbalanceOrig": 0.0, "nameDest": f"M{i % 20}",
                     "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFraud": fraud, "isFlaggedFraud": 0})
    rows.append({**rows[1], "nameOrig": "C_zero", "amount": 0.0})          # zero-amount row: dropped
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def sparkov(tmp_path):
    return write_sparkov(tmp_path / "fraudTrain.csv", fraud_days=(30,))


# ------------------------------------------------------------------ Sparkov

def test_sparkov_maps_into_the_raw_schema(sparkov):
    raw, notes = load_sparkov([sparkov])
    assert list(raw.columns) == RAW_COLUMNS
    validate_raw(raw)
    assert raw["transaction_id"].is_unique and raw["transaction_id"].iloc[0] == "txn_000000001"
    assert raw["timestamp"].is_monotonic_increasing
    assert set(raw["customer_id"]) == {f"cc_{4000000000000000 + c}" for c in range(6)}
    assert not raw["merchant_id"].str.startswith("fraud_").any()          # dataset's artefact prefix removed
    assert (raw["status"] == "success").all() and (raw["payment_method"] == "card").all()
    assert raw["is_fraud"].sum() == 1 and raw["is_fraud"].isin([0, 1]).all()
    assert notes["dataset"] == "sparkov" and notes["rows"] == len(raw)


def test_sparkov_never_reads_personal_columns(sparkov, monkeypatch):
    seen = {}
    real = pd.read_csv

    def spy(*a, **k):
        seen["usecols"] = k.get("usecols")
        return real(*a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    load_sparkov([sparkov])
    cols = set(seen["usecols"])
    assert not cols & {"first", "last", "street", "dob", "job", "gender", "city", "zip"}


def test_sparkov_devices_are_unknown_and_geo_comes_from_the_merchant_location(sparkov):
    raw, notes = load_sparkov([sparkov])
    assert raw.groupby("customer_id")["device_id"].nunique().max() == 1
    assert raw["geo_region"].str.startswith("g").all()
    assert "new_device_flag" in notes["inert_features"] or "device" in " ".join(notes["assumptions"]).lower()


def test_sparkov_accepts_several_files_and_samples_whole_customers(tmp_path):
    a = write_sparkov(tmp_path / "fraudTrain.csv", customers=6, days=20)
    b = write_sparkov(tmp_path / "fraudTest.csv", customers=6, days=20, start="2019-02-01")
    full, _ = load_sparkov([a, b])
    sample, notes = load_sparkov([a, b], max_customers=3, seed=7)
    again, _ = load_sparkov([a, b], max_customers=3, seed=7)
    assert sample["customer_id"].nunique() == 3 and notes["sampled_customers"] == 3
    assert sample["customer_id"].tolist() == again["customer_id"].tolist()           # deterministic
    for cust, g in sample.groupby("customer_id"):                                    # all of a customer's rows kept
        assert len(g) == (full["customer_id"] == cust).sum()


def test_a_missing_column_is_named_in_the_error(tmp_path):
    p = write_sparkov(tmp_path / "bad.csv")
    pd.read_csv(p).drop(columns=["merch_lat"]).to_csv(p, index=False)
    with pytest.raises(DatasetError, match="merch_lat"):
        load_sparkov([p])


# ----------------------------------------------------------------- TabFormer

def test_tabformer_maps_into_the_raw_schema(tmp_path):
    raw, notes = load_tabformer(write_tabformer(tmp_path / "card_transaction.v1.csv"))
    assert list(raw.columns) == RAW_COLUMNS
    validate_raw(raw)
    assert (raw["amount"] > 0).all() and notes["dropped_non_positive_amounts"] == 1
    assert set(raw["customer_id"]) == {f"u{u}_c0" for u in range(4)}
    assert (raw["status"] == "failed").sum() == 4                                    # one error day per user
    assert raw["is_fraud"].sum() == 1
    online = raw[raw["payment_method"] == "online"]
    assert not online.empty and (online["geo_region"] == "ONLINE").all()
    assert raw["amount"].dtype.kind == "f"


def test_tabformer_chunked_reading_gives_identical_results(tmp_path):
    p = write_tabformer(tmp_path / "t.csv")
    whole, _ = load_tabformer(p, chunksize=10_000)
    chunked, _ = load_tabformer(p, chunksize=7)
    pd.testing.assert_frame_equal(whole, chunked)
    sample, notes = load_tabformer(p, max_users=2, seed=3, chunksize=7)
    assert sample["customer_id"].nunique() == 2 and notes["sampled_customers"] == 2


# -------------------------------------------------------------------- PaySim

def test_paysim_maps_hours_to_timestamps_and_drops_zero_amounts(tmp_path):
    raw, notes = load_paysim(write_paysim(tmp_path / "paysim.csv"))
    assert list(raw.columns) == RAW_COLUMNS
    validate_raw(raw)
    assert notes["dropped_non_positive_amounts"] == 1 and (raw["amount"] > 0).all()
    assert raw["timestamp"].is_monotonic_increasing
    assert raw["merchant_category"].isin(["PAYMENT", "TRANSFER", "CASH_OUT"]).all()
    assert raw["is_fraud"].sum() == 6


# --------------------------------------------------------------------- IEEE

def test_ieee_wrapper_returns_the_same_schema_with_naive_timestamps(tmp_path, monkeypatch):
    import ml.datasets.loaders as loaders
    fake = pd.DataFrame({
        "transaction_id": ["a", "b"], "customer_id": ["c1", "c1"], "merchant_id": ["m", "m"],
        "merchant_category": ["W", "W"], "amount": [10.0, 20.0], "device_id": ["device_unknown"] * 2,
        "geo_region": ["g", "g"], "payment_method": ["card"] * 2, "status": ["success"] * 2,
        "timestamp": pd.to_datetime(["2017-12-01", "2017-12-02"], utc=True),
        "account_created": pd.to_datetime(["2017-11-01", "2017-11-01"], utc=True),
        "is_fraud": [0, 1], "TransactionDT": [1, 2]})
    monkeypatch.setattr(loaders, "_ieee_load_and_adapt", lambda *a, **k: (fake, {"degraded": ["x"]}))
    (tmp_path / "train_transaction.csv").write_text("x")
    raw, notes = load_ieee(tmp_path)
    assert list(raw.columns) == RAW_COLUMNS and raw["timestamp"].dt.tz is None
    assert raw["transaction_id"].iloc[0] == "txn_000000001"
    assert notes["dataset"] == "ieee"


# ------------------------------------------------------------ validate + profile

def good_raw(sparkov):
    return load_sparkov([sparkov])[0]


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.drop(columns=["device_id"]), "device_id"),
    (lambda d: d.assign(transaction_id="same"), "unique"),
    (lambda d: d.assign(amount=d["amount"] * 0), "positive"),
    (lambda d: d.assign(is_fraud=2), "is_fraud"),
    (lambda d: d.assign(customer_id=None), "customer_id"),
    (lambda d: d.assign(account_created=d["timestamp"] + pd.Timedelta(days=1)), "account_created"),
])
def test_validate_raw_rejects_each_kind_of_bad_data(sparkov, mutate, needle):
    with pytest.raises(DatasetError, match=needle):
        validate_raw(mutate(good_raw(sparkov)))


def test_profile_reports_size_history_depth_and_inert_features(sparkov):
    p = profile(good_raw(sparkov))
    assert p["rows"] == 480 and p["customers"] == 6 and p["days"] == 40
    assert p["share_rows_with_history"] > 0.8 and p["fit"] == "good"
    assert {"new_device_flag", "failed_ratio_trailing10"} <= set(p["inert_features"])
    assert p["median_amount"] > 0


def test_a_dataset_with_no_per_customer_history_is_rated_poor(tmp_path):
    raw, _ = load_paysim(write_paysim(tmp_path / "p.csv"))
    p = profile(raw)
    assert p["fit"] == "poor" and p["share_rows_with_history"] < 0.1
    assert "history" in p["fit_reason"].lower()


# ------------------------------------------------------------------- catalog

def test_catalog_lists_integrated_and_rejected_datasets_with_reasons():
    assert {"sparkov", "tabformer", "paysim", "ieee"} <= set(DATASETS)
    assert {"baf", "ulb"} <= set(NOT_INTEGRATED)
    assert all(len(reason) > 40 for reason in NOT_INTEGRATED.values())
    for d in DATASETS.values():
        assert d.source and d.files and callable(d.loader) and d.title


def test_find_files_gives_actionable_errors(tmp_path):
    with pytest.raises(DatasetError) as e:
        find_files("sparkov", tmp_path)
    msg = str(e.value)
    assert "fraudTrain.csv" in msg and "kaggle" in msg.lower()
    (tmp_path / "fraudTrain.csv").write_text("x")
    assert [p.name for p in find_files("sparkov", tmp_path)] == ["fraudTrain.csv"]
    with pytest.raises(DatasetError, match="Unknown dataset"):
        find_files("nope", tmp_path)


def test_tabformer_real_download_layout_is_found_and_other_files_are_ignored(tmp_path):
    """The real Kaggle download is credit_card_transactions-ibm_v2.csv, next to a one-user sample and two
    files holding names, addresses and card numbers that must never be loaded."""
    for name in ("credit_card_transactions-ibm_v2.csv", "User0_credit_card_transactions.csv",
                 "sd254_users.csv", "sd254_cards.csv"):
        (tmp_path / name).write_text("x")
    found = [p.name for p in find_files("tabformer", tmp_path)]
    assert found == ["credit_card_transactions-ibm_v2.csv"]


def test_tabformer_older_filename_still_works(tmp_path):
    (tmp_path / "card_transaction.v1.csv").write_text("x")
    assert [p.name for p in find_files("tabformer", tmp_path)] == ["card_transaction.v1.csv"]


def test_a_sample_file_alone_is_not_mistaken_for_the_dataset(tmp_path):
    (tmp_path / "User0_credit_card_transactions.csv").write_text("x")
    with pytest.raises(DatasetError, match="kaggle"):
        find_files("tabformer", tmp_path)


def test_paysim_and_ieee_primary_files_are_still_found(tmp_path):
    (tmp_path / "PS_20174392719_1491204439457_log.csv").write_text("x")
    assert find_files("paysim", tmp_path)[0].name.startswith("PS_")
    (tmp_path / "train_transaction.csv").write_text("x")
    (tmp_path / "train_identity.csv").write_text("x")
    assert [p.name for p in find_files("ieee", tmp_path)] == ["train_transaction.csv", "train_identity.csv"]


# ------------------------------------------------------------------- prepare

def test_prepare_writes_a_workspace_with_history_and_a_replay_stream(tmp_path, sparkov):
    ws_root = tmp_path / "ws"
    out = prepare("sparkov", sparkov.parent, ws_root, stream_days=10)
    ws = ws_root / "sparkov"
    hist, stream = pd.read_csv(ws / "raw_transactions.csv", parse_dates=["timestamp"]), \
        pd.read_csv(ws / "stream.csv", parse_dates=["timestamp"])

    assert len(hist) + len(stream) == 480 and out["history_rows"] == len(hist) and out["stream_rows"] == len(stream)
    assert hist["timestamp"].max() < stream["timestamp"].min()               # strictly earlier, no overlap
    assert (stream["timestamp"].max().normalize() - stream["timestamp"].min().normalize()).days <= 9
    assert set(hist["transaction_id"]).isdisjoint(stream["transaction_id"])
    cfg = json.loads((ws / "config.json").read_text())
    assert cfg["dataset"] == "sparkov" and cfg["fp_cost"] > 0 and cfg["fn_cost_fraction"] == 0.5
    assert cfg["fp_cost"] == pytest.approx(round(0.08 * hist["amount"].median(), 2), rel=0.5)
    assert json.loads((ws / "profile.json").read_text())["fit"] == "good"


def test_prepare_never_streams_more_than_a_third_of_the_span(tmp_path, sparkov):
    out = prepare("sparkov", sparkov.parent, tmp_path / "ws", stream_days=1000)
    assert out["stream_days"] <= 40 // 3


def test_prepare_refuses_a_poor_fit_unless_forced(tmp_path):
    write_paysim(tmp_path / "PS_20174392719_1491204439457_log.csv")
    with pytest.raises(DatasetError, match="history"):
        prepare("paysim", tmp_path, tmp_path / "ws")
    out = prepare("paysim", tmp_path, tmp_path / "ws", force=True)
    assert out["fit"] == "poor"

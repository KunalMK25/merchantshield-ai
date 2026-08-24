"""
Tests for ml/external/ieee/card_product_features.py — Phase 15.

DESIGN
------
All tests use small, fully deterministic DataFrames so expected values
can be hand-calculated. No external data files are required.

The tests verify:
  1. Correct count / seen_before / share values on known examples
  2. No future information can influence any row's feature value
  3. First occurrence of a (card, product) pair always gets count=0 / seen=0
  4. Labels (is_fraud) are never used during feature construction
  5. Chronological ordering is respected
  6. Share is within [0, 1] and sentinel (0.0) is applied correctly
  7. Multi-card, multi-product interaction is handled correctly
  8. Input validation raises on missing columns
  9. Input validation raises on unsorted data
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.external.ieee.card_product_features import (
    build_card_product_features,
    NEW_FEATURE_NAMES,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(rows):
    """
    Build a minimal test DataFrame from a list of
    (transaction_id, customer_id, merchant_category, TransactionDT, is_fraud).
    Already sorted by TransactionDT.
    """
    return pd.DataFrame(rows, columns=[
        "transaction_id", "customer_id", "merchant_category",
        "TransactionDT", "is_fraud",
    ]).sort_values("TransactionDT").reset_index(drop=True)


# ── 1. Basic correctness on a single card, single product ────────────────────

class TestSingleCardSingleProduct:
    """One card, always the same ProductCD — count should be 0,1,2,..."""

    def setup_method(self):
        self.df = _make_df([
            ("tx1", "card_A", "W", 100, 0),
            ("tx2", "card_A", "W", 200, 0),
            ("tx3", "card_A", "W", 300, 1),
            ("tx4", "card_A", "W", 400, 0),
        ])
        self.out = build_card_product_features(self.df)

    def test_prior_count_is_zero_for_first(self):
        row = self.out[self.out["transaction_id"] == "tx1"].iloc[0]
        assert row["card_product_prior_count"] == 0

    def test_prior_count_increments_correctly(self):
        counts = self.out.sort_values("TransactionDT")["card_product_prior_count"].tolist()
        assert counts == [0, 1, 2, 3], f"Expected [0,1,2,3], got {counts}"

    def test_seen_before_is_zero_for_first(self):
        row = self.out[self.out["transaction_id"] == "tx1"].iloc[0]
        assert row["card_product_seen_before"] == 0

    def test_seen_before_is_one_after_first(self):
        out_sorted = self.out.sort_values("TransactionDT")
        assert out_sorted["card_product_seen_before"].tolist() == [0, 1, 1, 1]

    def test_share_is_zero_for_first(self):
        row = self.out[self.out["transaction_id"] == "tx1"].iloc[0]
        assert row["card_product_share"] == 0.0

    def test_share_is_one_for_all_same_product(self):
        """When all prior transactions are the same product, share = 1.0."""
        out_sorted = self.out.sort_values("TransactionDT")
        # tx2: 1 prior W / 1 prior total = 1.0
        # tx3: 2 prior W / 2 prior total = 1.0
        # tx4: 3 prior W / 3 prior total = 1.0
        shares = out_sorted["card_product_share"].tolist()
        assert shares[0] == 0.0          # first transaction, no prior
        assert all(s == 1.0 for s in shares[1:]), f"Expected 1.0 for all non-first, got {shares}"

    def test_row_count_preserved(self):
        assert len(self.out) == len(self.df)

    def test_labels_not_used(self):
        """Flipping is_fraud should not change any feature value."""
        df_flipped = self.df.copy()
        df_flipped["is_fraud"] = 1 - df_flipped["is_fraud"]
        out_flipped = build_card_product_features(df_flipped)
        for feat in NEW_FEATURE_NAMES:
            pd.testing.assert_series_equal(
                self.out[feat].reset_index(drop=True),
                out_flipped[feat].reset_index(drop=True),
                check_names=False,
                obj=f"Feature '{feat}' changed when is_fraud was flipped",
            )


# ── 2. Single card, multiple products ────────────────────────────────────────

class TestSingleCardMultipleProducts:
    """
    card_A: W, W, C, W, C
    At each point, count/share must reflect only PRIOR transactions.
    """

    def setup_method(self):
        self.df = _make_df([
            ("tx1", "card_A", "W", 100, 0),   # first W: count=0, share=0
            ("tx2", "card_A", "W", 200, 0),   # second W: count=1, share=1/1=1.0
            ("tx3", "card_A", "C", 300, 0),   # first C: count=0, share=0/2=0.0
            ("tx4", "card_A", "W", 400, 1),   # third W: count=2, share=2/3≈0.667
            ("tx5", "card_A", "C", 500, 0),   # second C: count=1, share=1/4=0.25
        ])
        self.out = build_card_product_features(self.df).sort_values("TransactionDT").reset_index(drop=True)

    def test_prior_count_per_product(self):
        expected_counts = [0, 1, 0, 2, 1]
        assert self.out["card_product_prior_count"].tolist() == expected_counts, \
            f"Got {self.out['card_product_prior_count'].tolist()}"

    def test_seen_before_per_product(self):
        expected_seen = [0, 1, 0, 1, 1]
        assert self.out["card_product_seen_before"].tolist() == expected_seen

    def test_share_tx1(self):
        # First transaction — no prior history
        assert self.out.iloc[0]["card_product_share"] == pytest.approx(0.0)

    def test_share_tx2(self):
        # tx2: 1 prior W, 1 prior total → share = 1.0
        assert self.out.iloc[1]["card_product_share"] == pytest.approx(1.0)

    def test_share_tx3(self):
        # tx3: 0 prior C, 2 prior total → share = 0.0
        assert self.out.iloc[2]["card_product_share"] == pytest.approx(0.0)

    def test_share_tx4(self):
        # tx4: 2 prior W, 3 prior total → share = 2/3
        assert self.out.iloc[3]["card_product_share"] == pytest.approx(2 / 3)

    def test_share_tx5(self):
        # tx5: 1 prior C, 4 prior total → share = 1/4
        assert self.out.iloc[4]["card_product_share"] == pytest.approx(0.25)

    def test_share_within_bounds(self):
        assert self.out["card_product_share"].between(0.0, 1.0).all()


# ── 3. Multiple cards ─────────────────────────────────────────────────────────

class TestMultipleCards:
    """card_A and card_B interleaved — each card's features are independent."""

    def setup_method(self):
        self.df = _make_df([
            ("t1", "card_A", "W", 100, 0),
            ("t2", "card_B", "W", 150, 0),
            ("t3", "card_A", "W", 200, 1),
            ("t4", "card_B", "C", 250, 0),
            ("t5", "card_A", "C", 300, 0),
        ])
        self.out = build_card_product_features(self.df).sort_values("TransactionDT").reset_index(drop=True)

    def test_card_a_w_counts(self):
        a_w = self.out[
            (self.out["customer_id"] == "card_A") &
            (self.out["merchant_category"] == "W")
        ].sort_values("TransactionDT")
        assert a_w["card_product_prior_count"].tolist() == [0, 1]

    def test_card_b_w_count(self):
        b_w = self.out[
            (self.out["customer_id"] == "card_B") &
            (self.out["merchant_category"] == "W")
        ]
        assert b_w.iloc[0]["card_product_prior_count"] == 0

    def test_card_b_c_count(self):
        b_c = self.out[
            (self.out["customer_id"] == "card_B") &
            (self.out["merchant_category"] == "C")
        ]
        # First C for card_B — should be 0 even though card_A has used C
        assert b_c.iloc[0]["card_product_prior_count"] == 0

    def test_cards_are_independent(self):
        """card_A's product count must not be affected by card_B's transactions."""
        a_c = self.out[
            (self.out["customer_id"] == "card_A") &
            (self.out["merchant_category"] == "C")
        ]
        # tx5: card_A first C, prior W=2, total=2 → count=0
        assert a_c.iloc[0]["card_product_prior_count"] == 0
        assert a_c.iloc[0]["card_product_seen_before"] == 0


# ── 4. Future leakage tests ───────────────────────────────────────────────────

class TestNoFutureLeakage:
    """Prove that no future transaction can influence any past feature value."""

    def test_inserting_future_transaction_does_not_change_past(self):
        """
        Add a new transaction with DT=999 (future) to the dataset.
        All earlier features must remain identical.
        """
        df_base = _make_df([
            ("tx1", "card_A", "W", 100, 0),
            ("tx2", "card_A", "W", 200, 0),
            ("tx3", "card_A", "W", 300, 0),
        ])
        df_with_future = _make_df([
            ("tx1", "card_A", "W", 100, 0),
            ("tx2", "card_A", "W", 200, 0),
            ("tx3", "card_A", "W", 300, 0),
            ("tx_future", "card_A", "W", 999, 0),   # future transaction added
        ])
        out_base   = build_card_product_features(df_base)
        out_future = build_card_product_features(df_with_future)

        # Features for the original three rows must be identical
        for feat in NEW_FEATURE_NAMES:
            base_vals   = out_base[out_base["transaction_id"].isin(
                ["tx1","tx2","tx3"])].sort_values("TransactionDT")[feat].tolist()
            future_vals = out_future[out_future["transaction_id"].isin(
                ["tx1","tx2","tx3"])].sort_values("TransactionDT")[feat].tolist()
            assert base_vals == future_vals, (
                f"Feature '{feat}' changed when a future transaction was added: "
                f"base={base_vals}  with_future={future_vals}"
            )

    def test_first_transaction_count_is_always_zero(self):
        """First (card, product) pair appearance must always have count=0."""
        df = _make_df([
            ("tx1", "card_A", "W", 100, 1),   # fraud, but should still be 0
            ("tx2", "card_A", "W", 200, 0),
            ("tx3", "card_B", "C", 150, 1),   # different card, different product
            ("tx4", "card_B", "W", 250, 0),   # different product for card_B
        ])
        out = build_card_product_features(df)
        # First occurrence of each (card, product) pair
        first_pairs = out.groupby(["customer_id","merchant_category"]).first()
        assert (first_pairs["card_product_prior_count"] == 0).all(), \
            "Some first (card, product) pair has prior_count != 0"
        assert (first_pairs["card_product_seen_before"] == 0).all(), \
            "Some first (card, product) pair has seen_before != 0"

    def test_count_strictly_less_than_row_position_in_group(self):
        """
        The card_product_prior_count at position i (0-indexed within group)
        must equal i exactly.
        """
        df = _make_df([
            ("tx1", "card_A", "W", 100, 0),
            ("tx2", "card_A", "W", 200, 0),
            ("tx3", "card_A", "W", 300, 0),
            ("tx4", "card_A", "W", 400, 0),
        ])
        out = build_card_product_features(df).sort_values("TransactionDT").reset_index(drop=True)
        for i, row in out.iterrows():
            assert row["card_product_prior_count"] == i, \
                f"Row {i}: expected count={i}, got {row['card_product_prior_count']}"

    def test_current_transaction_not_in_own_count(self):
        """
        The current transaction MUST NOT count itself.
        Verify by checking that seen_before=0 on the FIRST occurrence of
        a (card, product) pair, regardless of label.
        """
        df = _make_df([
            ("only", "card_X", "R", 500, 1),  # only transaction, fraud
        ])
        out = build_card_product_features(df)
        assert out.iloc[0]["card_product_prior_count"] == 0
        assert out.iloc[0]["card_product_seen_before"] == 0
        assert out.iloc[0]["card_product_share"] == 0.0


# ── 5. Label independence ─────────────────────────────────────────────────────

class TestLabelIndependence:
    """Feature values must be identical regardless of is_fraud values."""

    def test_all_fraud_same_as_all_legit(self):
        rows = [
            ("tx1", "card_A", "W", 100),
            ("tx2", "card_A", "W", 200),
            ("tx3", "card_A", "C", 300),
            ("tx4", "card_A", "W", 400),
        ]
        df_fraud = pd.DataFrame(
            [r + (1,) for r in rows],
            columns=["transaction_id","customer_id","merchant_category","TransactionDT","is_fraud"]
        )
        df_legit = pd.DataFrame(
            [r + (0,) for r in rows],
            columns=["transaction_id","customer_id","merchant_category","TransactionDT","is_fraud"]
        )
        out_f = build_card_product_features(df_fraud.sort_values("TransactionDT").reset_index(drop=True))
        out_l = build_card_product_features(df_legit.sort_values("TransactionDT").reset_index(drop=True))
        for feat in NEW_FEATURE_NAMES:
            pd.testing.assert_series_equal(
                out_f[feat].reset_index(drop=True),
                out_l[feat].reset_index(drop=True),
                check_names=False,
            )

    def test_alternating_labels_same_features(self):
        rows = [("t1","c","W",10), ("t2","c","W",20), ("t3","c","W",30)]
        df_alt = pd.DataFrame(
            [r + (i % 2,) for i, r in enumerate(rows)],
            columns=["transaction_id","customer_id","merchant_category","TransactionDT","is_fraud"]
        )
        df_all0 = pd.DataFrame(
            [r + (0,) for r in rows],
            columns=["transaction_id","customer_id","merchant_category","TransactionDT","is_fraud"]
        )
        out_alt  = build_card_product_features(df_alt)
        out_all0 = build_card_product_features(df_all0)
        for feat in NEW_FEATURE_NAMES:
            pd.testing.assert_series_equal(
                out_alt[feat].reset_index(drop=True),
                out_all0[feat].reset_index(drop=True),
                check_names=False,
            )


# ── 6. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_transaction(self):
        df = _make_df([("only", "card_Z", "S", 42, 0)])
        out = build_card_product_features(df)
        assert out.iloc[0]["card_product_prior_count"] == 0
        assert out.iloc[0]["card_product_seen_before"] == 0
        assert out.iloc[0]["card_product_share"] == 0.0

    def test_all_different_products(self):
        df = _make_df([
            ("t1", "card_A", "W", 100, 0),
            ("t2", "card_A", "C", 200, 0),
            ("t3", "card_A", "R", 300, 0),
            ("t4", "card_A", "H", 400, 0),
            ("t5", "card_A", "S", 500, 0),
        ])
        out = build_card_product_features(df).sort_values("TransactionDT").reset_index(drop=True)
        # Every transaction is the first of its (card, product) → count=0, seen=0
        assert (out["card_product_prior_count"] == 0).all()
        assert (out["card_product_seen_before"] == 0).all()
        # share: always 0 because no prior matches
        assert (out["card_product_share"] == 0.0).all()

    def test_share_is_never_negative(self):
        df = _make_df([
            ("t1","c","W",1,0),("t2","c","W",2,0),("t3","c","C",3,0),
            ("t4","c","W",4,0),("t5","c","C",5,0),("t6","c","R",6,0),
        ])
        out = build_card_product_features(df)
        assert (out["card_product_share"] >= 0.0).all()

    def test_share_is_never_above_one(self):
        df = _make_df([
            ("t1","c","W",1,0),("t2","c","W",2,0),("t3","c","W",3,0),
        ])
        out = build_card_product_features(df)
        assert (out["card_product_share"] <= 1.0).all()

    def test_feature_names_are_as_declared(self):
        df = _make_df([("t1","c","W",1,0)])
        out = build_card_product_features(df)
        for feat in NEW_FEATURE_NAMES:
            assert feat in out.columns, f"Missing declared feature: {feat}"

    def test_output_row_count_matches_input(self):
        df = _make_df([
            ("t1","cA","W",1,0),("t2","cA","C",2,1),("t3","cB","W",3,0),
        ])
        out = build_card_product_features(df)
        assert len(out) == len(df)

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"customer_id": ["a"], "TransactionDT": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            build_card_product_features(df)

    def test_raises_on_unsorted_data(self):
        df = pd.DataFrame({
            "transaction_id": ["t2","t1"],
            "customer_id": ["c","c"],
            "merchant_category": ["W","W"],
            "TransactionDT": [200, 100],   # descending — unsorted
            "is_fraud": [0, 0],
        })
        with pytest.raises(ValueError, match="sorted by TransactionDT"):
            build_card_product_features(df)

    def test_same_dt_different_cards_handled(self):
        """Two different cards can have the same TransactionDT — each is independent."""
        df = _make_df([
            ("t1","cA","W",100,0),
            ("t2","cB","W",100,0),  # same DT as t1 but different card
            ("t3","cA","W",200,0),
        ])
        out = build_card_product_features(df)
        cA_rows = out[out["customer_id"]=="cA"].sort_values("TransactionDT")
        cB_rows = out[out["customer_id"]=="cB"]
        assert cA_rows.iloc[0]["card_product_prior_count"] == 0  # first for cA/W
        assert cB_rows.iloc[0]["card_product_prior_count"] == 0  # first for cB/W
        # cA's second W should have count=1 (after its own first W)
        assert cA_rows.iloc[1]["card_product_prior_count"] == 1

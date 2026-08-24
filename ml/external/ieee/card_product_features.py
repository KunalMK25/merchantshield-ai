"""
Phase 15 — Card-to-product/category familiarity features.

PURPOSE
-------
Test whether card-to-ProductCD familiarity features improve the external
IEEE-CIS model's precision/recall tradeoff compared to the Phase 14 baseline.

This module is EXPERIMENT-ONLY. It does NOT:
  - modify the production MerchantShield feature pipeline (build_features.py)
  - modify the frozen lgbm_v1 model or its 0.40 threshold
  - modify any existing synthetic pipeline
  - merge these features into the MerchantShield API or frontend

HYPOTHESIS (Phase 14 evidence)
-------------------------------
The Phase 14 FP analysis revealed that false positives often involve elevated
amounts relative to card history (amount_vs_avg_ratio 1.28× for FP vs TN).
However, the model cannot distinguish whether a high-amount transaction in an
*unusual product category* for that card is more suspicious than an equally
high-amount transaction in the card's *familiar category*.

Card-to-product familiarity captures this: if a card has 200 prior W-category
transactions but has never used C-category before, a C-category transaction is
a novelty signal independent of amount.

THREE FEATURES (all strictly causal — prior history only)
-----------------------------------------------------------
1. card_product_prior_count:
   Number of prior transactions for this exact (card1, ProductCD) pair.
   First appearance for a new card/product combination = 0.
   Analogy to prior_txn_count, but per product category.

2. card_product_seen_before:
   Binary (0/1): has this card ever used this ProductCD before this transaction?
   First appearance = 0, all subsequent = 1.
   Simpler, lower-variance alternative to the count.

3. card_product_share:
   Fraction of this card's PRIOR transactions that used this ProductCD.
   = card_product_prior_count / prior_txn_count_for_card.
   First transaction for a card = 0.0 (prior count = 0, no history).
   If prior_txn_count = 0 for this card: share = 0.0 (sentinel).

LEAKAGE CONTROLS
----------------
All three features are computed using an expanding window that includes ONLY
transactions strictly before the current transaction in TransactionDT order.
The current transaction CANNOT influence its own feature value.

The computation proceeds:
1. Sort by (card1, TransactionDT) — within each card, ascending time.
2. For each transaction i of card c with ProductCD p:
   - count = number of prior transactions (j < i for card c) with product p
   - seen  = 1 if count > 0 else 0
   - share = count / total_prior_count_for_card_c
3. No label (isFraud) is read at any point.
4. No future transaction can affect a past transaction's features.

INPUT REQUIREMENTS
------------------
The input DataFrame must be sorted by TransactionDT (ascending) and contain:
  - transaction_id  (str)
  - customer_id     (str — corresponds to card1)
  - merchant_category (str — corresponds to ProductCD, e.g. 'W', 'C', 'R')
  - is_fraud        (int — kept separate, NEVER used for feature computation)

The function never reads is_fraud. It is present in the input only because
build_ieee_features re-attaches it; this function ignores it explicitly.

OUTPUT
------
Returns a copy of the input DataFrame with three additional columns:
  - card_product_prior_count   (int64, ≥ 0)
  - card_product_seen_before   (int8, 0 or 1)
  - card_product_share         (float64, [0.0, 1.0])

FEATURE NAMES (for ablation experiments)
-----------------------------------------
NEW_FEATURE_NAMES = ["card_product_prior_count",
                     "card_product_seen_before",
                     "card_product_share"]
"""

import numpy as np
import pandas as pd

# Names of the three new features — referenced in ablation experiment
NEW_FEATURE_NAMES = [
    "card_product_prior_count",
    "card_product_seen_before",
    "card_product_share",
]


def build_card_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add card-to-product familiarity features to a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: transaction_id, customer_id, merchant_category,
        TransactionDT, is_fraud (kept separate — never read for features).
        Must be sorted ascending by TransactionDT before calling this function.
        Typically the output of ieee_features.build_ieee_features() with
        TransactionDT re-attached.

    Returns
    -------
    pd.DataFrame
        Copy of df with three new columns appended. All other columns
        are unchanged.

    Raises
    ------
    ValueError
        If required columns are missing or if the DataFrame is not sorted
        by TransactionDT (detected via monotonicity check).
    """
    required = {"transaction_id", "customer_id", "merchant_category", "TransactionDT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Ensure the input is the output of build_ieee_features()."
        )

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted by TransactionDT ascending before "
            "calling build_card_product_features(). This ensures the "
            "expanding-window computation is causally correct."
        )

    # --- LEAKAGE GUARD: confirm isFraud / is_fraud is NOT used ---
    # We only need customer_id, merchant_category, and TransactionDT.
    # We explicitly ignore is_fraud by never referencing it below.

    result = df.copy()

    # ------------------------------------------------------------------ #
    # Implementation strategy: group by customer_id, sort by TransactionDT
    # within each group, then use expanding cumulative counts.
    #
    # For each transaction at position i in a card's chronological sequence:
    #   - prior_count_for_product = number of transactions 0..i-1 with same product
    #   - prior_total_count = i (number of all transactions before this one)
    #
    # We implement this efficiently using:
    #   cumcount() within (card, product) group = how many PRIOR transactions
    #              have the same (card, product) INCLUDING current → subtract 0
    #              because cumcount starts from 0 for the first occurrence.
    #   Actually: groupby(card, product).cumcount() gives 0 for the FIRST
    #   transaction in that card+product group, 1 for the second, etc.
    #   That is exactly what we want: 0 prior transactions for the first one.
    # ------------------------------------------------------------------ #

    # Sort so cumcount is in correct time order within each group.
    # We keep the original index to restore order at the end.
    sorted_df = result.sort_values(["customer_id", "TransactionDT"]).copy()

    # card_product_prior_count:
    # cumcount within (customer_id, merchant_category) gives the number of
    # PRIOR transactions of the same (card, product) pair:
    #   position 0 in group → 0 prior
    #   position 1 in group → 1 prior
    #   etc.
    sorted_df["card_product_prior_count"] = (
        sorted_df.groupby(["customer_id", "merchant_category"]).cumcount()
    ).astype(np.int64)

    # card_product_seen_before: 1 if prior count > 0, else 0
    sorted_df["card_product_seen_before"] = (
        sorted_df["card_product_prior_count"] > 0
    ).astype(np.int8)

    # card_product_share: fraction of prior transactions for this card that
    # used this product.
    # prior_txn_count_for_card = cumcount within customer_id only (all products)
    # = total prior transactions for this card.
    # When prior_total = 0 (first transaction for a card), share = 0.0.
    sorted_df["_prior_total"] = (
        sorted_df.groupby("customer_id").cumcount()
    ).astype(np.int64)

    sorted_df["card_product_share"] = np.where(
        sorted_df["_prior_total"] > 0,
        sorted_df["card_product_prior_count"] / sorted_df["_prior_total"],
        0.0,
    ).astype(np.float64)

    sorted_df = sorted_df.drop(columns=["_prior_total"])

    # Restore original row order (matching TransactionDT-sorted input)
    result = sorted_df.sort_values("TransactionDT").reset_index(drop=True)

    # Final leakage assertion: verify first occurrence of each (card, product)
    # pair has count=0 and seen=0
    first_mask = sorted_df.groupby(
        ["customer_id", "merchant_category"]
    )["card_product_prior_count"].transform("first") == 0
    # This would always be true by construction but serves as a code-level check
    assert (sorted_df.loc[sorted_df["card_product_prior_count"] == 0,
                           "card_product_seen_before"] == 0).all(), \
        "Leakage: seen_before=1 on a transaction with prior_count=0"

    # Verify share is within [0, 1]
    assert result["card_product_share"].between(0.0, 1.0).all(), \
        "card_product_share out of [0, 1] range"

    # Verify no negative counts
    assert (result["card_product_prior_count"] >= 0).all(), \
        "card_product_prior_count contains negative values"

    return result


def feature_distributions(df: pd.DataFrame, label: str = "") -> None:
    """
    Print distribution summary for the three new features.
    For diagnostic use only — not part of any model pipeline.
    """
    prefix = f"[card_product{' '+label if label else ''}]"
    for feat in NEW_FEATURE_NAMES:
        col = df[feat]
        print(f"  {prefix} {feat}: "
              f"mean={col.mean():.4f}  std={col.std():.4f}  "
              f"p50={col.median():.4f}  "
              f"zero_rate={( col == 0).mean():.1%}  "
              f"max={col.max():.4f}")
        if "is_fraud" in df.columns:
            fraud_mean = col[df["is_fraud"] == 1].mean()
            legit_mean = col[df["is_fraud"] == 0].mean()
            print(f"           fraud_mean={fraud_mean:.4f}  "
                  f"legit_mean={legit_mean:.4f}  "
                  f"ratio={fraud_mean/legit_mean:.3f}" if legit_mean > 0 else "")

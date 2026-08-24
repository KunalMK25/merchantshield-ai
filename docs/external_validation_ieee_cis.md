# External Validation — IEEE-CIS Fraud Detection Dataset

This document describes the IEEE-CIS external validation track for MerchantShield AI.
It is a **separate research track** that does not modify any part of the existing
MerchantShield v1 production system (synthetic pipeline, frozen model, threshold,
decision engine, API, or frontend).

---

## Purpose

MerchantShield's model performance report concluded:

> **OBJECTIVE SUBSTANTIALLY ACHIEVED, EXTERNAL VALIDATION REMAINS.**

The primary open question is: *do the model's learned patterns generalise beyond the
synthetic data distribution?* This track attempts to answer two separate questions:

**Question A — Frozen model transfer:**
Does the frozen `lgbm_v1` model (trained on MerchantShield's synthetic data) produce
meaningful predictions on IEEE-CIS without retraining?

**Question B — Methodology transfer:**
Can MerchantShield's feature-engineering and training methodology be retrained from
scratch on IEEE-CIS, and does it outperform baselines on that dataset?

These are **not the same question**. A failed Answer A does not imply a failed Answer B.

---

## Dataset Provenance

**Dataset:** IEEE-CIS Fraud Detection  
**Provider:** Vesta Corporation (e-commerce payment processor)  
**Released via:** Kaggle competition (2019), organised by the IEEE Computational Intelligence Society  
**Competition URL:** https://www.kaggle.com/c/ieee-fraud-detection  

**Data type:** Anonymised real transaction data. Labels (`isFraud=1`) represent confirmed
chargebacks reported on a card, with all subsequent transactions linked to the same
card/email/billing address also labelled fraud. This is a real chargeback-based fraud label,
not a simulation.

**Size:** 590,540 labelled training transactions. The Kaggle test set (~506k rows) has
labels withheld by the competition host and is not used.

**Fraud rate:** 3.499% (20,663 fraud / 590,540 total)

---

## Access and Licensing

**Access requirement:** Kaggle account + acceptance of competition rules.

**License:** The data is provided for non-commercial research purposes per the Kaggle
competition terms. It **must not be redistributed** or published.

**Why the data is not committed to Git:**
The raw CSV files (`train_transaction.csv`, `train_identity.csv`) are excluded from the
repository by `.gitignore`. They must be downloaded locally by any developer who wishes
to run the experiments. See "Running the experiments" below for download instructions.

**Files expected locally at:**
```
C:\Users\...\Downloads\ieee_cis_inspect\train_transaction.csv   (~683 MB)
C:\Users\...\Downloads\ieee_cis_inspect\train_identity.csv      (~26.5 MB)
```
The adapter code accepts any path via command-line arguments — the paths above are the
defaults used during development.

---

## What This Track Does NOT Change

The following are completely untouched by this external validation track:

| Component | Status |
|---|---|
| `ml/data/generate_synthetic.py` | Unchanged |
| `ml/features/build_features.py` | Unchanged — called as-is |
| `ml/models/candidate_lgbm_v1.pkl` | Unchanged (frozen model) |
| `ml/models/lgbm_v1_metadata.json` | Unchanged (frozen threshold 0.40) |
| `ml/evaluation/decision_engine.py` | Unchanged |
| `ml/evaluation/policy.py` | Unchanged |
| `backend/` | Unchanged |
| `frontend/` | Unchanged |
| `tests/test_api.py` etc. | Unchanged (all 125 tests still pass) |

---

## Feature Mapping

MerchantShield uses 15 behavioural features. This section documents exactly what each
feature maps to in the IEEE-CIS adapter.

### Active features (non-trivial signal expected)

| Feature | IEEE-CIS source | Notes |
|---|---|---|
| `amount` | `TransactionAmt` | USD amounts (MerchantShield training used INR — relative ratios unaffected) |
| `amount_zscore` | `card1` + `TransactionAmt` + `TransactionDT` | Per-card expanding z-score; exact |
| `amount_vs_avg_ratio` | Same | Per-card expanding ratio; exact |
| `prior_txn_count` | `card1` + `TransactionDT` | Exact count of prior transactions for this card |
| `time_since_prev_txn_min` | `TransactionDT` ÷ 60 | TransactionDT is seconds — exact inter-transaction gap |
| `velocity_5min` | `card1` + `TransactionDT` | 300-second window; sub-minute gaps confirmed present |
| `velocity_30min` | Same | 1800-second window |
| `velocity_60min` | Same | 3600-second window |

### Degraded features (signal present but impaired)

| Feature | IEEE-CIS source | Degradation |
|---|---|---|
| `new_geo_flag` | `addr1` (billing region code) | 11.1% null → `geo_region="geo_unknown"` → flag=0. `addr1` is billing region (rarely changes), not per-transaction location. |
| `account_age_days` | `D11` (proxy) | Community-documented as "days since account opened". **47.3% null** (those rows get `account_age_days=0.0`). D11 Spearman monotonicity rho=0.244 (weak). Assessment: DEGRADED. |
| `hour_of_day` | `TransactionDT` + inferred reference | Reference datetime (2017-11-30) is not officially documented by Vesta. Hours are approximate. |
| `is_night` | Same | Same caveat |
| `day_of_week` | Same | Same caveat |

### Inert features (zero signal — constant sentinel inputs)

| Feature | Why inert | Adapter decision |
|---|---|---|
| `new_device_flag` | `DeviceInfo` is 79.9% missing after identity join; missingness is fraud-correlated (not MAR — 7.25% fraud when present vs 2.56% when absent). Using any value would introduce a confound, not genuine device novelty signal. | `device_id = "device_unknown"` (constant) → flag=0 for all rows |
| `failed_ratio_trailing10` | No transaction success/failure field exists in IEEE-CIS. | `status = "success"` (constant) → ratio=0.0 for all rows |

**Important:** these features are not fabricated — the value 0.0 correctly reflects
"no device novelty signal available" and "no failed transaction history available".
They are passed to `build_features()` exactly as coded; it is the sentinel *input*
that makes them produce zeros, not any special handling in the feature builder.

---

## Prohibited Columns

The following IEEE-CIS columns are **never loaded or used**, enforced by the adapter's
`adapt()` function which raises `ValueError` if any are present:

| Column group | Count | Reason excluded |
|---|---|---|
| `V1`–`V339` | 339 | Vesta-engineered opaque features. Unknown definitions — likely include post-transaction aggregates. |
| `C1`–`C14` | 14 | Undisclosed count/aggregate features. No missing values (suspicious for transaction-level data). Likely post-transaction signals. |
| `M1`–`M9` | 9 | Match flags. 45–90% missing. Undisclosed semantics. |
| `D2`–`D10`, `D12`–`D15` | 13 | Undisclosed timedelta features. Definitions not published. |
| `dist1`, `dist2` | 2 | 59.7% and 93.6% missing. Undisclosed distance metric. |

---

## Chronological Split

The split was declared **before any modeling** from TransactionDT quantiles during the
inspection phase.

| Split | TransactionDT range | Rows | Fraud rate | Days from start |
|---|---|---|---|---|
| Train | `< 9,614,666` | 383,851 | 3.420% | 0–110 |
| Validation | `[9,614,666, 12,192,853)` | 88,581 | 3.921% | 110–140 |
| Test | `≥ 12,192,853` | 118,108 | 3.441% | 140–182 |

The dataset spans 182 days (approximately 6 months, late 2017 through mid-2018).

**Causal construction:** Features are built from the full chronological card history
*before* the split is applied. A test-set transaction's features use all prior
transactions for that card, including those in the training split — this matches how
the model would operate in production (where historical context is available) and
matches the same methodology as MerchantShield's synthetic split.

---

## Experiment A — Frozen Model Transfer

**File:** `ml/external/ieee/experiment_a_frozen.py`

### Protocol

1. Load the frozen `lgbm_v1` model. Assert threshold = 0.40. Do not modify.
2. Score all three splits using the frozen model's `predict_proba`.
3. **Primary result:** evaluate at the frozen 0.40 threshold on the test set.
4. **Experiment A.5:** sweep thresholds on the *validation* set to determine
   the frozen model's best achievable external performance (informational only —
   does not update any threshold).
5. Report probability calibration statistics.

### What a "good" result means

The frozen model generalises if precision and recall are both materially above
the naïve all-positive baseline (precision = fraud prevalence ≈ 3.5%) at the
frozen threshold or at any threshold.

### What a "bad" result means

The frozen model fails to generalise if its probability outputs are not
discriminative — e.g. all probabilities near zero (the synthetic model learned
to flag a 1.6%-prevalence distribution; at 3.5% prevalence its calibration
may shift). A failed transfer does **not** mean the methodology failed — see
Experiment B.

### Cost model note

The same `CostAssumptions(fp_cost=50, fn_cost_fraction=0.5)` parameters are
used. IEEE-CIS amounts are USD; MerchantShield's were INR. Cost totals are
labelled "cost units" — absolute values are **not** comparable across datasets,
only within each dataset.

---

## Experiment B — External Retraining

**File:** `ml/external/ieee/experiment_b_retrain.py`

### Protocol

1. Train a **new** LightGBM model from scratch on the external training split.
   Same architecture: `num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=600,
   early_stopping(rounds=50, metric=auc)`. `scale_pos_weight` is recalculated from
   the external training set's class balance.
2. Train a Logistic Regression baseline (same `class_weight='balanced'` approach).
3. Select threshold using the **validation split only** — minimise expected cost
   subject to recall ≥ 80%. Do not inspect the test set during selection.
4. Evaluate **once** on the test set at the selected threshold.
5. Compare: naive baseline → LR → external LightGBM.
6. Save the external model as `ml/external/ieee/results/ieee_lgbm_v1.pkl`.
   This file **never overwrites** `ml/models/candidate_lgbm_v1.pkl`.

### Expected threshold behaviour

At 3.5% prevalence (vs 1.59% synthetic), the optimal threshold will be lower
than 0.40. The frozen model's sensitivity analysis predicted optimal thresholds
in the range 0.25–0.35 at 3% prevalence. The retrained model's scale_pos_weight
will also differ.

### What success means

Experiment B succeeds if the retrained LightGBM outperforms both baselines on
expected cost while maintaining recall ≥ 80%. This would demonstrate that
MerchantShield's feature design and training methodology is transportable to
a different (real-data) fraud dataset.

---

## Key Limitations

1. **Label noise:** Vesta's labelling propagates `isFraud=1` to all subsequent
   transactions linked to a compromised card. Not all fraud-labelled transactions
   are individually fraudulent acts.

2. **Missing features:** `new_device_flag` (inert) and `failed_ratio_trailing10` (inert)
   carry zero signal. The effective feature set is 13/15.

3. **D11 is a degraded proxy:** 47.3% of rows have `account_age_days=0.0`. The D11
   Spearman monotonicity test returned rho=0.244 — weak evidence that it actually
   measures account age.

4. **Reference datetime is inferred:** The exact TransactionDT reference is not
   published by Vesta. Hour-of-day, is_night, and day-of-week values are
   approximate.

5. **6-month window:** The dataset covers approximately 6 months. Seasonal patterns,
   evolving fraud tactics, and long-term customer behaviour are not captured.

6. **E-commerce context:** IEEE-CIS is card-not-present e-commerce fraud (Vesta
   Corporation). MerchantShield was designed for a merchant payment context. Fraud
   patterns may differ.

7. **Prior transaction history distribution:** The external dataset has cards with
   thousands of prior transactions (mean 820 in train, 2248 in test). This creates
   very large `amount_zscore` values for rare outliers on heavily-transacted cards.
   This is a legitimate data characteristic, not a pipeline error.

8. **Cost model is assumed:** `fp_cost=50` (USD in this context) and
   `fn_cost_fraction=0.5` are the same parameter values used for the synthetic
   experiment. They are not calibrated to real e-commerce merchant economics.

---

## Running the Experiments

### Prerequisites

1. Download the IEEE-CIS dataset from Kaggle (requires account + rule acceptance):
   https://www.kaggle.com/c/ieee-fraud-detection/data

2. Extract to the local path expected by the experiment runner
   (or pass `--txn` and `--idnt` arguments to override).

3. Ensure the MerchantShield Python environment is active:
   ```bash
   pip install -r requirements.txt
   ```

### Run Experiment A (frozen model transfer)

```bash
python ml/external/ieee/experiment_a_frozen.py \
  --txn "path/to/train_transaction.csv" \
  --idnt "path/to/train_identity.csv"
```

Results written to: `ml/external/ieee/results/experiment_a_results.json`

### Run Experiment B (external retraining)

```bash
python ml/external/ieee/experiment_b_retrain.py \
  --txn "path/to/train_transaction.csv" \
  --idnt "path/to/train_identity.csv"
```

Results written to: `ml/external/ieee/results/experiment_b_results.json`  
External model saved to: `ml/external/ieee/results/ieee_lgbm_v1.pkl`

### Run the test suite

```bash
# All tests (structural tests run without data; integration tests skipped in CI)
pytest tests/test_ieee_external.py -v

# Integration tests only (requires data files)
pytest tests/test_ieee_external.py -v -k "TestIntegration"
```

---

## Interpreting Results

Results from this track **do not replace** the MerchantShield synthetic benchmark.
They provide additional evidence about generalisability.

| Scenario | Interpretation |
|---|---|
| Exp A succeeds (recall > naïve baseline, precision > prevalence) | The frozen model's learned patterns partially transfer. The 8 active features carry real signal across datasets. |
| Exp A fails (recall ≈ 0 or precision ≈ prevalence) | The frozen model is calibrated to the synthetic distribution and does not transfer. This is expected given the ~2.2× prevalence difference. |
| Exp B succeeds (LightGBM beats LR baseline on cost) | MerchantShield's feature design is transportable. The methodology works on real data. |
| Exp B fails (LightGBM cannot beat LR baseline) | The 13-feature subset is insufficient for this dataset, or the label noise / missing features prevent the methodology from working. |

A failed Experiment A plus a successful Experiment B would be the most common expected
outcome: the frozen model does not calibrate across datasets, but the *methodology*
(features + training approach) is sound.

---

## Results Location

After running the experiments:

```
ml/external/ieee/results/
  experiment_a_results.json    — Experiment A metrics + threshold sweep
  experiment_b_results.json    — Experiment B metrics + feature importances
  ieee_lgbm_v1.pkl             — External retrained model (NOT lgbm_v1)
```

These result files are excluded from Git (see `.gitignore`). The experiment
scripts that produced them are tracked.

---

## Actual Results Summary

*(Produced by running the experiments on the locally available IEEE-CIS training
data. All numbers are from the JSON result files in `ml/external/ieee/results/`.)*

### Experiment A — Frozen lgbm_v1 at threshold 0.40 (TEST SET, PRIMARY RESULT)

| Metric | Value | Interpretation |
|---|---|---|
| Precision | 0.026 | Of all transactions flagged, only 2.6% are actually fraud |
| Recall | 0.005 | Only 0.5% of fraud cases are caught (19 of 4,064) |
| F1 | 0.008 | Near-zero — model is not discriminating on this data |
| ROC-AUC | **0.443** | **Below 0.50 — the frozen model is effectively anti-predictive** |
| PR-AUC | 0.030 | Trivial; barely above the 3.4% prevalence baseline |
| FP | 705 | 705 false positives |
| FN | 4,045 | 4,045 missed fraud cases |
| Fraud caught | 19 / 4,064 | 0.47% capture rate |
| Expected cost | 332,310 units | vs. naive all-positive: 5,702,200 units |

**Experiment A.5 — Best threshold for frozen model on external val set (INFORMATIONAL ONLY):**  
No threshold in the [0.05, 0.95] sweep achieves recall ≥ 80%. At the lowest
threshold (0.05), recall is only 3.1% — the frozen model's probability outputs
are not discriminative on IEEE-CIS data regardless of threshold.

**Probability calibration finding:** The frozen model's mean predicted probability
on the external val set is 0.011, with median 0.00027. The fraction of transactions
above the 0.40 threshold is only 0.64%. This confirms the model assigns near-zero
probabilities to nearly all IEEE-CIS transactions — its synthetic-data calibration
does not transfer. Mean fraud probability (0.0095) is actually *lower* than mean
legitimate probability (0.0111), explaining the sub-0.5 ROC-AUC.

**Conclusion for Experiment A: The frozen model does NOT generalise to IEEE-CIS.**

### Experiment B — External LightGBM retrained from scratch (TEST SET, PRIMARY RESULT)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Cost (units) |
|---|---|---|---|---|---|---|---|
| Naive (all-positive) | 0.50 | 0.034 | 1.000 | 0.067 | 0.500 | 0.034 | 5,702,200 |
| Logistic Regression | 0.45 | 0.046 | 0.848 | 0.087 | 0.662 | 0.055 | 3,614,602 |
| **LightGBM (external)** | **0.35** | **0.057** | **0.804** | **0.106** | **0.753** | **0.114** | **2,763,187** |

Confusion matrix for external LightGBM at threshold 0.35 (test set):
- TP: 3,266  FP: 54,079  FN: 798  TN: 59,965
- Fraud caught: 3,266 of 4,064 (80.4%)
- Selected threshold: 0.35 (selected on val set only — test set never seen during selection)

Feature importances (top 5 by gain):
1. `amount` (1,731) — transaction amount
2. `prior_txn_count` (1,007) — behavioural history depth
3. `amount_vs_avg_ratio` (918) — amount relative to card average
4. `amount_zscore` (760) — amount deviation from card history
5. `account_age_days` (600) — D11 proxy, despite 47.3% null rate

Inert features confirmed: `new_device_flag` = 0 gain, `failed_ratio_trailing10` = 0 gain.

**Conclusion for Experiment B: The methodology PARTIALLY transfers.**
The retrained LightGBM achieves ROC-AUC 0.753 and PR-AUC 0.114 (vs. 0.034
prevalence baseline), beating both the naive and LR baselines on expected cost.
However, precision (5.7%) is very low — reflecting IEEE-CIS's fraud complexity
and the missing device/failure features.

### Comparison with synthetic MerchantShield v1

| Metric | Synthetic (lgbm_v1 at 0.40) | External B (ieee_lgbm at 0.35) |
|---|---|---|
| ROC-AUC | 0.901 (reported) | 0.753 |
| PR-AUC | 0.901 (reported) | 0.114 |
| Precision | 0.785 | 0.057 |
| Recall | 0.882 | 0.804 |
| F1 | 0.830 | 0.106 |
| Fraud prevalence | ~1.6% | ~3.4% |
| Active features | 15/15 | 13/15 (2 inert) |

The performance gap is substantial but expected for three reasons:
1. The synthetic pipeline had 15 active features including device novelty and
   failure signals; the external experiment has only 13 effective features.
2. IEEE-CIS fraud (e-commerce chargebacks, label propagation) is a harder and
   noisier classification problem than the MerchantShield synthetic generator.
3. Precision is especially low because IEEE-CIS at 3.4% prevalence with a broad
   feature base has many ambiguous legitimate transactions near the decision boundary.

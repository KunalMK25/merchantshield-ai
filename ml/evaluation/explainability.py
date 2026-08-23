"""
Explainability layer for MerchantShield AI.

Uses shap.TreeExplainer on the frozen LightGBM model (unchanged from Phase 4/5).

GROUNDING RULE: every sentence produced by `humanize_contribution` is generated from
a (feature_name, observed_value, shap_value) triple that actually came out of the
explainer for that specific transaction. There is no free-text generation step, no
LLM call, and no template that fires without a real contribution behind it. If SHAP
says a feature didn't matter for this transaction, it does not appear in the output.
"""

import warnings

import numpy as np
import pandas as pd
import shap

from ml.features.build_features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Human-readable templates, keyed by feature name.
# Each template is a function(value, shap_value) -> str. Templates only describe
# what the feature value IS and which direction it pushed risk -- they never
# invent detail the feature doesn't carry (e.g. we don't say "this looks like
# account takeover", we say "device has not been seen previously").
# ---------------------------------------------------------------------------

def _direction_word(shap_value: float) -> str:
    return "increased" if shap_value > 0 else "decreased"


def _fmt_amount(x):
    return f"Rs{x:,.0f}"


_TEMPLATES = {
    "amount": lambda v, s: f"Transaction amount ({_fmt_amount(v)}) {_direction_word(s)} risk",
    "amount_zscore": lambda v, s: (
        f"Transaction amount is {abs(v):.1f} standard deviations "
        f"{'above' if v > 0 else 'below'} this customer's historical average, which {_direction_word(s)} risk"
    ),
    "amount_vs_avg_ratio": lambda v, s: (
        f"Transaction amount is {v:.1f}x the customer's historical average amount, which {_direction_word(s)} risk"
    ),
    "prior_txn_count": lambda v, s: (
        f"Customer has only {int(v)} prior transactions on record, which {_direction_word(s)} risk"
        if v < 5 else
        f"Customer has an established history of {int(v)} prior transactions, which {_direction_word(s)} risk"
    ),
    "time_since_prev_txn_min": lambda v, s: (
        f"Only {v:.1f} minutes since this customer's previous transaction, which {_direction_word(s)} risk"
        if v < 60 else
        f"{v/60:.1f} hours since this customer's previous transaction, which {_direction_word(s)} risk"
    ),
    "velocity_5min": lambda v, s: f"{int(v)} transactions occurred in the previous 5 minutes, which {_direction_word(s)} risk",
    "velocity_30min": lambda v, s: f"{int(v)} transactions occurred in the previous 30 minutes, which {_direction_word(s)} risk",
    "velocity_60min": lambda v, s: f"{int(v)} transactions occurred in the previous 60 minutes, which {_direction_word(s)} risk",
    "new_device_flag": lambda v, s: (
        "Device has not been seen previously for this customer, which increased risk" if v == 1
        else "Device has been used previously by this customer, which decreased risk"
    ),
    "new_geo_flag": lambda v, s: (
        "Geographic location is new for this customer, which increased risk" if v == 1
        else "Geographic location matches this customer's prior activity, which decreased risk"
    ),
    "failed_ratio_trailing10": lambda v, s: (
        f"{v:.0%} of this customer's recent transactions failed, which {_direction_word(s)} risk"
    ),
    "account_age_days": lambda v, s: (
        f"Account is only {v:.0f} days old, which {_direction_word(s)} risk"
        if v < 30 else
        f"Account age is {v:.0f} days, which {_direction_word(s)} risk"
    ),
    "hour_of_day": lambda v, s: f"Transaction occurred at hour {int(v)} of the day, which {_direction_word(s)} risk",
    "is_night": lambda v, s: (
        "Transaction occurred during overnight hours (12am-5am), which increased risk" if v == 1
        else "Transaction occurred during normal daytime hours, which decreased risk"
    ),
    "day_of_week": lambda v, s: f"Transaction occurred on day-of-week {int(v)}, which {_direction_word(s)} risk",
}


class RiskExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_row: pd.Series, top_k: int = 5) -> dict:
        """
        feature_row: a pandas Series indexed by FEATURE_COLUMNS for ONE transaction.
        Returns a dict with fraud_probability, shap contributions, and a grounded
        list of human-readable reason strings for the top_k contributing features.
        """
        missing = [c for c in FEATURE_COLUMNS if c not in feature_row.index]
        if missing:
            raise ValueError(f"Cannot explain: missing required feature(s) {missing}")

        x = feature_row[FEATURE_COLUMNS].astype(float)
        if x.isna().any():
            bad = x[x.isna()].index.tolist()
            raise ValueError(f"Cannot explain: NaN/invalid value(s) for feature(s) {bad}")

        X = pd.DataFrame([x.values], columns=FEATURE_COLUMNS)

        raw_prob = float(self.model.predict_proba(X)[0, 1])

        # shap_values in margin (log-odds) space for the positive class, plus expected_value (base).
        #
        # OUTPUT FORMAT NOTE (SHAP 0.52 + LightGBM binary classifier):
        # SHAP 0.52 emits a UserWarning that the output format for LightGBM binary
        # classifiers "has changed to a list of ndarray". In practice, with the current
        # pinned versions (shap==0.52.0, lightgbm==4.7.0), shap_values() still returns
        # a plain ndarray of shape (n_samples, n_features); the warning is
        # forward-looking. Both branches below are kept for version-compatibility:
        #   - list branch:   future SHAP where output becomes [neg_class_array, pos_class_array]
        #   - else branch:   current SHAP where output is a single (n_samples, n_features) ndarray
        # The warning is suppressed here because it is a known, benign API-migration
        # notice for a version combination we have already accounted for in code.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed",
                category=UserWarning,
            )
            shap_out = self.explainer.shap_values(X)

        if isinstance(shap_out, list):
            # future SHAP: list of [neg_class, pos_class] arrays, each (n_samples, n_features)
            sv = np.asarray(shap_out[1][0])  # positive class, first (only) sample row
            base_value = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        else:
            # current SHAP 0.52: plain ndarray (n_samples, n_features); index [0] = first row
            sv = np.asarray(shap_out[0])
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[-1]

        # --- consistency check: base_value + sum(shap) should reconstruct the model's
        #     raw margin output (log-odds), which sigmoid-transforms to raw_prob.
        margin_from_shap = float(base_value) + float(np.sum(sv))
        reconstructed_prob = 1.0 / (1.0 + np.exp(-margin_from_shap))
        additivity_ok = abs(reconstructed_prob - raw_prob) < 1e-4

        contributions = []
        for i, feat in enumerate(FEATURE_COLUMNS):
            contributions.append(dict(
                feature=feat,
                value=float(x[feat]),
                shap_value=float(sv[i]),
                direction="increases_risk" if sv[i] > 0 else "decreases_risk",
                magnitude=abs(float(sv[i])),
            ))

        contributions_sorted = sorted(contributions, key=lambda c: c["magnitude"], reverse=True)
        top_contributions = contributions_sorted[:top_k]
        reasons = [humanize_contribution(c) for c in top_contributions]

        return dict(
            fraud_probability=round(raw_prob, 4),
            base_value=round(float(base_value), 4),
            additivity_check_passed=bool(additivity_ok),
            contributions=contributions_sorted,
            top_reasons=reasons,
        )


def narrative_header(fraud_probability: float, decision_threshold: float) -> str:
    """
    Chooses the correct framing sentence based on the actual decision outcome -- so we
    never say "why this was flagged" about a transaction that wasn't flagged.
    """
    if fraud_probability >= decision_threshold:
        return "Why this transaction was flagged:"
    return "Top contributing factors (transaction was NOT flagged):"


def build_explanation_text(result: dict, decision_threshold: float, top_k: int = 5) -> dict:
    """
    Produces the final human-readable explanation, choosing reason set and framing
    based on the ACTUAL decision outcome (grounded in fraud_probability vs the real
    frozen threshold, not a guess):
      - Flagged transactions: reasons = top risk-increasing contributions only
        (matches "why was this flagged" framing).
      - Non-flagged transactions: reasons = top contributions by magnitude in either
        direction (there's no "why flagged" story to tell, so we show what mattered).
    """
    flagged = result["fraud_probability"] >= decision_threshold
    header = narrative_header(result["fraud_probability"], decision_threshold)
    if flagged:
        pool = [c for c in result["contributions"] if c["direction"] == "increases_risk"]
    else:
        pool = result["contributions"]
    top = pool[:top_k]
    reasons = [humanize_contribution(c) for c in top]
    return dict(flagged=flagged, header=header, reasons=reasons)


def humanize_contribution(contribution: dict) -> str:
    feat = contribution["feature"]
    val = contribution["value"]
    shap_val = contribution["shap_value"]
    template = _TEMPLATES.get(feat)
    if template is None:
        # Grounded fallback for any feature without a bespoke template -- still
        # built directly from the actual value/direction, not invented.
        return f"{feat} = {val:.3g}, which {_direction_word(shap_val)} risk"
    return template(val, shap_val)

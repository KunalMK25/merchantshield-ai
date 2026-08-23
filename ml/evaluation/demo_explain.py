import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import joblib

from ml.features.build_features import FEATURE_COLUMNS
from ml.training.train_baseline import chronological_split
from ml.evaluation.risk_scoring import score_transaction
from ml.evaluation.explainability import RiskExplainer, build_explanation_text

FROZEN_THRESHOLD = 0.40  # from Phase 5, unchanged


def main():
    feats = pd.read_csv("ml/data/features.csv")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)

    model = joblib.load("ml/models/candidate_lgbm_v1.pkl")
    explainer = RiskExplainer(model)

    # pick a few illustrative examples from VALIDATION (not test) for this demo --
    # this is a demonstration of the explainability mechanism, not a model evaluation,
    # so touching validation rows here has no bearing on model selection/threshold.
    sample = val.sort_values("amount_zscore", ascending=False).head(2)   # likely-flagged examples
    sample = pd.concat([sample, val[val["is_fraud"] == 0].sample(1, random_state=3)])  # one legit example

    for _, row in sample.iterrows():
        result = explainer.explain(row)
        risk = score_transaction(result["fraud_probability"])
        explanation = build_explanation_text(result, FROZEN_THRESHOLD)
        decision = "FLAG (above 0.40 threshold)" if explanation["flagged"] else "ALLOW (below 0.40 threshold)"

        print("=" * 90)
        print(f"Transaction: {row['transaction_id']}   (actual label: {'FRAUD' if row['is_fraud'] == 1 else 'legit'})")
        print(f"Fraud probability: {risk['fraud_probability']}")
        print(f"Risk score: {risk['risk_score']}/100   Risk category: {risk['risk_category']}")
        print(f"Decision-threshold outcome: {decision}")
        print(f"SHAP additivity check passed: {result['additivity_check_passed']}")
        print(explanation["header"])
        for r in explanation["reasons"]:
            print(f"  - {r}")
        print()


if __name__ == "__main__":
    main()

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import joblib

from ml.features.build_features import FEATURE_COLUMNS
from ml.training.train_baseline import chronological_split
from ml.evaluation.explainability import RiskExplainer, build_explanation_text
from ml.evaluation.decision_engine import make_decision


def main():
    feats = pd.read_csv("ml/data/features.csv")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)

    model = joblib.load("ml/models/candidate_lgbm_v1.pkl")
    explainer = RiskExplainer(model)

    # one real example per category, drawn from validation data
    examples = [
        val[val["is_fraud"] == 0].sample(1, random_state=1).iloc[0],                      # likely LOW
        val.sort_values("failed_ratio_trailing10", ascending=False).iloc[50],              # likely MEDIUM
        val.sort_values("velocity_5min", ascending=False).iloc[30],                        # likely HIGH
        val.sort_values("amount_zscore", ascending=False).iloc[0],                         # likely CRITICAL
    ]

    for row in examples:
        result = explainer.explain(row)
        explanation = build_explanation_text(result, decision_threshold=0.40)
        decision = make_decision(
            transaction_id=row["transaction_id"],
            model_probability=result["fraud_probability"],
            amount=float(row["amount"]),
            model_explanation=explanation["reasons"],
            timestamp="2026-08-21T12:00:00Z",
        )
        print("=" * 90)
        print(json.dumps(decision.to_dict(), indent=2))
        print()


if __name__ == "__main__":
    main()

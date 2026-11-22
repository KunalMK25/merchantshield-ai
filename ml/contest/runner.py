"""
Runs the model contest: train every candidate on the training split, pick each
one's threshold on validation, and report validation (selection) and test
(reporting only) metrics. Test data is optional and never influences anything.
"""

import time

from ml.contest.candidates import CANDIDATES
from ml.contest.selection import evaluate_at, pick_threshold
from ml.evaluation.cost_model import CostAssumptions
from ml.features.build_features import FEATURE_COLUMNS


def run_contest(train, val, test=None, names=None, cost: CostAssumptions = None,
                min_recall: float = 0.8, seed: int = 42):
    cost = cost or CostAssumptions()
    names = list(names or CANDIDATES)
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidate(s): {unknown}")

    X, y = train[FEATURE_COLUMNS], train["is_fraud"].values
    Xv, yv, av = val[FEATURE_COLUMNS], val["is_fraud"].values, val["amount"].values

    entries = []
    for name in names:
        fit, tree_based = CANDIDATES[name]
        t0 = time.time()
        model = fit(X, y, Xv, yv, seed)
        train_seconds = time.time() - t0

        threshold, val_metrics = pick_threshold(yv, model.predict_proba(Xv)[:, 1], av, cost, min_recall)
        test_metrics = None
        if test is not None:
            test_prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            test_metrics = evaluate_at(test["is_fraud"].values, test_prob, test["amount"].values,
                                       cost, threshold)
        entries.append({
            "name": name, "tree_based": tree_based, "model": model,
            "train_seconds": train_seconds, "val": val_metrics, "test": test_metrics,
        })
    return entries

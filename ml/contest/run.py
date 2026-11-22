"""
Run the model contest on the project's feature data and update the registry.

    python -m ml.contest.run                 # all candidates, then try to promote the winner
    python -m ml.contest.run --models lightgbm xgboost
    python -m ml.contest.run --rollback      # restore the previous champion

The first run also imports the original lgbm_v1 as the starting champion, so a
challenger has to beat the model that is already serving.
"""

import argparse
import json
import os

import pandas as pd

from ml.contest.runner import run_contest
from ml.contest.selection import select_winner
from ml.evaluation.cost_model import CostAssumptions
from ml.registry.store import ModelRegistry, maybe_promote
from ml.training.train_baseline import chronological_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REGISTRY = os.path.join(ROOT, "ml", "registry_store")
FEATURES = os.path.join(ROOT, "ml", "data", "features.csv")
LGBM_V1 = os.path.join(ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
LGBM_V1_META = os.path.join(ROOT, "ml", "models", "lgbm_v1_metadata.json")


def bootstrap_champion(registry: ModelRegistry):
    """Seed the registry with the model that is already in production."""
    if registry.champion() is not None:
        return
    with open(LGBM_V1_META) as f:
        meta = json.load(f)
    metrics = dict(meta["validation_metrics_at_threshold"])
    metrics["cost"] = metrics.pop("total_cost")
    run_id = registry.import_existing(LGBM_V1, "lightgbm", True, meta["selected_threshold"], metrics,
                                      note="Original frozen model imported as the starting champion",
                                      metadata=meta)
    registry.promote(run_id, "lightgbm", reason="Original frozen model lgbm_v1 (starting champion).")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of candidates (default: all)")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--margin", type=float, default=0.03, help="required relative cost improvement")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args(argv)

    registry = ModelRegistry(args.registry)
    if args.rollback:
        print("Rolled back to:", json.dumps(registry.rollback(), indent=2))
        return

    bootstrap_champion(registry)

    feats = pd.read_csv(FEATURES)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}\n")

    cost = CostAssumptions()
    entries = run_contest(train, val, test, names=args.models, cost=cost)
    winner = select_winner(entries)

    print(f"{'model':22s}{'thr':>5}{'prec':>7}{'recall':>8}{'val cost':>11}{'test cost':>11}{'secs':>7}")
    for e in sorted(entries, key=lambda e: e["val"]["cost"]):
        star = " *" if winner and e["name"] == winner["name"] else ""
        print(f"{e['name']:22s}{e['val']['threshold']:>5.2f}{e['val']['precision']:>7.3f}"
              f"{e['val']['recall']:>8.3f}{e['val']['cost']:>11,.0f}{e['test']['cost']:>11,.0f}"
              f"{e['train_seconds']:>7.1f}{star}")

    run_id = registry.register_run(entries, data_note=f"ml/data/features.csv, {len(feats):,} rows")
    decision = maybe_promote(registry, run_id, val, cost, margin=args.margin)
    print(f"\nrun {run_id}: {decision['reason']}")
    print("champion:", registry.champion()["model_name"], "from", registry.champion()["run_id"])


if __name__ == "__main__":
    main()

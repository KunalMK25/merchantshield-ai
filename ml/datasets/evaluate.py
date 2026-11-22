"""
How do the models, and the account anomaly detector, do on a prepared dataset?

    python -m ml.datasets.evaluate sparkov

Runs the same contest as the daily job on a chronological split of the dataset's real labels
(history plus the held-back replay days), then scores the account detector on the most recent days
it never trained on. Writes ml/workspaces/<name>/evaluation.json. Costs are assumptions scaled to the
dataset's amounts; the printed numbers describe THIS dataset only.
"""

import argparse
import json

import pandas as pd

from ml.contest.runner import run_contest
from ml.contest.selection import select_winner
from ml.datasets.schema import profile
from ml.datasets.workspace import DEFAULT_ROOT, Workspace
from ml.features.build_features import FEATURE_COLUMNS, build_features
from ml.monitoring.anomaly import SEVERITIES, AnomalyConfig, AnomalyDetector, account_day_table
from ml.monitoring.supervised_alerts import account_day_scores, calibrate_thresholds, detect_supervised
from ml.registry.store import _clean
from ml.retrain.job import RetrainConfig, rolling_split


def _tier(flags, truth, sel):
    s = flags[flags["severity"].isin(sel)] if not flags.empty else flags
    hits = set(map(tuple, s[["customer_id", "day"]].to_numpy())) & truth if len(s) else set()
    return {"flagged": int(len(s)), "precision": len(hits) / len(s) if len(s) else None,
            "recall": len(hits) / len(truth) if truth else None}


def _anomaly(table, test_start_day, config, label):
    reference, current = table[table["day"] < test_start_day], table[table["day"] >= test_start_day]
    try:
        detector = AnomalyDetector(config=config).fit(reference)
    except ValueError as e:
        return {"config": label, "error": str(e)}
    flags = detector.detect(current)
    truth = set(map(tuple, current.loc[current["fraud"] == 1, ["customer_id", "day"]].to_numpy()))
    return {"config": label, "accounts_checked": int(len(current)), "compromised_account_days": len(truth),
            "tiers": {"critical": _tier(flags, truth, ["critical"]),
                      "critical_plus_high": _tier(flags, truth, ["critical", "high"]),
                      "everything_flagged": _tier(flags, truth, list(SEVERITIES))}}


def _supervised(winner, val, test, origin) -> dict:
    """Account alerts from the winning model's scores, thresholds chosen on validation, test scored once."""
    label = {"config": "supervised"}
    if winner is None:
        return {**label, "error": "no model met the recall floor, so there is no supervised model to score with"}
    model = winner["model"]
    pv = account_day_scores(val, model.predict_proba(val[FEATURE_COLUMNS])[:, 1], origin)
    pt = account_day_scores(test, model.predict_proba(test[FEATURE_COLUMNS])[:, 1], origin)
    try:
        cal = calibrate_thresholds(pv)
    except ValueError as e:
        return {**label, "error": str(e)}
    flags = detect_supervised(pt, cal["critical"], cal["high"])
    truth = set(map(tuple, pt.loc[pt["fraud"] == 1, ["customer_id", "day"]].to_numpy()))
    chosen = next(g for g in cal["grid"] if g["critical"] == cal["critical"] and g["high"] == cal["high"])
    return {**label, "model": winner["name"], "thresholds": {"critical": cal["critical"], "high": cal["high"]},
            "validation": {"critical": chosen["critical_tier"], "critical_plus_high": chosen["critical_plus_high"]},
            "accounts_checked": int(len(pt)), "compromised_account_days": len(truth),
            "tiers": {"critical": _tier(flags, truth, ["critical"]),
                      "critical_plus_high": _tier(flags, truth, ["critical", "high"])},
            "calibrated_on": "validation days only; the test days were scored once"}


def evaluate_workspace(ws: Workspace, models=None, val_days=None, test_days=None) -> dict:
    cfg = RetrainConfig.from_workspace(ws, models=models)
    parts = [pd.read_csv(ws.raw_path)] + ([pd.read_csv(ws.stream_path)] if ws.stream_path.exists() else [])
    raw = pd.concat(parts, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    prof = profile(raw)

    default = max(3, prof["days"] // 10)
    val_days, test_days = val_days or default, test_days or default

    calibration = ws.anomaly_calibration()
    if calibration and prof["days"] - test_days < calibration["test_start"]:
        raise ValueError(
            f"The test window starts at day {prof['days'] - test_days}, but the detector was calibrated on days up to "
            f"{calibration['test_start'] - 1}. Those windows overlap, so the test score would not be honest. Use a test "
            f"window of at most {prof['days'] - calibration['test_start']} days, or recalibrate with the same --test-days.")
    feats = build_features(raw)
    train, val, test = rolling_split(feats, val_days, test_days)
    entries = run_contest(train, val, test, names=cfg.models, cost=cfg.cost, seed=cfg.seed)
    winner = select_winner(entries)

    test_start = prof["days"] - test_days
    table = account_day_table(feats)
    meta = json.loads(ws.config_path.read_text())
    report = {
        "dataset": meta["dataset"], "title": meta["title"], "fit": prof["fit"], "fit_reason": prof["fit_reason"],
        "inert_features": prof["inert_features"], "rows": prof["rows"], "customers": prof["customers"],
        "fraud_rate": prof["fraud_rate"], "days": prof["days"], "val_days": val_days, "test_days": test_days,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "cost": {"fp_cost": cfg.cost.fp_cost, "fn_cost_fraction": cfg.cost.fn_cost_fraction},
        "cost_note": meta["cost_note"],
        "contest": [{"name": e["name"], "tree_based": e["tree_based"], "val": e["val"], "test": e["test"],
                     "train_seconds": e["train_seconds"]} for e in entries],
        "winner": winner["name"] if winner else None,
    }
    report["anomaly_supervised"] = _supervised(
        winner, val, test, pd.to_datetime(feats["timestamp"]).min().normalize())
    default_anomaly = _anomaly(table, test_start, AnomalyConfig(), "default")
    if calibration:
        cfg = AnomalyConfig(quantile=calibration["quantile"], iso_quantile=calibration["iso_quantile"])
        report["anomaly"] = _anomaly(table, test_start, cfg, "calibrated")
        report["anomaly_default"] = default_anomaly
        report["calibration"] = {k: calibration[k] for k in
                                 ("quantile", "iso_quantile", "score", "objective", "calibrated_on", "validation_compromised")}
    else:
        report["anomaly"] = default_anomaly
    (ws.path / "evaluation.json").write_text(json.dumps(_clean(report), indent=2))
    return report


def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--workspace-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--val-days", type=int)
    ap.add_argument("--test-days", type=int)
    a = ap.parse_args(argv)
    ws = Workspace(a.dataset, a.workspace_root)
    try:
        r = evaluate_workspace(ws, a.models, a.val_days, a.test_days)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}")
        raise SystemExit(2)

    print(f"{r['title']}\n{r['rows']:,} transactions, {r['customers']:,} customers, {r['days']} days, "
          f"{r['fraud_rate']:.2%} fraud. Fit: {r['fit']}.")
    if r["inert_features"]:
        print(f"Inert features (no signal here): {', '.join(r['inert_features'])}")
    print(f"Split: train {r['split']['train']:,} / validation {r['split']['val']:,} ({r['val_days']} days) / "
          f"test {r['split']['test']:,} ({r['test_days']} days). {r['cost_note']}\n")
    print(f"{'model':22s}{'thr':>5}{'precision':>11}{'recall':>8}{'test cost':>14}")
    for m in sorted(r["contest"], key=lambda m: m["val"]["cost"]):
        star = " *" if m["name"] == r["winner"] else ""
        print(f"{m['name']:22s}{m['val']['threshold']:>5.2f}{m['test']['precision']:>11.1%}"
              f"{m['test']['recall']:>8.1%}{m['test']['cost']:>14,.0f}{star}")
    for key, title in (("anomaly_default", "default thresholds"), ("anomaly", None), ("anomaly_supervised", None)):
        an = r.get(key)
        if an is None:
            continue
        if title is None and an["config"] == "supervised" and "thresholds" in an:
            t = an["thresholds"]
            title = f"supervised alerts ({an['model']}; critical >= {t['critical']}, high >= {t['high']}, chosen on validation)"
        title = title or ("calibrated thresholds" if an["config"] == "calibrated" else "default thresholds")
        if "error" in an:
            print(f"\nAccount detector ({title}): not run ({an['error']})")
            continue
        print(f"\nAccount detector, {title}, on the last {r['test_days']} days ({an['compromised_account_days']} compromised "
              f"account-days among {an['accounts_checked']:,}):")
        for name, t in an["tiers"].items():
            print(f"  {name:20s} flagged {t['flagged']:>5}  precision {_pct(t['precision']):>5}  recall {_pct(t['recall']):>5}")
    if "calibration" in r:
        print(f"  (calibrated on {r['calibration']['calibrated_on']}: quantile {r['calibration']['quantile']}, "
              f"forest quantile {r['calibration']['iso_quantile']})")
    print(f"\nSaved {ws.path / 'evaluation.json'}")


if __name__ == "__main__":
    main()

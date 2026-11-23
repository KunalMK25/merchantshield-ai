"""
Calibrate the account anomaly detector to a prepared dataset.

    python -m ml.datasets.calibrate sparkov

The detector's thresholds were tuned by hand on synthetic data, and on real data they flagged far too many
ordinary customers. This chooses, for one dataset, the quantile that sets its ratio thresholds and the
forest's flag threshold, by scoring a grid on a VALIDATION window of days.

The most recent days (the test window, the same one `evaluate` scores) are never read here: the window
split is computed first and only earlier days are passed to `calibrate`. The result, with the days it was
chosen on, is saved to ml/workspaces/<name>/anomaly_config.json; `evaluate` and the daily job pick it up,
and `evaluate` refuses a test window that overlaps the days used for tuning.
"""

import argparse
import json

import pandas as pd

from ml.datasets.schema import check_windows, profile
from ml.datasets.workspace import DEFAULT_ROOT, Workspace
from ml.features.build_features import build_features
from ml.monitoring.anomaly import account_day_table, calibrate
from ml.registry.store import _clean


def calibrate_workspace(ws: Workspace, val_days=None, test_days=None) -> dict:
    parts = [pd.read_csv(ws.raw_path)] + ([pd.read_csv(ws.stream_path)] if ws.stream_path.exists() else [])
    raw = pd.concat(parts, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    prof = profile(raw)

    default = max(3, prof["days"] // 10)
    val_days, test_days = val_days or default, test_days or default
    check_windows(prof["days"], val_days, test_days)
    test_start = prof["days"] - test_days
    val_start = test_start - val_days

    table = account_day_table(build_features(raw))
    reference = table[table["day"] < val_start]
    validation = table[(table["day"] >= val_start) & (table["day"] < test_start)]   # test days are filtered out here
    result = calibrate(reference, validation)
    result.update(dataset=json.loads(ws.config_path.read_text())["dataset"], val_start=val_start,
                  test_start=test_start, val_days=val_days, test_days=test_days,
                  calibrated_on=f"days {val_start} to {test_start - 1} (validation); days {test_start}+ not used")
    result = _clean(result)
    ws.anomaly_config_path.write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--workspace-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--val-days", type=int)
    ap.add_argument("--test-days", type=int)
    a = ap.parse_args(argv)
    ws = Workspace(a.dataset, a.workspace_root)
    try:
        ws.config()
        r = calibrate_workspace(ws, a.val_days, a.test_days)
    except FileNotFoundError as e:
        print(f"error: {e}")
        raise SystemExit(2)
    except ValueError as e:
        print(f"error: {e}")
        raise SystemExit(2)

    print(f"{r['dataset']}: calibrated on {r['calibrated_on']}")
    print(f"validation: {r['validation_account_days']:,} account-days, {r['validation_compromised']} compromised\n")
    print(f"{'quantile':>9}{'forest q':>10}{'critical':>22}{'critical+high':>34}")
    for g in sorted(r["grid"], key=lambda g: -g["score"])[:6]:
        c, h = g["critical"], g["critical_plus_high"]
        print(f"{g['quantile']:>9}{g['iso_quantile']:>10}   n={c['flagged']:<5} p={c['precision']:.0%} r={c['recall']:.0%}"
              f"      n={h['flagged']:<5} p={h['precision']:.0%} r={h['recall']:.0%}   score={g['score']:.2f}")
    print(f"\nchosen: quantile {r['quantile']}, forest quantile {r['iso_quantile']} ({r['objective']}: {r['score']:.2f})")
    print(f"Saved {ws.anomaly_config_path}")


if __name__ == "__main__":
    main()

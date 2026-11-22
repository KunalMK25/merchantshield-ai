"""
Turn a downloaded dataset into a workspace the rest of the system can run on.

    python -m ml.datasets.prepare sparkov --path data/raw/sparkov
    python -m ml.datasets.prepare tabformer --path data/raw/tabformer --max-customers 300
    python -m ml.datasets.catalog          # what is available and where to get it

It loads and validates the data, profiles it (and refuses a dataset that cannot support the
behavioural features unless --force), then writes, under ml/workspaces/<name>/:

    raw_transactions.csv   the history the models learn from
    stream.csv             the last few days, held back so the daily job can replay them as "new" real days
    config.json            cost assumptions scaled to this dataset, and the source
    profile.json, notes.json
"""

import argparse
import json

import pandas as pd

from ml.datasets.catalog import DATASETS, NOT_INTEGRATED, find_files
from ml.datasets.schema import DatasetError, profile, validate_raw
from ml.datasets.workspace import DEFAULT_ROOT, Workspace

FP_COST_SHARE_OF_MEDIAN = 0.08   # the original study: a flat 50 against a median amount of about 624
FN_COST_FRACTION = 0.5           # unchanged from the original study's assumption


def prepare(name, path, workspace_root=DEFAULT_ROOT, max_customers=None, stream_days=30, force=False, seed=0) -> dict:
    files = find_files(name, path)
    raw, notes = DATASETS[name].loader(files, max_customers, seed)
    validate_raw(raw)
    prof = profile(raw)
    if prof["fit"] == "poor" and not force:
        raise DatasetError(f"{name} is a poor fit: {prof['fit_reason']} Use --force to prepare it anyway, "
                           f"knowing the models will learn little beyond the amount.")

    ts = pd.to_datetime(raw["timestamp"])
    day0 = ts.min().normalize()
    day = (ts.dt.normalize() - day0).dt.days
    span = int(day.max()) + 1
    stream_days = max(0, min(int(stream_days), span // 3))       # never hold back more than a third of the span
    cut = span - stream_days
    history, stream = raw[day < cut], raw[day >= cut]

    ws = Workspace(name, workspace_root)
    ws.path.mkdir(parents=True, exist_ok=True)
    history.to_csv(ws.raw_path, index=False)
    stream.to_csv(ws.stream_path, index=False) if len(stream) else ws.stream_path.unlink(missing_ok=True)

    config = {
        "dataset": name, "title": DATASETS[name].title, "source": DATASETS[name].source,
        "currency": notes.get("currency"), "fp_cost": round(FP_COST_SHARE_OF_MEDIAN * float(history["amount"].median()), 2),
        "fn_cost_fraction": FN_COST_FRACTION, "day0": day0.isoformat(), "stream_days": stream_days,
        "cost_note": "Assumed, not measured: false-positive cost is 8% of the median amount, as in the original study.",
    }
    ws.config_path.write_text(json.dumps(config, indent=2))
    (ws.path / "profile.json").write_text(json.dumps(prof, indent=2))
    (ws.path / "notes.json").write_text(json.dumps(notes, indent=2, default=str))
    return {"workspace": str(ws.path), "dataset": name, "history_rows": len(history), "stream_rows": len(stream),
            "stream_days": stream_days, "fit": prof["fit"], "fit_reason": prof["fit_reason"],
            "inert_features": prof["inert_features"], "fp_cost": config["fp_cost"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="sparkov, tabformer, paysim or ieee")
    ap.add_argument("--path", required=True, help="folder (or file) holding the downloaded files")
    ap.add_argument("--max-customers", type=int, default=None, help="sample whole customers (needed for TabFormer)")
    ap.add_argument("--stream-days", type=int, default=30, help="final days held back to replay as new daily data")
    ap.add_argument("--force", action="store_true", help="prepare even a poor-fit dataset")
    ap.add_argument("--workspace-root", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    try:
        out = prepare(a.dataset, a.path, a.workspace_root, a.max_customers, a.stream_days, a.force)
    except DatasetError as e:
        print(f"error: {e}")
        raise SystemExit(2)
    print(f"{out['dataset']}: {out['history_rows']:,} history rows + {out['stream_rows']:,} replay rows "
          f"({out['stream_days']} days) -> {out['workspace']}")
    print(f"fit: {out['fit']} - {out['fit_reason']}")
    if out["inert_features"]:
        print(f"inert features (no signal in this data): {', '.join(out['inert_features'])}")


if __name__ == "__main__":
    main()

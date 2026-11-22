"""
The daily retraining job.

One run does, in order:
  1. generate the next simulated day(s) of transactions, mirroring the demo data
  2. rebuild features over history + all generated days
  3. run the model contest on a rolling chronological split (latest days held out)
  4. register the run and promote the winner only if it clearly beats the champion

Run it from cron, or let the in-app scheduler call it:

    python -m ml.retrain.job                    # one simulated day, then retrain
    python -m ml.retrain.job --days 3
    python -m ml.retrain.job --drift amount_scale=2.5   # shift the world on purpose

A file lock stops two runs overlapping (cron plus the in-app scheduler, say).
Every run, successful or not, appends one line to retrain_log.jsonl.
"""

import argparse
import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from ml.cases.store import CaseStore
from ml.contest.candidates import CANDIDATES
from ml.contest.run import bootstrap_champion
from ml.contest.runner import run_contest
from ml.contest.selection import select_winner
from ml.data.daily_batch import DailyStore, DriftConfig, fit_population, simulate_day, warm_up
from ml.evaluation.cost_model import CostAssumptions
from ml.features.build_features import FEATURE_COLUMNS, build_features
from ml.monitoring.anomaly import AnomalyDetector, account_day_table, summarize, to_records
from ml.monitoring.drift import drift_report
from ml.registry.store import ModelRegistry, _clean, maybe_promote
from ml.training.train_baseline import chronological_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class JobAlreadyRunning(RuntimeError):
    pass


@dataclass
class RetrainConfig:
    base_raw_path: str = os.path.join(ROOT, "ml", "data", "raw_transactions.csv")
    data_dir: str = os.path.join(ROOT, "ml", "data_store")
    registry_dir: str = os.path.join(ROOT, "ml", "registry_store")
    models: Optional[list] = None      # default: every candidate
    days: int = 1                      # simulated days to add per run
    margin: float = 0.03               # required relative cost improvement to replace the champion
    val_days: int = 10
    test_days: int = 10
    seed: int = 42
    drift: Optional[DriftConfig] = None
    bootstrap: bool = True             # seed the registry with the original lgbm_v1 champion


def rolling_split(feats: pd.DataFrame, val_days: int, test_days: int):
    """
    Chronological split where the latest days are held out. With the original 60 days and
    10/10 this reproduces the project's train<40, val 40-49, test>=50 boundaries exactly.
    """
    ts = pd.to_datetime(feats["timestamp"])
    n_days = int((ts.max().normalize() - ts.min().normalize()).days) + 1
    val_end = n_days - test_days
    return chronological_split(feats, train_end_day=val_end - val_days, val_end_day=val_end)


@contextmanager
def _exclusive(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    handle = open(os.path.join(data_dir, "job.lock"), "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise JobAlreadyRunning("Another retraining run is in progress.")
        yield
    finally:
        handle.close()  # closing releases the lock


def _log(data_dir: str, entry: dict):
    with open(os.path.join(data_dir, "retrain_log.jsonl"), "a") as f:
        f.write(json.dumps(_clean(entry)) + "\n")


RECENT_DAYS = 14        # trailing window used as the "recent" drift reference
BASELINE_DAYS = 40      # oldest history used as the slow-creep drift reference
ANOMALY_REFERENCE_DAYS = 30


def _monitor(feats, sim_days, registry, data_dir, job_id) -> dict:
    """
    Drift and account-anomaly checks on the newly generated days. Runs BEFORE promotion so
    the scores it watches come from the model that was actually serving. Writes the full
    detail to drift_log.jsonl and one anomalies/day_XXXX.json per day; returns a short summary.
    """
    ts = pd.to_datetime(feats["timestamp"])
    origin = ts.min().normalize()
    day = (ts.dt.normalize() - origin).dt.days
    first = min(sim_days)

    cur = feats[day.isin(sim_days)]
    recent = feats[(day >= first - RECENT_DAYS) & (day < first)]
    baseline = feats[day < min(BASELINE_DAYS, first)]

    score_ref = score_cur = None
    champ = registry.champion()
    if champ is not None:
        model = registry.load_model(champ["run_id"], champ["model_name"])
        score_ref = model.predict_proba(recent[FEATURE_COLUMNS])[:, 1]
        score_cur = model.predict_proba(cur[FEATURE_COLUMNS])[:, 1]
    drift = drift_report(recent, baseline, cur, score_ref, score_cur)

    table = account_day_table(feats, origin=origin)
    reference = table[(table["day"] >= first - ANOMALY_REFERENCE_DAYS) & (table["day"] < first)]
    detector = AnomalyDetector().fit(reference)
    current = table[table["day"].isin(sim_days)]
    flags = detector.detect(current)
    summary = summarize(flags, n_accounts=len(current), reference_rate=detector.reference_rate)

    os.makedirs(os.path.join(data_dir, "anomalies"), exist_ok=True)
    for d in sim_days:
        day_flags = flags[flags["day"] == d]
        day_summary = summarize(day_flags, n_accounts=int((current["day"] == d).sum()),
                                reference_rate=detector.reference_rate)
        with open(os.path.join(data_dir, "anomalies", f"day_{d:04d}.json"), "w") as f:
            json.dump(_clean({"day": d, "job_id": job_id, "summary": day_summary,
                              "records": to_records(day_flags)}), f)

    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(data_dir, "drift_log.jsonl"), "a") as f:
        f.write(json.dumps(_clean({"job_id": job_id, "at": finished, "sim_days": sim_days,
                                   "drift": drift, "anomalies": summary})) + "\n")

    # Serious flags join the analyst queue. A queue failure must not lose the monitoring result
    # (it is already on disk above), so it is reported separately.
    alerts_created, alerts_error = None, None
    try:
        alerts_created = CaseStore(os.path.join(data_dir, "cases.db")).ingest(to_records(flags), job_id=job_id)
    except Exception as e:
        alerts_error = f"{type(e).__name__}: {e}"

    return {
        "alerts_created": alerts_created, "alerts_error": alerts_error,
        "drift_status": drift["status"], "max_psi": drift["max_psi"],
        "flagged_features": [x["feature"] for x in drift["features"] if x["severity"] == "significant"],
        "score_psi": drift["score"]["psi"] if drift["score"] else None,
        "accounts_checked": summary["accounts_checked"], "flagged": summary["flagged"],
        **summary["by_severity"], "rate_vs_expected": summary["rate_vs_expected"],
    }


def run_daily_job(config: RetrainConfig = None) -> dict:
    c = config or RetrainConfig()
    names = list(c.models or CANDIDATES)
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidate(s): {unknown}")  # before any state changes

    drift = c.drift or DriftConfig()
    with _exclusive(c.data_dir):
        started = time.time()
        report = {"job_id": uuid.uuid4().hex[:12], "status": "failed",
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                  "drift": asdict(drift)}
        try:
            base = pd.read_csv(c.base_raw_path)
            base["timestamp"] = pd.to_datetime(base["timestamp"])
            profiles, merchants = fit_population(base)

            store = DailyStore(c.data_dir)
            state = store.state(base)
            if not store.has_state():
                state = warm_up(profiles, merchants, state, seed=c.seed)

            new_frames, sim_days = [], []
            for _ in range(c.days):
                day = state["next_day"]
                df, state = simulate_day(profiles, merchants, day, state, drift=drift, seed=c.seed)
                store.append_day(day, df, state)
                new_frames.append(df)
                sim_days.append(day)
            new = pd.concat(new_frames, ignore_index=True)

            raw_all = pd.concat([base, store.load_daily()], ignore_index=True)
            feats = build_features(raw_all)
            train, val, test = rolling_split(feats, c.val_days, c.test_days)

            registry = ModelRegistry(c.registry_dir)
            if c.bootstrap:
                bootstrap_champion(registry)
            try:
                monitoring = _monitor(feats, sim_days, registry, c.data_dir, report["job_id"])
            except Exception as e:  # monitoring must never stop retraining; the error is recorded
                monitoring = {"error": f"{type(e).__name__}: {e}"}
            cost = CostAssumptions()
            entries = run_contest(train, val, test, names=names, cost=cost, seed=c.seed)
            winner = select_winner(entries)
            run_id = registry.register_run(
                entries, data_note=f"{len(raw_all):,} rows through simulated day {sim_days[-1]}")
            decision = maybe_promote(registry, run_id, val, cost, margin=c.margin)

            report.update(
                status="ok", sim_days=sim_days, new_rows=len(new), new_fraud_rows=int(new["is_fraud"].sum()),
                total_rows=len(raw_all), fraud_rate_total=float(raw_all["is_fraud"].mean()),
                split={"train": len(train), "val": len(val), "test": len(test)},
                run_id=run_id, winner=winner["name"] if winner else None,
                decision=decision, champion=registry.champion(), monitoring=monitoring)
            return report
        except Exception as e:
            report["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            report["duration_seconds"] = round(time.time() - started, 2)
            _log(c.data_dir, report)


def _parse_drift(pairs):
    kw = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        kw[k] = float(v)
    return DriftConfig(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--drift", nargs="*", metavar="KEY=VALUE",
                    help="amount_scale, fraud_rate_multiplier, new_device_multiplier")
    args = ap.parse_args(argv)
    try:
        report = run_daily_job(RetrainConfig(days=args.days, models=args.models, margin=args.margin,
                                             drift=_parse_drift(args.drift)))
    except JobAlreadyRunning as e:
        print(e)
        raise SystemExit(75)  # EX_TEMPFAIL: cron can simply try again next time
    d = report["decision"]
    print(f"day(s) {report['sim_days']}: +{report['new_rows']:,} rows "
          f"({report['new_fraud_rows']} fraud), total {report['total_rows']:,}")
    print(f"contest winner: {report['winner']}  |  {'PROMOTED' if d['promoted'] else 'kept champion'}: {d['reason']}")
    print(f"serving champion: {report['champion']['model_name']} ({report['champion']['run_id']})")
    m = report["monitoring"]
    if "error" in m:
        print(f"monitoring FAILED: {m['error']}")
    else:
        print(f"drift: {m['drift_status']} (max PSI {m['max_psi']:.2f}"
              f"{', ' + ', '.join(m['flagged_features']) if m['flagged_features'] else ''})  |  "
              f"accounts flagged: {m['critical']} critical, {m['high']} high, {m['watch']} watch "
              f"of {m['accounts_checked']:,}")


if __name__ == "__main__":
    main()

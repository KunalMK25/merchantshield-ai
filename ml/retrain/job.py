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
from dataclasses import asdict, dataclass, field
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
from ml.monitoring.anomaly import AnomalyConfig, AnomalyDetector, account_day_table, summarize, to_records
from ml.monitoring.drift import MIN_REFERENCE_ROWS, drift_report
from ml.monitoring.supervised_alerts import account_day_scores, calibrate_thresholds, detect_supervised
from ml.datasets.workspace import Workspace
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
    margin: float = 0.10               # required relative cost improvement to replace the champion
    cooldown_runs: int = 7             # daily runs a champion keeps its place after promotion
    val_days: int = 10
    test_days: int = 10
    seed: int = 42
    drift: Optional[DriftConfig] = None
    bootstrap: bool = True             # seed the registry with the original lgbm_v1 champion
    stream_path: Optional[str] = None  # held-back real days to replay (a prepared dataset workspace)
    cost: CostAssumptions = field(default_factory=CostAssumptions)
    anomaly: Optional[AnomalyConfig] = None    # calibrated detector thresholds for a dataset workspace
    alert_source: str = "unsupervised"         # or "supervised": alerts from the serving model's scores (needs labels)

    @classmethod
    def from_workspace(cls, ws: Workspace, **overrides):
        """Run the lifecycle on a prepared public dataset, with its own registry, logs and cost scale."""
        meta = ws.config()
        cal = ws.anomaly_calibration()
        fields = dict(
            anomaly=AnomalyConfig(quantile=cal["quantile"], iso_quantile=cal["iso_quantile"]) if cal else None,
            base_raw_path=str(ws.raw_path), data_dir=str(ws.data_dir), registry_dir=str(ws.registry_dir),
            stream_path=str(ws.stream_path) if ws.stream_path.exists() else None,
            cost=CostAssumptions(fp_cost=meta["fp_cost"], fn_cost_fraction=meta["fn_cost_fraction"]),
            bootstrap=False)
        fields.update(overrides)
        return cls(**fields)


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


RECENT_DAYS = 14        # trailing window used as the drift reference when same-weekday days are unavailable
SAME_WEEKDAY_WEEKS = 4  # a single day is compared with this many earlier weeks of the same weekday
SAME_WEEKDAY_MIN_WEEKS = 3  # ...and at least this many of them must have data
BASELINE_DAYS = 60      # window just before the recent one; shown for information, never judged
ANOMALY_REFERENCE_DAYS = 30


def reference_window(feats, day, sim_days):
    """
    The days a new day is judged against. A single day is compared with the SAME WEEKDAY over the previous
    weeks, because real data has a weekly cycle (on Sparkov, Mondays and Tuesdays carry twice the volume of
    other days): against a mixed 14-day window an ordinary day looked drifted, and the model's score
    distribution shifted with it (median score PSI 0.11 against 0.00 for the same weekday). A run covering
    several days, or too little history, falls back to the trailing window. Returns (rows, label).
    """
    first = min(sim_days)
    if len(sim_days) == 1:
        same = feats[day.isin([first - 7 * k for k in range(1, SAME_WEEKDAY_WEEKS + 1)])]
        # a single earlier Saturday is too thin a reference: need most of the weeks to be there
        if day[same.index].nunique() >= SAME_WEEKDAY_MIN_WEEKS and len(same) >= MIN_REFERENCE_ROWS:
            return same, f"same weekday, last {SAME_WEEKDAY_WEEKS} weeks"
    return feats[(day >= first - RECENT_DAYS) & (day < first)], f"trailing {RECENT_DAYS} days"


def _monitor(feats, sim_days, registry, data_dir, job_id, anomaly_config=None,
             alert_source="unsupervised", val_feats=None) -> dict:
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
    recent, reference_label = reference_window(feats, day, sim_days)
    baseline = feats[(day >= first - RECENT_DAYS - BASELINE_DAYS) & (day < first - RECENT_DAYS)]

    score_ref = score_cur = None
    champ = registry.champion()
    if champ is not None:
        model = registry.load_model(champ["run_id"], champ["model_name"])
        score_ref = model.predict_proba(recent[FEATURE_COLUMNS])[:, 1]
        score_cur = model.predict_proba(cur[FEATURE_COLUMNS])[:, 1]
    drift = drift_report(recent, baseline, cur, score_ref, score_cur)
    drift["reference"] = reference_label

    used, fallback, thresholds = "unsupervised", None, None
    flags = n_by_day = reference_rate = None
    if alert_source == "supervised":
        if champ is None:
            fallback = "no champion model yet to score with"
        else:
            try:
                pv = account_day_scores(val_feats, model.predict_proba(val_feats[FEATURE_COLUMNS])[:, 1], origin)
                cal = calibrate_thresholds(pv)                       # validation days only, for this model
                pc = account_day_scores(cur, score_cur, origin)
                flags = detect_supervised(pc, cal["critical"], cal["high"])
                n_by_day, reference_rate = pc.groupby("day").size().to_dict(), 0.0
                used, thresholds = "supervised", {"critical": cal["critical"], "high": cal["high"]}
            except ValueError as e:
                fallback = str(e)
    if used == "unsupervised":
        table = account_day_table(feats, origin=origin)
        reference = table[(table["day"] >= first - ANOMALY_REFERENCE_DAYS) & (table["day"] < first)]
        detector = AnomalyDetector(config=anomaly_config).fit(reference)
        current = table[table["day"].isin(sim_days)]
        flags = detector.detect(current)
        n_by_day, reference_rate = current.groupby("day").size().to_dict(), detector.reference_rate
    summary = summarize(flags, n_accounts=sum(n_by_day.values()), reference_rate=reference_rate)

    os.makedirs(os.path.join(data_dir, "anomalies"), exist_ok=True)
    for d in sim_days:
        day_flags = flags[flags["day"] == d]
        day_summary = summarize(day_flags, n_accounts=int(n_by_day.get(d, 0)), reference_rate=reference_rate)
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
        "anomaly_quantile": anomaly_config.quantile if anomaly_config else None,
        "alert_source": used, "alert_fallback_reason": fallback, "alert_thresholds": thresholds,
        "drift_status": drift["status"], "max_psi": drift["max_psi"],
        "flagged_features": drift["flagged_features"],
        "score_psi": drift["score"]["psi"] if drift["score"] else None,
        "accounts_checked": summary["accounts_checked"], "flagged": summary["flagged"],
        **summary["by_severity"], "rate_vs_expected": summary["rate_vs_expected"],
    }


def _load_stream(path, day0):
    """Held-back real days with a day index relative to the history's first day, or None."""
    if not path or not os.path.exists(path):
        return None
    stream = pd.read_csv(path)
    if stream.empty:
        return None
    stream["timestamp"] = pd.to_datetime(stream["timestamp"])
    stream["_day"] = (stream["timestamp"].dt.normalize() - pd.Timestamp(day0)).dt.days
    return stream


def run_daily_job(config: RetrainConfig = None) -> dict:
    c = config or RetrainConfig()
    names = list(c.models or CANDIDATES)
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidate(s): {unknown}")  # before any state changes
    if c.alert_source not in ("unsupervised", "supervised"):
        raise ValueError(f"alert_source must be 'unsupervised' or 'supervised', got {c.alert_source!r}")

    drift = c.drift or DriftConfig()
    with _exclusive(c.data_dir):
        started = time.time()
        report = {"job_id": uuid.uuid4().hex[:12], "status": "failed",
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                  "drift": asdict(drift)}
        try:
            base = pd.read_csv(c.base_raw_path)
            base["timestamp"] = pd.to_datetime(base["timestamp"])
            store = DailyStore(c.data_dir)
            state = store.state(base)
            stream = _load_stream(c.stream_path, state["day0"])
            synth = None            # fitted lazily: a run that only replays real days never needs it

            new_frames, sim_days, sources = [], [], []
            for _ in range(c.days):
                later = stream[stream["_day"] >= state["next_day"]] if stream is not None else None
                if later is not None and len(later):
                    day = int(later["_day"].min())                    # real data: skip days with no transactions
                    df = later[later["_day"] == day].drop(columns="_day").reset_index(drop=True)
                    ids = df["transaction_id"].str.extract(r"_(\d+)$")[0].astype(int)
                    state = {**state, "next_day": day + 1, "txn_counter": max(state["txn_counter"], int(ids.max()))}
                    sources.append("replay")
                else:
                    if synth is None:
                        profiles, merchants = fit_population(base)
                        synth = (profiles, merchants)
                        if not store.has_state():
                            state = warm_up(profiles, merchants, state, seed=c.seed)
                    day = state["next_day"]
                    df, state = simulate_day(synth[0], synth[1], day, state, drift=drift, seed=c.seed)
                    sources.append("synthetic")
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
                monitoring = _monitor(feats, sim_days, registry, c.data_dir, report["job_id"], c.anomaly,
                                    c.alert_source, val)
            except Exception as e:  # monitoring must never stop retraining; the error is recorded
                monitoring = {"error": f"{type(e).__name__}: {e}"}
            cost = c.cost
            entries = run_contest(train, val, test, names=names, cost=cost, seed=c.seed)
            winner = select_winner(entries)
            run_id = registry.register_run(
                entries, data_note=f"{len(raw_all):,} rows through simulated day {sim_days[-1]}")
            decision = maybe_promote(registry, run_id, val, cost, margin=c.margin, cooldown_runs=c.cooldown_runs)

            report.update(
                status="ok", sim_days=sim_days, new_rows=len(new), cost=asdict(cost),
                source=sources[0] if len(set(sources)) == 1 else "mixed", new_fraud_rows=int(new["is_fraud"].sum()),
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
    ap.add_argument("--workspace", help="run on a prepared dataset workspace (see ml.datasets.prepare)")
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--alerts", choices=["unsupervised", "supervised"], default=None,
                    help="where account alerts come from (supervised needs fraud labels and a champion model)")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="relative cost improvement a challenger needs (sized to the measured noise)")
    ap.add_argument("--cooldown", type=int, default=7, help="daily runs a champion keeps its place (0 disables)")
    ap.add_argument("--drift", nargs="*", metavar="KEY=VALUE",
                    help="amount_scale, fraud_rate_multiplier, new_device_multiplier")
    args = ap.parse_args(argv)
    try:
        opts = dict(days=args.days, models=args.models, margin=args.margin, cooldown_runs=args.cooldown,
                    drift=_parse_drift(args.drift))
        if args.alerts:
            opts["alert_source"] = args.alerts
        if args.workspace:
            kw = {"root": args.workspace_root} if args.workspace_root else {}
            config = RetrainConfig.from_workspace(Workspace(args.workspace, **kw), **opts)
        else:
            config = RetrainConfig(**opts)
        report = run_daily_job(config)
    except JobAlreadyRunning as e:
        print(e)
        raise SystemExit(75)  # EX_TEMPFAIL: cron can simply try again next time
    d = report["decision"]
    print(f"day(s) {report['sim_days']} [{report['source']}]: +{report['new_rows']:,} rows "
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

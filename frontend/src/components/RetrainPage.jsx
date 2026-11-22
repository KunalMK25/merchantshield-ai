import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import ErrorBanner from "./ErrorBanner";

const CRON = "0 2 * * *  cd /path/to/merchantshield-ai && .venv/bin/python -m ml.retrain.job";

const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
const num = (v) => (typeof v === "number" ? v.toLocaleString("en-IN") : "—");

function Decision({ run }) {
  if (run.status !== "ok") return <span className="pill pill-bad">failed</span>;
  return run.decision?.promoted ? (
    <span className="pill pill-win">promoted</span>
  ) : (
    <span className="pill pill-muted">kept champion</span>
  );
}

export default function RetrainPage() {
  const [status, setStatus] = useState(null);
  const [runs, setRuns] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [s, l] = await Promise.all([api.getRetrainStatus(), api.getRetrainLog(30)]);
      setStatus(s);
      setRuns(l.runs);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load retraining status.");
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, [load]);

  const sch = status?.scheduler;
  const last = status?.last_run;

  return (
    <>
      <header className="page-head">
        <h1 className="page-title">Automatic retraining</h1>
        <p className="page-sub">
          Every cycle adds a new simulated day of transactions that mirrors the demo data, retrains
          every candidate model, and promotes the winner only if it clearly beats the current one.
        </p>
      </header>

      {error && <ErrorBanner message="Retraining status unavailable." detail={error} />}

      {status && (
        <div className="stat-row">
          <div className={`stat-card${sch?.enabled ? " tone-good" : ""}`}>
            <div className="stat-label">Schedule</div>
            <div className="stat-value mono">{sch?.enabled ? "On" : "Off"}</div>
            <div className="stat-hint">
              {sch?.enabled ? `Every ${Number(sch.interval_hours).toFixed(2).replace(/\.?0+$/, "")} h` : "Use cron or set an interval"}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Next run</div>
            <div className="stat-value mono small-value">{sch?.running ? "Running…" : when(sch?.next_run_at)}</div>
            <div className="stat-hint">In-app scheduler</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Next simulated day</div>
            <div className="stat-value mono">{status.next_simulated_day ?? "—"}</div>
            <div className="stat-hint">Day 0 = first day of history</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Last run</div>
            <div className="stat-value mono small-value">{last ? last.status : "—"}</div>
            <div className="stat-hint">{last ? when(last.finished_at) : "Nothing has run yet"}</div>
          </div>
        </div>
      )}

      <div className="stack">
        <section className="panel">
          <div className="panel-header">
            <span className="panel-title">Run history</span>
          </div>
          {runs && runs.length === 0 ? (
            <div className="empty-state">
              No retraining has run yet. Run <span className="mono">python -m ml.retrain.job</span> or turn
              on the schedule below.
            </div>
          ) : (
            <div className="panel-body table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Finished</th>
                    <th>Day</th>
                    <th>New transactions</th>
                    <th>Fraud</th>
                    <th>Contest winner</th>
                    <th>Outcome</th>
                    <th>Took</th>
                  </tr>
                </thead>
                <tbody>
                  {(runs || []).map((r) => (
                    <tr key={r.job_id} title={r.decision?.reason || r.error || ""}>
                      <td>{when(r.finished_at)}</td>
                      <td className="mono">{r.sim_days ? r.sim_days.join(", ") : "—"}</td>
                      <td className="mono">{num(r.new_rows)}</td>
                      <td className="mono">{num(r.new_fraud_rows)}</td>
                      <td className="mono">{r.winner || "—"}</td>
                      <td><Decision run={r} /></td>
                      <td className="mono">{r.duration_seconds != null ? `${r.duration_seconds}s` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <span className="panel-title">How to schedule it</span>
          </div>
          <div className="panel-body">
            <p className="page-sub" style={{ marginTop: 0 }}>
              One process: set <span className="mono">MS_RETRAIN_INTERVAL_HOURS=24</span> before starting the API
              and it retrains itself. Several workers, or a host that sleeps: use cron instead.
            </p>
            <pre className="code-block">{CRON}</pre>
            <p className="context-note" style={{ margin: "12px 0 0" }}>
              A promoted model starts serving within seconds, with no restart, and every decision records
              which model made it. The run history above is read-only on purpose.
            </p>
          </div>
        </section>
      </div>
    </>
  );
}

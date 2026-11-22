import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import ErrorBanner from "./ErrorBanner";

const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
const fixed = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "—");

const STATUS = {
  stable: { label: "Stable", tone: "low", text: "New data looks like the recent past." },
  watch: { label: "Watch", tone: "medium", text: "Some features moved a little. Nothing to act on yet." },
  drifted: { label: "Drift detected", tone: "critical", text: "The data has shifted enough to matter." },
  insufficient_data: { label: "Not enough data", tone: "medium", text: "Too few transactions to judge." },
};
const SEV_PILL = { significant: "pill-bad", moderate: "pill-warn", stable: "pill-muted", informational: "pill-muted" };
const ACCOUNT_PILL = { critical: "pill-bad", high: "pill-warn", watch: "pill-muted" };
const FILTERS = [
  { id: "serious", label: "Critical + high", sev: ["critical", "high"] },
  { id: "all", label: "Everything flagged", sev: [] },
];

function HistoryChart({ history }) {
  const rows = [...history].reverse().slice(-24);
  if (rows.length === 0) return null;
  const W = 640, H = 120, pad = 22;
  const top = Math.max(0.3, ...rows.map((r) => r.max_psi || 0)) * 1.1;
  const bw = Math.max(6, (W - pad * 2) / rows.length - 6);
  const y = (v) => H - pad - (Math.min(v, top) / top) * (H - pad * 2);
  const color = (s) => (s === "drifted" ? "var(--risk-critical)" : s === "watch" ? "var(--risk-medium)" : "var(--risk-low)");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="hist-chart" role="img" aria-label="Largest drift score per run">
      {[0.1, 0.25].map((t) => (
        <g key={t}>
          <line x1={pad} x2={W - pad} y1={y(t)} y2={y(t)} stroke="var(--border-strong)" strokeDasharray="4 4" />
          <text x={W - pad} y={y(t) - 3} textAnchor="end" fontSize="10" fill="var(--text-tertiary)">{t}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const x = pad + i * ((W - pad * 2) / rows.length) + 3;
        const yy = y(r.max_psi || 0);
        return (
          <rect key={r.job_id} x={x} y={yy} width={bw} height={Math.max(2, H - pad - yy)} rx="2" fill={color(r.status)}>
            <title>{`${when(r.at)} · ${r.status} · max PSI ${fixed(r.max_psi)}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export default function DriftPage() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState(null);
  const [anoms, setAnoms] = useState(null);
  const [filter, setFilter] = useState("serious");
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const sev = FILTERS.find((f) => f.id === filter).sev;
      const [l, h, a] = await Promise.all([api.getDriftLatest(), api.getDriftHistory(24), api.getAnomalies(sev)]);
      setLatest(l);
      setHistory(h.history);
      setAnoms(a);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load monitoring data.");
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  const d = latest?.drift;
  const st = d ? STATUS[d.status] || STATUS.stable : null;
  const by = latest?.anomalies?.by_severity;
  const flaggedFeatures = d?.features?.filter((f) => f.severity === "significant").map((f) => f.feature) || [];

  return (
    <>
      <header className="page-head">
        <h1 className="page-title">Drift &amp; anomalies</h1>
        <p className="page-sub">
          Is the incoming data different from what the model learned from, and which accounts look
          unusual for their own owner? Flags are recommendations for a person to review, never
          automatic account actions.
        </p>
      </header>

      {error && <ErrorBanner message="Monitoring unavailable." detail={error} />}

      {latest && !d && (
        <div className="panel">
          <div className="empty-state">
            No monitoring data yet. Run <span className="mono">python -m ml.retrain.job</span> to generate a
            day and check it. Try <span className="mono">--drift amount_scale=2.5</span> to see a shift get caught.
          </div>
        </div>
      )}

      {d && (
        <>
          <div className={`status-banner tone-${st.tone}`} role="status">
            <strong>{st.label}</strong>
            <span>
              Simulated day {latest.sim_days?.join(", ")} · {st.text}
              {flaggedFeatures.length > 0 && ` Shifted: ${flaggedFeatures.join(", ")}.`}
            </span>
          </div>

          <div className="stat-row">
            <div className="stat-card">
              <div className="stat-label">Largest feature shift</div>
              <div className="stat-value mono">{fixed(d.max_psi, 2)}</div>
              <div className="stat-hint">PSI · 0.10 watch · 0.25 drift</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Model score shift</div>
              <div className="stat-value mono">{fixed(d.score?.psi, 3)}</div>
              <div className="stat-hint">{d.score ? "Serving model's scores" : "No model scored yet"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Accounts to review</div>
              <div className="stat-value mono">{(by?.critical ?? 0) + (by?.high ?? 0)}</div>
              <div className="stat-hint">{by?.critical ?? 0} critical · {by?.high ?? 0} high</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Checked</div>
              <div className="stat-value mono">{latest.anomalies?.accounts_checked?.toLocaleString("en-IN") ?? "—"}</div>
              <div className="stat-hint">account-days, {by?.watch ?? 0} on watch</div>
            </div>
          </div>

          <div className="stack">
            <section className="panel">
              <div className="panel-header">
                <span className="panel-title">Flagged accounts</span>
                <div className="chip-row" role="group" aria-label="Filter">
                  {FILTERS.map((f) => (
                    <button key={f.id} type="button" className={`chip${filter === f.id ? " active" : ""}`}
                            onClick={() => setFilter(f.id)}>{f.label}</button>
                  ))}
                </div>
              </div>
              {d.status === "drifted" && (
                <p className="context-note drift-caveat">
                  The data shifted as a whole today, which can push many ordinary accounts over their own
                  thresholds. Treat today's flags with extra caution, especially those on watch.
                </p>
              )}
              {anoms && anoms.records.length === 0 ? (
                <div className="empty-state">No accounts flagged at this level for day {anoms.day ?? "—"}.</div>
              ) : (
                <div className="panel-body table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr><th>Level</th><th>Account</th><th>Why it was flagged</th><th>Payments</th><th>Recommendation</th></tr>
                    </thead>
                    <tbody>
                      {(anoms?.records || []).slice(0, 25).map((r) => (
                        <tr key={`${r.customer_id}-${r.day}`}>
                          <td><span className={`pill ${ACCOUNT_PILL[r.severity]}`} style={{ marginLeft: 0 }}>{r.severity}</span></td>
                          <td className="mono">{r.customer_id}</td>
                          <td className="why-cell">
                            {r.signals?.length
                              ? r.signals.map((s) => <div key={s.name}>{s.detail}</div>)
                              : <div className="text-tertiary">Behaviour unusual compared with other accounts</div>}
                          </td>
                          <td className="mono">{r.n_txns} · ₹{Math.round(r.total_amount).toLocaleString("en-IN")}</td>
                          <td>{r.recommended_action}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <section className="panel">
              <div className="panel-header">
                <span className="panel-title">Feature drift, latest run</span>
                <span className="small text-tertiary">{d.n_current?.toLocaleString("en-IN")} new vs {d.n_reference?.toLocaleString("en-IN")} recent</span>
              </div>
              <div className="panel-body table-scroll">
                <table className="data-table">
                  <thead>
                    <tr><th>Feature</th><th>Shift (PSI)</th><th>vs recent</th><th>vs baseline</th><th>KS</th><th>Verdict</th></tr>
                  </thead>
                  <tbody>
                    {[...d.features, ...(d.score ? [{ feature: "model score", ...d.score, psi_recent: d.score.psi, psi_baseline: null, ks_statistic: d.score.ks_statistic }] : [])]
                      .sort((a, b) => b.psi - a.psi)
                      .map((f) => (
                        <tr key={f.feature}>
                          <td className="mono">{f.feature}</td>
                          <td>
                            <div className="cost-cell">
                              <span className="mono">{fixed(f.psi)}</span>
                              <span className="cost-bar" aria-hidden="true">
                                <span style={{ width: `${Math.min(100, (f.psi / 0.5) * 100)}%` }} />
                              </span>
                            </div>
                          </td>
                          <td className="mono">{fixed(f.psi_recent)}</td>
                          <td className="mono">{f.psi_baseline == null ? "n/a" : fixed(f.psi_baseline)}</td>
                          <td className="mono">{fixed(f.ks_statistic)}</td>
                          <td><span className={`pill ${SEV_PILL[f.severity]}`} style={{ marginLeft: 0 }}>{f.severity}</span></td>
                        </tr>
                      ))}
                  </tbody>
                </table>
                <p className="context-note" style={{ marginTop: 12 }}>
                  Counts like <span className="mono">prior_txn_count</span> and <span className="mono">account_age_days</span> grow
                  every day by construction, so they are shown for information and never judged. Weekday is not monitored.
                </p>
              </div>
            </section>

            {history && history.length > 0 && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-title">Largest shift per run</span>
                  <span className="small text-tertiary">dashed lines: watch 0.10, drift 0.25</span>
                </div>
                <div className="panel-body"><HistoryChart history={history} /></div>
              </section>
            )}
          </div>
        </>
      )}
    </>
  );
}

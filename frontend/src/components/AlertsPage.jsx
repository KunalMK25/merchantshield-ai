import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import ErrorBanner from "./ErrorBanner";
import { money } from "../utils/currency";

const NAME_KEY = "ms-analyst";
const TOKEN_KEY = "ms-analyst-token";

const store = {
  get: (k) => {
    try { return window.localStorage.getItem(k) || ""; } catch { return ""; }
  },
  set: (k, v) => {
    try { window.localStorage.setItem(k, v); } catch { /* private mode: lasts for this page only */ }
  },
};

const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
const SEV_PILL = { critical: "pill-bad", high: "pill-warn" };
const ACCT_PILL = { active: "pill-muted", held: "pill-warn", closed: "pill-bad" };
const EVENT_LABEL = {
  alert_created: "Alert raised", hold_placed: "Temporary hold placed", hold_confirmed: "Hold confirmed (already held)",
  hold_released: "Hold released", account_closed: "Account closed", alert_dismissed: "Dismissed as false alarm",
  notice_created: "Customer notice prepared",
};
const TABS = [{ id: "queue", label: "Queue" }, { id: "held", label: "Held accounts" }, { id: "activity", label: "Activity" }];

function NoticeCard({ notice }) {
  if (!notice) return null;
  return (
    <div className="notice-card">
      <div className="notice-head">
        <strong>Customer notice ({notice.kind})</strong>
        <span className="pill pill-muted" style={{ marginLeft: 0 }}>simulated, not sent</span>
      </div>
      <p>{notice.message}</p>
    </div>
  );
}

function AlertDetail({ alert, analystOk, busy, onHold, onDismiss }) {
  const [note, setNote] = useState("");
  useEffect(() => setNote(""), [alert?.id]);
  if (!alert) return <div className="empty-state">Select an alert to review it.</div>;
  const open = alert.status === "open";
  const closed = alert.account_status === "closed";
  return (
    <div className="panel-body">
      <div className="detail-head">
        <span className={`pill ${SEV_PILL[alert.severity]}`} style={{ marginLeft: 0 }}>{alert.severity}</span>
        <span className="mono detail-id">{alert.customer_id}</span>
        <span className={`pill ${ACCT_PILL[alert.account_status]}`}>account {alert.account_status}</span>
      </div>
      <p className="page-sub" style={{ margin: "4px 0 12px" }}>
        Simulated day {alert.day} · {alert.n_txns} payments totalling {money(alert.total_amount)}
      </p>

      <div className="field-section-title">Why it was flagged</div>
      <ul className="signal-list">
        {alert.signals.length
          ? alert.signals.map((s) => <li key={s.name}>{s.detail}</li>)
          : <li>Behaviour unusual compared with other accounts</li>}
      </ul>
      <p className="context-note">{alert.recommended_action}. This is a recommendation; nothing happens until you decide.</p>

      {open ? (
        <div className="action-box">
          <label className="field-label" htmlFor="note">Note (kept in the audit trail)</label>
          <textarea id="note" className="field-input" rows={3} maxLength={500} value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="What did you check? e.g. called customer, reviewed the payments" />
          {!analystOk && <p className="field-hint">Enter your name above to enable actions.</p>}
          {closed && <p className="field-hint">This account is already closed, so it can only be dismissed.</p>}
          <div className="action-row">
            <button type="button" className="btn btn-primary" disabled={!analystOk || busy || closed}
                    onClick={() => onHold(alert.id, note)}>Place temporary hold</button>
            <button type="button" className="btn btn-ghost" disabled={!analystOk || busy}
                    onClick={() => onDismiss(alert.id, note)}>Dismiss as false alarm</button>
          </div>
        </div>
      ) : (
        <p className="decision-line">
          <strong>{alert.decision === "hold" ? "Hold placed" : "Dismissed"}</strong> by {alert.decided_by} on{" "}
          {when(alert.decided_at)}{alert.decision_note ? ` — “${alert.decision_note}”` : ""}
        </p>
      )}
      {alert.notices?.slice(-1).map((n) => <NoticeCard key={n.id} notice={n} />)}
    </div>
  );
}

function HeldTable({ accounts, analystOk, busy, onRelease, onClose }) {
  const [notes, setNotes] = useState({});
  const set = (id, v) => setNotes((n) => ({ ...n, [id]: v }));
  if (!accounts.length) return <div className="empty-state">No accounts are currently held.</div>;
  return (
    <div className="panel-body table-scroll">
      <table className="data-table">
        <thead><tr><th>Account</th><th>Status</th><th>Since</th><th>By</th><th>Decision</th></tr></thead>
        <tbody>
          {accounts.map((a) => {
            const note = notes[a.customer_id] || "";
            const canAct = analystOk && !busy && note.trim().length >= 3;
            return (
              <tr key={a.customer_id}>
                <td className="mono">{a.customer_id}</td>
                <td><span className={`pill ${ACCT_PILL[a.status]}`} style={{ marginLeft: 0 }}>{a.status}</span></td>
                <td>{when(a.updated_at)}</td>
                <td>{a.updated_by || "—"}</td>
                <td className="why-cell">
                  {a.status === "held" ? (
                    <>
                      <input className="field-input" value={note} maxLength={500}
                             placeholder="Reason (required), e.g. customer verified by phone"
                             onChange={(e) => set(a.customer_id, e.target.value)} aria-label={`Reason for ${a.customer_id}`} />
                      <div className="action-cell" style={{ marginTop: 8 }}>
                        <button type="button" className="btn btn-ghost" disabled={!canAct}
                                onClick={() => onRelease(a.customer_id, note)}>Release hold</button>
                        <button type="button" className="btn btn-danger" disabled={!canAct}
                                onClick={() => onClose(a.customer_id, note)}>Close account</button>
                      </div>
                    </>
                  ) : <span className="text-tertiary">{a.reason}</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function AlertsPage({ onChanged }) {
  const [tab, setTab] = useState("queue");
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [events, setEvents] = useState([]);
  const [name, setName] = useState(() => store.get(NAME_KEY));
  const [token, setToken] = useState(() => store.get(TOKEN_KEY));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const analystOk = name.trim().length >= 2;

  const load = useCallback(async () => {
    try {
      const [s, a, acc, ev] = await Promise.all([
        api.getCaseSummary(), api.getCaseAlerts("open"), api.getCaseAccounts(), api.getCaseEvents(40),
      ]);
      setSummary(s);
      setAlerts(a.alerts);
      setAccounts(acc.accounts.filter((x) => x.status !== "active"));
      setEvents(ev.events);
      setSelected((cur) => (cur && a.alerts.some((x) => x.id === cur) ? cur : cur ?? a.alerts[0]?.id ?? null));
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the alert queue.");
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (selected == null) { setDetail(null); return; }
    api.getCaseAlert(selected).then(setDetail).catch(() => setDetail(null));
  }, [selected, summary]);

  const run = async (fn) => {
    if (!analystOk) return;
    setBusy(true);
    try {
      await fn();
      setError(null);
      await load();
      if (onChanged) onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message}` : "That action failed.");
    } finally {
      setBusy(false);
    }
  };

  const who = (note) => ({ decided_by: name.trim(), note });
  const onHold = (id, note) => run(async () => { await api.holdAlert(id, who(note)); setSelected(id); });
  const onDismiss = (id, note) => run(async () => { await api.dismissAlert(id, who(note)); });
  const onRelease = (cid, note) => run(() => api.releaseAccount(cid, who(note)));
  const onClose = (cid, note) => {
    if (!window.confirm(`Close account ${cid}? Closing is final and cannot be undone here.`)) return;
    run(() => api.closeAccount(cid, who(note)));
  };

  const heldCount = accounts.filter((a) => a.status === "held").length;

  return (
    <>
      <header className="page-head">
        <h1 className="page-title">Alerts</h1>
        <p className="page-sub">
          Accounts the monitoring flagged as critical or high. The system only recommends. A person reviews
          each alert, decides, and every decision is recorded. Holds, customer notices and closures here are
          simulated; nothing reaches a real account or customer.
        </p>
      </header>

      {error && <ErrorBanner message="Alerts unavailable or action refused." detail={error} />}

      <div className="analyst-bar">
        <label htmlFor="analyst">Analyst</label>
        <input id="analyst" className="field-input" value={name} placeholder="Your name (required to act)"
               maxLength={80} onChange={(e) => { setName(e.target.value); store.set(NAME_KEY, e.target.value); }} />
        <label htmlFor="token">Token</label>
        <input id="token" type="password" className="field-input" value={token} placeholder="Only if the server sets one"
               onChange={(e) => { setToken(e.target.value); store.set(TOKEN_KEY, e.target.value); }} />
        <span className="small text-tertiary">Demo: there is no login. Set MS_ANALYST_TOKEN on the server to require a token.</span>
      </div>

      {summary && (
        <div className="stat-row">
          <div className={`stat-card${summary.open_critical ? " tone-warn" : ""}`}>
            <div className="stat-label">Open alerts</div>
            <div className="stat-value mono">{summary.open_alerts}</div>
            <div className="stat-hint">{summary.open_critical} critical · {summary.open_high} high</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Held accounts</div>
            <div className="stat-value mono">{summary.held_accounts}</div>
            <div className="stat-hint">Waiting on a decision</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Closed</div>
            <div className="stat-value mono">{summary.closed_accounts}</div>
            <div className="stat-hint">Final</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">False alarms</div>
            <div className="stat-value mono">{summary.dismissed_alerts}</div>
            <div className="stat-hint">Dismissed</div>
          </div>
        </div>
      )}

      <div className="chip-row tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id}
                  className={`chip${tab === t.id ? " active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}{t.id === "held" && heldCount ? ` (${heldCount})` : ""}
          </button>
        ))}
      </div>

      {tab === "queue" && (
        alerts && alerts.length === 0 ? (
          <div className="panel">
            <div className="empty-state">
              The queue is empty. Alerts appear after the daily job flags accounts. To load flags from
              days that already ran, use <span className="mono">python -m ml.cases.ingest</span>.
            </div>
          </div>
        ) : (
          <div className="split">
            <section className="panel">
              <div className="panel-header"><span className="panel-title">Open alerts</span></div>
              <div className="queue">
                {(alerts || []).map((a) => (
                  <button key={a.id} type="button" className={`queue-row${selected === a.id ? " selected" : ""}`}
                          onClick={() => setSelected(a.id)}>
                    <span className={`pill ${SEV_PILL[a.severity]}`} style={{ marginLeft: 0 }}>{a.severity}</span>
                    <span className="mono">{a.customer_id}</span>
                    <span className="queue-meta">day {a.day} · {a.n_txns} payments · {money(a.total_amount)}</span>
                    {a.account_status !== "active" && <span className={`pill ${ACCT_PILL[a.account_status]}`}>{a.account_status}</span>}
                  </button>
                ))}
              </div>
            </section>
            <section className="panel">
              <div className="panel-header"><span className="panel-title">Review</span></div>
              <AlertDetail alert={detail} analystOk={analystOk} busy={busy} onHold={onHold} onDismiss={onDismiss} />
            </section>
          </div>
        )
      )}

      {tab === "held" && (
        <section className="panel">
          <div className="panel-header">
            <span className="panel-title">Accounts acted on</span>
            <span className="small text-tertiary">Release returns the account to normal. Closing is final.</span>
          </div>
          <HeldTable accounts={accounts} analystOk={analystOk} busy={busy} onRelease={onRelease} onClose={onClose} />
        </section>
      )}

      {tab === "activity" && (
        <section className="panel">
          <div className="panel-header"><span className="panel-title">Audit trail</span></div>
          {events.length === 0 ? <div className="empty-state">Nothing recorded yet.</div> : (
            <ul className="timeline">
              {events.map((e) => (
                <li key={e.id} className={`timeline-item ev-${e.kind}`}>
                  <div className="timeline-head">
                    <strong>{EVENT_LABEL[e.kind] || e.kind}</strong>
                    <span className="small text-tertiary">{when(e.at)}</span>
                  </div>
                  <div className="small text-secondary">
                    <span className="mono">{e.customer_id}</span> · {e.actor}{e.detail?.note ? ` — “${e.detail.note}”` : ""}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </>
  );
}

import { tierFor, formatPercent, formatTimestamp, shortId } from "../utils/presentation";

export default function AuditLogTable({ entries, isLoading, error, onRefresh }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Recent audit log</span>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onRefresh}
          disabled={isLoading}
          aria-label={isLoading ? "Refreshing audit log" : "Refresh audit log"}
          style={{ padding: "4px 10px", fontSize: 12 }}
        >
          {isLoading ? "Refreshing…" : "↻ Refresh"}
        </button>
      </div>

      {error && (
        <div className="panel-body">
          <p className="small text-tertiary">Could not load audit log: {error}</p>
        </div>
      )}

      {!error && !isLoading && (!entries || entries.length === 0) && (
        <div className="empty-state">
          No decisions recorded yet. Evaluate a transaction to populate the log.
        </div>
      )}

      {!error && !isLoading && entries && entries.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table className="audit-table">
            <caption className="visually-hidden">Recent risk decisions, most recent first</caption>
            <thead>
              <tr>
                <th>Time</th>
                <th>Transaction</th>
                <th>Action</th>
                <th>Probability</th>
                <th>Score</th>
                <th>Rule</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(e => {
                const tier = tierFor(e.action);
                return (
                  <tr key={e.id}>
                    <td style={{ whiteSpace: "nowrap", color: "var(--text-tertiary)", fontSize: 11.5 }}>
                      {formatTimestamp(e.decided_at)}
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: 11.5 }}>
                        {shortId(e.transaction_id)}
                      </span>
                    </td>
                    <td>
                      <span className={`action-chip ${tier.tier}`}>
                        <span aria-hidden="true">{tier.icon}</span>
                        {tier.label}
                      </span>
                    </td>
                    <td>
                      <span className="prob-value">
                        {formatPercent(e.fraud_probability)}
                      </span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: 12 }}>{e.risk_score ?? "—"}</span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: 10.5, color: "var(--text-tertiary)" }}>
                        {e.policy_rule_id || "—"}
                      </span>
                    </td>
                    <td>
                      <span className="source-tag">
                        {e.source || "manual"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

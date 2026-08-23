import { tierFor, formatPercent, formatTimestamp, shortId } from "../utils/presentation";

export default function AuditLogTable({ entries, isLoading, error, onRefresh }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Recent audit log</span>
        <button type="button" className="btn btn-ghost" onClick={onRefresh} disabled={isLoading}>
          {isLoading ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {error && (
          <div className="error-banner" style={{ margin: 18 }}>
            ⚠ Could not load audit log: {error}
          </div>
        )}

        {!error && isLoading && (
          <div style={{ padding: 18 }}>
            {[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 32, marginBottom: 8 }} />)}
          </div>
        )}

        {!error && !isLoading && (!entries || entries.length === 0) && (
          <div className="empty-state">No decisions recorded yet. Evaluate a transaction to populate the audit log.</div>
        )}

        {!error && !isLoading && entries && entries.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table className="audit-table">
              <thead>
                <tr>
                  <th>Transaction</th>
                  <th>Source</th>
                  <th>Decision</th>
                  <th>Probability</th>
                  <th>Risk</th>
                  <th>Rule</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const tier = tierFor(entry.action);
                  return (
                    <tr key={entry.request_id}>
                      <td className="mono">{shortId(entry.transaction_id)}</td>
                      <td>
                        <span className="small text-secondary">
                          {entry.source === "demo" ? "Demo" : "Manual"}
                        </span>
                      </td>
                      <td>
                        <span className={`action-chip ${tier.tier}`}>{entry.action}</span>
                      </td>
                      <td className="mono">{formatPercent(entry.fraud_probability)}</td>
                      <td>{entry.risk_category}</td>
                      <td className="mono small">{entry.policy_rule_id}</td>
                      <td className="small text-secondary">{formatTimestamp(entry.decision_timestamp)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

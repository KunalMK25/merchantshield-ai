import { tierFor, RISK_CATEGORY_META, formatPercent } from "../utils/presentation";

export default function RiskResultCard({ result, isLoading }) {
  if (isLoading) {
    return (
      <div className="panel" aria-busy="true" aria-label="Risk decision loading">
        <div className="panel-header"><span className="panel-title">Risk decision</span></div>
        <div className="panel-body">
          <div className="skeleton" style={{ height: 64, marginBottom: 16 }} aria-hidden="true" />
          <div className="skeleton" style={{ height: 90, marginBottom: 16 }} aria-hidden="true" />
          <div className="skeleton" style={{ height: 60 }} aria-hidden="true" />
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel">
        <div className="panel-header"><span className="panel-title">Risk decision</span></div>
        <div className="empty-state" aria-live="polite">
          Select a demo scenario or enter a transaction to see a risk decision here.
        </div>
      </div>
    );
  }

  const tier = tierFor(result.action);
  const categoryMeta = RISK_CATEGORY_META[result.risk_category] || {};

  return (
    <div className="panel" aria-live="polite" aria-atomic="true">
      <div className="panel-header">
        <span className="panel-title">Risk decision</span>
        <span className="small text-tertiary mono" aria-label={`Transaction ID: ${result.transaction_id}`}>{result.transaction_id}</span>
      </div>
      <div className="panel-body">
        <div className={`action-banner ${tier.tier}`}>
          <span className="action-icon" aria-hidden="true">{tier.icon}</span>
          <div>
            <div className="action-text-main">
              {tier.label} — <span className="mono">{result.action}</span>
            </div>
            <div className="action-text-sub">Policy rule: <span className="mono">{result.policy_rule_id}</span></div>
            {tier.clarification && <div className="action-text-sub">{tier.clarification}</div>}
          </div>
        </div>

        <div className="metric-row">
          <div className="metric-tile">
            <div className="metric-tile-label">Fraud probability</div>
            <div className="metric-tile-value">{formatPercent(result.fraud_probability)}</div>
          </div>
          <div className="metric-tile">
            <div className="metric-tile-label">Risk score</div>
            <div className="metric-tile-value">{result.risk_score}/100</div>
          </div>
          <div className="metric-tile">
            <div className="metric-tile-label">Decision threshold</div>
            <div className="metric-tile-value">{result.threshold}</div>
          </div>
        </div>

        <div className="field-group">
          <span className={`risk-badge ${result.risk_category}`}>
            <span aria-hidden="true">{categoryMeta.icon}</span> {result.risk_category} RISK
          </span>
          <p className="small text-secondary" style={{ marginTop: 8, marginBottom: 0 }}>
            {categoryMeta.description}
          </p>
        </div>

        <div className="policy-box">
          <div><span className="rule-id">{result.policy_rule_id}</span></div>
          <p style={{ margin: "6px 0 0" }}>{result.policy_reason}</p>
        </div>

        <p className="small text-tertiary" style={{ marginTop: 12, marginBottom: 0 }}>
          Model {result.model_version} · Evaluated {new Date(result.timestamp).toLocaleString()}
          {result.audit_persisted === false && (
            <> · <span style={{ color: "var(--risk-medium)" }}>⚠ Audit record not saved</span></>
          )}
        </p>
      </div>
    </div>
  );
}

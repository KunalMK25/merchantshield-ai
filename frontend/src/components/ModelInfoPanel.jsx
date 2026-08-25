export default function ModelInfoPanel({ modelInfo, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="panel" aria-busy="true" aria-label="Model information loading">
        <div className="panel-header">
          <span className="panel-title">Model &amp; Policy</span>
        </div>
        <div className="panel-body">
          {[80, 240, 160].map((w, i) => (
            <div key={i} className="skeleton" style={{ height: 18, width: `${w}px`, marginBottom: 10 }} aria-hidden="true" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Model &amp; Policy</span>
        </div>
        <div className="panel-body">
          <p className="small text-tertiary">
            {error ? `Unable to load model info: ${error}` : "Model metadata unavailable."}
          </p>
        </div>
      </div>
    );
  }

  const m = modelInfo;
  const t = m.test_metrics_at_frozen_threshold || {};
  const hasMetrics = t.precision !== undefined;

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Model &amp; Policy</span>
      </div>

      <div className="panel-body">
        {/* Model identity tiles */}
        <div className="model-card-grid">
          <div className="model-metric-tile">
            <div className="metric-tile-label">Model</div>
            <div className="metric-tile-value" style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}>
              {m.model_version || "—"}
            </div>
          </div>
          <div className="model-metric-tile">
            <div className="metric-tile-label">Threshold</div>
            <div className="metric-tile-value" style={{ fontFamily: "var(--font-mono)" }}>
              {m.decision_threshold ?? "—"}
            </div>
          </div>
          <div className="model-metric-tile">
            <div className="metric-tile-label">Features</div>
            <div className="metric-tile-value">
              {m.feature_count ?? "15"}
            </div>
          </div>
        </div>

        {/* Selection note */}
        <p className="model-selection-note">
          Threshold selected by minimising expected cost on a validation set, subject to
          ≥ 80% recall — not the default 0.5. Evaluated once on a held-out test set.
        </p>

        {/* Held-out test metrics */}
        {hasMetrics && (
          <>
            <div className="held-out-label">
              Held-out test metrics
              <span className="held-out-pill">Synthetic benchmark</span>
            </div>
            <div className="model-perf-grid">
              {[
                ["Precision",  t.precision !== undefined ? `${(t.precision * 100).toFixed(1)}%`  : "—"],
                ["Recall",     t.recall    !== undefined ? `${(t.recall    * 100).toFixed(1)}%`  : "—"],
                ["F1",         t.f1        !== undefined ? `${(t.f1        * 100).toFixed(1)}%`  : "—"],
                ["PR-AUC",     t.pr_auc    !== undefined ? (t.pr_auc).toFixed(3)                 : "—"],
                ["FP",         t.fp        !== undefined ? t.fp.toLocaleString()                 : "—"],
                ["FN",         t.fn        !== undefined ? t.fn.toLocaleString()                 : "—"],
              ].map(([label, value]) => (
                <div key={label} className="model-metric-tile">
                  <div className="metric-tile-label">{label}</div>
                  <div className="metric-tile-value" style={{ fontSize: 14, fontFamily: "var(--font-mono)" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Decision policy summary */}
        <hr className="divider" />
        <div className="held-out-label" style={{ marginBottom: 8 }}>Decision policy</div>
        <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }} aria-label="Decision policy rules">
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: "4px 8px", color: "var(--text-tertiary)", fontWeight: 700, borderBottom: "1px solid var(--border-subtle)", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.05em" }}>Rule</th>
              <th style={{ textAlign: "left", padding: "4px 8px", color: "var(--text-tertiary)", fontWeight: 700, borderBottom: "1px solid var(--border-subtle)", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.05em" }}>Condition</th>
              <th style={{ textAlign: "left", padding: "4px 8px", color: "var(--text-tertiary)", fontWeight: 700, borderBottom: "1px solid var(--border-subtle)", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.05em" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["CRITICAL_BLOCK",          "p ≥ 0.80",                     "BLOCK"],
              ["HIGH_STEP_UP",            "p ≥ 0.40",                     "STEP_UP_VERIFICATION"],
              ["MEDIUM_AMOUNT_ESCALATION","p ≥ 0.15 and amt ≥ ₹25,000",  "STEP_UP_VERIFICATION"],
              ["MEDIUM_MONITOR",          "p ≥ 0.15",                     "ALLOW_WITH_MONITORING"],
              ["LOW_ALLOW",               "(catch-all)",                   "ALLOW"],
            ].map(([rule, cond, action]) => (
              <tr key={rule}>
                <td style={{ padding: "5px 8px", fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--accent)", borderBottom: "1px solid var(--border-subtle)" }}>{rule}</td>
                <td style={{ padding: "5px 8px", color: "var(--text-secondary)", borderBottom: "1px solid var(--border-subtle)" }}>{cond}</td>
                <td style={{ padding: "5px 8px", fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-primary)", borderBottom: "1px solid var(--border-subtle)" }}>{action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

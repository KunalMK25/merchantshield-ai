export default function ModelInfoPanel({ modelInfo, isLoading, error }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Model &amp; policy metadata</span>
      </div>
      <div className="panel-body">
        {error && <div className="error-banner">⚠ Could not load model metadata: {error}</div>}

        {!error && isLoading && <div className="skeleton" style={{ height: 100 }} />}

        {!error && !isLoading && modelInfo && (
          <>
            <p className="small text-secondary" style={{ marginTop: 0 }}>
              {modelInfo.selection_rule}
            </p>
            <div className="model-metrics-grid" style={{ marginBottom: 16 }}>
              <div className="metric-tile">
                <div className="metric-tile-label">Version</div>
                <div className="metric-tile-value mono" style={{ fontSize: 14 }}>{modelInfo.model_version}</div>
              </div>
              <div className="metric-tile">
                <div className="metric-tile-label">Threshold</div>
                <div className="metric-tile-value">{modelInfo.decision_threshold}</div>
              </div>
              <div className="metric-tile">
                <div className="metric-tile-label">Features used</div>
                <div className="metric-tile-value">{modelInfo.feature_columns?.length ?? "—"}</div>
              </div>
            </div>

            <p className="field-label">Held-out test set performance</p>
            <div className="model-metrics-grid">
              {Object.entries(modelInfo.test_metrics_at_frozen_threshold || {})
                .filter(([k]) => ["precision", "recall", "f1", "fp", "fn"].includes(k))
                .map(([key, value]) => (
                  <div className="metric-tile" key={key}>
                    <div className="metric-tile-label">{key.toUpperCase()}</div>
                    <div className="metric-tile-value">
                      {typeof value === "number" && value <= 1 ? `${(value * 100).toFixed(1)}%` : value}
                    </div>
                  </div>
                ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

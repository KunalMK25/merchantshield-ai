export default function StatusBar({ health, modelInfo, healthError }) {
  const isOk = health?.status === "ok" && health?.model_loaded;

  return (
    <div className="status-bar" role="status">
      <span className="status-item">
        <span className={`status-dot ${isOk ? "ok" : "degraded"}`} />
        System: <strong>{healthError ? "Unreachable" : isOk ? "Operational" : "Degraded"}</strong>
      </span>

      <span className="status-item">
        Model: <strong className="mono">{health?.model_version || modelInfo?.model_version || "—"}</strong>
      </span>

      {modelInfo && (
        <>
          <span className="status-item">
            Decision threshold: <strong className="mono">{modelInfo.decision_threshold}</strong>
          </span>
          <span className="status-item">
            Test precision:{" "}
            <strong className="mono">
              {modelInfo.test_metrics_at_frozen_threshold?.precision != null
                ? `${(modelInfo.test_metrics_at_frozen_threshold.precision * 100).toFixed(1)}%`
                : "—"}
            </strong>
          </span>
          <span className="status-item">
            Test recall:{" "}
            <strong className="mono">
              {modelInfo.test_metrics_at_frozen_threshold?.recall != null
                ? `${(modelInfo.test_metrics_at_frozen_threshold.recall * 100).toFixed(1)}%`
                : "—"}
            </strong>
          </span>
        </>
      )}

      {!isOk && !healthError && (
        <span className="status-item" style={{ color: "var(--risk-critical)" }}>
          Model is not loaded — risk scoring is unavailable.
        </span>
      )}
      {healthError && (
        <span className="status-item" style={{ color: "var(--risk-critical)" }}>
          Cannot reach the API — check that the backend is running.
        </span>
      )}
    </div>
  );
}

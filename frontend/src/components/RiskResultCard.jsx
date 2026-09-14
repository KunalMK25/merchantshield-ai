import { tierFor, RISK_CATEGORY_META, formatPercent, formatTimestamp, shortId } from "../utils/presentation";

const CIRCUMFERENCE = 2 * Math.PI * 40; // r=40

function ScoreRing({ score, category }) {
  const filled = Math.min(Math.max(score || 0, 0), 100);
  const offset = CIRCUMFERENCE - (filled / 100) * CIRCUMFERENCE;

  return (
    <div className="score-ring-container">
      <div className="score-ring" role="img" aria-label={`Risk score ${filled} out of 100`}>
        <svg viewBox="0 0 96 96" width="96" height="96">
          <circle className="score-ring-bg" cx="48" cy="48" r="40" />
          <circle
            className={`score-ring-fg ${category || "LOW"}`}
            cx="48" cy="48" r="40"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="score-ring-label" aria-hidden="true">
          <span className="score-ring-number">{filled}</span>
          <span className="score-ring-denom">/100</span>
        </div>
      </div>
      <div className={`score-ring-category ${category || ""}`} aria-hidden="true">
        {category || "—"}
      </div>
    </div>
  );
}

// V2: Signal quality badge for historical context
function SignalQualityBadge({ signalQuality }) {
  if (!signalQuality) return null;

  const levelColors = {
    MINIMAL: "#d4524f",
    LIMITED: "#e8a536",
    MODERATE: "#a8b820",
    ESTABLISHED: "#4a9d6f",
  };

  const levelLabels = {
    MINIMAL: "No History",
    LIMITED: "Limited",
    MODERATE: "Moderate",
    ESTABLISHED: "Established",
  };

  return (
    <div
      style={{
        display: "inline-block",
        padding: "4px 8px",
        borderRadius: 4,
        backgroundColor: levelColors[signalQuality.level] || "#ccc",
        color: "white",
        fontSize: 11,
        fontWeight: 600,
        marginTop: 8,
      }}
      title={signalQuality.message}
    >
      {levelLabels[signalQuality.level] || signalQuality.level}
    </div>
  );
}

export default function RiskResultCard({ result, isLoading }) {
  if (isLoading) {
    return (
      <div className="panel" aria-busy="true" aria-label="Risk decision loading">
        <div className="panel-header">
          <span className="panel-title">Risk Decision</span>
        </div>
        <div className="panel-body">
          <div className="skeleton" style={{ height: 96, width: 96, borderRadius: "50%", margin: "0 auto 12px" }} aria-hidden="true" />
          <div className="skeleton" style={{ height: 56, marginBottom: 12 }} aria-hidden="true" />
          <div className="skeleton" style={{ height: 40, marginBottom: 8 }} aria-hidden="true" />
          <div className="skeleton" style={{ height: 40 }} aria-hidden="true" />
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Risk Decision</span>
        </div>
        <div className="empty-state" aria-live="polite">
          Select a demo scenario or enter a transaction to see the risk decision here.
        </div>
      </div>
    );
  }

  const tier   = tierFor(result.action);
  const catMeta = RISK_CATEGORY_META[result.risk_category] || {};
  const isLimitedContext = result.signal_quality && (result.signal_quality.level === "MINIMAL" || result.signal_quality.level === "LIMITED");

  return (
    <div className="panel" aria-live="polite" aria-atomic="true">
      <div className="panel-header">
        <span className="panel-title">Risk Decision</span>
        <span
          className="small text-tertiary mono"
          aria-label={`Transaction ID: ${result.transaction_id}`}
        >
          {shortId(result.transaction_id)}
        </span>
      </div>

      <div className="panel-body">
        {/* Signal quality warning (V2) */}
        {isLimitedContext && (
          <div
            style={{
              backgroundColor: "#fff3cd",
              border: "1px solid #ffc107",
              borderRadius: 4,
              padding: "8px 12px",
              marginBottom: 12,
              fontSize: 13,
              color: "#856404",
              lineHeight: 1.4,
            }}
            role="alert"
          >
            <strong>⚠️ {result.signal_quality.message}</strong>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              Behavioral patterns cannot be reliably assessed.
            </div>
          </div>
        )}

        {/* Score ring */}
        <ScoreRing score={result.risk_score} category={result.risk_category} />

        {/* Signal quality badge (V2) */}
        {result.signal_quality && (
          <div style={{ textAlign: "center" }}>
            <SignalQualityBadge signalQuality={result.signal_quality} />
          </div>
        )}

        {/* Action banner */}
        <div
          className={`action-banner ${tier.tier}`}
          role="region"
          aria-label={`Recommended action: ${result.action}`}
        >
          <div className="action-icon-circle" aria-hidden="true">
            {tier.icon}
          </div>
          <div>
            <div className="action-text-main">{tier.label}</div>
            <div className="action-mono">{result.action}</div>
            {tier.clarification && (
              <div className="action-clarification">{tier.clarification}</div>
            )}
          </div>
        </div>

        {/* Metric row - V2: Clarity on risk vs action */}
        <div className="metric-row">
          <div className="metric-tile">
            <div className="metric-tile-label">Fraud Probability</div>
            <div className="metric-tile-value">
              {formatPercent(result.fraud_probability)}
            </div>
          </div>
          <div className="metric-tile">
            <div className="metric-tile-label">Risk Level</div>
            <div className="metric-tile-value" style={{ fontSize: 13 }}>
              {result.risk_category || "—"}
            </div>
          </div>
          <div className="metric-tile">
            <div className="metric-tile-label">Risk Score</div>
            <div className="metric-tile-value">{result.risk_score ?? "—"}</div>
          </div>
        </div>

        {/* V2: Clarify score vs action relationship */}
        <div
          style={{
            fontSize: 12,
            color: "var(--text-secondary)",
            backgroundColor: "#f5f5f5",
            padding: 8,
            borderRadius: 4,
            marginBottom: 12,
            lineHeight: 1.5,
          }}
        >
          Risk level is informational. The action is determined by the merchant policy thresholds.
        </div>

        {/* Policy box */}
        <div className="policy-box">
          <div className="rule-id">{result.policy_rule_id}</div>
          <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
            {result.policy_reason}
          </div>
        </div>

        {/* Footer meta */}
        <div className="result-meta">
          <div>
            <strong>Model:</strong>{" "}
            <span className="mono">{result.model_version}</span>
            {"  ·  "}
            <strong>Threshold:</strong>{" "}
            <span className="mono">{result.threshold}</span>
          </div>
          {result.evaluated_at && (
            <div>
              <strong>Evaluated:</strong> {formatTimestamp(result.evaluated_at)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

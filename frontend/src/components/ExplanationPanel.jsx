import { friendlyFeatureLabel } from "../utils/presentation";

function ContributionBar({ contribution }) {
  const isIncrease = contribution.direction === "increases_risk";
  const widthPct   = Math.min(50, Math.abs(contribution.magnitude) * 40 + 6);

  return (
    <div className="contribution-bar-row">
      <div className="contribution-bar-label">
        <span>{friendlyFeatureLabel(contribution.feature)}</span>
        <span
          className={`contribution-bar-direction ${isIncrease ? "contribution-increase" : "contribution-decrease"}`}
        >
          {isIncrease ? "↑ increases risk" : "↓ decreases risk"}
        </span>
      </div>
      {/* Bar is visual only; label text above carries the meaning for AT */}
      <div className="contribution-bar-track" aria-hidden="true">
        <div className="contribution-bar-midline" />
        <div
          className={`contribution-bar-fill ${isIncrease ? "increase" : "decrease"}`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </div>
  );
}

// V2: Historical transaction timeline visualization
function TransactionHistory({ payload }) {
  if (!payload || !payload.prior_transactions || payload.prior_transactions.length === 0) {
    return null;
  }

  const priorTxns = payload.prior_transactions;
  const currentTxn = payload.transaction;

  // Extract amounts for visualization
  const amounts = priorTxns.map(t => t.amount).concat([currentTxn.amount]);
  const minAmount = Math.min(...amounts);
  const maxAmount = Math.max(...amounts);
  const range = maxAmount - minAmount || 1;

  // Calculate average of prior transactions
  const avgPrior = priorTxns.length > 0
    ? priorTxns.reduce((sum, t) => sum + t.amount, 0) / priorTxns.length
    : currentTxn.amount;

  const formatCurrency = (amt) => {
    if (amt >= 100000) return `₹${(amt / 100000).toFixed(1)}L`;
    if (amt >= 1000) return `₹${(amt / 1000).toFixed(1)}K`;
    return `₹${amt}`;
  };

  return (
    <div style={{ marginBottom: 16, paddingTop: 12, borderTop: "1px solid #eee" }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "#333", marginBottom: 8 }}>
        Transaction History ({priorTxns.length} prior)
      </div>

      {/* Timeline visualization */}
      <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 40, marginBottom: 8 }}>
        {priorTxns.map((t, i) => {
          const normalized = (t.amount - minAmount) / range;
          const height = Math.max(10, normalized * 35);
          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${height}px`,
                backgroundColor: "#c9d4e8",
                borderRadius: 2,
                cursor: "default",
                title: formatCurrency(t.amount),
              }}
              title={formatCurrency(t.amount)}
            />
          );
        })}
        {/* Current transaction - highlighted */}
        <div
          style={{
            flex: 1,
            height: `${Math.max(10, ((currentTxn.amount - minAmount) / range) * 35)}px`,
            backgroundColor: "#4a9d6f",
            borderRadius: 2,
            fontWeight: 600,
            title: formatCurrency(currentTxn.amount),
          }}
          title={formatCurrency(currentTxn.amount)}
        />
      </div>

      {/* Stats row */}
      <div style={{ fontSize: 11, color: "#666", display: "flex", gap: 16, justifyContent: "space-between" }}>
        <div>
          <strong>Average:</strong> {formatCurrency(avgPrior)}
        </div>
        <div>
          <strong>Current:</strong> {formatCurrency(currentTxn.amount)}
        </div>
        <div>
          <strong>Ratio:</strong> {(currentTxn.amount / avgPrior).toFixed(2)}x
        </div>
      </div>
    </div>
  );
}

export default function ExplanationPanel({ explanation, isLoading, payload }) {
  if (isLoading) {
    return (
      <div className="panel" aria-busy="true" aria-label="Explanation loading">
        <div className="panel-header">
          <span className="panel-title">Why this decision?</span>
        </div>
        <div className="panel-body">
          <div className="skeleton" style={{ height: 18, width: "60%", marginBottom: 12 }} aria-hidden="true" />
          {[0, 1, 2].map(i => (
            <div key={i} className="skeleton" style={{ height: 36, marginBottom: 8 }} aria-hidden="true" />
          ))}
        </div>
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Why this decision?</span>
        </div>
        <div className="empty-state">
          Explanation appears here after evaluation.
        </div>
      </div>
    );
  }

  const topContributions = explanation.contributions
    ? [...explanation.contributions]
        .sort((a, b) => b.magnitude - a.magnitude)
        .slice(0, 6)
    : [];

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Why this decision?</span>
      </div>

      <div className="panel-body">
        {/* V2: Transaction history timeline */}
        <TransactionHistory payload={payload} />

        {/* V2: Cold-start contextual signal (NOT a SHAP contribution) */}
        {explanation.cold_start_context && (
          <div
            style={{
              backgroundColor: "#f0f4f8",
              border: "1px solid #b3c5d9",
              borderLeft: "4px solid #b3c5d9",
              borderRadius: 4,
              padding: "8px 12px",
              marginBottom: 12,
              fontSize: 12,
              color: "#334455",
              lineHeight: 1.5,
              fontStyle: "italic",
            }}
            role="region"
            aria-label="Cold-start context"
          >
            <strong>⚠️ Historical Context Not Available:</strong>
            <div style={{ marginTop: 4, fontSize: 11 }}>
              {explanation.cold_start_context}
            </div>
            <div style={{ marginTop: 6, fontSize: 10, color: "#556677", fontStyle: "normal" }}>
              <em>Note: This is a contextual risk factor, not a model contribution (SHAP).</em>
            </div>
          </div>
        )}

        {/* Grounded reason list */}
        {explanation.header && (
          <p className="explanation-header">{explanation.header}</p>
        )}

        {explanation.reasons && explanation.reasons.length > 0 && (
          <ul className="reason-list" aria-label="Top risk factors">
            {explanation.reasons.map((r, i) => (
              <li key={i} className="reason-item">
                <span className="reason-bullet" aria-hidden="true" />
                <span>{r}</span>
              </li>
            ))}
          </ul>
        )}

        {/* SHAP contribution bars */}
        {topContributions.length > 0 && (
          <>
            <div className="factor-section-title">Feature contributions</div>
            {topContributions.map(c => (
              <ContributionBar key={c.feature} contribution={c} />
            ))}
          </>
        )}

        <p className="explanation-disclaimer">
          Bars show how much each factor pushed the model's estimate up or down
          for this specific transaction — not a general rule about what always
          matters most.
        </p>
      </div>
    </div>
  );
}

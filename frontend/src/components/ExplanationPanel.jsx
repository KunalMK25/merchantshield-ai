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

export default function ExplanationPanel({ explanation, isLoading }) {
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

import { friendlyFeatureLabel } from "../utils/presentation";

function ContributionBar({ contribution }) {
  const isIncrease = contribution.direction === "increases_risk";
  // scale bar width by magnitude relative to the largest contribution shown;
  // capped so a single outlier doesn't make every other bar invisible
  const widthPct = Math.min(50, Math.abs(contribution.magnitude) * 40 + 6);

  return (
    <div className="contribution-bar-row">
      <div className="contribution-bar-label">
        <span>{friendlyFeatureLabel(contribution.feature)}</span>
        <span style={{ color: isIncrease ? "var(--risk-high)" : "var(--risk-low)" }}>
          {isIncrease ? "↑ increases risk" : "↓ decreases risk"}
        </span>
      </div>
      {/* Bar is a visual representation only; the label text above carries the
          full meaning for assistive technologies */}
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
        <div className="panel-header"><span className="panel-title">Why this decision</span></div>
        <div className="panel-body">
          <div className="skeleton" style={{ height: 18, width: "60%", marginBottom: 12 }} aria-hidden="true" />
          {[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 36, marginBottom: 8 }} aria-hidden="true" />)}
        </div>
      </div>
    );
  }

  if (!explanation) return null;

  const { header, reasons, contributions } = explanation;
  const topContributions = (contributions || []).slice(0, 5);

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Why this decision</span>
      </div>
      <div className="panel-body">
        <p style={{ fontWeight: 600, marginTop: 0, marginBottom: 10 }}>{header}</p>

        <ul className="reason-list" style={{ marginBottom: 20 }}>
          {reasons.map((reason, i) => (
            <li key={i} className="reason-item">
              <span className="reason-icon" aria-hidden="true">•</span>
              <span>{reason}</span>
            </li>
          ))}
        </ul>

        {topContributions.length > 0 && (
          <>
            <p className="field-label" style={{ marginBottom: 10 }}>Factor breakdown</p>
            {topContributions.map((c) => <ContributionBar key={c.feature} contribution={c} />)}
            <p className="small text-tertiary" style={{ marginTop: 4, marginBottom: 0 }}>
              Bars show how much each factor pushed the model's estimate up or down for this
              specific transaction — not a general rule about what always matters most.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

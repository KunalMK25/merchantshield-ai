// Presentation-only helpers.
//
// IMPORTANT: nothing here changes or re-derives a business decision. The `action`
// string from the API is always displayed verbatim and is the source of truth.
// ACTION_TIERS below is a purely visual grouping for at-a-glance color/iconography
// (approved per the Phase 9 plan) -- it never substitutes for the real action value.

// NOTE on ALLOW_WITH_MONITORING's label: this system does not run any background
// monitoring job, alerting, or re-scoring after a decision is made -- "MONITORED"
// here describes a recommended review posture for this transaction (matching the
// real policy_reason text from the decision engine), not an active running service.
// Worded to avoid implying capability that doesn't exist (Phase 9.5 remediation).
export const ACTION_TIERS = {
  ALLOW: { tier: "tier-approve", icon: "✓", label: "Allow" },
  ALLOW_WITH_MONITORING: {
    tier: "tier-approve",
    icon: "✓",
    label: "Allow — Monitored",
    clarification: "Flagged for review; no automated monitoring is performed by this system.",
  },
  STEP_UP_VERIFICATION: { tier: "tier-review", icon: "⚠", label: "Review" },
  BLOCK: { tier: "tier-block", icon: "✕", label: "Block" },
};

export function tierFor(action) {
  return ACTION_TIERS[action] || { tier: "tier-review", icon: "?", label: action };
}

export const RISK_CATEGORY_META = {
  LOW: { icon: "●", description: "Minimal indicators of fraud risk." },
  MEDIUM: { icon: "●", description: "Some elevated risk indicators present." },
  HIGH: { icon: "▲", description: "Multiple strong risk indicators present." },
  CRITICAL: { icon: "■", description: "Very high confidence of fraudulent activity." },
};

// Human-friendly axis captions for SHAP contribution bars. These are display
// labels ONLY -- the actual explanatory sentence for each factor always comes
// verbatim from the backend's grounded `reasons` text, never fabricated here.
export const FEATURE_DISPLAY_LABELS = {
  amount: "Transaction amount",
  amount_zscore: "Amount vs. usual spending pattern",
  amount_vs_avg_ratio: "Amount vs. historical average",
  prior_txn_count: "Account transaction history",
  time_since_prev_txn_min: "Time since last transaction",
  velocity_5min: "Transactions in last 5 minutes",
  velocity_30min: "Transactions in last 30 minutes",
  velocity_60min: "Transactions in last 60 minutes",
  new_device_flag: "Device familiarity",
  new_geo_flag: "Location familiarity",
  failed_ratio_trailing10: "Recent failed-transaction rate",
  account_age_days: "Account age",
  hour_of_day: "Time of day",
  is_night: "Overnight transaction",
  day_of_week: "Day of week",
};

export function friendlyFeatureLabel(featureKey) {
  return FEATURE_DISPLAY_LABELS[featureKey] || featureKey;
}

export function formatCurrency(amount) {
  if (amount === null || amount === undefined || Number.isNaN(amount)) return "—";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(amount);
}

export function formatPercent(probability, digits = 1) {
  if (probability === null || probability === undefined) return "—";
  return `${(probability * 100).toFixed(digits)}%`;
}

export function formatTimestamp(isoString) {
  if (!isoString) return "—";
  try {
    return new Date(isoString).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return isoString;
  }
}

// Displays only the last 4 characters of long identifiers where full precision
// isn't needed for the reviewer to act -- avoids putting full raw IDs on screen
// unnecessarily (see Phase 9 requirement: don't expose sensitive identifiers
// unnecessarily). Full IDs remain available in the underlying data/audit log,
// this only affects compact table/summary display.
export function shortId(id, keep = 6) {
  if (!id) return "—";
  return id.length <= keep + 3 ? id : `…${id.slice(-keep)}`;
}

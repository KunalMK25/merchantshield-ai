import { useState } from "react";
import { DEMO_SCENARIOS, emptyTransactionDraft } from "../data/sampleScenarios";

const STATUS_OPTIONS  = ["success", "failed"];
const PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"];

/** Map tierHint string → CSS class suffix */
function tierClass(hint) {
  if (!hint) return "";
  const h = hint.toLowerCase();
  if (h.includes("critical")) return "tier-critical";
  if (h.includes("high"))     return "tier-high";
  if (h.includes("medium"))   return "tier-medium";
  if (h.includes("low"))      return "tier-low";
  return "";
}

/** Short badge label from tierHint */
function tierLabel(hint) {
  if (!hint) return "";
  const h = hint.toLowerCase();
  if (h.includes("critical")) return "Critical";
  if (h.includes("high"))     return "High";
  if (h.includes("medium"))   return "Medium";
  if (h.includes("low"))      return "Low";
  return hint;
}

export default function TransactionForm({ onSubmit, isSubmitting }) {
  const [mode, setMode]           = useState("scenario");
  const [selectedId, setSelectedId] = useState(null);
  const [draft, setDraft]         = useState(emptyTransactionDraft());
  const [formError, setFormError] = useState(null);

  function updateDraft(field, value) {
    setDraft(d => ({ ...d, [field]: value }));
  }

  // Scenario submit — payload matches /risk/evaluate contract exactly
  function handleScenarioSubmit(scenario) {
    setSelectedId(scenario.id);
    onSubmit({
      transaction: scenario.transaction,
      prior_transactions: scenario.prior_transactions,
      source: "demo",
    });
  }

  // Manual submit — identical validation to original TransactionForm
  function handleManualSubmit(e) {
    e.preventDefault();
    setFormError(null);

    if (
      !draft.transaction_id.trim() ||
      !draft.customer_id.trim()    ||
      !draft.merchant_id.trim()    ||
      !draft.device_id.trim()      ||
      !draft.geo_region.trim()
    ) {
      setFormError("Please fill in all required fields.");
      return;
    }
    const amount = Number(draft.amount);
    if (!Number.isFinite(amount) || amount <= 0) {
      setFormError("Amount must be a positive number.");
      return;
    }
    if (!draft.timestamp || !draft.account_created) {
      setFormError("Please provide both the transaction time and the account creation date.");
      return;
    }

    const transaction = {
      ...draft,
      amount,
      timestamp:       new Date(draft.timestamp).toISOString(),
      account_created: new Date(draft.account_created).toISOString(),
    };
    // No prior_transactions in manual mode — scores as first-ever observed
    // transaction (documented, correct behavior per docs/api.md).
    onSubmit({ transaction, prior_transactions: [], source: "manual" });
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Transaction Evaluation</span>
      </div>

      <div className="panel-body">
        {/* ── Mode toggle ───────────────────────────────────── */}
        <div className="mode-toggle" role="group" aria-label="Evaluation mode">
          <button
            type="button"
            className={`btn btn-ghost ${mode === "scenario" ? "active" : ""}`}
            onClick={() => setMode("scenario")}
            aria-pressed={mode === "scenario"}
          >
            Demo scenarios
          </button>
          <button
            type="button"
            className={`btn btn-ghost ${mode === "manual" ? "active" : ""}`}
            onClick={() => setMode("manual")}
            aria-pressed={mode === "manual"}
          >
            Manual entry
          </button>
        </div>

        {/* ── Scenario selection ────────────────────────────── */}
        {mode === "scenario" && (
          <>
            <div className="demo-badge" aria-label="Synthetic demo data — not real transactions">
              <span aria-hidden="true">⚗</span>
              Synthetic demo data — not real transactions
            </div>

            <div className="scenario-list" role="list">
              {DEMO_SCENARIOS.map(s => {
                const tc = tierClass(s.tierHint);
                return (
                  <button
                    key={s.id}
                    type="button"
                    role="listitem"
                    className={`scenario-card ${tc} ${selectedId === s.id ? "selected" : ""}`}
                    onClick={() => handleScenarioSubmit(s)}
                    disabled={isSubmitting}
                    aria-pressed={selectedId === s.id}
                    aria-label={`${s.label} — expected ${tierLabel(s.tierHint)} risk`}
                  >
                    <div className="scenario-card-header">
                      <span className="scenario-card-label">{s.label}</span>
                      {s.tierHint && (
                        <span className={`scenario-tier-badge ${tc}`} aria-hidden="true">
                          {tierLabel(s.tierHint)}
                        </span>
                      )}
                    </div>
                    {s.description && (
                      <div className="scenario-card-desc">{s.description}</div>
                    )}
                  </button>
                );
              })}
            </div>
          </>
        )}

        {/* ── Manual entry form ─────────────────────────────── */}
        {mode === "manual" && (
          <form onSubmit={handleManualSubmit} aria-label="Manual transaction entry">
            <p className="field-hint" style={{ marginBottom: 12 }}>
              Scored as this customer's first observed transaction (no prior history supplied).
            </p>

            {formError && (
              <div className="error-banner" role="alert" aria-live="assertive">
                <span aria-hidden="true">⚠</span> {formError}
              </div>
            )}

            <div className="field-section-title">Transaction</div>

            <div className="field-group">
              <label className="field-label" htmlFor="txn-transaction-id">Transaction ID</label>
              <input
                id="txn-transaction-id"
                className="field-input"
                value={draft.transaction_id}
                onChange={e => updateDraft("transaction_id", e.target.value)}
              />
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label" htmlFor="txn-amount">Amount (INR)</label>
                <input
                  id="txn-amount"
                  className="field-input"
                  type="number"
                  min="0.01"
                  step="0.01"
                  placeholder="e.g. 5000"
                  value={draft.amount}
                  onChange={e => updateDraft("amount", e.target.value)}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="txn-payment-method">Payment method</label>
                <select
                  id="txn-payment-method"
                  className="field-select"
                  value={draft.payment_method}
                  onChange={e => updateDraft("payment_method", e.target.value)}
                >
                  {PAYMENT_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label" htmlFor="txn-merchant-category">Merchant category</label>
                <input
                  id="txn-merchant-category"
                  className="field-input"
                  value={draft.merchant_category}
                  onChange={e => updateDraft("merchant_category", e.target.value)}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="txn-status">Status</label>
                <select
                  id="txn-status"
                  className="field-select"
                  value={draft.status}
                  onChange={e => updateDraft("status", e.target.value)}
                >
                  {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>

            <div className="field-section-title">Customer &amp; Device</div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label" htmlFor="txn-customer-id">Customer ID</label>
                <input
                  id="txn-customer-id"
                  className="field-input"
                  value={draft.customer_id}
                  onChange={e => updateDraft("customer_id", e.target.value)}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="txn-merchant-id">Merchant ID</label>
                <input
                  id="txn-merchant-id"
                  className="field-input"
                  value={draft.merchant_id}
                  onChange={e => updateDraft("merchant_id", e.target.value)}
                />
              </div>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label" htmlFor="txn-device-id">Device ID</label>
                <input
                  id="txn-device-id"
                  className="field-input"
                  value={draft.device_id}
                  onChange={e => updateDraft("device_id", e.target.value)}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="txn-geo-region">Geo region</label>
                <input
                  id="txn-geo-region"
                  className="field-input"
                  value={draft.geo_region}
                  onChange={e => updateDraft("geo_region", e.target.value)}
                />
              </div>
            </div>

            <div className="field-section-title">Timing</div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label" htmlFor="txn-timestamp">Transaction time</label>
                <input
                  id="txn-timestamp"
                  className="field-input"
                  type="datetime-local"
                  value={draft.timestamp}
                  onChange={e => updateDraft("timestamp", e.target.value)}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="txn-account-created">Account created</label>
                <input
                  id="txn-account-created"
                  className="field-input"
                  type="datetime-local"
                  value={draft.account_created}
                  onChange={e => updateDraft("account_created", e.target.value)}
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-block"
              disabled={isSubmitting}
              style={{ marginTop: 8 }}
            >
              {isSubmitting ? "Evaluating…" : "Evaluate transaction"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

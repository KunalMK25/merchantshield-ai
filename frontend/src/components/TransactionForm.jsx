import { useState } from "react";
import { DEMO_SCENARIOS, emptyTransactionDraft } from "../data/sampleScenarios";

const STATUS_OPTIONS = ["success", "failed"];
const PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"];

export default function TransactionForm({ onSubmit, isSubmitting }) {
  const [mode, setMode] = useState("scenario"); // "scenario" | "manual"
  const [selectedScenarioId, setSelectedScenarioId] = useState(null);
  const [draft, setDraft] = useState(emptyTransactionDraft());
  const [formError, setFormError] = useState(null);

  function updateDraft(field, value) {
    setDraft((d) => ({ ...d, [field]: value }));
  }

  function handleScenarioSubmit(scenario) {
    setSelectedScenarioId(scenario.id);
    onSubmit({
      transaction: scenario.transaction,
      prior_transactions: scenario.prior_transactions,
      source: "demo",
    });
  }

  function handleManualSubmit(e) {
    e.preventDefault();
    setFormError(null);

    if (!draft.transaction_id.trim() || !draft.customer_id.trim() || !draft.merchant_id.trim()
        || !draft.device_id.trim() || !draft.geo_region.trim()) {
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
      timestamp: new Date(draft.timestamp).toISOString(),
      account_created: new Date(draft.account_created).toISOString(),
    };

    // No prior_transactions in manual mode -- per docs/api.md, this scores the
    // transaction as the customer's first-ever observed transaction, which is
    // documented, correct behavior, not a workaround.
    onSubmit({ transaction, prior_transactions: [], source: "manual" });
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Evaluate a transaction</span>
      </div>
      <div className="panel-body">
        <div className="mode-toggle">
          <button
            type="button"
            className={`btn btn-ghost ${mode === "scenario" ? "active" : ""}`}
            onClick={() => setMode("scenario")}
          >
            Demo scenarios
          </button>
          <button
            type="button"
            className={`btn btn-ghost ${mode === "manual" ? "active" : ""}`}
            onClick={() => setMode("manual")}
          >
            Manual entry
          </button>
        </div>

        {mode === "scenario" && (
          <div>
            <span className="demo-badge">Synthetic demo data — not real transactions</span>
            <div className="scenario-list">
              {DEMO_SCENARIOS.map((scenario) => (
                <button
                  key={scenario.id}
                  type="button"
                  className={`scenario-card ${selectedScenarioId === scenario.id ? "selected" : ""}`}
                  onClick={() => handleScenarioSubmit(scenario)}
                  disabled={isSubmitting}
                >
                  <div className="scenario-card-label">{scenario.label}</div>
                  <div className="scenario-card-hint">{scenario.tierHint}</div>
                  <div className="scenario-card-desc">{scenario.description}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === "manual" && (
          <form onSubmit={handleManualSubmit}>
            <p className="field-hint" style={{ marginBottom: 12 }}>
              Scored as this customer's first observed transaction (no prior history supplied).
            </p>

            {formError && <div className="error-banner">⚠ {formError}</div>}

            <div className="field-group">
              <label className="field-label">Transaction ID</label>
              <input className="field-input" value={draft.transaction_id}
                     onChange={(e) => updateDraft("transaction_id", e.target.value)} />
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Customer ID</label>
                <input className="field-input" value={draft.customer_id}
                       onChange={(e) => updateDraft("customer_id", e.target.value)} />
              </div>
              <div className="field-group">
                <label className="field-label">Merchant ID</label>
                <input className="field-input" value={draft.merchant_id}
                       onChange={(e) => updateDraft("merchant_id", e.target.value)} />
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Merchant category</label>
              <input className="field-input" value={draft.merchant_category}
                     onChange={(e) => updateDraft("merchant_category", e.target.value)} />
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Amount (INR)</label>
                <input className="field-input" type="number" min="0.01" step="0.01" value={draft.amount}
                       onChange={(e) => updateDraft("amount", e.target.value)} />
              </div>
              <div className="field-group">
                <label className="field-label">Payment method</label>
                <select className="field-select" value={draft.payment_method}
                        onChange={(e) => updateDraft("payment_method", e.target.value)}>
                  {PAYMENT_METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Device ID</label>
                <input className="field-input" value={draft.device_id}
                       onChange={(e) => updateDraft("device_id", e.target.value)} />
              </div>
              <div className="field-group">
                <label className="field-label">Geo region</label>
                <input className="field-input" value={draft.geo_region}
                       onChange={(e) => updateDraft("geo_region", e.target.value)} />
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Status</label>
              <select className="field-select" value={draft.status}
                      onChange={(e) => updateDraft("status", e.target.value)}>
                {STATUS_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Transaction time</label>
                <input className="field-input" type="datetime-local" value={draft.timestamp}
                       onChange={(e) => updateDraft("timestamp", e.target.value)} />
              </div>
              <div className="field-group">
                <label className="field-label">Account created</label>
                <input className="field-input" type="datetime-local" value={draft.account_created}
                       onChange={(e) => updateDraft("account_created", e.target.value)} />
              </div>
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={isSubmitting}>
              {isSubmitting ? "Evaluating…" : "Evaluate transaction"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

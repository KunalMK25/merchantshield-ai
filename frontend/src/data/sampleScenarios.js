// Demo scenarios for MerchantShield AI.
//
// IMPORTANT: these are entirely SYNTHETIC, hand-built transactions for demoing the
// dashboard -- they are NOT real transactions and are NOT pulled from any live
// database. Each one bundles a transaction plus a short prior-transaction history
// for the same fictitious customer, matching the /risk/evaluate request contract
// documented in docs/api.md (see the "prior_transactions" design note there for why
// history is needed: the feature pipeline computes velocity/behavioral features from
// a customer's recent transactions, which a real-time API can't look up on its own
// in this prototype).
//
// The probabilities each scenario produces were verified at build time by running
// them through the real, frozen model -- but this file itself contains no model
// logic; it is static request payloads only.

function buildPriorHistory(customerId, { n = 8, device = "dev_known", geo = "region_03", amount = 520 } = {}) {
  return Array.from({ length: n }, (_, i) => ({
    transaction_id: `demo_ctx_${customerId}_${i}`,
    customer_id: customerId,
    merchant_id: "merch_demo_001",
    merchant_category: "electronics",
    timestamp: `2026-05-${String(10 + i).padStart(2, "0")}T10:00:00Z`,
    amount: amount + i * 5,
    device_id: device,
    geo_region: geo,
    payment_method: "card",
    status: "success",
    account_created: "2024-01-01T00:00:00Z",
  }));
}

function buildScenario({ id, label, tierHint, description, customerId, amount, device, geo, timestamp }) {
  const priorTransactions = buildPriorHistory(customerId);
  const transaction = {
    transaction_id: `demo_txn_${id}`,
    customer_id: customerId,
    merchant_id: "merch_demo_002",
    merchant_category: "electronics",
    timestamp: timestamp || "2026-05-20T10:05:00Z",
    amount,
    device_id: device,
    geo_region: geo,
    payment_method: "card",
    status: "success",
    account_created: "2024-01-01T00:00:00Z",
  };
  return { id, label, tierHint, description, transaction, prior_transactions: priorTransactions };
}

export const DEMO_SCENARIOS = [
  buildScenario({
    id: "low",
    label: "Routine repeat purchase",
    tierHint: "Typically scores LOW",
    description: "Same device, same region, an amount consistent with this customer's usual spending.",
    customerId: "demo_cust_low",
    amount: 515.0,
    device: "dev_known",
    geo: "region_03",
  }),
  buildScenario({
    id: "medium",
    label: "Unfamiliar region, known device",
    tierHint: "Typically scores MEDIUM",
    description: "A larger-than-usual purchase from a region this customer hasn't transacted in before, but on their known device.",
    customerId: "demo_cust_medium",
    amount: 3500.0,
    device: "dev_known",
    geo: "region_15",
  }),
  buildScenario({
    id: "high",
    label: "New device, unfamiliar region",
    tierHint: "Typically scores HIGH",
    description: "A moderate purchase, but from both a device and a region never seen for this customer before.",
    customerId: "demo_cust_high",
    amount: 2500.0,
    device: "dev_new_high",
    geo: "region_15",
  }),
  buildScenario({
    id: "critical",
    label: "Large purchase, new device and region",
    tierHint: "Typically scores CRITICAL",
    description: "A very large purchase far outside this customer's normal spending, from a device and region never seen before.",
    customerId: "demo_cust_critical",
    amount: 45000.0,
    device: "dev_never_seen",
    geo: "region_19",
  }),
];

export function emptyTransactionDraft() {
  const now = new Date().toISOString().slice(0, 16); // yyyy-MM-ddTHH:mm for <input type=datetime-local>
  return {
    transaction_id: "",
    customer_id: "",
    merchant_id: "",
    merchant_category: "electronics",
    timestamp: now,
    amount: "",
    device_id: "",
    geo_region: "",
    payment_method: "card",
    status: "success",
    account_created: "",
  };
}

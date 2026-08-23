# API Documentation (Phase 8)

## Request → response flow

```
HTTP request
    ↓
Pydantic validation (backend/schemas/transaction.py)
    ↓
Feature engineering (ml/features/build_features.py -- Phase 1, UNCHANGED)
    ↓
LightGBM inference (frozen model artifact, loaded once at startup)
    ↓
Risk score/category (ml/evaluation/risk_scoring.py -- Phase 6, UNCHANGED)
    ↓
SHAP explanation (ml/evaluation/explainability.py -- Phase 6, UNCHANGED)
    ↓
Decision engine (ml/evaluation/decision_engine.py -- Phase 7, UNCHANGED)
    ↓
Audit persistence (backend/services/audit_service.py -- new this phase)
    ↓
Structured JSON response
```

`backend/services/risk_service.py` is the only file that wires these together. It
contains no policy numbers, no scoring formulas, and no decision rules of its own —
every one of those is imported from the module that already owns it.

## A real design decision worth calling out: feature engineering needs history

`build_features()` computes velocity, historical-average deviation, and
device/geo novelty from a **customer's transaction history**, not from a single
isolated transaction. This prototype has no live transaction database, so the API
can't look up "this customer's last 10 transactions" on its own.

**Resolution:** `RiskRequest` accepts the transaction to be scored plus an optional
`prior_transactions` list — the same customer's recent raw transaction history,
supplied by the caller. The API concatenates them and runs the *exact same*
`build_features()` used throughout training/evaluation, then reads off the row for
the transaction being scored. If `prior_transactions` is omitted, the transaction is
scored as this customer's first-ever observed transaction (which is also exactly how
`build_features()` already handles a brand-new customer during training).

This was the correct call rather than writing a second, simplified feature
computation for "live" scoring — a second implementation would be an unmonitored
second copy of Phase 1's leakage-sensitive logic, which is exactly the kind of
duplication Phase 8 was told to avoid.

**Guardrails enforced by `RiskRequest` validation** (not by the model or decision
engine — this is pure request-shape validation):
- every `prior_transactions[i]` must have the same `customer_id` as the transaction
  being scored,
- every `prior_transactions[i].timestamp` must be strictly before the scored
  transaction's timestamp — feeding same-or-future "prior" transactions would leak
  information a real-time system would never actually have, so it's rejected as a
  malformed request (422), not silently accepted.

## Endpoint table

| Method | Endpoint | Purpose | Key response fields |
|---|---|---|---|
| GET | `/health` | Liveness/readiness, reports whether the model loaded | `status`, `model_loaded`, `model_version` |
| GET | `/model/info` | Frozen model metadata (read-only, no inference) | `decision_threshold`, `feature_columns`, validation/test metrics |
| POST | `/risk/score` | Probability + risk score/category only | `fraud_probability`, `risk_score`, `risk_category` |
| POST | `/risk/explain` | Probability + full SHAP explanation | `reasons`, `contributions`, `additivity_check_passed` |
| POST | `/risk/evaluate` | Full pipeline: score + explain + decide + audit | `action`, `policy_rule_id`, `reasons`, `audit_persisted` |
| GET | `/audit-log` | Recent persisted decision records | list of audit entries |

`/risk/score` and `/risk/explain` are intentionally lightweight and **do not write
audit records** — they're for exploration/what-if use, not decisions on real
transactions. Only `/risk/evaluate` runs the decision engine and persists an audit
entry, because only it represents an actual system decision.

## Example request/response (real transaction, real model)

Request to `/risk/evaluate`:
```json
{
  "transaction": {
    "transaction_id": "txn_live_test_001",
    "customer_id": "cust_00042",
    "merchant_id": "merch_0099",
    "merchant_category": "electronics",
    "timestamp": "2026-05-20T10:05:00Z",
    "amount": 45000.0,
    "device_id": "dev_BRAND_NEW",
    "geo_region": "region_19",
    "payment_method": "card",
    "status": "success",
    "account_created": "2025-01-01T00:00:00Z"
  },
  "prior_transactions": [ /* 6 prior transactions for cust_00042, all card purchases ~Rs500-560 */ ],
  "source": "manual"
}
```

Response (actual, unedited):
```json
{
  "request_id": "d9c5acd2-7f2f-47a7-be5e-68a82cfe37ed",
  "transaction_id": "txn_live_test_001",
  "model_version": "lgbm_v1",
  "fraud_probability": 0.9997,
  "threshold": 0.4,
  "risk_score": 100,
  "risk_category": "CRITICAL",
  "action": "BLOCK",
  "policy_rule_id": "CRITICAL_BLOCK",
  "policy_reason": "Fraud probability is at or above the configured blocking boundary (0.8), where validation precision is high enough that most flagged transactions at this confidence level are genuinely fraudulent.",
  "explanation_header": "Why this transaction was flagged:",
  "reasons": [
    "Transaction amount is 85.7x the customer's historical average amount, which increased risk",
    "Transaction amount (Rs45,000) increased risk",
    "Device has not been seen previously for this customer, which increased risk",
    "Geographic location is new for this customer, which increased risk",
    "Account age is 504 days, which increased risk"
  ],
  "timestamp": "2026-08-22T06:51:08.575938+00:00",
  "audit_persisted": true,
  "audit_error": null
}
```

## Error responses

All errors return structured JSON, never a raw stack trace:

| Failure | Status | Body shape |
|---|---|---|
| Malformed JSON / missing field / invalid enum / bad type | 422 | `{"error": "validation_error", "detail": "..."}` |
| `prior_transactions` violates leakage/consistency rules | 422 | `{"error": "validation_error", "detail": "..."}` |
| Feature engineering fails on valid-shaped input | 422 | `{"detail": "Feature engineering failed: ..."}` |
| Decision engine rejects input | 422 | `{"detail": "Decision engine rejected input: ..."}` |
| Model not loaded (down/corrupt at startup) | 503 | `{"detail": "Model is not loaded; service is degraded."}` |
| Unexpected internal failure inside `/risk/evaluate`'s own try/except | 500 | `{"detail": "Internal risk evaluation failure: ..."}` |
| Unexpected internal failure anywhere else (true bug) | 500 | `{"error": "internal_error", "detail": "An unexpected error occurred."}` |
| Audit database write fails | **200** | Decision still returned; `audit_persisted: false`, `audit_error: "..."` |

The last row is deliberate: a database hiccup should never cause the merchant to
lose an already-computed fraud decision. The decision is the valuable, expensive-to-
recompute artifact; the audit write is a side effect that can fail independently and
be retried/reconciled later without blocking the response.

## What gets audited, and what deliberately doesn't

Persisted per decision: `request_id`, `transaction_id`, `model_version`,
`fraud_probability`, `threshold`, `risk_score`, `risk_category`, `action`,
`policy_rule_id`, `policy_reason`, decision `timestamp`, `source` (`"demo"` or
`"manual"` — request-origin metadata only, see below), and the SHAP-derived
`top_reasons` (stored in its own column, separate from the decision fields — the
decision is the durable governance fact, the explanation is attached narrative).

**`source` field:** set from the request body's optional `source` field (defaults to
`"manual"` if omitted, so any direct API caller that doesn't know about the
dashboard's demo concept is correctly treated as a real entry). This lets a reviewer
of `/audit-log` tell dashboard demo-scenario evaluations apart from hand-entered or
externally-integrated ones. It is purely descriptive: the decision engine never reads
it and it has no bearing on the computed action.

**Not persisted:** `customer_id`, `device_id`, `geo_region`, `payment_method`, or any
other raw transaction context. The audit trail's job is proving what the risk system
decided and why — not duplicating a copy of the merchant's transaction data.

## Interactive docs

FastAPI auto-generates OpenAPI docs at `/docs` (Swagger UI) and `/redoc` from the
endpoint descriptions and Pydantic schemas above — no separate documentation to keep
in sync.

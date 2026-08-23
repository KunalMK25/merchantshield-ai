# Frontend — Design Notes (Phase 9)

## Stack

React 19 + Vite, plain CSS (no UI component library, no Tailwind). Dependencies are
exactly what `npm create vite -- --template react` installs, nothing added — the
brief asked for "a simple, maintainable dependency set," and the dashboard's actual
needs (forms, a couple of fetch calls, conditional rendering) don't justify a
component library or CSS framework.

## Decoupling from the backend

The frontend never imports Python, model files, policy thresholds, or
feature-engineering code. Everything it knows about the system comes through
`src/api/client.js`, which speaks HTTP/JSON against the exact contract in
`docs/api.md`. If the decision engine's rules change, the frontend needs zero
changes unless the *response shape* changes — verified by construction (there is no
Python import anywhere under `frontend/`), not just by convention.

The one place this could go wrong is `src/utils/presentation.js`, which maps the
real `action` values to a display tier and gives feature keys human-readable
captions. This is presentation-only: `ACTION_TIERS` never substitutes for the real
`action` string (always shown verbatim alongside the tier), and
`FEATURE_DISPLAY_LABELS` only supplies an axis caption — the actual explanatory
sentence for every reason shown on screen still comes verbatim from the backend's
grounded `reasons` text, never fabricated client-side.

## Why /risk/evaluate AND /risk/explain are both called

`RiskEvaluateResponse` (the authoritative decision) includes human-readable
`reasons` but not the raw per-feature SHAP `contributions` needed for the bar-chart
breakdown — that level of detail only exists in `RiskExplainResponse`. Rather than
inventing a new endpoint or a client-side re-derivation of SHAP values, the
dashboard calls both existing endpoints in parallel for the same payload:
`/risk/evaluate` for the actual decision + audit write, `/risk/explain` for the
contribution bars. Both reflect the same deterministic computation (confirmed by
`test_evaluate_reasons_are_grounded_in_real_shap_output` in the Phase 8 test suite),
so this is redundancy-in-computation for a richer UI, not two different sources of
truth.

## Why prior_transactions matters here too

The feature-engineering pipeline needs a customer's transaction history to compute
velocity/behavioral features (see `docs/api.md`). The dashboard's demo scenarios
each bundle a realistic short history for a fictitious customer so the model has
something to compare against; manual entry mode omits it and is scored as the
customer's first-ever transaction (documented, correct behavior, not a limitation
being hidden).

## Demo scenario data

`src/data/sampleScenarios.js` contains four **synthetic, clearly-labeled** demo
transactions (labeled "Synthetic demo data — not real transactions" in the UI).
Each scenario's parameters (amount, device, geo pattern) were verified at
development time by running them through the real, frozen model — not guessed:

| Scenario | Verified probability | Resulting action |
|---|---|---|
| Routine repeat purchase | 0.0001 | ALLOW |
| Unfamiliar region, known device | 0.3551 | ALLOW_WITH_MONITORING |
| New device, unfamiliar region | 0.6583 | STEP_UP_VERIFICATION |
| Large purchase, new device+region | 0.9986 | BLOCK |

The data file itself contains no model logic — it's a static array of request
payloads. The verification was a one-time dev-time check (same pattern used for
Phase 7/8 test fixtures), not a runtime dependency.

## Audit source tagging (Phase 9.5)

Every `/risk/evaluate` request from the dashboard now includes a `source` field:
`"demo"` when submitted via a scenario card, `"manual"` when submitted via the
manual-entry form. This is purely descriptive metadata passed straight through to
the backend's audit record (see `docs/api.md`) — it has no effect on the computed
decision. The audit log table displays a "Source" column so a reviewer can tell a
synthetic demo evaluation apart from a real hand-entered one at a glance.

## Action-tier display mapping (approved design)

The real `action` value is always shown, verbatim, in every view (risk card, audit
table). A secondary visual grouping adds color/iconography for at-a-glance scanning:

| Real action | Display label | Visual tier | Icon |
|---|---|---|---|
| `ALLOW` | Allow | Approve (green) | ✓ |
| `ALLOW_WITH_MONITORING` | Allow — Monitored | Approve (green) | ✓ |
| `STEP_UP_VERIFICATION` | Review | Review (amber) | ⚠ |
| `BLOCK` | Block | Block (red) | ✕ |

**`ALLOW_WITH_MONITORING` wording (Phase 9.5 remediation):** the original label
"Approve (monitored)" was flagged in review as potentially implying an active
background monitoring service exists — it doesn't; this system runs no scheduled
job, alerting, or re-scoring after a decision is made. The label was changed to
"Allow — Monitored" and a short clarification line ("Flagged for review; no
automated monitoring is performed by this system.") is shown alongside it on the
risk result card, so the wording describes a recommended review posture rather than
a running capability.

Risk category (LOW/MEDIUM/HIGH/CRITICAL) and severity are never conveyed by color
alone — every badge/chip pairs color with an icon and the text label itself.

## Making SHAP legible to a non-ML user

Two layers, both grounded in real backend output:
1. **Grounded sentences** (`reasons`) — plain-English, backend-generated text like
   "Device has not been seen previously for this customer, which increased risk."
   This is the primary explanation surface.
2. **Contribution bars** — a diverging bar per top factor, colored by direction
   (red = increases risk, green = decreases risk), labeled with a friendly caption
   ("Amount vs. historical average" instead of `amount_vs_avg_ratio`) rather than
   the raw feature key or a bare SHAP float. A caption under the bars explicitly
   states these are per-transaction, not a general importance ranking — directly
   reflecting the "local, not global" limitation documented in
   `docs/explainability.md`.

No raw feature keys or unexplained numbers are shown to the user anywhere in the
primary decision flow.

## Sensitive identifiers

The audit table shows a shortened transaction ID (`shortId()`, keeps the last 6
characters) rather than the full raw ID, and never displays `customer_id`,
`device_id`, or `geo_region` at all — matching what the backend audit log itself
withholds (see `docs/api.md`, "What gets audited"). The risk result card does show
the full `transaction_id` of the transaction just evaluated, since the person just
entered or selected it themselves — that's not exposing anything they didn't
already have.

## Failure/degraded-state handling

Each data source (`/health`, `/model/info`, `/risk/evaluate`+`/risk/explain`,
`/audit-log`) has its own independent loading/error state — a failure in one (e.g.
audit log unreachable) never blanks out the rest of the dashboard. `StatusBar`
surfaces "Degraded" / "Cannot reach the API" distinctly from a healthy state.
`RiskResultCard` shows `audit_persisted: false` inline if the backend's own
graceful-degradation path (Phase 8) kicks in — the frontend doesn't need special
handling for that case since the backend already returns 200 with a decision either
way.

## Serving

`npm run build` produces `frontend/dist/`, mounted by FastAPI via `StaticFiles` at
`/` (same-origin, no CORS needed in production). The old root JSON
service-identification payload moved to `/api`. CORS middleware was added to
`backend/main.py` only to support running `npm run dev` against the live API during
development on a different port — a config addition, not a decision-logic change.

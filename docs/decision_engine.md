# Decision Engine — Design Notes (Phase 7)

## Why the decision layer is rule-based, not ML-based

The LightGBM model's only job is estimating `P(fraud)` for a transaction. What the
system *does* with that number — allow it, watch it, ask for more verification, or
block it — is a business policy decision, not a pattern-recognition problem. Rule-based
logic is used here because:

- **Auditability.** Every action must be traceable to an exact, readable condition
  ("probability >= 0.80") that a risk analyst, auditor, or reviewer can check by hand.
  A second ML model choosing the action would just move the "why" question one level
  deeper without answering it.
- **Reproducibility.** `evaluate_policy(probability, amount)` is a pure function —
  same inputs, same output, forever, unless someone deliberately edits the rule table.
  This is required for the audit trail (Phase 10) to mean anything.
- **Bounded, reviewable change.** Changing when the system blocks a transaction should
  be a one-line, reviewable diff to `policy.py`, not a retraining run with all the
  attendant uncertainty about what else might shift.
- **Separation of concerns matches the project's core architecture**: ML estimates
  probability → deterministic logic converts probability to a risk category (Phase 6)
  → separate deterministic logic decides the action (this phase). Three independent,
  independently-auditable layers.

## The exact action policy

Rules are evaluated in order; the **first match wins**. Source of truth is
`ml/evaluation/policy.py`.

| Order | Rule ID | Condition | Action |
|---|---|---|---|
| 1 | `CRITICAL_BLOCK` | `probability >= 0.80` | `BLOCK` |
| 2 | `HIGH_STEP_UP` | `probability >= 0.40` | `STEP_UP_VERIFICATION` |
| 3 | `MEDIUM_AMOUNT_ESCALATION` | `probability >= 0.15` AND `amount >= Rs25,000` | `STEP_UP_VERIFICATION` |
| 4 | `MEDIUM_MONITOR` | `probability >= 0.15` | `ALLOW_WITH_MONITORING` |
| 5 | `LOW_ALLOW` | (catch-all) | `ALLOW` |

**Where each number comes from:**
- `0.40` (`DECISION_THRESHOLD`) is **frozen** from Phase 5's cost-minimization
  analysis on the validation set. Not re-derived here.
- `0.80` (`CRITICAL_MIN`) is a **new policy assumption** introduced in this phase,
  chosen using evidence already in hand: Phase 5's threshold sweep showed validation
  precision of 0.941 at threshold 0.80 — i.e., at that confidence level the model is
  right about fraud the overwhelming majority of the time, which is the bar set for
  recommending an outright block rather than a lower-friction step-up check. This is
  explicitly a policy choice layered on top of the model, not a re-run of the cost
  optimization from Phase 5.
- `0.15` (`LOW_MAX`) and `Rs25,000` (`LARGE_AMOUNT_CUTOFF`) are simple, documented
  policy assumptions — not fit to data — reflecting "don't add friction for
  near-zero-risk transactions" and "a very large transaction deserves a second look
  even at moderate model confidence." Both are easy to challenge and change; that's
  the point of keeping the policy this small.

**Why the risk-score bucket (LOW/MEDIUM/HIGH/CRITICAL, 0–100) is not used for
action selection:** it's a display/audit convenience computed independently from
probability by `risk_scoring.py` (Phase 6). Using it to drive actions would create a
second, redundant threshold system and make "why did this transaction get blocked"
ambiguous between two competing definitions of risk level.
`tests/test_decision_engine.py::test_risk_category_and_policy_action_are_independent`
demonstrates this concretely: a transaction at probability 0.25 displays as risk
category **LOW** (score 25, ≤30) while the policy engine still recommends
`ALLOW_WITH_MONITORING` — because 0.25 clears the *policy's* 0.15 floor. This looks
like an inconsistency at first glance; it is intentional and tested.

## Why bounded actions instead of unconstrained autonomous actions

Every action in `ALLOWED_ACTIONS` is a *recommendation*, and every one is reversible
or low-stakes on its own:
- `ALLOW` / `ALLOW_WITH_MONITORING` — no friction, transaction proceeds; monitoring
  is passive.
- `STEP_UP_VERIFICATION` — asks for more proof before proceeding; doesn't move money
  or deny anyone permanently.
- `BLOCK` — the strongest action, but it's a hold/refusal, not an irreversible
  financial action (e.g. not "issue a chargeback" or "close the account"). A human
  reviewer can always override it.

The engine has no access to a payments API, no ability to freeze funds, close
accounts, or contact a customer directly. It can only return a label. This is
intentional, and matches the buildathon brief's "recommend or simulate, never perform
irreversible real-world financial actions" requirement — and it's also just good risk
engineering: the system that assigns the risk score should not be the same system
that has unilateral authority to act on it.

## What happens at threshold boundaries

All conditions use `>=`, so the boundary value itself belongs to the *higher-friction*
side:
- `probability == 0.40` exactly → `HIGH_STEP_UP` (not `MEDIUM_MONITOR`). This matches
  the "flagged" framing already established in Phase 5/6 (a transaction at exactly the
  decision threshold was already being treated as "flag" there).
- `probability == 0.80` exactly → `CRITICAL_BLOCK` (not `HIGH_STEP_UP`).
- `probability == 0.15` exactly → `MEDIUM_MONITOR` (not `LOW_ALLOW`).
- `amount == 25,000` exactly → escalation rule fires.

Every one of these exact-boundary cases has a dedicated test
(`test_exactly_at_decision_threshold_is_step_up`, `test_exactly_at_critical_min_is_block`,
`test_exactly_at_low_max_is_medium_monitor`, `test_amount_exactly_at_escalation_cutoff`),
plus a matching "just below" test for each, rather than relying on representative
mid-range examples alone.

## What the engine does with invalid inputs

`make_decision()` validates before touching the policy:
- `probability` must be numeric (bool explicitly rejected, since `bool` is a Python
  `int` subclass and `True`/`False` silently coercing to `1.0`/`0.0` would be a
  dangerous, unannounced substitution), non-NaN, and in `[0, 1]`. Out-of-range values
  (including `inf`/`-inf`) are rejected, not clamped — silently clamping an invalid
  upstream probability would hide a bug in whatever produced it.
- `amount` must be numeric, non-NaN, and non-negative.
- `transaction_id` must be a non-empty string.

All violations raise `InvalidTransactionError` (a `ValueError` subclass) — the
function never guesses a default or silently substitutes a "safe-looking" value.

`evaluate_policy()` itself also fails safe: if a rule ever produced an action string
outside `ALLOWED_ACTIONS`, or if (due to a future bug) no rule matched at all despite
`LOW_ALLOW` being an unconditional catch-all, a `RuntimeError` is raised rather than
returning an undefined or best-guess action. Both cases are covered by tests using a
monkeypatched, deliberately broken policy table.

## Explainability vs. decision-making: why they're kept separate

`make_decision()` accepts `model_explanation` as a parameter, but it is **never read**
by anything that determines the action — `evaluate_policy()`'s signature is literally
`(probability, amount)`, structurally incapable of receiving the explanation at all
(verified by `test_evaluate_policy_signature_does_not_accept_explanation`, which
inspects the function signature directly rather than just testing behavior).
`test_shap_values_cannot_alter_action` goes further: it calls `make_decision()` with
the same probability/amount but wildly different (including deliberately
self-contradictory) SHAP explanations, and asserts the action and rule ID never move.

This matters because SHAP explains *the model's reasoning*, which can be nuanced,
correlated, or occasionally misleading in edge cases (see `docs/explainability.md`
limitations). The action taken on real money needs a harder, simpler guarantee than
"the explanation looked convincing." Keeping them structurally separate means a bug or
oddity in the explanation layer can never silently change what happens to a
transaction — at worst, it produces a confusing audit note next to an otherwise
correct decision.

---

## Final policy table

*(see table above — reproduced here for the deliverable)*

| Order | Rule ID | Condition | Action |
|---|---|---|---|
| 1 | CRITICAL_BLOCK | probability >= 0.80 | BLOCK |
| 2 | HIGH_STEP_UP | probability >= 0.40 | STEP_UP_VERIFICATION |
| 3 | MEDIUM_AMOUNT_ESCALATION | probability >= 0.15 AND amount >= Rs25,000 | STEP_UP_VERIFICATION |
| 4 | MEDIUM_MONITOR | probability >= 0.15 | ALLOW_WITH_MONITORING |
| 5 | LOW_ALLOW | (catch-all) | ALLOW |

## Example decisions

**LOW** (real validation transaction, txn_0118343):
```json
{
  "fraud_probability": 0.0005, "action": "ALLOW", "policy_rule_id": "LOW_ALLOW",
  "risk_score": 0, "risk_category": "LOW"
}
```

**MEDIUM** (illustrative — hand-set probability to demonstrate this branch cleanly):
```json
{
  "fraud_probability": 0.28, "action": "ALLOW_WITH_MONITORING", "policy_rule_id": "MEDIUM_MONITOR",
  "risk_score": 28, "risk_category": "LOW"
}
```
Note risk_category shows LOW (score 28 ≤ 30) while the action is ALLOW_WITH_MONITORING —
this is the intentional score/policy independence described above, not an error.

**HIGH** (illustrative):
```json
{
  "fraud_probability": 0.62, "action": "STEP_UP_VERIFICATION", "policy_rule_id": "HIGH_STEP_UP",
  "risk_score": 62, "risk_category": "HIGH"
}
```

**CRITICAL** (real validation transaction, txn_0110465):
```json
{
  "fraud_probability": 0.9929, "action": "BLOCK", "policy_rule_id": "CRITICAL_BLOCK",
  "risk_score": 99, "risk_category": "CRITICAL"
}
```

## Test count and results

`tests/test_decision_engine.py`: **50/50 passing**, covering all 12 required
categories from the brief plus additional boundary/edge cases (amount-escalation
boundaries, bool-as-probability rejection, immutability, fail-safe behavior under a
monkeypatched broken policy table).

Full project test suite (`tests/`): **79/79 passing** (3 leakage sanity checks +
26 explainability tests + 50 decision engine tests) — no regressions introduced in
frozen components.

## Edge cases discovered during implementation

1. **My own test assumptions were wrong, not the code.** Two early tests assumed
   `risk_category` and the policy action would always "agree" (e.g. expected MEDIUM
   category alongside a MEDIUM_MONITOR action). They failed — correctly — because the
   two systems are deliberately decoupled (see above). Fixed by picking probabilities
   where they do align for the simple representative tests, and adding a dedicated
   test that asserts the divergence explicitly, so this isn't accidentally "fixed"
   into false coupling later.
2. **`bool` silently satisfying `isinstance(x, (int, float))`.** Python's `bool` is an
   `int` subclass, so an accidental `True`/`False` passed as a probability would
   otherwise silently become `1.0`/`0.0` — a dangerous silent coercion for a fraud
   probability. Added an explicit `isinstance(x, bool)` check ahead of the numeric
   check to reject this outright rather than accept it.
3. **`inf`/`-inf` pass Python's `0 <= x <= 1` check incorrectly in some numeric edge
   cases if not handled explicitly** — verified this is actually handled correctly
   by the straightforward range check (`inf` fails `<= 1.0` as expected), but added
   explicit test coverage for it rather than assuming.

# Explainability — Design Notes (Phase 6)

## Why SHAP TreeExplainer

The selected model (Phase 4) is LightGBM, a tree-ensemble. `shap.TreeExplainer` is
appropriate here specifically because:

- It computes **exact** Shapley values for tree ensembles in polynomial time (not the
  sampling-based approximation required for arbitrary black-box models). There is no
  approximation error to caveat for this model class.
- It decomposes the model's raw margin (log-odds) output as
  `base_value + sum(shap_values) = model_margin`, and `sigmoid(model_margin) = predicted_probability`.
  This is a mathematical identity we verify per-transaction (see "Consistency
  verification" below), not an assumption.
- It uses the model's actual learned tree structure, so contributions reflect what the
  model actually does, not a simplified surrogate.

We evaluated whether to use it at all before committing to it (per project philosophy:
"don't add SHAP just because it sounds impressive"). Two simpler alternatives were
considered and rejected for this component:
- **Feature importance (global, model-level):** doesn't explain an *individual*
  transaction, which is what the audit trail and analyst UI actually need.
- **Coefficient-based explanation (as used for the Logistic Regression baseline):**
  not applicable — LightGBM is non-linear, so there are no fixed per-feature
  coefficients to read off.

Given the model is tree-based and per-transaction, per-decision explanations were
a hard requirement (Section 9/11 of the brief), TreeExplainer is the correct tool,
not a default reflex.

## Consistency verification

Every call to `RiskExplainer.explain()` reconstructs the model's predicted probability
from `base_value + sum(shap_values)` via the sigmoid function and compares it to the
model's own `predict_proba` output, asserting they agree to within `1e-4`. This is
returned as `additivity_check_passed` in the explanation payload and is covered by
`tests/test_explainability.py::test_shap_additivity_holds_for_all_samples`, run
against 20 real transactions. If this check ever fails in production, that's a signal
something is wrong with the explainer/model pairing (e.g. a version mismatch) — not
something the system silently papers over.

## How reasons are grounded (no fabrication)

`humanize_contribution()` never generates language independent of the actual
`(feature, value, shap_value)` triple it's given:
- The **feature name** and **observed value** come directly from the transaction's
  real feature vector — see `test_contribution_values_match_input_features`.
- The **direction** ("increased"/"decreased" risk) is derived directly from the sign
  of the SHAP value — see `test_direction_field_matches_shap_sign`.
- The **magnitude/ranking** used to pick the "top" reasons is `abs(shap_value)`,
  nothing else.
- There is no LLM in this path. Templates are fixed Python functions keyed by feature
  name; a feature without a bespoke template still gets a grounded fallback sentence
  built from its real value and direction, never a generic or invented one.

Additionally, the framing sentence ("Why this transaction was flagged" vs. "Top
contributing factors — NOT flagged") is chosen from the actual decision outcome
(probability vs. the frozen 0.40 threshold), not assumed. Early in this phase, the
first implementation showed a "why this was flagged" narrative for a transaction that
scored well *below* the decision threshold — the reasons themselves were accurate
SHAP contributions, but the framing was misleading. This was caught by manually
inspecting the demo output (`ml/evaluation/demo_explain.py`) before writing tests, and
fixed by having `build_explanation_text()` branch explicitly on the real decision
outcome rather than always adopting a "why flagged" tone.

## Limitations of this approach

- **SHAP explains the model, not the world.** A high positive contribution from
  `new_device_flag` means the *model* leans on that signal heavily — it is not proof
  the transaction is actually fraud, and the explanation should not be read as a
  certainty statement. This is exactly why every action downstream is a
  *recommendation for human review*, not an automated financial action (Section 10 of
  the brief, implemented in Phase 10).
- **Correlated features can split credit.** `amount` and `amount_vs_avg_ratio` are
  related; SHAP can distribute "credit" for an unusual amount across both rather than
  cleanly attributing it to one, which can make the top-5 list slightly redundant
  (visible in the demo output above). We accept this rather than manually
  de-duplicating, since silently merging SHAP attributions would itself be a form of
  fabrication.
- **Local, not global, explanation.** Each explanation is specific to one transaction's
  feature values and the model's local behavior near that point; it should not be
  generalized into a blanket statement like "the model always weighs velocity most."
- **Depends on feature quality.** If a feature pipeline bug feeds a wrong value in
  (e.g. a stale rolling average), SHAP will faithfully explain the model's reaction to
  that *wrong* value — explainability doesn't validate feature correctness, which is
  why the leakage/correctness tests from Phase 1 remain a separate, necessary layer.

## Separation from the decision layer

This is a deliberate architectural boundary, not an implementation detail:

```
LightGBM model            -> fraud_probability          (learned, Phase 3/4)
risk_scoring.py            -> risk_score (0-100), category (rule-based, Phase 6)
explainability.py          -> SHAP contributions + grounded reasons (Phase 6)
[Phase 10] decision.py     -> recommended action (rule-based, driven by the frozen
                               0.40 probability threshold, NOT by the risk score bucket)
```

SHAP and the risk score/category are **explanatory and informational** — they answer
"how risky does the model think this is, and why." They do **not** decide the bounded
action taken. The actual flag/allow decision uses the frozen, cost-optimized
probability threshold (0.40) from Phase 5 directly. This means:

- A transaction can display as "MEDIUM" risk (score 40–60) while sitting on either
  side of the 0.40 decision boundary — the score is for human legibility in the
  dashboard/audit trail; the threshold is what actually governs the bounded action.
- Changing the risk-score bucket boundaries (e.g. moving MEDIUM's ceiling from 60 to
  55) never changes what action gets taken on any transaction — only how it's
  displayed. Changing the decision threshold does change actions, and doing so
  requires rerunning the cost analysis from Phase 5, not a config tweak.
- This split means a reviewer can audit "why did the model think this was risky"
  (explainability) completely separately from "why did the system flag/allow this
  transaction" (deterministic policy) — matching the project's core architectural
  principle: ML estimates probability, rule-based logic converts probability into
  risk category, and separate rule-based logic determines the action.

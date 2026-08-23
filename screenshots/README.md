# Screenshots — capture guide

This directory has no screenshots yet — none were fabricated for this repository.
Below is exactly what to capture and why, so the portfolio repo has real evidence of
the working system rather than staged images.

**How:** run the app locally (`README.md` → Local setup), open
`http://127.0.0.1:8000`, and use the dashboard as described in each shot below.
Save as PNG, name them as listed, place them in this directory, and reference them
from the root `README.md` once captured.

| # | Filename | What to capture | Why it matters |
|---|---|---|---|
| 1 | `01-dashboard-overview.png` | The full dashboard on first load: status bar (model loaded, threshold), transaction form with demo scenarios visible, empty result panel | Shows the overall layout and that the system is live/connected in one shot — the first thing a recruiter sees |
| 2 | `02-low-risk-allow.png` | After clicking the "Routine repeat purchase" demo scenario: the green ALLOW result, probability, risk score, and policy reason | Shows the low-friction path and that low-risk transactions aren't over-flagged |
| 3 | `03-high-risk-stepup.png` | After clicking "New device, unfamiliar region": the amber STEP_UP_VERIFICATION result with policy rule `HIGH_STEP_UP` visible | Shows the mid-tier decision and that the real `action` string is displayed verbatim |
| 4 | `04-critical-block.png` | After clicking "Large purchase, new device and region": the red BLOCK result, high fraud probability, `CRITICAL_BLOCK` rule | Shows the system correctly escalating a clear fraud pattern to the strongest action |
| 5 | `05-shap-explanation.png` | The "Why this decision" panel for the CRITICAL scenario: grounded reason list + the contribution bars | This is the differentiator — proves explanations are real, per-transaction, and legible to a non-ML reader |
| 6 | `06-audit-trail.png` | The audit log table after evaluating a few scenarios, showing the Source column (demo/manual), action chips, and timestamps | Shows the auditability story — every decision is logged and traceable |
| 7 | `07-model-info.png` | The model & policy metadata panel: version, threshold, held-out test precision/recall/F1 | Backs up the README's metrics claims with a live view of the same numbers |

**Optional, if useful:** `08-api-docs.png` — the FastAPI Swagger UI at `/docs`,
showing the endpoint list. Useful for a software-engineer-focused reviewer who wants
to see the API contract without reading `docs/api.md`.

## Existing visual asset (not a UI screenshot)

`ml/models/threshold_analysis_chart.png` already exists in the repo — it's a real
matplotlib chart from Phase 5 (precision/recall/F1 and cost vs. threshold on the
validation set), not a dashboard screenshot. It's referenced from
`ml/models/lgbm_v1_metadata.json`'s companion analysis and can be linked directly
from the README's "Model & evaluation results" section if a visual is wanted there,
separately from the dashboard screenshots above.

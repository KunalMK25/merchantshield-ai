# Screenshots — capture guide

This directory has no UI screenshots committed yet — none were fabricated for this
repository. Below is exactly what to capture and why, so the portfolio repo has real
evidence of the working system rather than staged images.

**How to run the app for capture:**

Option A (Docker — simplest):
```bash
docker compose up --build
# open http://localhost:8000
```

Option B (local):
```bash
# from the repo root, after running generate_synthetic.py + build_features.py
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Save captures as PNG, name them as listed below, place them in this directory
(`screenshots/`), and the root `README.md` will automatically render them once they
exist (the image references are already written).

---

## UI screenshots to capture

| # | Filename | What to capture | Why it matters |
|---|---|---|---|
| 1 | `01-dashboard-overview.png` | The full dashboard on first load: status bar (model loaded, threshold, test precision/recall), transaction form with demo scenarios visible, empty result panel | Shows the overall layout and that the system is live/connected in one shot — the first thing a recruiter sees |
| 2 | `02-low-risk-allow.png` | After clicking "Routine repeat purchase": the green ALLOW result, fraud probability, risk score, and policy reason | Shows the low-friction path and that low-risk transactions aren't over-flagged |
| 3 | `03-medium-risk-monitor.png` | After clicking "Unfamiliar region, known device": the ALLOW_WITH_MONITORING result and its policy reason | Shows the MEDIUM tier and the monitoring clarification text |
| 4 | `04-high-risk-stepup.png` | After clicking "New device, unfamiliar region": the amber STEP_UP_VERIFICATION result with policy rule `HIGH_STEP_UP` visible | Shows the mid-tier decision and that the real `action` string is displayed verbatim |
| 5 | `05-critical-block.png` | After clicking "Large purchase, new device and region": the red BLOCK result, high fraud probability, `CRITICAL_BLOCK` rule | Shows the system correctly escalating a clear fraud pattern to the strongest action |
| 6 | `06-shap-explanation.png` | The "Why this decision" panel for the CRITICAL scenario: grounded reason list + the contribution bars | This is the differentiator — proves explanations are real, per-transaction, and legible to a non-ML reader |
| 7 | `07-audit-trail.png` | The audit log table after evaluating a few scenarios, showing the Source column (Demo/Manual), action chips, and timestamps | Shows the auditability story — every decision is logged and traceable |
| 8 | `08-model-info.png` | The model & policy metadata panel: version, threshold, held-out test precision/recall/F1 | Backs up the README's metrics claims with a live view of the same numbers |

**Optional:** `09-api-docs.png` — the FastAPI Swagger UI at `/docs`, showing the
endpoint list. Useful for a software-engineer-focused reviewer who wants to see the
API contract without reading `docs/api.md`.

---

## Existing visual asset (already in the repo)

`ml/models/threshold_analysis_chart.png` is the real matplotlib output from Phase 5
— precision, recall, F1, and estimated cost vs. threshold on the **validation set**.
It is already embedded in the root `README.md` under "Model & evaluation results"
and does not need to be re-captured. It is not a dashboard screenshot.

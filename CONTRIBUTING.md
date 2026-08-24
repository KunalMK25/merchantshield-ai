# Contributing to MerchantShield AI

This document covers the developer workflow: environment setup, the commands you
need to know, what CI checks, commit conventions, and the rules that protect the
project's ML integrity.

---

## Environment setup

**Python** — 3.11 or newer (CI runs 3.13; developed on 3.12.3).

```bash
pip install -r requirements.txt
```

> If pip raises an "externally managed environment" error (common on Ubuntu 23+),
> add `--break-system-packages` or use a virtual environment:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

**Node.js** — 20.19+ or 22.12+ (Vite 8 requires Node ≥ 20.19; Node 18 is not
sufficient). CI pins 22.12.0.

```bash
cd frontend && npm ci
```

**Docker** (optional) — if you want to skip local Python/Node setup entirely:

```bash
docker compose up --build
```

---

## Generating the dataset

The generated CSVs (`ml/data/*.csv`) are excluded from Git — they are large (~30 MB)
and fully reproducible from the committed generator source with a fixed RNG seed.
The frozen model artifact (`ml/models/candidate_lgbm_v1.pkl`) **is** committed, so
you only need to regenerate data to run the test suite or to retrain.

```bash
python ml/data/generate_synthetic.py     # → ml/data/raw_transactions.csv
python ml/features/build_features.py     # → ml/data/features.csv
```

---

## Running tests

```bash
pytest tests/
```

The suite has six files:

| File | What it covers | Approximate run time |
|---|---|---|
| `test_decision_engine.py` | All policy branches, boundary conditions, fail-safe behavior, SHAP-cannot-alter-action proof | ~1 s |
| `test_explainability.py` | SHAP grounding, additivity, determinism, direction correctness | ~20 s |
| `test_api.py` | Valid requests, every documented error path, audit persistence, no-logic-duplication check | ~15 s |
| `test_no_leakage.py` | Strictly-prior feature computation verified on sampled customers | ~3 min (reads 213k-row CSV) |
| `test_ieee_external.py` | IEEE-CIS adapter schema, prohibited column guards, future-leakage prevention, label independence | ~1 min (non-integration); integration tests skipped without data |
| `test_card_product_features.py` | card_product_share and sibling features — correctness, future-leakage, label independence | ~4 s |

`test_decision_engine.py` and `test_api.py` are fast enough to run on every change.
`test_no_leakage.py` is slow by nature (it iterates over customer histories) — run
it before committing anything that touches `ml/features/build_features.py` or
`ml/data/generate_synthetic.py`.

The `model`, `explainer`, `sample_rows`, and `client` fixtures are defined at
**session scope** in `tests/conftest.py` — the LightGBM model and SHAP explainer
load once per `pytest tests/` invocation, not once per test.

---

## Building the frontend

```bash
cd frontend && npm run build
```

The built assets go to `frontend/dist/`, which FastAPI serves same-origin at `/`.
`frontend/dist/` is excluded from Git — rebuild whenever you change frontend source.

For live development against a running backend:

```bash
# terminal 1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2
cd frontend && npm run dev   # Vite dev server, proxies API calls to port 8000
```

---

## What CI checks

On every push to `main` and every pull request, GitHub Actions runs:

**Backend job** (Python 3.13, ubuntu-latest):
1. `pip install -r requirements.txt`
2. `python ml/data/generate_synthetic.py`
3. `python ml/features/build_features.py`
4. `npm ci` + `npm run build` (frontend)
5. `python -m pytest tests/`

**Frontend job** (Node 22.12.0, ubuntu-latest):
1. `npm ci`
2. `npm run build`

Both jobs must pass for a PR to be mergeable. Never push a commit that breaks CI
— fix the root cause rather than skipping tests or weakening assertions.

---

## Before committing

1. **Run the relevant tests.** At minimum, `pytest tests/test_decision_engine.py`
   for any Python change. For ML pipeline changes, run the full suite.
2. **Run the frontend build** if you changed anything under `frontend/`.
3. **Check `git status`** — confirm no generated files are staged:
   - `ml/data/*.csv` — generated, not tracked
   - `backend/data/*.db` — runtime state, not tracked
   - `frontend/dist/` — build output, not tracked
   - `ml/external/ieee/results/` — experiment result JSONs + external model PKL, not tracked
   - Raw IEEE-CIS CSV files (`train_transaction.csv`, `train_identity.csv`) — never commit
   - `__pycache__/`, `*.pyc` — never tracked
4. **No secrets.** Never commit `.env` files, API keys, or credentials. The
   `.env.example` documents what variables exist but contains no real values.

```bash
git diff --staged    # review exactly what you're about to commit
git status           # confirm nothing unintended is staged
```

---

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

Common types:

| Type | Use for |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `test` | Adding or correcting tests |
| `docs` | Documentation only |
| `refactor` | Code restructuring with no behavior change |
| `chore` | Tooling, config, dependency updates |
| `ci` | CI workflow changes |

Examples from this repo's history:
- `feat(api): add FastAPI risk evaluation service`
- `fix(ci): correct workflow syntax`
- `docs: add project documentation and verification`

Keep the subject line under 72 characters. Use the body for anything that needs
more context — especially the *why* behind a decision, not just the *what*.

---

## ML integrity rules — read before touching the pipeline

These rules exist because ML correctness is more important than any individual
feature addition. Breaking them silently degrades the system in ways that may not
be caught by the test suite.

### Frozen artifacts — do not change casually

| Artifact | Location | Why frozen |
|---|---|---|
| LightGBM model | `ml/models/candidate_lgbm_v1.pkl` | Selected via validation-set cost minimization; re-selecting requires re-running the full Phase 3–5 pipeline |
| Decision threshold | `ml/models/lgbm_v1_metadata.json` → `selected_threshold` (0.40) | Cost-optimized on validation, evaluated once on test; changing it requires justification + rerunning `threshold_analysis.py` |
| `MODEL_VERSION` constant | `ml/evaluation/decision_engine.py` | Must match `lgbm_v1_metadata.json`'s `model_name`; a dedicated test enforces this |
| Policy rule boundaries | `ml/evaluation/policy.py` | `DECISION_THRESHOLD` is imported from metadata, not re-derived here |

### Leakage rules

- Every feature in `FEATURE_COLUMNS` must depend only on transactions **strictly
  before** the one being scored, for that customer.
- `build_features.py` is the **single implementation** of feature engineering —
  never write a second version for "live" scoring. The API (`backend/services/risk_service.py`)
  calls the same function.
- If you add a feature: add a corresponding leakage test in `test_no_leakage.py`.

### What requires a deliberate evaluation run

If you change any of the following, you must rerun `ml/evaluation/threshold_analysis.py`
on fresh data and justify the new threshold before committing:

- The model (retraining `train_candidates.py`)
- The feature set (`FEATURE_COLUMNS` in `build_features.py`)
- The cost function (`CostAssumptions` in `cost_model.py`)
- The dataset generator parameters (affects train/val/test distributions)

Do **not** casually change the frozen threshold to improve demo results. Do **not**
report synthetic-data metrics as real-world performance.

---

## Adding a new API endpoint

1. Add the route to `backend/api/routes.py` — **before** the `StaticFiles` mount at
   the bottom of `backend/main.py` (see the comment there; anything registered after
   the mount is silently shadowed).
2. Add a Pydantic response model to `backend/schemas/response.py`.
3. Business logic goes in `backend/services/` — routes stay thin.
4. Add tests covering the happy path, at least one validation failure, and the
   model-unavailable (503) path.

## Adding a new frontend component

The frontend speaks HTTP/JSON only — never import Python modules or duplicate policy
thresholds in JavaScript. If you need a value that currently only lives in
`ml/evaluation/policy.py`, surface it through an existing API endpoint (e.g.
`/model/info` already exposes the threshold) rather than hardcoding it client-side.

---

## Questions about design decisions

Most "why is it done this way?" questions are answered in the `docs/` directory:

- `docs/api.md` — API contract, the `prior_transactions` design decision, audit field choices
- `docs/decision_engine.md` — why rule-based (not ML), boundary semantics, SHAP isolation
- `docs/explainability.md` — why TreeExplainer, the additivity check, SHAP limitations
- `docs/frontend.md` — decoupling from backend, demo scenario verification, action-tier display
- `docs/external_validation_ieee_cis.md` — IEEE-CIS dataset, Experiment A (frozen transfer), Experiment B (retrain), PARTIAL TRANSFER conclusion, full feature-compatibility analysis
- `docs/phase15_card_product_experiment.md` — card_product_share feature, ablation results, leakage proof, test-set evaluation

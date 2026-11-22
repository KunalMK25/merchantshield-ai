# MerchantShield AI — Documentation Index

This directory contains technical documentation for the MerchantShield AI project.
For the project overview, quick-start, and results summary see the root
[README.md](../README.md).

---

## System documentation

| Document | What it covers |
|---|---|
| [api.md](api.md) | Full API contract, endpoint reference, error table, `prior_transactions` design note, audit field choices |
| [decision_engine.md](decision_engine.md) | Why rule-based (not ML), the five-rule policy, boundary semantics, SHAP structural isolation, fail-safe design |
| [explainability.md](explainability.md) | SHAP TreeExplainer rationale, additivity check, human-readable grounding, known limitations |
| [frontend.md](frontend.md) | Frontend/backend decoupling, demo scenario verification, action-tier display, accessibility |

---

## Model lifecycle

The contest and registry, daily retraining, drift and anomaly monitoring, and the alert workflow are
documented in the root [README.md](../README.md) (sections "Model contest and registry" through
"Alerts and account holds"). The complete, always-current endpoint reference for those routes is the
interactive API documentation served at `/docs`; [api.md](api.md) covers the original scoring endpoints.

---

## External validation documentation

| Document | What it covers |
|---|---|
| [external_validation_ieee_cis.md](external_validation_ieee_cis.md) | IEEE-CIS dataset provenance, licensing, feature mapping (active/degraded/inert), Experiment A (frozen transfer — fails), Experiment B (retrained — PARTIAL TRANSFER), Phase 14 root-cause analysis |
| [phase15_card_product_experiment.md](phase15_card_product_experiment.md) | card_product_share feature, leakage proof (35 tests), ablation results (5 variants), test-set evaluation, KEEP/REJECT decision |
| [phase17_heldout_real_data_validation.md](phase17_heldout_real_data_validation.md) | **Canonical held-out evaluation.** Phase 17 results on real IEEE-CIS data: ROC-AUC 0.771, recall 79.8%, precision 6.3% at threshold 0.35. Objective assessment. Full comparison table (Phases 14–17). |

---

## Reading order for code reviewers

1. Root `README.md` — project overview, architecture, all key results
2. `docs/decision_engine.md` — why the system is a decisioning layer, not just a classifier
3. `docs/explainability.md` — how SHAP explanations are grounded and verified
4. `docs/api.md` — the full API contract
5. `docs/external_validation_ieee_cis.md` — honest external validation story (Phases 14–15)
6. `docs/phase15_card_product_experiment.md` — the targeted improvement experiment
7. `docs/phase17_heldout_real_data_validation.md` — canonical real-data held-out evaluation

---

## What is not in this directory

- **ML pipeline scripts** — in `ml/` (see `ml/training/`, `ml/evaluation/`, `ml/features/`)
- **Test suite** — in `tests/` (see `CONTRIBUTING.md` for test descriptions)
- **Screenshots** — in `screenshots/README.md` (none committed; capture guide provided)
- **API interactive docs** — available at `/docs` when the backend is running

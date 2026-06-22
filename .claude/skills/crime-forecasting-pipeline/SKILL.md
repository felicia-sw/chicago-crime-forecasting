---
name: crime-forecasting-pipeline
description: Master orchestration guide for the Chicago crime forecasting paper (horizon-adaptive weighted ensemble). Use whenever working in this repo on modeling, data, evaluation, or the IEEE Access manuscript, to follow the locked plan and guardrails.
---

# Crime Forecasting Pipeline (master skill)

**Goal:** IEEE Access paper. **Contribution:** a horizon- & regime-adaptive weighted ensemble
(SARIMA + Prophet + XGBoost) that beats the best single model AND the static inverse-RMSE ensemble —
advantage largest at long horizons and across the 2020 COVID break. The contribution is the
*mechanism + finding* (the optimal model mix is not constant), not assembling off-the-shelf models.

## Guardrails (never violate)
- Temporal split only; no shuffling. Window 2015+; train ≤2020, val 2021–2022, test 2023→ (rolling-origin CV).
- Ensemble weights fit on **validation only** (no test leakage); features lagged behind the split.
- Missing ≠ zero (panel already 0-filled). Seasonal-naive baseline mandatory.
- Metrics: MAE/RMSE primary; MAPE/SMAPE masked at 0; MASE vs naive. Diebold–Mariano per horizon.
- SHAP on the XGBoost member only. Ethics section required.

## Build order (each step has a dedicated skill)
1. **crime-data-prep** — load the weekly panel; trim to 2015 at modeling time (don't rebuild).
2. **horizon-adaptive-ensemble** — base models + the 3 ensemble schemes + rolling-origin backtest.
3. **forecast-evaluation** — metrics, DM tests, robustness (weight-trajectory across 2020), SHAP.
4. **Manuscript** — IMRaD in `manuscript/`, IEEE Access two-column template; lead with the adaptivity
   result and the 2020 weight-trajectory figure, not the model list. Cite prior art from `CONTRIBUTION.md` §10.

## Key files
- `config.yaml` (single source of truth) · `CONTRIBUTION.md` (full plan) · `HANDOVER.md` (state + steps).
- `data/processed/weekly_district_panel.csv` (modeling data) · `data/processed/weekly_type_panel.csv` (secondary).

## Research questions (answer these)
RQ1 best single model & its stability across horizons; RQ2 ensemble vs best single (DM) and adaptive vs
static; RQ3 how weights shift across the 2020 regime and across districts (is adaptivity the source of gain).

# Chicago Crime Forecasting — Project Memory (for Claude Code & other agents)

This repo is an **IEEE Access** paper: a **horizon- and regime-adaptive weighted ensemble**
(SARIMA + Prophet + XGBoost) for weekly, district-level Chicago crime forecasting.

## Start here
1. `HANDOVER.md` — full state + ordered next steps (the master brief).
2. `CONTRIBUTION.md` — locked contribution, RQs, hypotheses, scope.
3. `config.yaml` — single source of truth (districts, horizons, window, split, ensemble).
4. Project skills live in `.claude/skills/` (auto-discovered): `crime-forecasting-pipeline`
   (master), `crime-data-prep`, `horizon-adaptive-ensemble`, `forecast-evaluation`.

## Data is ready
`data/processed/weekly_district_panel.csv` — 22 districts × 1327 weeks (2001–2026).
- Read with `dtype={'district': str}` (districts are zero-padded, e.g. "001").
- `week` is the Monday of the week (W-MON); absent weeks are already 0-filled (missing ≠ zero).

## Non-negotiable guardrails
- **Temporal split only**, no shuffling. Modeling window **2015+**; train ≤2020, val 2021–2022, test 2023+.
- **Ensemble weights fit on validation ONLY** (no test leakage); features lagged behind the split.
- Metrics: **MAE/RMSE primary**; MAPE/SMAPE masked at actual=0; **MASE** vs seasonal-naive.
- **Seasonal-naive baseline is mandatory.** **Diebold–Mariano** per horizon. **SHAP on the XGBoost member only.**
- Ethics & Responsible Use section required (aggregate workload forecasting, not targeting).

## Next
Modeling phase — Step 1 = feature engineering (`HANDOVER.md` §6). The data, contribution, and scope are locked.

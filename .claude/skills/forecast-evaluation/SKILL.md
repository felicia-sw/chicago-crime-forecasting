---
name: forecast-evaluation
description: Evaluation, significance testing, robustness, and interpretability conventions for the crime forecasting paper. Use when computing metrics, running Diebold–Mariano, doing robustness checks, or SHAP.
---

# Forecast Evaluation & Robustness

## Metrics (report per horizon; per district and pooled)
- **Primary: MAE, RMSE.**
- MAPE / SMAPE: report but **mask where actual = 0** (state the caveat).
- **MASE**: error scaled by the in-sample seasonal-naive — expresses skill relative to the floor.

## Significance — Diebold–Mariano
- Run **per horizon**: ensemble vs the best single model (and adaptive vs static ensemble).
- Report the DM statistic and p-value; state the loss differential used (squared or absolute error).
- Use a small-sample correction (Harvey–Leybourne–Newbold) where appropriate.

## Robustness & stability (answers RQ3)
- Sensitivity across districts, random seeds, weight schemes, and **training-window length**.
- Compare pre / within / post-2020 regimes.
- **Centerpiece figure:** weight trajectories over time showing authority shifting between members
  (e.g., SARIMA → XGBoost) across the 2020 break, then reverting.

## Interpretability
- **SHAP on the XGBoost member only**; frame explicitly as explaining that component, not the ensemble.
  Surface the dominant lag / rolling / calendar drivers.

## Reporting discipline
- Always show the seasonal-naive floor.
- Honestly report where the ensemble does NOT beat the best single model (often short horizons) — that is
  itself a finding (RQ2), not a failure to hide.

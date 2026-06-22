---
name: horizon-adaptive-ensemble
description: How to build the base forecasters and the horizon-/regime-adaptive weighted ensemble (this paper's core contribution), with no-leakage weighting and a rolling-origin backtest. Use when implementing or changing models or the ensemble.
---

# Horizon-Adaptive Weighted Ensemble (core contribution)

## Base models — for each horizon h ∈ {1, 4, 8, 12} weeks
- **Seasonal-naive** (mandatory floor): ŷ(t+h) = y(t+h−52) (same week last year).
- **SARIMA**: `pmdarima.auto_arima` per district; cache fitted orders; parallelize with joblib (slowest part).
- **Prophet**: per district; yearly seasonality; add COVID changepoint / holiday regressors.
- **XGBoost**: ONE **global** model across the whole panel (district as a feature + lag/rolling/calendar
  features) — cross-learning beats 22 separate models and is cheaper.

## Forecast strategy
Direct (a separate model/target per horizon) as primary; recursive as a sensitivity.

## Backtest — rolling-origin, temporal only
Roll origins across validation then test (no shuffling). Emit forecasts keyed by
`(district, horizon, origin)` so every scheme is evaluated on identical points.

## Ensemble schemes — weights estimated on VALIDATION ONLY
1. `static_inverse_rmse` (baseline): wₘ ∝ 1/RMSEₘ, one weight vector for all horizons.
2. `horizon_adaptive` (**main contribution**): a SEPARATE constrained weight vector **per horizon**
   (wₘ ≥ 0, Σw = 1) minimizing validation RMSE at that horizon (e.g. `scipy.optimize.minimize`, SLSQP).
3. `regime_adaptive`: re-estimate weights on a rolling 52-week window → time-varying weights
   (produces the 2020 weight-trajectory figure that is the paper's memorable visual).

## Must-haves
- No test leakage — weights never see the test set.
- Report the weight vectors per horizon and over time (a finding, not just a metric).
- Persist per-(district, horizon) forecasts for the evaluation step.
- Config: `ensemble.schemes`, `ensemble.constraints`, `ensemble.rolling_window_weeks` in `config.yaml`.

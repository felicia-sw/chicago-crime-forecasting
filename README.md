# Horizon- & Regime-Adaptive Weighted Ensemble — Chicago Crime Forecasting

A reproducible study of **when** combining forecasters helps for multi-horizon,
district-level weekly crime forecasting, and whether the optimal model mix is stable
across forecast horizon and across the 2020 COVID structural break.

**Members:** seasonal-naive (baseline) + SARIMA + Prophet + global XGBoost + global LSTM,
blended under three schemes (static inverse-RMSE, horizon-adaptive, regime-adaptive),
with weights fit on validation only, a rolling-origin backtest, Diebold–Mariano
significance (BH-corrected), MASE, and SHAP.

**Honest headline finding (the forecast-combination puzzle):** a weighted ensemble
significantly beats the best single model at all horizons (the diversification benefit),
but *adaptive* weighting does **not** significantly beat a simple static inverse-RMSE
blend in any regime — the optimal mix shifts with horizon/regime descriptively, yet that
does not translate into a significant accuracy gain. See `manuscript/REVIEW_REPORT.md`.

## Project layout
```
chicago-crime-forecasting/
├── config.yaml               # single source of truth (districts, horizons, split, ensemble)
├── config_regime2020.yaml    # alt split placing the 2020 shock in the test window
├── requirements.txt          # env deps (run in the `crime-fcst` conda env; numpy<2 for pmdarima)
├── data/
│   ├── raw/                  # (gitignored) raw 1.9 GB incidents CSV
│   └── processed/            # source panels + committed results
│       ├── weekly_district_panel.csv   # MODELING DATA (22 districts × weeks)
│       ├── base_forecasts.csv          # base-model forecasts (the expensive SARIMA output)
│       ├── ensemble_weights.csv        # fitted ensemble weights
│       ├── metrics_by_horizon.csv / dm_tests.csv   # evaluation + significance
│       #  (model_frame.csv & ensemble_forecasts.csv regenerate from the pipeline; gitignored)
│       ├── robustness_*.csv / shap_importance.csv
│       ├── checkpoints/                # per-district SARIMA cache (resume; gitignored)
│       └── regime2020/                 # isolated outputs of the 2020-regime run
├── src/                      # the pipeline (see below)
├── figures/                  # paper-ready PNG/PDF figures
├── notebooks/results.ipynb   # executed results notebook (tables + figures inline)
└── manuscript/               # LaTeX draft, outline, peer-review report
```

## Pipeline (`src/`)
| Step | Script | Output |
|---|---|---|
| Data | `build_panel_from_raw.py` | weekly district panel from the raw CSV |
| EDA | `eda_overview.py`, `eda_map.py` | Figs 1–4 |
| 1. Features | `features.py` | `model_frame.csv` (lags/rolling/calendar, direct multi-horizon targets) |
| 2. Base models | `models_base.py`, `models_lstm.py` | `base_forecasts.csv` (naive/SARIMA/Prophet/XGBoost/LSTM) |
| 3. Ensemble | `ensemble.py` | `ensemble_forecasts.csv`, `ensemble_weights.csv` |
| 4. Evaluate | `evaluate.py` | `metrics_by_horizon.csv`, `dm_tests.csv` |
| 5. Robustness | `robustness.py` | `robustness_*.csv`, H4 figure |
| 6. SHAP | `shap_analysis.py` | `shap_importance.csv`, SHAP figures |
| Figure | `figures_weight_trajectory.py` | the 2020 weight-trajectory figure |

## Reproduce
```bash
conda activate crime-fcst          # Python 3.11; numpy<2 (pmdarima ABI); torch for the LSTM
pip install -r requirements.txt    # if setting up fresh

python src/features.py                                   # -> model_frame.csv
bash   src/run_base_backtest.sh                          # SARIMA+Prophet, memory-safe, ~1.5h, --jobs 1
python src/models_lstm.py                                # add the LSTM member (~80s, reuses cached forecasts)
python src/ensemble.py && python src/evaluate.py         # ensemble + metrics + DM (seconds)
python src/robustness.py && python src/shap_analysis.py  # robustness + interpretability
python src/figures_weight_trajectory.py                  # centerpiece figure
```
The 2020-regime experiment reuses the same scripts with an alternate config and writes to
`data/processed/regime2020/`:
```bash
CRIME_CONFIG=config_regime2020.yaml python src/features.py
bash src/run_base_backtest.sh config_regime2020.yaml
CRIME_CONFIG=config_regime2020.yaml python src/models_lstm.py
CRIME_CONFIG=config_regime2020.yaml python src/ensemble.py
CRIME_CONFIG=config_regime2020.yaml python src/evaluate.py
CRIME_CONFIG=config_regime2020.yaml python src/figures_weight_trajectory.py
```

**Compute note:** SARIMA with weekly seasonality (m=52) costs ~4 GB per district; the driver
runs one district at a time (`--jobs 1`) with per-district checkpointing so it is safe on a
16 GB machine. Everything downstream of `base_forecasts.csv` is light and runs in seconds.

## Status
- [x] Data, EDA, features, base models (incl. LSTM), adaptive ensemble
- [x] Evaluation (MAE/RMSE/MAPE/SMAPE/MASE + Diebold–Mariano, BH-corrected), robustness, SHAP
- [x] 2020-regime extension + centerpiece figure; executed results notebook
- [x] Simulated peer review (`manuscript/REVIEW_REPORT.md`)
- [ ] Manuscript finalization (reframe around the honest finding) + venue (Scopus-indexed; under discussion)

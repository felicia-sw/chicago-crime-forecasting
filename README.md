# Horizon-Adaptive Weighted Ensemble — Chicago Crime Forecasting

Reproducible pipeline for an IEEE Access / Scopus-style study on multi-horizon,
district-level crime forecasting. Headline contribution: a **horizon- and
regime-adaptive** weighted ensemble of SARIMA + Prophet + XGBoost that beats the
best single model and the static inverse-RMSE ensemble, with Diebold–Mariano
significance, and is robust across the 2020 COVID structural break.

See `CONTRIBUTION.md` for the locked contribution, RQs, scope, and guardrails.

## Project layout
```
crime-forecasting/
├── CONTRIBUTION.md          # locked contribution, RQs, scope, methodology guardrails
├── config.yaml              # seeds, paths, districts, horizons, split dates
├── requirements.txt
├── data/
│   ├── raw/                 # (gitignored) downloaded incidents / API cache
│   ├── processed/           # weekly district x week panel
│   ├── chicago_monthly_citywide.csv   # EDA aggregate (Fig 1)
│   └── chicago_district_2024.csv      # EDA aggregate (Fig 2)
├── src/
│   ├── eda_overview.py      # macro overview + district profile figures
│   ├── data_acquisition.py  # pull weekly-per-district counts from Socrata API
│   └── aggregate_weekly.py  # build clean district x week panel (zero-fill)
├── figures/                 # paper-ready PNGs
├── notebooks/               # (to be added) EDA -> modeling -> ensemble -> eval
└── manuscript/              # (to be added) IEEE Access IMRaD manuscript
```

## Reproduce
```bash
pip install -r requirements.txt
python src/data_acquisition.py      # downloads weekly counts per district -> data/raw/
python src/aggregate_weekly.py      # -> data/processed/weekly_district_panel.csv
python src/eda_overview.py          # regenerates Fig 1 & Fig 2
# modeling scripts (base models, ensemble, evaluation) added in the next phase
```

A free Socrata app token (env var `SOCRATA_APP_TOKEN`) raises rate limits but is
optional. The Kaggle mirror (`chicago/chicago-crime`) is an alternative source.

## Status
- [x] Feasibility, contribution, scope locked
- [x] EDA overview + district profile
- [x] Data acquisition + weekly aggregation pipeline
- [ ] Feature engineering · base models · adaptive ensemble · DM tests · robustness · SHAP
- [ ] Figures, tables, IEEE Access manuscript

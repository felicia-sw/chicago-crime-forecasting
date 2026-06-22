# Project Handover — Chicago Crime Forecasting (Horizon-Adaptive Ensemble)

Self-contained handover so a fresh agent (e.g., Claude Code) can continue with zero prior context.
**Tip for Claude Code:** copy this file to `CLAUDE.md` (or paste its key points there) so it auto-loads as project memory.

---

## 0. Read these first (in order)
1. `HANDOVER.md` (this file) — state + next steps.
2. `CONTRIBUTION.md` — the locked contribution, RQs, scope, methodology guardrails.
3. `config.yaml` — districts, horizons, window, split, ensemble schemes, metrics (single source of truth).
4. `data/processed/weekly_district_panel.csv` — the modeling dataset (already built & validated).

**Kickoff prompt to paste into Claude Code:**
> Read HANDOVER.md, CONTRIBUTION.md, and config.yaml. The weekly panel is already built at
> data/processed/weekly_district_panel.csv (22 districts × 1327 weeks). Start at "Next steps → Step 1
> (feature engineering)" in HANDOVER.md. Respect the guardrails: temporal split only, weights fit on
> validation only, modeling window 2015+, seasonal-naive baseline mandatory.

---

## 1. Goal & venue
A research paper for **IEEE Access (Scopus-indexed)** on multi-horizon, district-level crime forecasting
for Chicago. **Headline contribution:** a *horizon- and regime-adaptive weighted ensemble* (of SARIMA +
Prophet + XGBoost) that beats both the best single model and the conventional static inverse-RMSE
ensemble — with the advantage largest at long horizons and across the 2020 COVID structural break. The
contribution is the **mechanism + finding** (optimal model mix is not constant; it shifts by horizon and
regime), not the assembly of off-the-shelf models.

## 2. Current status

**DONE**
- Feasibility + literature scan (space is saturated for plain ensembles → reframed around adaptivity & robustness).
- Contribution, RQs, scope **locked** (see CONTRIBUTION.md). Professor approved ("udah sesuai").
- EDA on real data: monthly overview + district profile + 2020-break maps (see `figures/`).
- Reproducible scaffold + data pipeline.
- **Full data ingested**: raw 1.9 GB CSV → clean weekly panels (validated, see §4).
- `Research_Plan_Summary.docx` (shared with & approved by professor).

**NEXT (modeling phase — not started)**: feature engineering → base models → adaptive ensemble →
multi-horizon eval + Diebold–Mariano → robustness (incl. 2020 weight-trajectory) → SHAP → figures/tables
→ IEEE Access manuscript. Detailed in §6.

## 3. Locked decisions

**Research questions (from professor, refined):**
- RQ1. Among SARIMA, Prophet, XGBoost, which best forecasts next-week district counts, and does the ranking hold at 4/8/12 weeks?
- RQ2. Does a weighted ensemble beat the best single model across horizons (Diebold–Mariano), and does horizon-adaptive weighting beat static inverse-RMSE?
- RQ3. How do the optimal model/weights shift across the 2020 regime break and across districts — is adaptivity the source of the gain?

**Hypotheses (author-derived, PROPOSED — approved in principle):**
- H1. Ensemble advantage over best single model grows with horizon (single models competitive at h=1).
- H2. Optimal weights are not constant across horizons.
- H3. Across 2020, regime/rolling-adaptive ensemble > static; weight mass shifts SARIMA→XGBoost during the shock, reverts after.
- H4. Ensemble gains larger in higher-volatility districts.
- H5. Adaptive ensemble beats seasonal-naive at all horizons (report where it doesn't).

**Scope:** 22 active police districts; target = weekly **total** count per district; top-5 crime types
(THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, DECEPTIVE PRACTICE) as a **secondary** heterogeneity slice.
Horizons 1/4/8/12 weeks (direct strategy; recursive as sensitivity).

**Window & split (per professor — trim pre-2015, train before 2021):**
- Modeling window: **2015 → present** (drop 2001–2014; different/higher-volume regime + lighter).
- Train ≤ **2020-12-31**, validation **2021–2022**, test **2023 →**. Rolling-origin CV, no shuffling.
- Train-window length is itself a **sensitivity check**.

**Guardrails (do not violate):**
- Temporal split only; **no shuffling**.
- Ensemble weights fit on **validation only** (no test leakage); features lagged strictly behind split.
- **Missing ≠ zero**: absent week-buckets are true 0 (already handled in panel build); nothing deleted.
- Metrics: **MAE/RMSE primary**; MAPE/SMAPE with zero-count caveat (mask where actual=0); **MASE** vs seasonal-naive.
- **Diebold–Mariano** per horizon (ensemble vs best single).
- **SHAP on the XGBoost member only** (frame as explaining that component, not the ensemble).
- Include an **Ethics & Responsible Use** section (CLEAR reporting bias; aggregate workload forecasting, not targeting).

## 4. DATA — state & paths (important)

**Raw:** `Crimes_-_2001_to_Present.csv` (1.9 GB, 8,577,314 rows) at project root. *Do not commit; gitignore it.*

**Built & validated panels (USE THESE):**
- `data/processed/weekly_district_panel.csv` — **22 districts × 1327 weeks = 29,194 rows**, columns `district,week,count`.
  - Span **2001-01-01 → 2026-06-01**; weeks are Monday-labeled (W-MON); **0% zero-count weeks** (min 52, median 272, max 1117/wk).
  - District codes are zero-padded strings (`"001"`). **Read with `dtype={'district':str}`** or pandas drops leading zeros.
  - Top districts by total: 008 (574,552), 011 (542,442), 006 (500,636).
- `data/processed/weekly_type_panel.csv` — 5 crime types × 1327 weeks (secondary analysis).

**EDA aggregates (already pulled from API):** `data/chicago_monthly_citywide.csv`,
`data/chicago_district_2024.csv`, `data/district_year_counts.csv`, `data/district_centroids.csv`.

**Rebuild panel from raw if needed:** `python src/build_panel_from_raw.py "Crimes_-_2001_to_Present.csv"`
(run from `data/processed/`; ~20s; date-only parse).

**Note on the modeling window:** the panel is full 2001–2026; **trim to `window_start` (2015-01-01) at modeling time** (don't rebuild).

## 5. Repo structure
```
chicago-crime-forecasting/
├── HANDOVER.md                  # this file
├── CONTRIBUTION.md              # locked contribution / RQs / scope / guardrails
├── config.yaml                  # single source of truth (districts, horizons, window, split, ensemble)
├── README.md, requirements.txt
├── Research_Plan_Summary.docx   # the doc shared with the professor
├── Crimes_-_2001_to_Present.csv # raw 1.9GB (gitignore)
├── data/
│   ├── raw/ (gitignored)
│   ├── processed/weekly_district_panel.csv   # <-- MODELING DATA
│   ├── processed/weekly_type_panel.csv
│   └── *.csv (EDA aggregates)
├── figures/ fig1_monthly_overview, fig2_district_2024, fig3_map_2020_break, fig4_map_levels
├── src/
│   ├── build_panel_from_raw.py  # raw CSV -> weekly panels (DONE, used)
│   ├── eda_overview.py          # fig1/fig2
│   ├── eda_map.py               # fig3/fig4 (centroid bubble maps)
│   ├── data_acquisition.py      # Socrata API daily pull (alt to raw CSV)
│   └── aggregate_weekly.py      # daily->weekly panel (alt path)
├── notebooks/ (empty)           # add EDA->modeling->ensemble->eval
└── manuscript/ (empty)          # IEEE Access IMRaD goes here
```

## 6. Next steps (ordered, concrete)
1. **`src/features.py`** — from the panel (trimmed to 2015+): lags (1,2,3,4,8,12,52), rolling mean/std (4,8,12),
   calendar (month, week-of-year), US holiday flag (`holidays`), COVID indicator (`covid_window`), district id.
   Leakage-safe. Output a modeling frame for XGBoost.
2. **`src/models_base.py`** — seasonal-naive (lag-52); SARIMA via `pmdarima.auto_arima` **per district** (cache
   orders, parallelize); Prophet per district; **one global XGBoost** across the panel (district as feature, not
   22 separate models). Build a **rolling-origin backtest harness** that emits per-(district, horizon, origin) forecasts.
3. **`src/ensemble.py`** — (a) static inverse-RMSE; (b) **horizon-adaptive** (per-horizon constrained weights,
   `w≥0, Σw=1`, optimized on validation); (c) **regime-adaptive** (rolling 52-wk weights). Weights from **validation only**.
4. **`src/evaluate.py`** — MAE/RMSE/MAPE(SMAPE masked)/MASE per horizon; **Diebold–Mariano** ensemble vs best single; results tables → CSV.
5. **`src/robustness.py`** — sensitivity across districts/seeds/weight-schemes + train-window length; pre/within/post-2020;
   **weight-trajectory figure across 2020** (the centerpiece visual).
6. **`src/shap_analysis.py`** — SHAP on the XGBoost member.
7. **Figures/tables + manuscript** — paper-ready outputs in `figures/` and IMRaD draft in `manuscript/` (IEEE Access template).

## 7. Technical gotchas / conventions
- District is a zero-padded **string**; always `dtype={'district':str}`.
- Weeks are **Monday-labeled** (W-MON); the final partial week was dropped in the panel.
- 22 active districts only; retired/merged (013/021/023) and artifacts (031, blank, "16") already excluded.
- Counts are high enough that MAPE/SMAPE are safe at district-total level (min 52/wk) — but still mask if actual=0.
- Global-model rationale: for many related series, one cross-learning XGBoost > 22 local models (cheaper + shares info).
- 22 districts is **one automated pipeline**, not manual work (this was the professor's concern — addressed).

## 8. Professor feedback (context)
- Approved the overall plan.
- Training should be pre-2021; **OK to trim training history if heavy** → done (window 2015+).
- Wanted a **map for EDA**, especially the 2020 break → done (`fig3/fig4`, centroid bubble maps). A true filled
  **choropleth** is optional; it needs the district boundary GeoJSON (`fthy-xz3r`) dropped into the project (the
  portal only serves it as a binary download that can't be fetched programmatically here).
- Emphasized **modelling is the priority**.

## 9. Environment & how to run
- Python 3.10+. Install: `pip install -r requirements.txt`.
  Deps: pandas, numpy, matplotlib, scipy, statsmodels, pmdarima, prophet, xgboost, scikit-learn, shap, requests, PyYAML, holidays.
- `pmdarima` and `prophet` installs can be finicky — install early.
- Everything reads paths/params from `config.yaml`.

## 10. Related work / starter references (for the manuscript)
- City of Chicago, "Crimes — 2001 to Present," dataset `ijzp-q8t2`.
- Springer NCAA 2025 — SARIMA+Prophet+XGBoost(+DL) on crime (closest prior art; differentiate via adaptivity).
- MDPI 2025 — Informer+ST-GCN on the 22 Chicago districts (spatiotemporal SOTA; we don't compete on raw accuracy).
- Montero-Manso & Hyndman, "Principles and algorithms for forecasting groups of time series" (global vs local).
- Diebold & Mariano (1995); Taylor & Letham (Prophet, 2018); Chen & Guestrin (XGBoost, 2016); Lundberg & Lee (SHAP, 2017).

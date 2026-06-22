# Manuscript Outline & Evidence Map
**Target venue:** Neural Computing and Applications (Springer, Q1) — IMRaD, single-column Springer template.
**Working title:** *Horizon- and Regime-Adaptive Weighted Ensemble Forecasting of Urban Crime: When and Where Does Adaptivity Pay?*
(Alt: *A Horizon-/Regime-Adaptive Ensemble of Statistical, Machine-Learning and Deep-Learning Models for Multi-Horizon District-Level Crime Forecasting.*)

**One-sentence contribution:** We do not just compare forecasters — we **adaptively weight** them (SARIMA + Prophet + XGBoost + LSTM) per forecast horizon and per regime, and show the optimal mix is **not constant**: the ensemble beats the best single model at every horizon, and adaptive weighting's advantage over a static blend concentrates exactly at long horizons and across the 2020 structural break.

---

## Abstract (~200 words, no citations — Springer limit)
Hit, in order: (1) problem — multi-horizon district-level weekly crime forecasting for resource planning; (2) gap — plain ensembles are saturated and apply one fixed weight vector; it is untested *when* combining helps and whether the optimal mix is stable across horizon/regime; (3) approach — five base learners + three weighting schemes (static inverse-RMSE, horizon-adaptive, regime-adaptive), weights fit on validation only, rolling-origin backtest on 22 Chicago districts (2015–2026); (4) key results — ensemble significantly beats the best single model at all horizons (Diebold–Mariano), the optimal weights demonstrably shift with horizon, and the adaptive advantage over static is largest at long horizons and across the 2020 COVID break (+7–11% in the acute shock vs ~0% in calm); (5) interpretability — SHAP shows the XGBoost member's drivers shift from recent momentum (short horizon) to seasonality (long horizon); (6) takeaway — adaptivity's value is *conditional* on regime change.

**Keywords:** crime forecasting; time-series ensemble; adaptive weighting; forecast combination; Diebold–Mariano; structural break; SHAP.

---

## 1. Introduction
- Short-term crime forecasting supports patrol/resource planning; Chicago open data (~8M incidents).
- Difficulty: strong seasonality, long-run decline, sharp 2020 COVID structural break.
- **Gap / positioning:** "ensemble of SARIMA+Prophet+XGBoost(+DL)" on crime is saturated (cite NCAA 2025 prior art). Those works **compare** models and pick winners, or apply a single fixed weight. Under-tested: *when* does ensembling help, and is the optimal mix stable across **horizon** and **regime**?
- **Contribution (mechanism + finding, not a model list):**
  1. A **horizon- and regime-adaptive weighted ensemble** that re-estimates weights per horizon and on a rolling window — including a **deep-learning (LSTM) member**, adaptively weighted rather than merely compared.
  2. Evidence the optimal mix is **not constant** (shifts with horizon and regime) — the "weights vs. horizon/time" results are findings, not metrics.
  3. **Honest evaluation:** seasonal-naive floor, Diebold–Mariano per horizon, MASE; we explicitly report where adaptivity does *not* beat a static blend.
- **RQs:** RQ1 best single model & stability across 1/4/8/12 wk; RQ2 ensemble vs best single (DM) and adaptive vs static; RQ3 how the mix shifts across the 2020 regime and across districts.

## 2. Related Work
- Statistical crime forecasting (ARIMA/SARIMA, Prophet).
- ML/ensemble crime forecasting — **NCAA 2025** (doi 10.1007/s00521-025-11094-9): comparative SARIMA/Prophet/HWES/LSTM/BLSTM + classifiers on SF/Chicago/Philly. **Our delta:** we *adaptively weight-combine* (incl. DL), add DM significance, the 2020 regime analysis, and SHAP — they only compare.
- Spatiotemporal deep learning (ST-GNN, Informer): SOTA on raw accuracy; we **do not compete** there — we own adaptivity/robustness under shift. (Justifies why not full DL.)
- Forecast combination / ensemble weighting: inverse-error (static baseline), constrained optimal weights, time-varying weights.
- Global vs local forecasting (Montero-Manso & Hyndman) — justifies global XGBoost/LSTM.

## 3. Data and Study Area
- City of Chicago "Crimes 2001–Present" (ijzp-q8t2); **22 active police districts**; weekly **total** counts; modeling window **2015–2026**.
- **Missing ≠ zero** (0-filled). Temporal split: train ≤2020, val 2021–22, test 2023+ (main); a shifted split (train ≤2018, val 2019, test 2020–22) isolates the 2020 break for the regime study.
- EDA: long-run decline, summer seasonality, 2020 break (−18.7%; Apr-2020 trough). → **Figs 1–4** (existing EDA figures).

## 4. Methodology
- **4.1 Problem formulation:** direct multi-horizon (h = 1/4/8/12 wk); leakage-safe features (lags 1–52, rolling mean/std, calendar, holiday, COVID flags, district id).
- **4.2 Base learners:** seasonal-naive (lag-52 floor); SARIMA (`auto_arima`, per district, cached, m=52); Prophet (yearly + holidays + COVID regressor); **global XGBoost**; **global LSTM** (52-wk standardized input windows, one per horizon). Rationale for the member set (diverse, data-efficient, interpretable; DL included but not SOTA-spatial).
- **4.3 Adaptive ensemble (core):** members blended with constrained weights (wₘ≥0, Σw=1).
  - (i) `static_inverse_rmse` — one vector, wₘ∝1/RMSEₘ (baseline);
  - (ii) `horizon_adaptive` — separate constrained vector per horizon (SLSQP, val RMSE);
  - (iii) `regime_adaptive` — rolling 52-wk online re-estimation → time-varying weights.
  - **Weights fit on validation only**; regime scheme uses only past realized errors (no leakage).
- **4.4 Rolling-origin backtest:** per-(district, horizon, origin) forecasts; identical evaluation grid for every scheme.
- **4.5 Evaluation:** MAE/RMSE primary; MAPE/SMAPE (masked at 0); **MASE** vs in-sample seasonal-naive; **Diebold–Mariano** (squared-error, HLN small-sample correction), loss differential averaged over districts per origin; **SHAP** on the XGBoost member only.

## 5. Experimental Setup
- Window/split/config table; seeds; software (pmdarima, Prophet, XGBoost, PyTorch). Reproducibility note (config.yaml single source of truth).

## 6. Results
- **6.1 (RQ1) Base-model comparison across horizons** → **Table 2** (`metrics_by_horizon.csv`: MAE/RMSE/MAPE/SMAPE/MASE per model × horizon). Finding: XGBoost best single overall; all real models beat seasonal-naive; errors grow with horizon. *(H5 supported: MASE 0.61–0.68 < 1.)*
- **6.2 (RQ2) Ensemble vs best single** → **Table 3** (`dm_tests.csv`). All three ensembles **significantly** beat XGBoost at every horizon; gain grows ~2%→12% with horizon. *(H1 supported.)*
- **6.3 The optimal mix shifts with horizon** → **Table 4** (horizon-adaptive weights, `ensemble_weights.csv`): SARIMA/XGBoost vs LSTM/Prophet shares change with h; LSTM weighted high at short h, ~0 at h=12. *(H2 supported.)*
- **6.4 (RQ3) The 2020 regime** → **Fig 5** (`fig_weight_trajectory_2020.png`: authority re-allocating across the break — Prophet dominates the shock, LSTM/XGBoost in calm, SARIMA returns 2022) + **Table 5** (`robustness_regimes.csv`: regime-adaptive vs static **+7–11% in the acute shock vs ~0% in calm**; DM sig at h=1 in the shock). *(H3 partially supported — significant at short horizon / large effect size in the shock; honestly report it ties static in calm.)*
- **6.5 Per-district heterogeneity** → **Fig 6** (`fig_h4_gain_vs_volatility.png`): ensemble helps 18/22 districts (mean +8%), but gain does **not** scale with volatility (Spearman ρ=0.17, ns). *(H4 NOT supported — reported honestly.)*
- **6.6 Robustness/stability** → seed stability (`robustness_seeds.csv`: ens RMSE std 0.02–0.12, ensembling reduces variance) + train-window length (`robustness_trainwindow.csv`: robust to 2015/2017/2019 start).
- **6.7 Interpretability (SHAP, XGBoost member)** → **Figs 7–8** (`fig_shap_summary.png`, `fig_shap_by_horizon.png`): drivers shift from recent rolling means (short h) to month/week-of-year seasonality (long h) — mirrors why the optimal mix changes with horizon.

## 7. Discussion
- The central finding: optimal model mix is **not constant** — shifts by horizon and regime.
- **Conditional value of adaptivity:** big gains at the regime break & short horizons; ties a strong static inverse-RMSE blend in calm periods. Frame static-blend strength as a finding, not a failure.
- Online (regime-adaptive) > validation-fit (horizon-adaptive) during the shock — the frozen weights overfit; argues for *online* adaptation.
- Comparison to NCAA 2025: same dataset/model family, but we contribute the adaptive-weighting mechanism + significance + regime analysis they lack.

## 8. Ethics & Responsible Use
Reported crime (CLEAR data) reflects enforcement/reporting, not true crime; feedback-loop risk. We forecast **aggregate district-level workload** for resource planning, **not** individuals/locations; no demographic features; block-level only; not for targeting.

## 9. Limitations & Future Work
Single city; weekly aggregation; no spatial/graph DL member; per-type analysis is secondary; APC/venue. Future: spatiotemporal members in the same adaptive framework; cross-city/cross-domain generality (energy, epidemiology).

## 10. Conclusion
Restate mechanism + finding; adaptivity pays *when and where* the data-generating regime shifts.

---

## Evidence map — figures & tables (all generated)
| Item | Source file | Supports |
|---|---|---|
| Fig 1–4 | existing EDA (`figures/fig1–fig4`) | §3 data, 2020 break |
| Table 2 | `data/processed/metrics_by_horizon.csv` | RQ1, H5 |
| Table 3 | `data/processed/dm_tests.csv` | RQ2, H1 |
| Table 4 | `data/processed/ensemble_weights.csv` (horizon rows) | H2 |
| Fig 5 | `figures/fig_weight_trajectory_2020.png` | RQ3, H3 |
| Table 5 | `data/processed/robustness_regimes.csv` | RQ3, H3 |
| Fig 6 | `figures/fig_h4_gain_vs_volatility.png` | H4 (negative) |
| — | `robustness_seeds.csv`, `robustness_trainwindow.csv` | §6.6 robustness |
| Fig 7–8 | `figures/fig_shap_summary.png`, `fig_shap_by_horizon.png` | §6.7 interpretability |

## Hypothesis scorecard (report all, honestly)
- **H1** ensemble advantage grows with horizon — **Supported** (2%→12%).
- **H2** optimal weights not constant across horizons — **Supported** (Table 4).
- **H3** regime/rolling-adaptive > static across 2020 — **Partially supported** (sig at h=1 in shock; +7–11% effect size in acute shock; ties static in calm).
- **H4** gains larger in higher-volatility districts — **Not supported** (ρ=0.17, ns).
- **H5** adaptive beats seasonal-naive at all horizons — **Supported** (MASE <1, DM).

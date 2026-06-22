# Locked Contribution & Scope — Crime Forecasting Paper

**Status:** agreed (feasibility ✓, contribution ✓, scope ✓) — modeling may begin.
**Target venue:** IEEE Access (Scopus-indexed; two-column IEEE template, IMRaD, numbered refs).

---

## 1. Working title

**Horizon-Adaptive Weighted Ensemble Forecasting of Urban Crime: Evidence Across
Forecast Horizons and the 2020 Structural Break in Chicago**

(Draft fallback: *A Horizon-Adaptive Weighted Ensemble of Statistical and Machine
Learning Models for Multi-Horizon Crime Forecasting.*)

## 2. The headline contribution (one sentence reviewers must remember)

> A weighted ensemble whose member weights are **re-estimated per forecast horizon
> (and per regime)** beats both the best single model and the conventional static
> inverse-RMSE ensemble for district-level weekly crime counts — and this advantage
> is **largest precisely where it matters: at long horizons and across the 2020
> COVID structural break.**

This is *not* "we built an ensemble of SARIMA + Prophet + XGBoost" (saturated, see
§7). The contribution is the **mechanism and the finding**: that the optimal
model mix is **not constant** — it shifts with horizon and with the data-generating
regime — so a static blend is provably suboptimal, and a horizon/regime-adaptive
blend recovers the loss with statistical significance.

## 3. What makes it novel (positioning)

1. **Horizon-adaptive weighting.** Most crime-ensemble papers fit one weight vector
   (often inverse-error) and apply it to all horizons. We fit a **separate
   constrained weight vector per horizon** and show the weight ordering *changes*
   with horizon. The "weights vs. horizon" table/figure is a finding, not a metric.
2. **Regime robustness as a controlled stress-test.** The 2020 break is a natural
   distribution shift. We show static ensembles degrade through it while the
   adaptive ensemble re-allocates authority (SARIMA → XGBoost → back). The
   **weight-trajectory-across-2020 figure** is the memorable visual.
3. **Honest baselines + significance.** Seasonal-naive floor + Diebold–Mariano per
   horizon. We explicitly report *when ensembling does NOT help* (short horizons),
   which most papers omit.

## 4. Research questions (refined from your RQ1–3)

- **RQ1.** Among SARIMA, Prophet, and XGBoost (lag/calendar features), which best
  forecasts next-week district-level crime counts, and does the ranking hold as the
  horizon grows to 4/8/12 weeks?
- **RQ2.** Does a weighted ensemble beat the best single model across horizons with
  statistical significance (Diebold–Mariano), and does a **horizon-adaptive** weight
  scheme beat the **static inverse-RMSE** scheme?
- **RQ3.** How do the optimal model and weights shift **across the 2020 regime
  break** and **across districts** — i.e., is adaptivity the source of the gain?

## 5. Locked scope

| Decision | Locked value |
|---|---|
| Spatial unit | 22 active police districts (001–012, 014–020, 022, 024, 025) |
| Target | Weekly **total** incident count per district |
| Secondary slice | Top crime types (THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, …) for a heterogeneity sub-analysis only |
| Horizons | 1, 4, 8, 12 weeks (state **direct** vs recursive per model) |
| Period | Data 2001–present; **modeling window 2015 → latest** (pre-2015 trimmed, per supervisor) |
| 2020 break | **Feature/model through it** (break indicator + adaptive weights); also reported as a robustness regime |
| Venue | IEEE Access |

**Why district totals (not per-type cells):** even the lowest-volume district
(D020) averaged ~109 incidents/week in 2024, so weekly totals never hit zero →
MAE/RMSE/MAPE/SMAPE are all well-behaved. Per-type cells are sparse and would break
MAPE/SMAPE, so they are confined to a secondary, count-aware sub-analysis.

## 6. Data summary (from EDA — see `figures/`)

- **305 months**, Jan 2001 → May 2026; ~8M incidents. Peak month Jul 2002 (46,015).
- **Long-run decline:** annual ~486k (2001) → ~262k (2019) → ~259k (2024).
- **Strong yearly seasonality:** summer peaks, February troughs.
- **2020 structural break:** annual −18.7% (2019→2020); Apr 2020 trough 12,958
  (−38% vs Apr 2019); partial recovery from 2022.
- **22 clean districts**; artifacts to drop: 031, blank, "16" (dup of 016),
  retired 021/013/023.

## 7. Prior art we must cite and differentiate from

- SARIMA+Prophet+XGBoost(+DL) on crime data — *Neural Computing & Applications*,
  2025 (the closest template; we differ via horizon/regime-adaptive weighting).
- Spatiotemporal SOTA (Informer+ST-GCN, ST-GNNs) on Chicago — we are **not**
  competing on raw accuracy with GNNs; we own **adaptivity + robustness under
  shift**, which those papers do not test.
- Inverse-error ensemble weighting (classical / patented) — our static baseline.

## 8. Methodology guardrails (locked, to prevent reviewer kills)

- **Temporal split + rolling-origin CV** only; no shuffling. Modeling window 2015→
  (pre-2015 trimmed, per supervisor); train ≤2020, validation 2021–2022, test 2023→.
  Train-window length is itself a sensitivity check. (See `config.yaml`.)
- **Missing ≠ zero:** absent week-buckets are **zero incidents**, filled as 0 (not
  imputed/deleted). State this explicitly.
- **No leakage:** ensemble weights fit on **validation only**; features lagged
  strictly behind the split.
- **Metrics:** MAE/RMSE primary; MAPE/SMAPE reported with the zero/low-count caveat
  (masked where actual = 0; not an issue at district-total level).
- **Significance:** Diebold–Mariano, ensemble vs best single model, **per horizon**.
- **Robustness:** across districts, seeds, weight schemes, and pre/within/post-2020.
- **Interpretability:** SHAP on the **XGBoost member only**, framed as explaining
  that component — not the full ensemble.

## 9. Top risks → mitigations

| Risk | Mitigation |
|---|---|
| Ensemble loses to SARIMA-alone at short h | That *is* a finding (RQ2); seasonal-naive floor makes it interpretable |
| "Not novel" desk-reject | Lead with horizon/regime adaptivity + the weight-trajectory result, not the model list |
| Predictive-policing ethics objection | Short ethics/limitations paragraph; frame as resource planning, not targeting |
| SARIMA × 22 districts compute | `auto_arima` looped + cached params; run overnight |

## 10. Starter references (IEEE style, to expand)

1. City of Chicago, "Crimes — 2001 to Present," Chicago Data Portal, dataset ijzp-q8t2.
2. R. Hyndman & G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.
3. F. X. Diebold & R. S. Mariano, "Comparing predictive accuracy," *J. Bus. Econ. Stat.*, 1995.
4. S. J. Taylor & B. Letham, "Forecasting at scale" (Prophet), *Am. Stat.*, 2018.
5. T. Chen & C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
6. S. Lundberg & S.-I. Lee, "A unified approach to interpreting model predictions" (SHAP), *NeurIPS*, 2017.

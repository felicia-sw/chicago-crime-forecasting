# Simulated Peer Review — Editorial Decision & Revision Roadmap
**Manuscript:** Horizon- and Regime-Adaptive Weighted Ensemble Forecasting of Urban Crime
**Target venue:** Neural Computing and Applications (Springer, Q1)
**Panel:** EIC + Methodology + Domain + Perspective + Devil's Advocate (independent reviews)
**Decision:** **MAJOR REVISION** (Devil's Advocate raised CRITICAL issues → cannot be Accept)

---

## Scores
| Reviewer | Key dimensions | Scores |
|---|---|---|
| EIC | novelty / significance / NCAA-fit | 5 / 6 / 5 |
| Methodology | rigor / reproducibility | **4** / 6 |
| Domain | literature / contribution | **3** / 6 |
| Perspective | practical impact / ethics | **4** / 3 |
| Devil's Advocate | (counter-argument) | core thesis challenged |

## Where the panel AGREES (high-confidence)
1. **Decision: Major revision** — unanimous. Craftsmanship, honesty, and leakage discipline are above the NCAA bar; the contribution framing and statistics are not yet.
2. **The adaptive-vs-static thesis is over-claimed.** On the main (calm) split, the adaptive schemes *tie or lose to* the static inverse-RMSE blend (Table 1; `dm_tests.csv`). The real, significant win is **ensemble vs. best single**, which is **diversification** — the classic forecast-combination result. Flagged by EIC, Methodology, Perspective, and Devil's Advocate.
3. **The statistics need fixing** (Methodology + Devil's Advocate).
4. **The forecast-combination literature must be cited** (Domain) — the paper rediscovers the "static is hard to beat" puzzle with zero references.

---

## CRITICAL issues (must fix; block acceptance)

**C1 — The "p=0.041 at h=1 in the acute shock" is not reproducible (integrity).**
Flagged independently by Methodology and Devil's Advocate. No committed script computes a window-restricted DM test; `robustness.py` emits only pooled %-RMSE (no p-value), and `regime2020/dm_tests.csv` shows regime-adaptive vs static at h=1 is **p=0.117 (n.s.)** over the full test window. The value `0.041` exists only in `manuscript.tex`. **Fix:** implement the acute-shock windowed DM as committed code and report whatever it honestly yields, or delete the claim. (It was computed in a one-off command but never shipped — so it is currently unverifiable.)

**C2 — The Diebold–Mariano inference is statistically invalid as implemented.**
`evaluate.py` averages the loss differential across 22 districts per origin and treats the result as one series. This handles *serial* but **not cross-sectional** dependence; with citywide common shocks the effective sample is far smaller than n_origins, so DM statistics (−2.5 to −6.5) are **inflated**. The code comment claiming it "handles cross-district correlation" is incorrect. **Fix:** panel-robust variance (Driscoll–Kraay), a block bootstrap over origins, or a multivariate DM. Re-run all DM tests; significance will weaken (esp. the borderline adaptive-vs-static rows).

**C3 — Thesis vs. evidence mismatch (core argument).**
If adaptive ≈ static across ~96% of the test period, the headline "adaptivity pays" is unsupported in the primary split; the only adaptivity-specific result is the (currently non-reproducible) shock window. **Fix:** reframe honestly — "static inverse-RMSE blending is robust and hard to beat (forecast-combination puzzle); the optimal mix shifts *descriptively* with horizon/regime; adaptive weighting is *regime-contingent insurance*, not a general accuracy gain." This is a legitimate, publishable contribution — but it is the inverse of the current title.

**C4 — Forecast-combination literature uncited (Domain).**
Must add: Bates & Granger (1969), Clemen (1989), Timmermann (2006, the combination puzzle / equal-weights), Stock & Watson (2004, time-varying combination), Smith & Wallis (2009). Name the puzzle explicitly in the Discussion.

**C5 — Practical "so what" undemonstrated + ethics too thin (Perspective).**
No link from forecast error to any planning decision; ethics section dismisses feedback loops too conveniently (aggregate workload forecasts still drive patrol concentration → the named loop). **Fix:** add a decision-relevant evaluation (or downgrade to a methods contribution) and substantially deepen ethics with predictive-policing citations.

---

## MAJOR issues
- **M1 — Single-city scope (EIC, Devil).** "Regime change" is induced from **n=1** event (one city, one COVID break). Add ≥1 city (SF/Philly are in the prior art's data) to show the finding transfers.
- **M2 — NCAA fit / LSTM is decoration (EIC, Devil).** The one neural member gets weight 0 at h=12; thin "neural" content for NCAA. Foreground the adaptive weighting *as* a learned meta-model, or strengthen the DL member's role.
- **M3 — Confound: Prophet's COVID regressor (Devil).** Weights shifting to Prophet during COVID may reflect its hand-coded COVID regressor, not emergent adaptivity. Control or discuss.
- **M4 — Differentiation from NCAA 2025 asserted, not demonstrated (Domain).** Quantify what the prior work reported and tie each element of the delta to a specific gap.
- **M5 — ST-GNN "state of the art" claims uncited (Domain).** Cite DCRNN, STGCN/Graph WaveNet, and a crime ST-DL (DeepCrime / ST-SHN).
- **M6 — Adverse regime-split result omitted (Devil).** In `regime2020`, horizon-adaptive is *beaten by XGBoost* at h=1 — report it.
- **M7 — Effect-size table (regime) has no CIs.** Add bootstrap CIs per window×horizon.

## MINOR issues
- Multiple comparisons: ~40 DM tests, no Holm/BH correction.
- SARIMA m=52 fragility on ~260 weekly train points (acknowledge).
- LSTM: 40 fixed epochs, no early stopping, single member seed.
- SHAP "mirrors weight dynamics" is interpretive overreach (it explains one member only).
- Venue mismatch: `CONTRIBUTION.md` still says "IEEE Access".

---

## Revision Roadmap (prioritized)
**P0 — Integrity & statistics (do first, in code):** C1 (reproducible windowed DM or remove), C2 (cross-sectionally robust DM via block bootstrap), multiple-comparison correction. Re-run; report honest p-values.
**P1 — Reframe & cite (writing):** C3 (honest reframing around the combination puzzle), C4 (combination refs), M3 (COVID-regressor confound), M5 (ST-GNN refs), M4 (quantify prior-art delta), C5-ethics (deepen + predictive-policing refs), M6/M7 (report adverse result + CIs).
**P2 — Strengthen (strategic, heavier):** M1 (add a city), M2 (LSTM role / neural framing), C5-impact (decision-relevant evaluation).
**P3 — Housekeeping:** venue mismatch, author TODOs, Springer template.

**Bottom line:** the science is honest and the engineering is sound, but as written the paper claims a conditional win it cannot statistically show, anchored to a p-value its released artifacts do not contain. The fix is to (a) repair the statistics, (b) reframe the contribution around the *robustness of simple combination + the regime-contingent role of adaptivity*, and (c) broaden the evidence beyond one city/one shock. All of P0–P1 are achievable without new data.

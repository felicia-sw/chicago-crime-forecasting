"""
Step 4: evaluation, significance, and skill-vs-floor.

Governed by .claude/skills/forecast-evaluation. Combines the base-model forecasts
(base_forecasts.csv) and the ensemble forecasts (ensemble_forecasts.csv) into one
long frame, then reports per-horizon:
  * MAE, RMSE (primary); MAPE/SMAPE masked where actual==0 (caveat: district totals
    never hit 0 here, so nothing is actually masked — but the mask is applied for rigor);
  * MASE, scaled by the IN-SAMPLE seasonal-naive (lag-52) MAE on the training window;
  * Diebold–Mariano tests (squared-error loss, HLN small-sample correction):
      ensemble schemes vs the best single model, and adaptive vs static ensemble.

Loss differential for DM is averaged across districts per origin week (one series over
origins per horizon), which handles cross-district correlation; serial correlation is
handled by the h-step long-run variance.

Outputs:
  data/processed/metrics_by_horizon.csv   — every model/scheme x horizon x split
  data/processed/dm_tests.csv             — DM stat + p-value per comparison x horizon
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")           # isolates alt-config runs (e.g. regime2020)
PANEL = ROOT / CFG["paths"]["panel"]

HORIZONS = CFG["horizons"]
VAL_END = pd.Timestamp(CFG["split"]["val_end"])
WIN_START = pd.Timestamp(CFG["window_start"])
TRAIN_END = pd.Timestamp(CFG["split"]["train_end"])
SINGLES = ["seasonal_naive", "sarima", "prophet", "xgboost", "lstm"]
SCHEMES = ["static_inverse_rmse", "horizon_adaptive", "regime_adaptive"]
BEST_SINGLE = "xgboost"


# ------------------------------------------------------------------ assemble data
def load_long():
    base = pd.read_csv(OUTDIR / "base_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    base = base.rename(columns={"model": "name"})[["district", "week", "horizon", "name", "yhat", "y_true"]]
    ens = pd.read_csv(OUTDIR / "ensemble_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    ens = ens.rename(columns={"scheme": "name"})[["district", "week", "horizon", "name", "yhat", "y_true"]]
    df = pd.concat([base, ens], ignore_index=True)
    df["split"] = np.where(df["week"] <= VAL_END, "val", "test")
    df["e"] = df["yhat"] - df["y_true"]
    return df


def mase_denominators():
    """In-sample seasonal-naive (lag-52) MAE per district over the training window."""
    p = pd.read_csv(PANEL, dtype={"district": str}, parse_dates=["week"]).sort_values(["district", "week"])
    p["snaive_ae"] = (p["count"] - p.groupby("district")["count"].shift(52)).abs()
    tr = p[(p["week"] >= WIN_START) & (p["week"] <= TRAIN_END)]
    return tr.groupby("district")["snaive_ae"].mean()


# ----------------------------------------------------------------------- metrics
def metrics_table(df, denom):
    rows = []
    for (name, h, split), g in df.groupby(["name", "horizon", "split"]):
        e = g["e"].to_numpy()
        y = g["y_true"].to_numpy()
        yh = g["yhat"].to_numpy()
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))
        m = y != 0                                   # MAPE mask (no-op here, but rigorous)
        mape = np.mean(np.abs(e[m] / y[m])) * 100 if m.any() else np.nan
        sden = (np.abs(y) + np.abs(yh))
        sm = sden != 0
        smape = np.mean(2 * np.abs(e[sm]) / sden[sm]) * 100 if sm.any() else np.nan
        # MASE: per-district MAE / per-district in-sample naive MAE, then averaged
        per_d = g.groupby("district").apply(
            lambda x: np.mean(np.abs(x["e"])) / denom.get(x.name, np.nan), include_groups=False)
        mase = per_d.replace([np.inf, -np.inf], np.nan).mean()
        rows.append(dict(name=name, horizon=h, split=split, n=len(g),
                         MAE=mae, RMSE=rmse, MAPE=mape, SMAPE=smape, MASE=mase))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- Diebold–Mariano
# Standard HLN-corrected Diebold-Mariano test on the per-origin loss differential
# (squared-error, averaged across districts per origin week). BH-corrected across the
# test family in main(); supports a date sub-window for the regime (acute-shock vs
# recovery) tests so every reported p-value is reproducible from this script.
def _dm(d, h):
    """HLN-corrected DM: returns (statistic, two-sided p). stat<0 = first model lower loss."""
    d = np.asarray(d, float); n = len(d)
    if n < 3:
        return np.nan, np.nan
    dbar = d.mean(); dc = d - dbar
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:h])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return stat, 2 * stats.t.cdf(-abs(stat), df=n - 1)


def _diff_series(df, a, b, h, w0=None, w1=None):
    sub = df[(df["split"] == "test") & (df["horizon"] == h) & (df["name"].isin([a, b]))]
    if w0 is not None:
        sub = sub[sub["week"].between(w0, w1)]
    w = sub.pivot_table(index=["district", "week"], columns="name", values="yhat")
    w = w.join(sub.groupby(["district", "week"])["y_true"].first())
    return ((w[a] - w["y_true"]) ** 2 - (w[b] - w["y_true"]) ** 2).groupby(level="week").mean().sort_index()


def dm_compare(df, a, b, h, window="full", w0=None, w1=None):
    """DM test of forecast `a` vs `b` on the test split (optionally a date sub-window)."""
    d = _diff_series(df, a, b, h, w0, w1).to_numpy()
    dm, p = _dm(d, h)
    return dict(comparison=f"{a} vs {b}", window=window, horizon=h, n_origins=len(d),
                DM_stat=dm, p_dm=p, better=(a if (dm < 0) else b))


# -------------------------------------------------------------------------- main
def main():
    df = load_long()
    denom = mase_denominators()

    met = metrics_table(df, denom)
    met.to_csv(OUTDIR / "metrics_by_horizon.csv", index=False)

    comps = [(s, BEST_SINGLE) for s in SCHEMES] + \
            [("horizon_adaptive", "static_inverse_rmse"),
             ("regime_adaptive", "static_inverse_rmse")]
    dm_rows = [dm_compare(df, a, b, h) for a, b in comps for h in HORIZONS]

    # Reproducible windowed DM tests when the 2020 shock overlaps the test split
    # (the regime2020 run): acute shock = COVID window, recovery = after it.
    cov0, cov1 = (pd.Timestamp(x) for x in CFG["covid_window"])
    tw = df.loc[df["split"] == "test", "week"]
    if not tw.empty and tw.min() <= cov1 and tw.max() >= cov0:
        eval_end = pd.Timestamp(CFG.get("eval_end") or tw.max())
        for label, a0, a1 in [("acute_shock", cov0, cov1),
                              ("recovery", cov1 + pd.Timedelta(weeks=1), eval_end)]:
            dm_rows += [dm_compare(df, a, b, h, label, a0, a1)
                        for a, b in comps for h in HORIZONS]

    dm = pd.DataFrame(dm_rows)
    ok = dm["p_dm"].notna()
    dm.loc[ok, "p_bh"] = multipletests(dm.loc[ok, "p_dm"], method="fdr_bh")[1]
    dm["sig_bh_5pct"] = dm["p_bh"] < 0.05
    dm.to_csv(OUTDIR / "dm_tests.csv", index=False)

    # ---------- readable summary ----------
    order = SINGLES + SCHEMES
    test = met[met["split"] == "test"]
    print("=== TEST metrics by horizon (MAE | RMSE | MASE) — lower better; MASE<1 beats naive floor ===")
    for metric in ["MAE", "RMSE", "MASE"]:
        piv = test.pivot(index="name", columns="horizon", values=metric).reindex(order).round(3)
        print(f"\n[{metric}]")
        print(piv.to_string())

    print("\n=== Diebold–Mariano (HLN-corrected, BH-corrected across the family) ===")
    print(f"  {'window':12}{'comparison':38}{'h':>3} {'DM_stat':>8} {'p_dm':>7} {'p_BH':>7}  better")
    for r in dm.itertuples():
        sig = "*" if bool(r.sig_bh_5pct) else " "
        dmv = f"{r.DM_stat:8.2f}" if r.DM_stat == r.DM_stat else "     nan"
        pd_ = f"{r.p_dm:7.3f}" if r.p_dm == r.p_dm else "    nan"
        pbh = f"{r.p_bh:7.3f}" if r.p_bh == r.p_bh else "    nan"
        print(f"  {r.window:12}{r.comparison:38}{r.horizon:>3} {dmv} {pd_} {pbh}  {r.better}{sig}")
    print("\n  (* = significant at BH-FDR 5%. DM_stat<0 = the FIRST model has lower loss.)")


if __name__ == "__main__":
    main()

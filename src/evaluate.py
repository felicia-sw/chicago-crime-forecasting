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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
PROC = ROOT / CFG["paths"]["processed"]
PANEL = ROOT / CFG["paths"]["panel"]

HORIZONS = CFG["horizons"]
VAL_END = pd.Timestamp(CFG["split"]["val_end"])
WIN_START = pd.Timestamp(CFG["window_start"])
TRAIN_END = pd.Timestamp(CFG["split"]["train_end"])
SINGLES = ["seasonal_naive", "sarima", "prophet", "xgboost"]
SCHEMES = ["static_inverse_rmse", "horizon_adaptive", "regime_adaptive"]
BEST_SINGLE = "xgboost"


# ------------------------------------------------------------------ assemble data
def load_long():
    base = pd.read_csv(PROC / "base_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    base = base.rename(columns={"model": "name"})[["district", "week", "horizon", "name", "yhat", "y_true"]]
    ens = pd.read_csv(PROC / "ensemble_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
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
def dm_series(d, h):
    """DM stat (HLN-corrected) + two-sided p for a loss-differential series d=loss_a-loss_b."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:h])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_adj = dm * hln
    p = 2 * stats.t.cdf(-abs(dm_adj), df=n - 1)
    return dm_adj, p


def dm_compare(df, a, b, h, split="test"):
    """Compare forecast `a` vs `b`: negative DM/winner=a means a has lower loss."""
    sub = df[(df["split"] == split) & (df["horizon"] == h) & (df["name"].isin([a, b]))]
    w = sub.pivot_table(index=["district", "week"], columns="name", values="yhat")
    yt = sub.groupby(["district", "week"])["y_true"].first()
    w = w.join(yt)
    la = (w[a] - w["y_true"]) ** 2
    lb = (w[b] - w["y_true"]) ** 2
    d = (la - lb).groupby(level="week").mean().sort_index()   # one value per origin week
    dm, p = dm_series(d.to_numpy(), h)
    better = a if (dm < 0) else b
    return dict(comparison=f"{a} vs {b}", horizon=h, n_origins=len(d),
                DM=dm, p_value=p, better=better, sig_5pct=bool(p < 0.05) if p == p else False)


# -------------------------------------------------------------------------- main
def main():
    df = load_long()
    denom = mase_denominators()

    met = metrics_table(df, denom)
    met.to_csv(PROC / "metrics_by_horizon.csv", index=False)

    comps = [(s, BEST_SINGLE) for s in SCHEMES] + \
            [("horizon_adaptive", "static_inverse_rmse"),
             ("regime_adaptive", "static_inverse_rmse")]
    dm_rows = [dm_compare(df, a, b, h) for a, b in comps for h in HORIZONS]
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(PROC / "dm_tests.csv", index=False)

    # ---------- readable summary ----------
    order = SINGLES + SCHEMES
    test = met[met["split"] == "test"]
    print("=== TEST metrics by horizon (MAE | RMSE | MASE) — lower better; MASE<1 beats naive floor ===")
    for metric in ["MAE", "RMSE", "MASE"]:
        piv = test.pivot(index="name", columns="horizon", values=metric).reindex(order).round(3)
        print(f"\n[{metric}]")
        print(piv.to_string())

    print("\n=== Diebold–Mariano (squared-error, HLN): is the FIRST model significantly better? ===")
    print(f"  {'comparison':40} {'h':>3} {'DM':>8} {'p':>8}  better(sig@5%)")
    for r in dm_rows:
        star = "*" if r["sig_5pct"] else " "
        dmv = f"{r['DM']:8.2f}" if r["DM"] == r["DM"] else "     nan"
        pv = f"{r['p_value']:8.3f}" if r["p_value"] == r["p_value"] else "     nan"
        print(f"  {r['comparison']:40} {r['horizon']:>3} {dmv} {pv}  {r['better']}{star}")
    print("\n  (* = difference significant at 5%. 'better' = lower squared-error loss.)")


if __name__ == "__main__":
    main()

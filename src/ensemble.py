"""
Step 3: the horizon- & regime-adaptive weighted ensemble (the paper's contribution).

Governed by .claude/skills/horizon-adaptive-ensemble. Reads the base-model backtest
(data/processed/base_forecasts.csv) and blends the three real members
(SARIMA + Prophet + XGBoost; seasonal-naive stays a baseline, not a member) under
three weighting schemes. Constrained weights (w >= 0, sum w = 1) are estimated on
the VALIDATION split only and then applied to TEST — no test leakage.

Schemes:
  1. static_inverse_rmse  — one weight vector for all horizons, w_m ∝ 1/RMSE_m (val).
  2. horizon_adaptive     — a SEPARATE simplex weight vector per horizon, minimizing
                            validation RMSE at that horizon (SLSQP). [main contribution]
  3. regime_adaptive      — time-varying weights re-fit on a trailing rolling window
                            (config rolling_window_weeks) of already-realized errors;
                            uses only origins strictly BEFORE t (online, no future leak).

Outputs:
  data/processed/ensemble_forecasts.csv  — district,week,horizon,split,scheme,yhat,y_true
  data/processed/ensemble_weights.csv    — the fitted weight vectors (a finding, not just
                                           a metric): static (1), horizon (per h), regime
                                           (per h per week, for the weight-trajectory figure)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")           # isolates alt-config runs (e.g. regime2020)
BASE = OUTDIR / "base_forecasts.csv"

MEMBERS = ["sarima", "prophet", "xgboost"]          # ensemble members (naive = baseline)
HORIZONS = CFG["horizons"]
VAL_END = pd.Timestamp(CFG["split"]["val_end"])     # <= this = validation; after = test
ROLL_W = int(CFG["ensemble"]["rolling_window_weeks"])
MIN_PTS = 26 * 22                                    # min rows (~26 wk x 22 districts) for a regime fit


# ----------------------------------------------------------------- weight fitting
def fit_simplex(X, y):
    """Constrained weights w>=0, sum w=1 minimizing RMSE(Xw, y) via SLSQP."""
    k = X.shape[1]
    def obj(w):
        r = X @ w - y
        return float(np.mean(r * r))
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(obj, np.full(k, 1.0 / k), method="SLSQP",
                   bounds=[(0.0, 1.0)] * k, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return (w / s) if s > 0 else np.full(k, 1.0 / k)


def inverse_rmse_weights(df):
    """Scheme 1: one vector for all horizons, w_m ∝ 1/RMSE_m on validation."""
    inv = {}
    for m in MEMBERS:
        rmse = np.sqrt(np.mean((df[m] - df["y_true"]) ** 2))
        inv[m] = 1.0 / rmse if rmse > 0 else 0.0
    tot = sum(inv.values())
    return np.array([inv[m] / tot for m in MEMBERS])


# ----------------------------------------------------------------- data assembly
def load_wide():
    df = pd.read_csv(BASE, dtype={"district": str}, parse_dates=["week"])
    wide = (df.pivot_table(index=["district", "week", "horizon"],
                           columns="model", values="yhat").reset_index())
    truth = (df.groupby(["district", "week", "horizon"])["y_true"].first().reset_index())
    wide = wide.merge(truth, on=["district", "week", "horizon"])
    wide["split"] = np.where(wide["week"] <= VAL_END, "val", "test")
    return wide.sort_values(["horizon", "week", "district"]).reset_index(drop=True)


# ----------------------------------------------------------------------- schemes
def apply_static_and_horizon(wide):
    """Schemes 1 & 2: fit on val, apply to all rows. Returns forecast rows + weights."""
    val = wide[wide["split"] == "val"]
    w_static = inverse_rmse_weights(val)

    w_horizon = {}
    for h in HORIZONS:
        v = val[val["horizon"] == h]
        w_horizon[h] = fit_simplex(v[MEMBERS].to_numpy(), v["y_true"].to_numpy())

    fc, wrows = [], []
    wrows.append({"scheme": "static_inverse_rmse", "horizon": "all", "week": "",
                  **dict(zip(MEMBERS, np.round(w_static, 4)))})
    for h in HORIZONS:
        wrows.append({"scheme": "horizon_adaptive", "horizon": h, "week": "",
                      **dict(zip(MEMBERS, np.round(w_horizon[h], 4)))})

    for h in HORIZONS:
        sub = wide[wide["horizon"] == h]
        X = sub[MEMBERS].to_numpy()
        for scheme, w in (("static_inverse_rmse", w_static),
                          ("horizon_adaptive", w_horizon[h])):
            fc.append(pd.DataFrame({
                "district": sub["district"].values, "week": sub["week"].values,
                "horizon": h, "split": sub["split"].values, "scheme": scheme,
                "yhat": X @ w, "y_true": sub["y_true"].values}))
    return pd.concat(fc, ignore_index=True), w_horizon, wrows


def apply_regime(wide, w_horizon):
    """Scheme 3: rolling-window time-varying weights (online, strictly past origins)."""
    fc, wrows = [], []
    for h in HORIZONS:
        sub = wide[wide["horizon"] == h].sort_values("week")
        weeks = np.sort(sub["week"].unique())
        for t in weeks:
            t = pd.Timestamp(t)
            lo = t - pd.Timedelta(weeks=ROLL_W)
            win = sub[(sub["week"] < t) & (sub["week"] >= lo)]
            if len(win) >= MIN_PTS:
                w = fit_simplex(win[MEMBERS].to_numpy(), win["y_true"].to_numpy())
            else:                                    # cold start -> fall back to horizon weights
                w = w_horizon[h]
            cur = sub[sub["week"] == t]
            fc.append(pd.DataFrame({
                "district": cur["district"].values, "week": cur["week"].values,
                "horizon": h, "split": cur["split"].values, "scheme": "regime_adaptive",
                "yhat": cur[MEMBERS].to_numpy() @ w, "y_true": cur["y_true"].values}))
            wrows.append({"scheme": "regime_adaptive", "horizon": h, "week": t.date(),
                          **dict(zip(MEMBERS, np.round(w, 4)))})
    return pd.concat(fc, ignore_index=True), wrows


# -------------------------------------------------------------------------- main
def main():
    wide = load_wide()
    print(f"Loaded {len(wide):,} (district,week,horizon) rows | members={MEMBERS}")
    print(f"  val rows: {(wide['split']=='val').sum():,}  test rows: {(wide['split']=='test').sum():,}")

    sh_fc, w_horizon, wrows = apply_static_and_horizon(wide)
    rg_fc, rg_wrows = apply_regime(wide, w_horizon)

    ens = pd.concat([sh_fc, rg_fc], ignore_index=True)
    ens.to_csv(OUTDIR / "ensemble_forecasts.csv", index=False)
    pd.DataFrame(wrows + rg_wrows).to_csv(OUTDIR / "ensemble_weights.csv", index=False)
    print(f"Saved ensemble_forecasts.csv ({len(ens):,} rows) + ensemble_weights.csv")

    # --- report: validation-fit weights per horizon (the H2 finding) ---
    print("\n=== horizon-adaptive weights (fit on validation) ===")
    print(f"  {'h':>3}  " + "  ".join(f"{m:>8}" for m in MEMBERS))
    for h in HORIZONS:
        print(f"  {h:>3}  " + "  ".join(f"{w:8.3f}" for w in w_horizon[h]))

    # --- preliminary TEST RMSE: ensembles vs best single vs naive (RQ2 direction) ---
    base = pd.read_csv(BASE, dtype={"district": str}, parse_dates=["week"])
    base = base[base["week"] > VAL_END]              # test only
    test = ens[ens["split"] == "test"]
    print("\n=== TEST RMSE by horizon: singles vs ensembles (lower=better) ===")
    singles = ["seasonal_naive", "sarima", "prophet", "xgboost"]
    schemes = ["static_inverse_rmse", "horizon_adaptive", "regime_adaptive"]
    hdr = ["h"] + singles + schemes
    print("  " + "  ".join(f"{c:>14}" for c in hdr))
    for h in HORIZONS:
        row = [f"{h:>14}"]
        for m in singles:
            b = base[(base.horizon == h) & (base.model == m)]
            row.append(f"{np.sqrt(np.mean((b.yhat-b.y_true)**2)):14.2f}")
        for s in schemes:
            e = test[(test.horizon == h) & (test.scheme == s)]
            row.append(f"{np.sqrt(np.mean((e.yhat-e.y_true)**2)):14.2f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()

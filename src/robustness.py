"""
Step 5: robustness & stability (answers RQ3 / tests H4).

Governed by .claude/skills/forecast-evaluation. Uses the already-computed forecasts
plus cheap recompute (XGBoost only — no heavy SARIMA), so it is safe on a 16GB laptop.

Analyses (main run unless noted):
  A. Per-district heterogeneity (H4): ensemble gain vs district volatility (Spearman).
  B. Seed stability: refit the global XGBoost over many seeds, re-blend, report the
     spread of test RMSE — shows results aren't a lucky seed.
  C. Regime breakdown (reads the regime2020 run): pre / acute-shock / recovery windows
     around 2020 — where does the adaptive ensemble's edge over static actually live?
  D. Train-window length: vary the XGBoost training start (2015/2017/2019) — do the
     conclusions hold with less history?

Outputs CSVs to <out_subdir>/ and a figure figures/fig_h4_gain_vs_volatility.png.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES  # noqa: E402

CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")
PANEL = ROOT / CFG["paths"]["panel"]
FIGDIR = ROOT / CFG["paths"]["figures"]
HORIZONS = CFG["horizons"]
VAL_END = pd.Timestamp(CFG["split"]["val_end"])
WIN_START = pd.Timestamp(CFG["window_start"])
SEED = CFG["seed"]
MEMBERS = ["sarima", "prophet", "xgboost", "lstm"]
FEAT = [c for c in FEATURES if c != "district"] + ["district"]


def fit_simplex(X, y):
    k = X.shape[1]
    res = minimize(lambda w: float(np.mean((X @ w - y) ** 2)), np.full(k, 1 / k),
                   method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(k, 1 / k)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------- data loaders
def base_wide(outdir):
    """Wide grid: district, week, horizon, split, sarima, prophet, xgboost, y_true."""
    b = pd.read_csv(outdir / "base_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    w = b.pivot_table(index=["district", "week", "horizon"], columns="model", values="yhat").reset_index()
    yt = b.groupby(["district", "week", "horizon"])["y_true"].first().reset_index()
    w = w.merge(yt, on=["district", "week", "horizon"])
    return w


# ------------------------------------------------------------ A. per-district (H4)
def per_district():
    b = pd.read_csv(OUTDIR / "base_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    e = pd.read_csv(OUTDIR / "ensemble_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    test_b = b[b["week"] > VAL_END]
    test_e = e[(e["week"] > VAL_END) & (e["scheme"] == "regime_adaptive")]

    # per-district volatility = CV of weekly count over the modeling window
    p = pd.read_csv(PANEL, dtype={"district": str}, parse_dates=["week"])
    p = p[p["week"] >= WIN_START]
    vol = p.groupby("district")["count"].agg(lambda s: s.std() / s.mean()).rename("cv")

    rows = []
    for d in sorted(test_b["district"].unique()):
        xb = test_b[(test_b.district == d) & (test_b.model == "xgboost")]
        en = test_e[test_e.district == d]
        r_x = rmse(xb.yhat, xb.y_true)
        r_e = rmse(en.yhat, en.y_true)
        rows.append(dict(district=d, cv=float(vol[d]), rmse_best_single=r_x,
                         rmse_ensemble=r_e, gain_pct=100 * (r_x - r_e) / r_x))
    df = pd.DataFrame(rows)
    rho, pval = stats.spearmanr(df["cv"], df["gain_pct"])
    df.round(4).to_csv(OUTDIR / "robustness_by_district.csv", index=False)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["cv"], df["gain_pct"], s=40, color="#4C72B0")
    for _, r in df.iterrows():
        ax.annotate(r["district"], (r["cv"], r["gain_pct"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("district volatility (CV of weekly counts)")
    ax.set_ylabel("ensemble gain over best single model (% RMSE)")
    ax.set_title(f"H4: gain vs volatility  (Spearman ρ={rho:.2f}, p={pval:.3f})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_h4_gain_vs_volatility.png", dpi=150)
    print(f"[A] per-district -> robustness_by_district.csv | mean gain "
          f"{df.gain_pct.mean():.1f}% (range {df.gain_pct.min():.1f}..{df.gain_pct.max():.1f}%)")
    print(f"    H4 Spearman(gain, volatility) rho={rho:.3f} p={pval:.3f} "
          f"-> {'supports H4' if (rho > 0 and pval < 0.05) else 'not significant'}")


# ----------------------------------------------------------------- B. seed stability
def seed_stability(seeds=range(10)):
    import xgboost as xgb
    frame = pd.read_csv(OUTDIR / "model_frame.csv", dtype={"district": str}, parse_dates=["week"])
    cats = frame["district"].astype("category").cat.categories
    frame["district"] = frame["district"].astype("category")
    grid = frame[frame["split"].isin(["val", "test"])].copy()
    grid["district"] = grid["district"].astype(pd.CategoricalDtype(categories=cats))
    sp = base_wide(OUTDIR)[["district", "week", "horizon", "sarima", "prophet", "lstm", "y_true"]]

    recs = []
    for s in seeds:
        wide_parts = []
        for h in HORIZONS:
            tr = frame[(frame["split"] == "train") & frame[f"y_h{h}"].notna()]
            m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=s,
                                 enable_categorical=True, tree_method="hist", n_jobs=4)
            m.fit(tr[FEAT], tr[f"y_h{h}"])
            g = pd.DataFrame({"district": grid["district"].astype(str).values,
                              "week": grid["week"].values, "horizon": h,
                              "xgboost": m.predict(grid[FEAT])})
            wide_parts.append(g)
        xgw = pd.concat(wide_parts, ignore_index=True)
        wide = sp.merge(xgw, on=["district", "week", "horizon"])
        wide["split"] = np.where(wide["week"] <= VAL_END, "val", "test")
        for h in HORIZONS:
            v = wide[(wide.horizon == h) & (wide.split == "val")]
            t = wide[(wide.horizon == h) & (wide.split == "test")]
            w = fit_simplex(v[MEMBERS].to_numpy(), v.y_true.to_numpy())
            recs.append(dict(seed=s, horizon=h,
                             xgb_rmse=rmse(t.xgboost, t.y_true),
                             ens_rmse=rmse(t[MEMBERS].to_numpy() @ w, t.y_true)))
    df = pd.DataFrame(recs)
    summ = df.groupby("horizon").agg(
        xgb_mean=("xgb_rmse", "mean"), xgb_std=("xgb_rmse", "std"),
        ens_mean=("ens_rmse", "mean"), ens_std=("ens_rmse", "std")).round(3)
    df.to_csv(OUTDIR / "robustness_seeds.csv", index=False)
    print(f"\n[B] seed stability ({len(list(seeds))} seeds) -> robustness_seeds.csv")
    print(summ.to_string())


# ------------------------------------------------------- C. regime breakdown (2020)
def regime_breakdown():
    rdir = PROC / "regime2020"
    if not (rdir / "ensemble_forecasts.csv").exists():
        print("\n[C] regime breakdown skipped (no regime2020 run found)")
        return
    e = pd.read_csv(rdir / "ensemble_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    b = pd.read_csv(rdir / "base_forecasts.csv", dtype={"district": str}, parse_dates=["week"])
    cov0, cov1 = (pd.Timestamp(x) for x in CFG["covid_window"])
    windows = {"pre   2020-01..02": ("2020-01-01", "2020-02-29"),
               "shock 2020-03..2021-06": (cov0, cov1),
               "recov 2021-07..2022-12": ("2021-07-01", "2022-12-31")}
    rows = []
    for label, (w0, w1) in windows.items():
        w0, w1 = pd.Timestamp(w0), pd.Timestamp(w1)
        for h in HORIZONS:
            xb = b[(b.model == "xgboost") & (b.horizon == h) & b.week.between(w0, w1)]
            st = e[(e.scheme == "static_inverse_rmse") & (e.horizon == h) & e.week.between(w0, w1)]
            rg = e[(e.scheme == "regime_adaptive") & (e.horizon == h) & e.week.between(w0, w1)]
            rows.append(dict(window=label, horizon=h,
                             rmse_xgb=rmse(xb.yhat, xb.y_true),
                             rmse_static=rmse(st.yhat, st.y_true),
                             rmse_regime=rmse(rg.yhat, rg.y_true)))
    df = pd.DataFrame(rows)
    df["regime_vs_static_%"] = (100 * (df.rmse_static - df.rmse_regime) / df.rmse_static).round(2)
    df.round(3).to_csv(OUTDIR / "robustness_regimes.csv", index=False)
    print("\n[C] regime breakdown (2020 run) -> robustness_regimes.csv")
    print(df.round(2).to_string(index=False))


# ----------------------------------------------------- D. train-window length (XGB)
def train_window_sensitivity(starts=("2015-01-01", "2017-01-01", "2019-01-01")):
    import xgboost as xgb
    frame = pd.read_csv(OUTDIR / "model_frame.csv", dtype={"district": str}, parse_dates=["week"])
    cats = frame["district"].astype("category").cat.categories
    frame["district"] = frame["district"].astype("category")
    grid = frame[frame["split"].isin(["val", "test"])].copy()
    grid["district"] = grid["district"].astype(pd.CategoricalDtype(categories=cats))
    sp = base_wide(OUTDIR)
    rows = []
    for start in starts:
        s0 = pd.Timestamp(start)
        for h in HORIZONS:
            tr = frame[(frame["split"] == "train") & (frame["week"] >= s0) & frame[f"y_h{h}"].notna()]
            m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                 enable_categorical=True, tree_method="hist", n_jobs=4)
            m.fit(tr[FEAT], tr[f"y_h{h}"])
            g = pd.DataFrame({"district": grid["district"].astype(str).values,
                              "week": grid["week"].values, "horizon": h,
                              "xgboost": m.predict(grid[FEAT])})
            wide = sp.merge(g, on=["district", "week", "horizon"], suffixes=("_old", ""))
            wide["split"] = np.where(wide["week"] <= VAL_END, "val", "test")
            v = wide[(wide.horizon == h) & (wide.split == "val")]
            t = wide[(wide.horizon == h) & (wide.split == "test")]
            w = fit_simplex(v[MEMBERS].to_numpy(), v.y_true.to_numpy())
            rows.append(dict(train_start=start, horizon=h, n_train=len(tr),
                             xgb_rmse=rmse(t.xgboost, t.y_true),
                             ens_rmse=rmse(t[MEMBERS].to_numpy() @ w, t.y_true)))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUTDIR / "robustness_trainwindow.csv", index=False)
    print("\n[D] train-window sensitivity -> robustness_trainwindow.csv")
    print(df.pivot(index="train_start", columns="horizon", values="ens_rmse").round(2).to_string())


def main():
    print(f"Robustness on out_subdir='{CFG.get('out_subdir','')}' (val_end={VAL_END.date()})")
    per_district()
    seed_stability()
    regime_breakdown()
    train_window_sensitivity()


if __name__ == "__main__":
    main()

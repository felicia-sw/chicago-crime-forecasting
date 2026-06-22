"""
Step 2: base forecasters + rolling-origin backtest harness.

Governed by .claude/skills/horizon-adaptive-ensemble. Produces per-(district,
horizon, origin) forecasts for every base model, on an identical evaluation grid,
so Step 3 (ensemble) and Step 4 (evaluation) compare schemes on the same points.

Models (direct strategy; recursive as a later sensitivity):
  * seasonal_naive  ŷ(t+h) = y(t+h-52)            (mandatory floor)
  * sarima          pmdarima.auto_arima per district; order cached on train,
                    rolled forward with cheap .update() (no full refit per origin)
  * prophet         per district; yearly seasonality + US holidays + COVID regressor
  * xgboost         ONE global model per horizon across the whole panel

Leakage rules: every fit uses only weeks <= origin t (t observed at forecast time);
SARIMA/Prophet train on the modeling window [window_start, t]; XGBoost trains on the
'train' split only and predicts val+test. The evaluation grid is the val+test rows of
model_frame.csv, with y_true taken from its direct targets y_h{h}.

Usage:
    python src/models_base.py                          # full run, all 22 districts
    python src/models_base.py --districts 008 020 --max-origins 20   # smoke test
    python src/models_base.py --models naive,xgb       # subset of models
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES  # noqa: E402  (reuse the locked feature list)

CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")           # isolates alt-config runs (e.g. regime2020)
PANEL = ROOT / CFG["paths"]["panel"]
FRAME = OUTDIR / "model_frame.csv"
HORIZONS = CFG["horizons"]
MAXH = max(HORIZONS)
WINDOW_START = pd.Timestamp(CFG["window_start"])
TRAIN_END = pd.Timestamp(CFG["split"]["train_end"])
SEED = CFG["seed"]

# SARIMA seasonal spec is the contested/expensive choice -> kept here as one knob.
SARIMA_SEASONAL = True
SARIMA_M = 52


# --------------------------------------------------------------------------- IO
def load_data():
    panel = (pd.read_csv(PANEL, dtype={"district": str}, parse_dates=["week"])
             .sort_values(["district", "week"]).reset_index(drop=True))
    frame = pd.read_csv(FRAME, dtype={"district": str}, parse_dates=["week"])
    # global lookup incl. pre-2015 history (seasonal-naive needs t+h-52)
    plook = panel.set_index(["district", "week"])["count"]
    # per-district modeling-window series for SARIMA/Prophet
    win = panel[panel["week"] >= WINDOW_START]
    series = {d: g.set_index("week")["count"].asfreq("W-MON")
              for d, g in win.groupby("district")}
    return panel, frame, plook, series


def make_grid(frame, districts, max_origins):
    grid = frame[frame["split"].isin(["val", "test"])].copy()
    grid = grid[grid["district"].isin(districts)]
    if max_origins:                       # smoke: cap origins per district
        grid = (grid.sort_values(["district", "week"])
                .groupby("district").head(max_origins))
    return grid.sort_values(["district", "week"]).reset_index(drop=True)


def long_targets(grid):
    """Melt the direct targets into (district, week, horizon, y_true) rows."""
    rows = []
    for h in HORIZONS:
        sub = grid[["district", "week", f"y_h{h}"]].dropna(subset=[f"y_h{h}"])
        sub = sub.rename(columns={f"y_h{h}": "y_true"})
        sub["horizon"] = h
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ base models
def forecast_seasonal_naive(grid, plook):
    recs = []
    for h in HORIZONS:
        wk = grid["week"] + pd.to_timedelta((h - 52) * 7, unit="D")
        idx = pd.MultiIndex.from_arrays([grid["district"].values, wk.values])
        yhat = plook.reindex(idx).values
        recs.append(pd.DataFrame({"district": grid["district"].values,
                                  "week": grid["week"].values,
                                  "horizon": h, "model": "seasonal_naive",
                                  "yhat": yhat}))
    return pd.concat(recs, ignore_index=True)


# Each per-district worker CHECKPOINTS its result to disk before returning, so an
# interruption (crash / thermal / OOM) loses at most the in-flight districts and a
# re-run resumes by skipping any district whose checkpoint already exists. Workers are
# module-level so joblib can pickle them; heavy deps import inside the subprocess.
CKPT = OUTDIR / "checkpoints"
_COLS = ["district", "week", "horizon", "model", "yhat"]


def _ckpt_path(model, d):
    return CKPT / f"{model}_{d}.csv"


def _sarima_district(d, s, origins):
    import pmdarima as pm
    out = _ckpt_path("sarima", d)
    if out.exists():                                  # resume: already done
        return
    s = s.dropna()
    t0 = origins[0]
    try:
        m = pm.auto_arima(s[s.index <= t0], seasonal=SARIMA_SEASONAL, m=SARIMA_M,
                          max_p=2, max_q=2, max_P=1, max_Q=1, D=1 if SARIMA_SEASONAL else 0,
                          stepwise=True, suppress_warnings=True, error_action="ignore")
        recs, prev = [], t0
        for t in origins:
            if t > prev:
                new = s[(s.index > prev) & (s.index <= t)]
                if len(new):
                    m.update(new)
                prev = t
            fc = np.asarray(m.predict(MAXH))
            recs += [(d, t, h, "sarima", float(fc[h - 1])) for h in HORIZONS]
        meta = {"order": list(m.order), "seasonal_order": list(m.seasonal_order)}
    except Exception as e:                            # never let one district kill the run
        recs = [(d, t, h, "sarima", np.nan) for t in origins for h in HORIZONS]
        meta = {"error": str(e)}
    pd.DataFrame(recs, columns=_COLS).to_csv(out, index=False)
    (CKPT / f"sarima_{d}.order.json").write_text(json.dumps(meta))


def _prophet_district(d, s, origins, refit_every):
    from prophet import Prophet
    out = _ckpt_path("prophet", d)
    if out.exists():                                  # resume: already done
        return
    cov0, cov1 = (pd.Timestamp(x) for x in CFG["covid_window"])
    s = s.dropna()

    def _df(idx, y=None):
        f = pd.DataFrame({"ds": idx})
        f["covid"] = f["ds"].between(cov0, cov1).astype(int)
        if y is not None:
            f["y"] = np.asarray(y)
        return f

    recs, last_fc, since = [], None, 0
    for t in origins:
        if last_fc is None or since >= refit_every:
            hist = s[s.index <= t]
            try:
                mp = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                             daily_seasonality=False)
                mp.add_country_holidays(country_name="US")
                mp.add_regressor("covid")
                mp.fit(_df(hist.index, hist.values))
                fut = mp.make_future_dataframe(periods=MAXH, freq="W-MON",
                                               include_history=False)
                fut["covid"] = fut["ds"].between(cov0, cov1).astype(int)
                last_fc = mp.predict(fut)["yhat"].to_numpy()
            except Exception as e:
                print(f"    [prophet] {d}@{t.date()} failed: {e}", file=sys.stderr)
                last_fc = np.full(MAXH, np.nan)
            since = 0
        else:
            since += 1
        recs += [(d, t, h, "prophet", float(last_fc[h - 1])) for h in HORIZONS]
    pd.DataFrame(recs, columns=_COLS).to_csv(out, index=False)


def _tasks(grid, series):
    for d, gd in grid.groupby("district"):
        origins = [pd.Timestamp(t) for t in sorted(gd["week"].unique())]
        yield d, series[d], origins


def _assemble(model, districts):
    dfs = []
    for d in districts:
        f = _ckpt_path(model, d)
        if f.exists():
            dfs.append(pd.read_csv(f, dtype={"district": str}, parse_dates=["week"]))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=_COLS)


def forecast_sarima(grid, series, n_jobs=1):
    from joblib import Parallel, delayed
    CKPT.mkdir(parents=True, exist_ok=True)
    tasks = list(_tasks(grid, series))
    todo = [t for t in tasks if not _ckpt_path("sarima", t[0]).exists()]
    print(f"    sarima: {len(tasks) - len(todo)} cached, {len(todo)} to compute", flush=True)
    if todo:
        Parallel(n_jobs=n_jobs)(delayed(_sarima_district)(d, s, o) for d, s, o in todo)
    districts = [t[0] for t in tasks]
    orders = {d: json.loads((CKPT / f"sarima_{d}.order.json").read_text())
              for d in districts if (CKPT / f"sarima_{d}.order.json").exists()}
    (OUTDIR / "sarima_orders.json").write_text(
        json.dumps(orders, indent=2))
    return _assemble("sarima", districts)


def forecast_prophet(grid, series, refit_every=1, n_jobs=1):
    from joblib import Parallel, delayed
    CKPT.mkdir(parents=True, exist_ok=True)
    tasks = list(_tasks(grid, series))
    todo = [t for t in tasks if not _ckpt_path("prophet", t[0]).exists()]
    print(f"    prophet: {len(tasks) - len(todo)} cached, {len(todo)} to compute", flush=True)
    if todo:
        Parallel(n_jobs=n_jobs)(
            delayed(_prophet_district)(d, s, o, refit_every) for d, s, o in todo)
    return _assemble("prophet", [t[0] for t in tasks])


def forecast_xgboost(frame, grid):
    import xgboost as xgb
    feat = [c for c in FEATURES if c != "district"] + ["district"]
    # consistent category coding across train/predict
    frame = frame.copy()
    frame["district"] = frame["district"].astype("category")
    grid = grid.copy()
    grid["district"] = grid["district"].astype(
        pd.CategoricalDtype(categories=frame["district"].cat.categories))

    recs = []
    for h in HORIZONS:
        tr = frame[(frame["split"] == "train") & frame[f"y_h{h}"].notna()]
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            enable_categorical=True, tree_method="hist", n_jobs=4)
        model.fit(tr[feat], tr[f"y_h{h}"])
        pred = model.predict(grid[feat])
        recs.append(pd.DataFrame({"district": grid["district"].astype(str).values,
                                  "week": grid["week"].values, "horizon": h,
                                  "model": "xgboost", "yhat": pred}))
    return pd.concat(recs, ignore_index=True)


# ------------------------------------------------------------------------- main
def run(models, districts, max_origins, refit_every, out, n_jobs=1):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel, frame, plook, series = load_data()
    grid = make_grid(frame, districts, max_origins)
    truth = long_targets(grid)
    print(f"Grid: {grid['district'].nunique()} districts x {grid['week'].nunique()} "
          f"origins -> {len(truth):,} (district,horizon,origin) eval points "
          f"[n_jobs={n_jobs}]")

    parts, timings = [], {}
    runners = {
        "naive": lambda: forecast_seasonal_naive(grid, plook),
        "sarima": lambda: forecast_sarima(grid, series, n_jobs),
        "prophet": lambda: forecast_prophet(grid, series, refit_every, n_jobs),
        "xgb": lambda: forecast_xgboost(frame, grid),
    }
    for name in models:
        t0 = time.time()
        print(f"  [{name}] running...", flush=True)
        parts.append(runners[name]())
        timings[name] = time.time() - t0
        print(f"  [{name}] done in {timings[name]:.1f}s")

    fc = pd.concat(parts, ignore_index=True)
    fc["week"] = pd.to_datetime(fc["week"])
    out_df = truth.merge(fc, on=["district", "week", "horizon"], how="left")

    out_path = OUTDIR / out
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved base forecasts -> {out_path}  ({len(out_df):,} rows)")
    # quick coverage + sanity (RMSE per model/horizon on this grid)
    for name in out_df["model"].dropna().unique():
        sub = out_df[out_df["model"] == name]
        miss = sub["yhat"].isna().mean() * 100
        rmse = np.sqrt(((sub["yhat"] - sub["y_true"]) ** 2).mean())
        print(f"    {name:14s} missing={miss:5.1f}%  RMSE(all-h)={rmse:8.2f}")
    n_orig = grid["week"].nunique()
    print("  timings/origin: " + ", ".join(
        f"{k}={v / max(n_orig,1):.2f}s" for k, v in timings.items()))
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="naive,sarima,prophet,xgb")
    ap.add_argument("--districts", nargs="*", default=CFG["districts"])
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="base_forecasts.csv")
    a = ap.parse_args()
    run([m.strip() for m in a.models.split(",")], a.districts,
        a.max_origins, a.refit_every, a.out, a.jobs)


if __name__ == "__main__":
    main()

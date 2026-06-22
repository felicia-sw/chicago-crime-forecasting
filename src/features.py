"""
Build the leakage-safe modeling frame from the weekly district panel.

This is Step 1 of the pipeline (see .claude/skills/crime-forecasting-pipeline).
We turn the long panel (district, week, count) into a per-(district, origin-week)
feature matrix with DIRECT multi-horizon targets y(t+h) for h in config.horizons.

Key semantics (state these in the paper):
  * DIRECT strategy: features are computed once at the origin week t; for each
    horizon h we attach a separate target column y_h = count(t+h). One model per
    horizon is trained downstream (config.forecast_strategy = 'direct').
  * No leakage: every feature uses only counts at weeks <= t (lags and rolling
    windows end at and include the origin t, which is observed at forecast time).
    Targets are strictly in the future. Split is assigned by the ORIGIN week.
  * Feature history vs. modeling window: features are computed on the FULL 2001+
    history, then we keep only origin weeks >= window_start (2015-01-01). Early
    2015 origins therefore get complete lag_52 / rolling-52 features, while
    pre-2015 weeks are used ONLY as predictor history -- never as training rows.
  * Missing != zero: the panel is already 0-filled; nothing is imputed here.

Input : data/processed/weekly_district_panel.csv   (district, week, count)
Output: data/processed/model_frame.csv             (one row per district x origin-week)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import holidays

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
PANEL = ROOT / CFG["paths"]["panel"]
OUT = ROOT / CFG["paths"]["processed"] / "model_frame.csv"

LAGS = [1, 2, 3, 4, 8, 12, 52]
ROLL_WINDOWS = [4, 8, 12]
HORIZONS = CFG["horizons"]

# Feature columns produced below (district is the global-model id; encoded downstream).
FEATURES = (
    ["count"]
    + [f"lag_{k}" for k in LAGS]
    + [f"rmean_{w}" for w in ROLL_WINDOWS]
    + [f"rstd_{w}" for w in ROLL_WINDOWS]
    + ["month", "weekofyear", "is_holiday_week", "is_covid", "district"]
)
TARGETS = [f"y_h{h}" for h in HORIZONS]


def _holiday_weeks(weeks: pd.Series) -> set:
    """Set of Monday-labeled weeks that contain a US federal holiday."""
    years = range(weeks.min().year, weeks.max().year + 1)
    hol = pd.to_datetime(list(holidays.UnitedStates(years=list(years)).keys()))
    # snap each holiday to its W-MON week (Monday of that week)
    mondays = (hol - pd.to_timedelta(hol.weekday, unit="D")).normalize()
    return set(mondays)


def build_features() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"district": str}, parse_dates=["week"])
    df = df.sort_values(["district", "week"]).reset_index(drop=True)
    g = df.groupby("district", sort=False)["count"]

    # --- lags: count(t-k), strictly past observations ---
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)

    # --- rolling mean/std ending at and including t (t is observed at origin) ---
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = g.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"rstd_{w}"] = g.transform(lambda s, w=w: s.rolling(w).std())

    # --- calendar / regime flags ---
    df["month"] = df["week"].dt.month
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["is_holiday_week"] = df["week"].isin(_holiday_weeks(df["week"])).astype(int)
    cov0, cov1 = (pd.Timestamp(x) for x in CFG["covid_window"])
    df["is_covid"] = df["week"].between(cov0, cov1).astype(int)

    # --- direct multi-horizon targets: y_h = count(t+h) ---
    for h in HORIZONS:
        df[f"y_h{h}"] = df.groupby("district", sort=False)["count"].shift(-h)

    # --- temporal split by ORIGIN week (no shuffling) ---
    te = pd.Timestamp(CFG["split"]["train_end"])
    ve = pd.Timestamp(CFG["split"]["val_end"])
    ts = pd.Timestamp(CFG["split"]["test_start"])
    df["split"] = np.select(
        [df["week"] <= te, df["week"] <= ve, df["week"] >= ts],
        ["train", "val", "test"],
        default="gap",
    )

    # --- restrict to the modeling window; pre-2015 kept only as feature history ---
    df = df[df["week"] >= pd.Timestamp(CFG["window_start"])].copy()
    return df.reset_index(drop=True)


def main() -> None:
    df = build_features()
    cols = ["district", "week", "split"] + [c for c in FEATURES if c != "district"] + TARGETS
    df = df[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved modeling frame -> {OUT}")
    print(f"  rows: {len(df):,}  districts: {df['district'].nunique()}  "
          f"origin weeks: {df['week'].nunique()}")
    print(f"  origin span: {df['week'].min().date()} .. {df['week'].max().date()}")
    print(f"  split rows: " + ", ".join(
        f"{k}={v:,}" for k, v in df['split'].value_counts().items()))
    print(f"  features ({len(FEATURES)}): {FEATURES}")
    # leakage-safe completeness: lag/rolling features should be fully populated in-window
    feat_na = df[[c for c in FEATURES if c != 'district']].isna().sum()
    print(f"  feature NaNs (expect 0 for lags/rolling in-window):\n"
          f"{feat_na[feat_na > 0].to_string() if feat_na.any() else '    none'}")
    print(f"  target NaNs (expected near series end): " + ", ".join(
        f"{t}={int(df[t].isna().sum())}" for t in TARGETS))


if __name__ == "__main__":
    main()

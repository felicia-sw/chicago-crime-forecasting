"""
Build the clean weekly district x week panel from daily counts.

Key semantics (state these in the paper):
  * A week with no recorded incidents is ZERO, not missing. We reindex every
    district onto the full weekly calendar and fill absent weeks with 0.
    No rows are deleted.
  * Weeks start Monday (W-MON). The final partial week is dropped.

Input : data/raw/daily_district_counts.csv   (from data_acquisition.py)
Output: data/processed/weekly_district_panel.csv   (long format: district, week, count)
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / CFG["paths"]["raw"] / "daily_district_counts.csv"
OUT = ROOT / CFG["paths"]["panel"]
OUT.parent.mkdir(parents=True, exist_ok=True)
WEEK_RULE = CFG["week_rule"]


def build_panel() -> pd.DataFrame:
    df = pd.read_csv(RAW, dtype={"district": str})
    df["day"] = pd.to_datetime(df["day"])
    df["district"] = df["district"].str.zfill(3)
    df = df[df["district"].isin(CFG["districts"])]

    # weekly sum per district
    wk = (df.set_index("day")
            .groupby("district")["n"]
            .resample(WEEK_RULE).sum()
            .reset_index()
            .rename(columns={"day": "week", "n": "count"}))

    # full district x week grid -> zero-fill absent weeks (missing != deleted)
    weeks = pd.date_range(wk["week"].min(), wk["week"].max(), freq=WEEK_RULE)
    grid = pd.MultiIndex.from_product([CFG["districts"], weeks],
                                      names=["district", "week"])
    panel = (wk.set_index(["district", "week"])
               .reindex(grid, fill_value=0)
               .reset_index())

    # drop the final partial week (portal excludes the most recent ~7 days)
    last_full = panel["week"].max() - pd.Timedelta(weeks=1)
    panel = panel[panel["week"] <= last_full].copy()
    panel["count"] = panel["count"].astype(int)
    return panel.sort_values(["district", "week"]).reset_index(drop=True)


def main() -> None:
    panel = build_panel()
    panel.to_csv(OUT, index=False)
    n_weeks = panel["week"].nunique()
    zero = (panel["count"] == 0).mean() * 100
    print(f"Saved weekly panel -> {OUT}")
    print(f"  districts: {panel['district'].nunique()}  weeks: {n_weeks}  rows: {len(panel):,}")
    print(f"  span: {panel['week'].min().date()} .. {panel['week'].max().date()}")
    print(f"  zero-count weeks: {zero:.3f}%  (expected ~0 at district-total level)")
    print(f"  weekly count: min {panel['count'].min()}, "
          f"median {int(panel['count'].median())}, max {panel['count'].max()}")


if __name__ == "__main__":
    main()

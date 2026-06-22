"""
Crush the full raw Chicago portal CSV (~2 GB, "Crimes - 2001 to Present") into the
small weekly panels we model. Reads in chunks so it works on a normal laptop.

Usage:
    python build_panel_from_raw.py  "Crimes_-_2001_to_Present.csv"

Outputs (tiny, upload these):
    weekly_district_panel.csv   ->  district, week, count   (~30k rows, ~1 MB)
    weekly_type_panel.csv       ->  primary_type, week, count  (top types; secondary analysis)
"""
import sys
from pathlib import Path
import pandas as pd

# 22 active police districts (retired/merged 013/021/023 and artifacts excluded)
DISTRICTS = ["001","002","003","004","005","006","007","008","009","010","011",
             "012","014","015","016","017","018","019","020","022","024","025"]
TOP_TYPES = ["THEFT","BATTERY","CRIMINAL DAMAGE","ASSAULT","DECEPTIVE PRACTICE"]
WEEK_RULE = "W-MON"
CHUNK = 250_000


def main(src: str) -> None:
    src = Path(src)
    assert src.exists(), f"file not found: {src}"
    usecols = ["Date", "District", "Primary Type"]
    dist_parts, type_parts = [], []

    n = 0
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=CHUNK,
                             dtype={"District": "string", "Primary Type": "string"}):
        n += len(chunk)
        chunk["dt"] = pd.to_datetime(chunk["Date"].str[:10], format="%m/%d/%Y",
                                     errors="coerce")  # date-only parse = much faster
        chunk = chunk.dropna(subset=["dt"])
        chunk["district"] = (chunk["District"].str.extract(r"(\d+)")[0]
                             .str.zfill(3))
        # Monday-of-week label (weekday: Mon=0), aligns with date_range(freq="W-MON")
        chunk["week"] = (chunk["dt"]
                         - pd.to_timedelta(chunk["dt"].dt.weekday, unit="D")).dt.normalize()

        d = chunk[chunk["district"].isin(DISTRICTS)]
        dist_parts.append(d.groupby(["district", "week"]).size()
                          .rename("count").reset_index())

        t = chunk[chunk["Primary Type"].isin(TOP_TYPES)]
        type_parts.append(t.groupby(["Primary Type", "week"]).size()
                          .rename("count").reset_index()
                          .rename(columns={"Primary Type": "primary_type"}))
        print(f"  processed {n:,} rows...", file=sys.stderr)

    # ---- district panel: collapse chunks, zero-fill the full grid ----
    dist = (pd.concat(dist_parts).groupby(["district", "week"])["count"].sum()
            .reset_index())
    weeks = pd.date_range(dist["week"].min(), dist["week"].max(), freq=WEEK_RULE)
    grid = pd.MultiIndex.from_product([DISTRICTS, weeks], names=["district", "week"])
    dist = (dist.set_index(["district", "week"]).reindex(grid, fill_value=0)
            .reset_index())
    last_full = dist["week"].max() - pd.Timedelta(weeks=1)   # drop partial final week
    dist = dist[dist["week"] <= last_full].copy()
    dist["count"] = dist["count"].astype(int)
    dist.sort_values(["district", "week"]).to_csv("weekly_district_panel.csv", index=False)

    # ---- type panel (secondary) ----
    typ = (pd.concat(type_parts).groupby(["primary_type", "week"])["count"].sum()
           .reset_index())
    tgrid = pd.MultiIndex.from_product([TOP_TYPES, weeks], names=["primary_type", "week"])
    typ = (typ.set_index(["primary_type", "week"]).reindex(tgrid, fill_value=0)
           .reset_index())
    typ = typ[typ["week"] <= last_full].copy()
    typ["count"] = typ["count"].astype(int)
    typ.sort_values(["primary_type", "week"]).to_csv("weekly_type_panel.csv", index=False)

    print(f"\nDone. {n:,} incident rows read.")
    print(f"  weekly_district_panel.csv : {dist['district'].nunique()} districts x "
          f"{dist['week'].nunique()} weeks = {len(dist):,} rows")
    print(f"  weekly_type_panel.csv     : {typ['primary_type'].nunique()} types x "
          f"{typ['week'].nunique()} weeks = {len(typ):,} rows")
    print(f"  span: {dist['week'].min().date()} .. {dist['week'].max().date()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Crimes_-_2001_to_Present.csv")

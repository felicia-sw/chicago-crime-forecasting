"""
Pull daily incident counts per police district from the City of Chicago Socrata
API (dataset ijzp-q8t2) and cache them to data/raw/daily_district_counts.csv.

We aggregate server-side (GROUP BY day, district) so the download is a few hundred
thousand small rows rather than ~8M incident records. aggregate_weekly.py then
resamples this to the clean weekly district x week panel.

Usage:
    python src/data_acquisition.py
Optional: set SOCRATA_APP_TOKEN to raise rate limits (free token from the portal).
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / CFG["paths"]["raw"]
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "daily_district_counts.csv"

DOMAIN = CFG["socrata"]["domain"]
DATASET = CFG["socrata"]["dataset_id"]
ENDPOINT = f"https://{DOMAIN}/resource/{DATASET}.json"
TOKEN = os.environ.get("SOCRATA_APP_TOKEN")
PAGE = 50000  # rows per request


def fetch_daily_district_counts() -> pd.DataFrame:
    """Server-side daily x district counts, paginated."""
    headers = {"X-App-Token": TOKEN} if TOKEN else {}
    select = "date_trunc_ymd(date) as day, district, count(1) as n"
    frames, offset = [], 0
    while True:
        params = {
            "$select": select,
            "$group": "date_trunc_ymd(date), district",
            "$order": "day",
            "$limit": PAGE,
            "$offset": offset,
        }
        for attempt in range(5):
            r = requests.get(ENDPOINT, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        else:
            r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        offset += PAGE
        print(f"  fetched {offset:,} rows...", file=sys.stderr)
        if len(batch) < PAGE:
            break
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["district"] = df["district"].astype(str).str.zfill(3)
    return df


def main() -> None:
    print("Downloading daily district counts from Socrata...", file=sys.stderr)
    df = fetch_daily_district_counts()
    keep = set(CFG["districts"])
    before = len(df)
    df = df[df["district"].isin(keep)].copy()      # drop artifact/retired districts
    df = df.sort_values(["district", "day"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows ({before - len(df):,} artifact rows dropped) -> {OUT}")
    print(f"Date range: {df['day'].min().date()} .. {df['day'].max().date()}")
    print(f"Districts: {df['district'].nunique()}")


if __name__ == "__main__":
    main()

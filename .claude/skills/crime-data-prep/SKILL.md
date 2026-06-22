---
name: crime-data-prep
description: Data conventions for the Chicago weekly crime panel — schema, gotchas, modeling-window trimming, and how to rebuild from the raw 1.9GB CSV. Use when loading, cleaning, slicing, or regenerating the panel.
---

# Crime Data Prep

## Modeling dataset
`data/processed/weekly_district_panel.csv` — columns `district, week, count`.
22 districts × 1327 weeks (2001-01-01 → 2026-06-01).

**Gotchas (important):**
- `pd.read_csv(path, dtype={'district': str})` — districts are zero-padded ("001"); pandas drops leading zeros otherwise.
- `week` = Monday of the week (W-MON); `pd.to_datetime(df['week'])`.
- Absent weeks are true **0** (missing ≠ zero); nothing deleted. No zero-count weeks at district level (min 52/wk) → MAPE/SMAPE safe (still mask if actual=0).

## Trim to the modeling window AT LOAD TIME (don't rebuild)
```python
import yaml, pandas as pd
cfg = yaml.safe_load(open('config.yaml'))
df = pd.read_csv(cfg['paths']['panel'], dtype={'district': str}, parse_dates=['week'])
df = df[df['week'] >= pd.Timestamp(cfg['window_start'])]   # 2015-01-01
```

## Split (from config.yaml)
train ≤ `split.train_end` (2020-12-31); validation up to `split.val_end` (2022-12-31); test ≥ `split.test_start` (2023-01-01).

## Active districts (22)
001–012, 014–020, 022, 024, 025. Excluded: retired/merged 013/021/023; artifacts 031, blank, "16".

## Secondary panel
`data/processed/weekly_type_panel.csv` — top-5 crime types × week (THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, DECEPTIVE PRACTICE) for the heterogeneity sub-analysis only.

## Rebuild from raw (only if the panel is missing)
```bash
cd data/processed && python ../../src/build_panel_from_raw.py "../../Crimes_-_2001_to_Present.csv"
```
Date-only parse, W-MON labeling, zero-fill, drops the partial final week. ~20s for 8.6M rows.
The raw 1.9GB CSV and `data/raw/` should be gitignored.

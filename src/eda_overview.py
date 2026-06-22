"""
EDA overview for the Chicago crime forecasting study.

Generates the macro-level overview figure (monthly citywide counts, 2001-present,
with the 2020 COVID structural break marked) and a district-volume profile for
2024 that justifies the 'district-level weekly totals' scope (no zero-count weeks).

Data here are server-side aggregates pulled from the City of Chicago Socrata API
(dataset ijzp-q8t2). The full weekly x district modeling panel is built separately
by src/data_acquisition.py + src/aggregate_weekly.py.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGD = ROOT / "figures"
DATD = ROOT / "data"
FIGD.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------------
# Monthly citywide totals, Jan 2001 - May 2026 (Socrata: date_trunc_ym + count).
# Jun 2026 dropped (partial: portal excludes most recent 7 days).
# ----------------------------------------------------------------------------
monthly = {
    2001: [38129,33790,40576,40100,41847,41743,44706,44049,41521,43039,39613,36853],
    2002: [38426,33908,38591,40041,42916,42841,46015,44218,42393,43150,37157,37180],
    2003: [36731,32511,38652,39790,41198,40822,43421,44272,41427,43330,38057,35793],
    2004: [35116,34071,39860,38272,40784,41137,43242,43049,41216,41530,36843,34324],
    2005: [33883,31990,36907,38872,40477,40059,41813,41544,39631,39961,35984,32671],
    2006: [36775,31302,36984,36481,39619,38825,41555,40499,37777,38666,35572,34147],
    2007: [34131,27139,36357,35641,40094,38997,40996,39850,38268,39600,34427,31610],
    2008: [33416,29053,33988,35602,38068,37709,40500,40546,37459,37895,33565,29423],
    2009: [30275,28243,33693,32576,35249,34288,35683,35835,33876,33548,31406,28196],
    2010: [29279,24959,32347,31678,33415,32739,33515,34165,31912,32404,28903,25256],
    2011: [27242,22246,28705,29119,31604,32333,33265,32609,29948,30295,27662,27035],
    2012: [26344,23895,28584,27197,30116,31106,31988,30069,27789,27998,26052,25246],
    2013: [25575,21430,25000,25547,28027,27404,28648,28679,26363,25495,23578,21880],
    2014: [20106,18055,22205,22946,24895,25468,26587,25910,23912,24015,20785,20998],
    2015: [21020,16410,21693,21747,23731,23198,24249,24832,23141,23116,20606,21157],
    2016: [20806,18763,22075,21193,23567,24098,24872,24942,23712,23782,21767,20410],
    2017: [22234,19329,20600,21721,23409,23890,24889,24763,22872,22955,21523,21130],
    2018: [20627,17390,21282,21178,24776,24290,25307,25517,23151,22879,20726,22047],
    2019: [19846,18465,20481,21061,23707,23661,24904,24472,22514,21754,20006,20850],
    2020: [20089,18226,16794,12958,17640,17716,19690,19955,17989,18478,16673,16549],
    2021: [16312,13159,16211,15708,17888,18901,19307,18635,19299,19421,17490,17387],
    2022: [15778,15295,18446,18086,20230,20871,22362,22405,22441,23296,20890,20010],
    2023: [21346,18493,20814,20814,22292,22771,24069,24242,22658,23106,21439,21333],
    2024: [19659,19950,20917,20494,22997,23209,24103,23007,22988,22496,19736,19613],
    2025: [18520,16544,19787,19671,20510,21122,22665,21335,20355,20992,18422,17507],
    2026: [16903,16460,19004,18952,20700],  # Jan-May 2026 (Jun partial, excluded)
}
rows = []
for y, vals in monthly.items():
    for m, n in enumerate(vals, start=1):
        rows.append((pd.Timestamp(y, m, 1), n))
ts = pd.DataFrame(rows, columns=["month", "count"]).set_index("month")
ts.to_csv(DATD / "chicago_monthly_citywide.csv")

# 2024 incidents per police district (Socrata, year='2024'); 031 artifact dropped.
district_2024 = {
    "008":17279,"012":16210,"001":14909,"006":14870,"019":14129,"004":13730,
    "011":13659,"018":13569,"025":13282,"002":13190,"003":13147,"009":11361,
    "007":10775,"010":10554,"005":10254,"014":9897,"016":9086,"024":8829,
    "015":8773,"017":7992,"022":7982,"020":5679,
}
dist = (pd.Series(district_2024, name="count").rename_axis("district")
        .sort_values(ascending=False).to_frame())
dist["per_week"] = (dist["count"] / 52).round(0)
dist.to_csv(DATD / "chicago_district_2024.csv")

# ----------------------------------------------------------------------------
# Figure 1 - macro overview with the 2020 structural break
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
ax.plot(ts.index, ts["count"], color="#1f3b63", lw=1.4)

covid_start, covid_end = pd.Timestamp(2020, 3, 1), pd.Timestamp(2021, 6, 1)
ax.axvspan(covid_start, covid_end, color="#d9534f", alpha=0.12)
ax.axvline(covid_start, color="#d9534f", lw=1.2, ls="--")
apr20 = pd.Timestamp(2020, 4, 1)
ax.scatter([apr20], [12958], color="#d9534f", zorder=5, s=28)
ax.annotate("Apr 2020: 12,958\n(COVID stay-at-home;\n-38% vs 2019 Apr)",
            xy=(apr20, 12958), xytext=(pd.Timestamp(2013, 6, 1), 14600),
            fontsize=8.5, color="#8a2a25",
            arrowprops=dict(arrowstyle="->", color="#8a2a25", lw=0.9))
ax.annotate("2020 structural break", xy=(pd.Timestamp(2020, 9, 1), 44000),
            fontsize=8.5, color="#8a2a25", ha="center")
ax.set_title("Chicago reported crime, monthly citywide totals (2001–2026)",
             fontsize=12.5, weight="bold", loc="left")
ax.set_ylabel("Incidents per month")
ax.set_xlabel("")
ax.set_ylim(0, 48000)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y", alpha=0.3)
ax.margins(x=0.01)
fig.text(0.125, -0.02,
         "Source: City of Chicago Open Data (ijzp-q8t2). Long-run decline + strong summer seasonality; "
         "sharp 2020 COVID break and partial recovery.",
         fontsize=7.5, color="#555")
fig.tight_layout()
fig.savefig(FIGD / "fig1_monthly_overview.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2 - district volume profile (2024) -> scope justification
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
colors = ["#1f3b63"] * len(dist)
colors[-1] = "#d9534f"  # smallest district highlighted
ax.bar(dist.index, dist["count"], color=colors)
ax.set_title("Incidents per police district, 2024  (22 active districts)",
             fontsize=12.5, weight="bold", loc="left")
ax.set_ylabel("Incidents in 2024")
ax.set_xlabel("Police district")
ax.grid(axis="y", alpha=0.3)
mn = dist.iloc[-1]
ax.annotate(f"Smallest: District {dist.index[-1]} = {int(mn['count']):,}/yr\n"
            f"≈ {int(mn['per_week'])}/week  → no zero-count weeks",
            xy=(len(dist) - 1, mn["count"]),
            xytext=(len(dist) - 8.5, 12500), fontsize=8.5, color="#8a2a25",
            arrowprops=dict(arrowstyle="->", color="#8a2a25", lw=0.9))
fig.text(0.125, -0.02,
         "Even the lowest-volume district averages ~109 incidents/week in 2024, so weekly district totals "
         "are well-behaved for MAE/RMSE/MAPE/SMAPE.", fontsize=7.5, color="#555")
fig.tight_layout()
fig.savefig(FIGD / "fig2_district_2024.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Profiling summary
# ----------------------------------------------------------------------------
ann = ts.resample("YE").sum()["count"]
peak = ts["count"].idxmax(); trough = ts.loc["2020"]["count"].idxmin()
print("=== PROFILE ===")
print(f"Months covered: {ts.index.min():%Y-%m} -> {ts.index.max():%Y-%m}  ({len(ts)} months)")
print(f"Peak month:   {peak:%Y-%m} = {int(ts['count'].max()):,}")
print(f"2020 trough:  {trough:%Y-%m} = {int(ts.loc['2020','count'].min()):,}")
print(f"Annual 2001:  {int(ann.loc['2001-12-31']):,}   | Annual 2019: {int(ann.loc['2019-12-31']):,}")
print(f"Annual 2024:  {int(ann.loc['2024-12-31']):,}")
print(f"2019->2020 annual change: {100*(ann.loc['2020-12-31']/ann.loc['2019-12-31']-1):+.1f}%")
print(f"Districts (2024): {len(dist)}  | max {dist['count'].max():,} (D{dist.index[0]})"
      f"  min {dist['count'].min():,} (D{dist.index[-1]})")
print(f"Min district weekly avg 2024: {dist['per_week'].min():.0f}/week")
print("Figures written:", [p.name for p in sorted(FIGD.glob('*.png'))])

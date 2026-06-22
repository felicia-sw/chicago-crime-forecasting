"""
Spatial EDA for the 2020 regime-break story — a centroid "bubble map" of the 22
Chicago police districts (no boundary shapefile needed). District centroids are the
incident-weighted mean lat/lon (from Socrata avg(latitude/longitude)).

Outputs:
  figures/fig3_map_2020_break.png  - bubble map, % decline 2019->2020 (COVID footprint)
  figures/fig4_map_levels.png      - small multiples: 2019 / 2020 / 2022 levels
  data/district_year_counts.csv, data/district_centroids.csv
"""
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIGD = ROOT / "figures"; DATD = ROOT / "data"
FIGD.mkdir(exist_ok=True, parents=True)

# incident-weighted centroids (Socrata avg lat/lon), 22 active districts
CENT = {
 "001":(41.8722,-87.6288),"002":(41.8104,-87.6123),"003":(41.7712,-87.5965),
 "004":(41.7338,-87.5637),"005":(41.6874,-87.6226),"006":(41.7455,-87.6323),
 "007":(41.7756,-87.6535),"008":(41.7783,-87.7159),"009":(41.8155,-87.6650),
 "010":(41.8535,-87.7126),"011":(41.8823,-87.7192),"012":(41.8803,-87.6713),
 "014":(41.9165,-87.6931),"015":(41.8861,-87.7583),"016":(41.9652,-87.7975),
 "017":(41.9604,-87.7207),"018":(41.9023,-87.6352),"019":(41.9476,-87.6599),
 "020":(41.9785,-87.6718),"022":(41.7087,-87.6587),"024":(42.0057,-87.6775),
 "025":(41.9192,-87.7525),
}
CNT = {
 "2019":{"001":15314,"002":11278,"003":12534,"004":14087,"005":11396,"006":16918,"007":13757,"008":15750,"009":11045,"010":12517,"011":18747,"012":13204,"014":9300,"015":10027,"016":8327,"017":6771,"018":15184,"019":11854,"020":4396,"022":8281,"024":8038,"025":12989},
 "2020":{"001":8428,"002":9557,"003":11067,"004":12359,"005":10296,"006":14141,"007":11803,"008":13214,"009":9308,"010":10139,"011":15089,"012":10223,"014":6681,"015":8933,"016":7600,"017":6062,"018":8888,"019":9449,"020":4107,"022":7198,"024":6855,"025":11356},
 "2022":{"001":13113,"002":11846,"003":11969,"004":14017,"005":9841,"006":14765,"007":10350,"008":14878,"009":10416,"010":10036,"011":13532,"012":14420,"014":8194,"015":8339,"016":9280,"017":7024,"018":12496,"019":12321,"020":4990,"022":7724,"024":8511,"025":12019},
}
dist = sorted(CENT)
df = pd.DataFrame({"district": dist,
                   "lat":[CENT[d][0] for d in dist], "lon":[CENT[d][1] for d in dist],
                   "y2019":[CNT["2019"][d] for d in dist],
                   "y2020":[CNT["2020"][d] for d in dist],
                   "y2022":[CNT["2022"][d] for d in dist]})
df["pct_20"] = (df["y2020"]/df["y2019"]-1)*100
df["pct_rec"] = (df["y2022"]/df["y2020"]-1)*100
df.to_csv(DATD/"district_year_counts.csv", index=False)
df[["district","lat","lon"]].to_csv(DATD/"district_centroids.csv", index=False)

ASPECT = 1/math.cos(math.radians(41.85))  # correct lon/lat aspect at Chicago
plt.rcParams.update({"font.size":10})

def label(ax):
    for _,r in df.iterrows():
        ax.annotate(r["district"], (r["lon"], r["lat"]), fontsize=6.5, ha="center",
                    va="center", color="black", weight="bold")

# ---- Fig 3: COVID footprint (% decline 2019->2020) ----
fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=200)
sizes = (df["y2019"]/df["y2019"].max())*1300 + 120
sc = ax.scatter(df["lon"], df["lat"], s=sizes, c=df["pct_20"], cmap="Reds_r",
                vmin=-45, vmax=0, edgecolor="#333", linewidth=0.6, zorder=2)
label(ax)
ax.set_aspect(ASPECT); ax.set_title("Spatial footprint of the 2020 COVID break\nChicago police districts — % change in reported crime, 2019→2020",
            fontsize=12, weight="bold")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02); cb.set_label("% change 2019→2020")
# size legend
for val in [5000, 12000, 18000]:
    ax.scatter([], [], s=(val/df["y2019"].max())*1300+120, c="#bbb", edgecolor="#333",
               linewidth=0.6, label=f"{val:,}")
ax.legend(title="2019 incidents", loc="lower right", fontsize=7, title_fontsize=7, labelspacing=1.1, borderpad=0.8)
ax.annotate("Downtown/central districts (001, 018)\ndropped most (~ -45%); periphery least",
            xy=(df.loc[df.district=='001','lon'].iloc[0], df.loc[df.district=='001','lat'].iloc[0]),
            xytext=(-87.92, 41.74), fontsize=7.5, color="#8a2a25",
            arrowprops=dict(arrowstyle="->", color="#8a2a25", lw=0.9))
fig.text(0.5, 0.02, "Markers = district centroids (incident-weighted). All 22 districts fell in 2020; the decline was steepest downtown.",
         ha="center", fontsize=7, color="#555")
fig.tight_layout(rect=[0,0.03,1,1])
fig.savefig(FIGD/"fig3_map_2020_break.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: levels 2019 / 2020 / 2022 ----
fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=200, sharex=True, sharey=True)
vmin, vmax = 4000, 19000
for ax, yr in zip(axes, ["2019","2020","2022"]):
    col = {"2019":"y2019","2020":"y2020","2022":"y2022"}[yr]
    sc = ax.scatter(df["lon"], df["lat"], s=300, c=df[col], cmap="YlOrRd",
                    vmin=vmin, vmax=vmax, edgecolor="#333", linewidth=0.5)
    ax.set_aspect(ASPECT); ax.set_title(f"{yr}", fontsize=12, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
labels = {"2019":"pre-COVID","2020":"COVID shock","2022":"recovery"}
for ax, yr in zip(axes, ["2019","2020","2022"]):
    ax.text(0.5,-0.06, labels[yr], transform=ax.transAxes, ha="center", fontsize=9, color="#555")
fig.suptitle("Annual reported crime by district: pre-COVID → 2020 shock → recovery", fontsize=13, weight="bold")
cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.02); cb.set_label("Incidents per year")
fig.savefig(FIGD/"fig4_map_levels.png", bbox_inches="tight"); plt.close(fig)

print("Wrote fig3_map_2020_break.png, fig4_map_levels.png")
print(f"Biggest 2020 drop: D{df.loc[df.pct_20.idxmin(),'district']} ({df.pct_20.min():.0f}%);"
      f" smallest: D{df.loc[df.pct_20.idxmax(),'district']} ({df.pct_20.max():.0f}%)")

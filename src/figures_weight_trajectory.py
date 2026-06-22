"""
Centerpiece figure: the regime-adaptive ensemble's member weights over time, showing
authority re-allocating between SARIMA / Prophet / XGBoost across the 2020 COVID break.

Reads the regime_adaptive rows of ensemble_weights.csv (one weight vector per origin
week per horizon) and draws a stacked-area trajectory per horizon, with the COVID window
shaded. Intended for the 2020-regime run:

    CRIME_CONFIG=config_regime2020.yaml python src/figures_weight_trajectory.py

Output: figures/fig_weight_trajectory_2020.png  (+ .pdf for the manuscript)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
OUTDIR = ROOT / CFG["paths"]["processed"] / CFG.get("out_subdir", "")
FIGDIR = ROOT / CFG["paths"]["figures"]
MEMBERS = ["sarima", "prophet", "xgboost"]
COLORS = {"sarima": "#4C72B0", "prophet": "#DD8452", "xgboost": "#55A868"}
HORIZONS = CFG["horizons"]


def main():
    w = pd.read_csv(OUTDIR / "ensemble_weights.csv")
    w = w[w["scheme"] == "regime_adaptive"].copy()
    if w.empty:
        raise SystemExit("No regime_adaptive weights found — run ensemble.py first.")
    # horizon is read as str because the static-weights row uses horizon="all"; coerce back
    w["horizon"] = w["horizon"].astype(int)
    w["week"] = pd.to_datetime(w["week"])
    cov0, cov1 = (pd.Timestamp(x) for x in CFG["covid_window"])

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), HORIZONS):
        sub = w[w["horizon"] == h].sort_values("week")
        ax.stackplot(sub["week"], *[sub[m] for m in MEMBERS],
                     labels=[m.upper() for m in MEMBERS],
                     colors=[COLORS[m] for m in MEMBERS], alpha=0.9)
        ax.axvspan(cov0, cov1, color="red", alpha=0.12, lw=0)         # COVID regime
        ax.axvline(cov0, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(f"h = {h} week(s)", fontsize=11)
        ax.set_ylim(0, 1)
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].set_ylabel("ensemble weight")
    axes[1, 0].set_ylabel("ensemble weight")
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    fig.suptitle("Regime-adaptive ensemble weights across the 2020 COVID break "
                 "(shaded = COVID window)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "pdf"):
        out = FIGDIR / f"fig_weight_trajectory_2020.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()

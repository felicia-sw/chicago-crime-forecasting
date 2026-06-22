"""
Step 6: interpretability via SHAP — on the XGBoost member ONLY.

Governed by .claude/skills/forecast-evaluation. Per the guardrail, SHAP is framed
explicitly as explaining the XGBoost *component*, not the full ensemble. We surface the
dominant lag / rolling / calendar drivers and show how they shift with horizon (which
echoes why the optimal member mix changes with horizon).

District is integer-coded (district_code) for a clean TreeExplainer pass — a control/id
feature, not a driver we interpret.

Outputs:
  <out_subdir>/shap_importance.csv          — mean |SHAP| per feature per horizon
  figures/fig_shap_summary.png              — beeswarm for a representative horizon
  figures/fig_shap_by_horizon.png           — top drivers, short vs long horizon
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES  # noqa: E402

CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")
FIGDIR = ROOT / CFG["paths"]["figures"]
HORIZONS = CFG["horizons"]
SEED = CFG["seed"]
FEAT = [c for c in FEATURES if c != "district"] + ["district_code"]


def load_frame():
    f = pd.read_csv(OUTDIR / "model_frame.csv", dtype={"district": str}, parse_dates=["week"])
    f["district_code"] = f["district"].astype("category").cat.codes
    return f


def train_xgb(frame, h):
    tr = frame[(frame["split"] == "train") & frame[f"y_h{h}"].notna()]
    m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         tree_method="hist", n_jobs=4)
    m.fit(tr[FEAT], tr[f"y_h{h}"])
    return m


def tree_shap(m, X):
    """TreeSHAP values via XGBoost's native pred_contribs (avoids shap-vs-xgboost
    model-format mismatch). Returns (n_samples, n_features); the bias column is dropped."""
    contribs = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contribs[:, :-1]


def main():
    frame = load_frame()
    test = frame[frame["split"] == "test"]
    imp_rows, store = [], {}
    for h in HORIZONS:
        m = train_xgb(frame, h)
        Xt = test[FEAT].reset_index(drop=True)
        sv = tree_shap(m, Xt)
        for f, v in zip(FEAT, np.abs(sv).mean(axis=0)):
            imp_rows.append(dict(horizon=h, feature=f, mean_abs_shap=float(v)))
        store[h] = (sv, Xt)

    imp = pd.DataFrame(imp_rows)
    imp.to_csv(OUTDIR / "shap_importance.csv", index=False)
    print("SHAP on the XGBoost member — top drivers by horizon (mean |SHAP|):")
    for h in HORIZONS:
        top = imp[imp.horizon == h].nlargest(6, "mean_abs_shap")
        print(f"  h={h:>2}: " + ", ".join(f"{r.feature}={r.mean_abs_shap:.2f}"
                                          for r in top.itertuples()))

    FIGDIR.mkdir(parents=True, exist_ok=True)

    # beeswarm for a representative horizon
    hr = HORIZONS[1] if len(HORIZONS) > 1 else HORIZONS[0]
    sv, Xt = store[hr]
    idx = np.random.RandomState(SEED).choice(len(Xt), size=min(2000, len(Xt)), replace=False)
    shap.summary_plot(sv[idx], Xt.iloc[idx], max_display=12, show=False)
    plt.title(f"SHAP — XGBoost member, h = {hr} weeks (explains the component, not the ensemble)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # how drivers shift: top features at the shortest vs longest horizon
    hmin, hmax = HORIZONS[0], HORIZONS[-1]
    feats = pd.unique(pd.concat([
        imp[imp.horizon == hmin].nlargest(6, "mean_abs_shap").feature,
        imp[imp.horizon == hmax].nlargest(6, "mean_abs_shap").feature]))
    piv = (imp[imp.feature.isin(feats)]
           .pivot(index="feature", columns="horizon", values="mean_abs_shap")
           .loc[feats][[hmin, hmax]])
    ax = piv.plot(kind="barh", figsize=(8, 5),
                  color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("mean |SHAP| (impact on XGBoost forecast)")
    ax.set_ylabel("")
    ax.legend([f"h = {hmin} wk", f"h = {hmax} wk"], title="horizon")
    ax.set_title("XGBoost drivers shift with horizon (short vs long)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_shap_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved shap_importance.csv + fig_shap_summary.png + fig_shap_by_horizon.png")


if __name__ == "__main__":
    main()

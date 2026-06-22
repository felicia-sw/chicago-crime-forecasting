"""
Add a deep-learning member (global LSTM) to the ensemble.

Motivation: the closest prior art (NCAA 2025) uses LSTM/BLSTM but only *compares*
models. We instead add an LSTM as a fourth ENSEMBLE member and let the adaptive
weighting blend it with SARIMA/Prophet/XGBoost — which is exactly our contribution.

This reuses the cached base forecasts: it does NOT retrain SARIMA. It trains one global
LSTM per horizon (direct strategy) on a window of the past L weekly counts per district,
predicts the val+test grid, and APPENDS those forecasts (model='lstm') to
<out_subdir>/base_forecasts.csv. Re-running is idempotent (old lstm rows are replaced).

Config-aware via CRIME_CONFIG (run for both the main and regime2020 configs).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / os.environ.get("CRIME_CONFIG", "config.yaml")).read_text())
PROC = ROOT / CFG["paths"]["processed"]
OUTDIR = PROC / CFG.get("out_subdir", "")
PANEL = ROOT / CFG["paths"]["panel"]
HORIZONS = CFG["horizons"]
WINDOW_START = pd.Timestamp(CFG["window_start"])
TRAIN_END = pd.Timestamp(CFG["split"]["train_end"])
SEED = CFG["seed"]
L = 52                                               # input sequence length (1 year)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)                             # don't peg all cores on a 16GB laptop


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=48, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                            # x: (B, L, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def build_sequences(panel, frame):
    """Per horizon: standardized input windows + targets, with split + de-standardize stats."""
    data = {h: {"X": [], "y": [], "yt": [], "split": [], "mu": [], "sd": [],
                "district": [], "week": []} for h in HORIZONS}
    for d, g in panel.groupby("district"):
        g = g.sort_values("week").reset_index(drop=True)
        cnt = g["count"].to_numpy(dtype="float32")
        pos = {w: i for i, w in enumerate(g["week"])}
        m = (g["week"] >= WINDOW_START) & (g["week"] <= TRAIN_END)
        mu, sd = float(cnt[m.to_numpy()].mean()), float(cnt[m.to_numpy()].std()) + 1e-6
        fd = frame[frame["district"] == d]
        for r in fd.itertuples():
            i = pos.get(r.week)
            if i is None or i - L + 1 < 0:
                continue
            seq = (cnt[i - L + 1:i + 1] - mu) / sd
            for h in HORIZONS:
                yt = getattr(r, f"y_h{h}")
                if yt != yt:                          # NaN target (series tail)
                    continue
                dd = data[h]
                dd["X"].append(seq); dd["y"].append((yt - mu) / sd); dd["yt"].append(yt)
                dd["split"].append(r.split); dd["mu"].append(mu); dd["sd"].append(sd)
                dd["district"].append(d); dd["week"].append(r.week)
    return data


def train_predict(dd, h):
    X = torch.tensor(np.asarray(dd["X"], dtype="float32")).unsqueeze(-1)   # (n, L, 1)
    y = torch.tensor(np.asarray(dd["y"], dtype="float32"))
    split = np.asarray(dd["split"])
    tr = split == "train"
    Xtr, ytr = X[tr], y[tr]

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n, bs = len(Xtr), 256
    for epoch in range(40):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    pred_mask = np.isin(split, ["val", "test"])
    with torch.no_grad():
        pred_std = model(X[pred_mask]).numpy()
    mu = np.asarray(dd["mu"])[pred_mask]
    sd = np.asarray(dd["sd"])[pred_mask]
    return pd.DataFrame({
        "district": np.asarray(dd["district"])[pred_mask],
        "week": np.asarray(dd["week"])[pred_mask],
        "y_true": np.asarray(dd["yt"])[pred_mask],
        "horizon": h, "model": "lstm",
        "yhat": pred_std * sd + mu})


def main():
    panel = pd.read_csv(PANEL, dtype={"district": str}, parse_dates=["week"])
    frame = pd.read_csv(OUTDIR / "model_frame.csv", dtype={"district": str}, parse_dates=["week"])
    data = build_sequences(panel, frame)

    parts = []
    for h in HORIZONS:
        df = train_predict(data[h], h)
        rmse = float(np.sqrt(np.mean((df.yhat - df.y_true) ** 2)))
        print(f"  lstm h={h:>2}: trained, val+test RMSE={rmse:.2f}  (n={len(df)})")
        parts.append(df)
    lstm = pd.concat(parts, ignore_index=True)

    bpath = OUTDIR / "base_forecasts.csv"
    base = pd.read_csv(bpath, dtype={"district": str}, parse_dates=["week"])
    base = base[base["model"] != "lstm"]             # idempotent re-run
    cols = ["district", "week", "y_true", "horizon", "model", "yhat"]
    out = pd.concat([base[cols], lstm[cols]], ignore_index=True)
    out.to_csv(bpath, index=False)
    print(f"Appended lstm to {bpath} -> models now: {sorted(out['model'].unique())}")


if __name__ == "__main__":
    main()

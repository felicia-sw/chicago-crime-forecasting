#!/usr/bin/env bash
# Memory-safe driver for the SARIMA+Prophet base backtest on a 16 GB machine.
#
# Why per-district subprocesses: SARIMA with m=52 peaks ~4.2 GB PER district. To
# guarantee memory can never accumulate (and re-OOM the machine), each district runs
# in its OWN short-lived python process that exits and reclaims all memory before the
# next starts. models_base.py checkpoints each district, so this is fully resumable:
# re-running skips any district already done. naive + xgboost are instant and added in
# the final assemble pass.
#
# Usage:  bash src/run_base_backtest.sh [config.yaml]
#   bash src/run_base_backtest.sh                       # main run -> data/processed/
#   bash src/run_base_backtest.sh config_regime2020.yaml  # 2020 run -> data/processed/regime2020/
set -u
cd "$(dirname "$0")/.."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate crime-fcst

export CRIME_CONFIG="${1:-config.yaml}"             # which experiment config the python steps read
echo "Using CRIME_CONFIG=$CRIME_CONFIG"

# 22 active districts; checkpointed, so any already-done district is skipped on a re-run
DISTRICTS="001 002 003 004 005 006 007 008 009 010 011 012 014 015 016 017 018 019 020 022 024 025"

i=0; n=$(echo $DISTRICTS | wc -w | tr -d ' ')
for d in $DISTRICTS; do
  i=$((i+1))
  echo "=== [$i/$n] district $d  $(date +%H:%M:%S) ==="
  python src/models_base.py --models sarima,prophet --districts "$d" --jobs 1 \
      --out base_forecasts_partial.csv 2>&1 | grep -vE "cmdstanpy|plotly|INFO -"
done

echo "=== final assemble (naive + sarima + prophet + xgboost, all 22) $(date +%H:%M:%S) ==="
python src/models_base.py --jobs 1 --out base_forecasts.csv 2>&1 | grep -vE "cmdstanpy|plotly|INFO -"
echo "=== ALL DONE $(date +%H:%M:%S) ==="

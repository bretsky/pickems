#!/bin/bash
# Fit every learned piece on data before a Worlds and simulate it: PICKEMS_ASOF backtest.
#   scripts/backtest_pipeline.sh 2025 2025-10-14 [sims] [worlds weight]
# (needs vendor/dp_asof_<year> from scripts/make_asof.py and data/manual/worlds<year>.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
YEAR=$1; CUT=$2; SIMS=${3:-1000}; WW=${4:-0.9}
export PICKEMS_ASOF=$YEAR
PY=.venv/bin/python
mkdir -p ".cache/asof_$YEAR" "data/derived/asof_$YEAR"
LOG=".cache/asof_$YEAR/logs"; mkdir -p "$LOG"
$PY -W ignore scripts/analysis/patch_shift.py 2>&1 | tee "$LOG/patch_shift.log"
$PY -W ignore scripts/train_draft.py --gbm --final --worlds-weight "$WW" --cutoff "$((YEAR))-06-01" 2>&1 | tee "$LOG/train_draft.log"
$PY -W ignore scripts/train_stats.py --worlds-weight "$WW" 2>&1 | tee "$LOG/train_stats.log"
$PY -W ignore scripts/simulate.py --sims "$SIMS" --worlds-weight "$WW" --name "bt${YEAR}_${SIMS}" 2>&1 | tee "$LOG/simulate.log"

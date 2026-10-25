#!/bin/bash
# Calibrate the sims against one finished Worlds, end to end: its pre-Worlds data copy
# and models, then the backtest sweep (scripts/backtest_progress.sh) at the pre-Worlds
# point and at eighths of the event's games played, plain and perturbed. The rows
# land in .cache/calibration.jsonl, which scripts/recommend_orders.py reads (every
# year in it, averaged at the live run's fraction played).
#   scripts/calibrate_year.sh 2026 [cuts] [sims] [sigmas] [worlds weight]
# cuts default to "pre 0.125 0.25 0.375 0.5 0.625 0.75 0.875", run in that order (put
# the ones needed soonest first).
# Needs data/manual/worlds<year>.yaml and that year's full Oracle's Elixir CSV in
# vendor/dp (python -m pickems.sync_model). Steps already done are skipped, so it
# can be rerun to add fractions or sigmas. Several hours at the defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
YEAR=$1; CUTS=${2:-"pre 0.125 0.25 0.375 0.5 0.625 0.75 0.875"}; SIMS=${3:-400}; SIGMAS=${4:-"0 0.4 0.6 0.9"}; WW=${5:-0.9}
PY=.venv/bin/python
CONFIG=data/manual/worlds$YEAR.yaml
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
START=$($PY -c "import yaml; print(min(str(e['date']) for e in yaml.safe_load(open('$CONFIG'))['schedule']))")

# pre-Worlds data copy (games before the first day) and models trained on it
[ -d "vendor/dp_asof_$YEAR" ] || $PY scripts/make_asof.py "$YEAR" "$START"
if [ ! -f ".cache/asof_$YEAR/draft_model_ww$WW.pkl" ] || [ ! -f ".cache/asof_$YEAR/stats_model_ww$WW.pkl" ]; then
	export PICKEMS_ASOF=$YEAR
	mkdir -p ".cache/asof_$YEAR/logs" "data/derived/asof_$YEAR"
	LOG=".cache/asof_$YEAR/logs"
	$PY -W ignore scripts/analysis/patch_shift.py > "$LOG/patch_shift.log" 2>&1
	$PY -W ignore scripts/train_draft.py --gbm --final --worlds-weight "$WW" --cutoff "$YEAR-06-01" > "$LOG/train_draft.log" 2>&1
	$PY -W ignore scripts/train_stats.py --worlds-weight "$WW" > "$LOG/train_stats.log" 2>&1
	unset PICKEMS_ASOF
fi

scripts/backtest_progress.sh "$YEAR" "$CUTS" "$SIMS" "$SIGMAS" "$WW"

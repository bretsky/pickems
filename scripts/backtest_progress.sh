#!/bin/bash
# How far to trust a run made part-way through a Worlds: re-run a past Worlds cut at
# several points of the event, plain and perturbed, and score every run's whole-event
# champion totals against what happened (scripts/calibrate_picks.py).
#   scripts/backtest_progress.sh 2025 "2025-10-16 2025-10-17 2025-10-20" [sims] [sigmas] [worlds weight]
# Each cut keeps the games dated before it (scripts/make_asof.py --mid, reusing the
# year's pre-Worlds models, so backtest_pipeline.sh for that year must have run).
# Runs that already exist are not redone. Then:
#   .venv/bin/python scripts/calibrate_picks.py --table .cache/calibration.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
YEAR=$1; CUTS=$2; SIMS=${3:-400}; SIGMAS=${4:-"0 0.6"}; WW=${5:-0.9}
PY=.venv/bin/python
SUMMARY=.cache/calibration.jsonl
for CUT in $CUTS; do
	KEY="$YEAR-${CUT:5:2}${CUT:8:2}"
	[ -d "vendor/dp_asof_$KEY" ] || $PY scripts/make_asof.py "$YEAR" "$CUT" --mid
	LOG=".cache/asof_$KEY/logs"; mkdir -p "$LOG"
	RUNS=()
	for S in $SIGMAS; do
		NAME="perturb_bt_${SIMS}_p$S"
		RUNS+=("$NAME")
		[ -f ".cache/asof_$KEY/sims/$NAME.pkl" ] && continue
		echo "== $KEY sigma $S"
		PICKEMS_ASOF=$KEY $PY -W ignore scripts/simulate_perturbed.py --sims "$SIMS" --worlds-weight "$WW" \
			--sigma-pick "$S" --name "$NAME" > "$LOG/$NAME.log" 2>&1
	done
	PICKEMS_ASOF=$KEY $PY -W ignore scripts/calibrate_picks.py "${RUNS[@]}" --summary "$SUMMARY" | tee "$LOG/calibrate.log"
done
$PY scripts/calibrate_picks.py --table "$SUMMARY"

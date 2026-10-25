#!/bin/bash
# How far to trust a run made part-way through a Worlds: re-run a past Worlds cut at
# several points of the event, plain and perturbed, and score every run's whole-event
# champion totals against what happened (scripts/calibrate_picks.py).
#   scripts/backtest_progress.sh 2025 "pre 0.25 0.5 0.75" [sims] [sigmas] [worlds weight]
#   scripts/backtest_progress.sh 2025 "2025-10-16 2025-10-20T04:00" ...
# A cut is "pre" (the pre-Worlds backtest itself, PICKEMS_ASOF=<year>), a fraction of
# the event's games played (scripts/backtest_cuts.py turns it into a time after that
# session), a day, or a UTC time; each mid-event cut keeps the games before it
# (scripts/make_asof.py --mid, reusing the year's pre-Worlds models). So the year's
# pre-Worlds models must exist first: scripts/calibrate_year.sh does all of it.
# Every cut gets the same sigmas (default 0 0.4 0.6 0.9). Runs that already exist
# are not redone. Then:
#   .venv/bin/python scripts/calibrate_picks.py --table .cache/calibration.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
YEAR=$1; CUTS=$2; SIMS=${3:-400}; SIGMAS=${4:-"0 0.4 0.6 0.9"}; WW=${5:-0.9}
PY=.venv/bin/python
SUMMARY=.cache/calibration.jsonl
for CUT in $CUTS; do
	if [ "$CUT" = pre ]; then
		KEY=$YEAR
	else
		[[ $CUT =~ ^0?\.[0-9]+$ ]] && CUT=$($PY scripts/backtest_cuts.py "$YEAR" "$CUT")
		KEY=$($PY scripts/make_asof.py "$YEAR" "$CUT" --mid --key)
		[ -d "vendor/dp_asof_$KEY" ] || $PY scripts/make_asof.py "$YEAR" "$CUT" --mid
	fi
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

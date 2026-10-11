#!/bin/bash
# The whole Worlds-weighted flow, end to end:
#
#   1. sync  -- create vendor/dp (the patched one-way copy of draftpredict) or
#               bring it to draftpredict's latest commit (pickems.sync_model)
#   2. tune  -- Optuna retune of draftpredict's params in vendor/dp at the
#               Worlds weight, adopted with --snapshot into its own
#               optuna_best_params_ww<W>.json (never touches ~/draftpredict)
#   3. train -- draft model (dataset rebuilt) and stats model at the Worlds weight
#   4. sim   -- full simulation; the weighted win model is rebuilt from the
#               retuned params on the way in, and the dashboard is rendered
#
#   scripts/full_pipeline.sh                                  # w=0.9, 2000 sims, 2400-trial retune on 5 workers
#   scripts/full_pipeline.sh --worlds-weight 0.95 --sims 1000 --name run_1000_ww095
#   scripts/full_pipeline.sh --skip-tune                      # reuse the current retuned params
#
# Options:
#   --worlds-weight W   0 < W <= 1 (default 0.9)
#   --sims N            simulations (default 2000)
#   --name NAME         run name (default run_<sims>_ww<W>_<draftpredict commit>_<YYYYmmdd-HHMM>)
#   --tune-workers N    parallel Optuna workers (default 5)
#   --tune-trials N     Optuna trial budget (default 2400; scaled down after the flag ablation)
#   --skip-tune         keep the current retune for this weight (a sync that moves the code archives it)
#   --skip-train        keep the current draft / stats models for this weight
#   --no-plot           don't open draftpredict's live Optuna plot during the retune
#
# Every step is logged to .cache/pipeline/<name>/; stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"

WW=0.9
SIMS=2000
NAME=""
TUNE_WORKERS=5
TUNE_TRIALS=2400
SKIP_TUNE=""
SKIP_TRAIN=""
NO_PLOT=""
while [ $# -gt 0 ]; do
	case "$1" in
		--worlds-weight) WW="$2"; shift 2 ;;
		--sims) SIMS="$2"; shift 2 ;;
		--name) NAME="$2"; shift 2 ;;
		--tune-workers) TUNE_WORKERS="$2"; shift 2 ;;
		--tune-trials) TUNE_TRIALS="$2"; shift 2 ;;
		--skip-tune) SKIP_TUNE=1; shift ;;
		--skip-train) SKIP_TRAIN=1; shift ;;
		--no-plot) NO_PLOT=1; shift ;;
		-h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
		*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
	esac
done
"$PY" -c "import sys; w = float(sys.argv[1]); sys.exit(0 if 0 < w <= 1 else 1)" "$WW" \
	|| { echo "--worlds-weight must be in (0, 1]; for unweighted runs use scripts/simulate.py directly" >&2; exit 2; }

fmt() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }
TIMES=()
step() {  # step <label> <log name> <command...>: run, tee to the log, record the time
	local label="$1" log="$LOGS/$2.log"; shift 2
	local t0=$SECONDS
	echo
	echo "=== $label  (log: ${log#$ROOT/})"
	"$@" 2>&1 | tee "$log"
	TIMES+=("$label $(fmt $((SECONDS - t0)))")
}

# 1. sync
LOGS="$ROOT/.cache/pipeline/_pending"
rm -rf "$LOGS"
mkdir -p "$LOGS"
step "sync vendor/dp" sync "$PY" -m pickems.sync_model
COMMIT=$(git -C vendor/dp rev-parse --short upstream)
NAME="${NAME:-run_${SIMS}_ww$(echo "$WW" | tr -d .)_${COMMIT}_$(date +%Y%m%d-%H%M)}"
LOGS="$ROOT/.cache/pipeline/$NAME"
if [ -d "$LOGS" ]; then
	echo "logs for $NAME already exist; pick another --name" >&2
	exit 1
fi
mv "$ROOT/.cache/pipeline/_pending" "$LOGS"
echo "run $NAME: draftpredict $COMMIT, worlds weight $WW"

# 2. tune
# The live plot (draftpredict's scripts/plot_optuna_live.py, a matplotlib
# window) opens in the background once the Optuna workers launch: the runner
# deletes and recreates the study db right before that, after the flag
# ablation, and a reader opened earlier would keep watching the deleted file.
# Worker 1's log appearing (newer than this step's start) marks the new study.
start_live_plot() {
	local marker="$LOGS/.tune_started" worker_log="vendor/dp/.cache/tune_optuna_workers/worker_1.log"
	touch "$marker"
	(
		until [ -f "$worker_log" ] && [ "$worker_log" -nt "$marker" ]; do sleep 5; done
		echo "opening the live Optuna plot (log: ${LOGS#$ROOT/}/plot.log)" >&2
		cd vendor/dp && exec .venv/bin/python -m scripts.plot_optuna_live >"$LOGS/plot.log" 2>&1
	) &
	PLOT_PID=$!
	disown "$PLOT_PID"
}
stop_plot_waiter() {  # the window itself stays open; only a still-waiting watcher is stopped
	if [ -n "${PLOT_PID:-}" ] && ! ps -p "$PLOT_PID" -o command= | grep -q plot_optuna_live; then
		kill "$PLOT_PID" 2>/dev/null || true
	fi
}
if [ -z "$SKIP_TUNE" ]; then
	[ -z "$NO_PLOT" ] && start_live_plot
	trap stop_plot_waiter EXIT
	step "Optuna retune at w=$WW" tune \
		bash -c 'cd vendor/dp && DP_WORLDS_WEIGHT="$1" scripts/tune_optuna_parallel.sh "$2" "$3" --warm-start --snapshot' _ "$WW" "$TUNE_WORKERS" "$TUNE_TRIALS"
	stop_plot_waiter
fi
"$PY" - "$WW" <<'EOF'
import sys
from pickems.model import tuned_params
w = float(sys.argv[1])
t = tuned_params(w)
if t:
	print(f"win model params: Worlds-retuned v{t['version']} (weighted CV {t['best_value']:.5f}, {t['n_trials']} trials, draftpredict {t['base_commit'][:7]})")
else:
	print(f"win model params: draftpredict's own -- no retune adopted at w={w:g} for this code yet")
EOF

# 3. train
if [ -z "$SKIP_TRAIN" ]; then
	step "train draft model" train_draft "$PY" -u -W ignore scripts/train_draft.py --gbm --final --worlds-weight "$WW"
	step "train stats model" train_stats "$PY" -u -W ignore scripts/train_stats.py --worlds-weight "$WW"
fi

# 4. simulate
step "simulate $SIMS" simulate "$PY" -u -W ignore scripts/simulate.py --sims "$SIMS" --worlds-weight "$WW" --name "$NAME"

echo
echo "=== done: $NAME"
printf '  %s\n' "${TIMES[@]}"
echo "  dashboard: dashboard/out/runs/$NAME.html"
echo "  compare:   .venv/bin/python -W ignore scripts/compare_runs.py <other run> $NAME"

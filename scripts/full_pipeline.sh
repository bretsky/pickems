#!/bin/bash
# The whole Worlds-weighted flow, end to end:
#
#   1. copy  -- create vendor/dp_tune (tune/setup_dp_tune.sh) or bring it up to
#               draftpredict's latest commit (tune/update_dp_tune.sh)
#   2. sync  -- vendor/dp from that same commit, so simulation and retune agree
#   3. tune  -- Optuna retune of draftpredict's params in vendor/dp_tune at the
#               Worlds weight, adopted with --snapshot (never touches ~/draftpredict)
#   4. train -- draft model (dataset rebuilt) and stats model at the Worlds weight
#   5. sim   -- full simulation; the weighted win model is rebuilt from the
#               retuned params on the way in, and the dashboard is rendered
#
#   scripts/full_pipeline.sh                                  # w=0.9, 2000 sims, 5x400-trial retune
#   scripts/full_pipeline.sh --worlds-weight 0.95 --sims 1000 --name run_1000_ww095
#   scripts/full_pipeline.sh --skip-tune                      # reuse the current retuned params
#
# Options:
#   --worlds-weight W   0 < W <= 1 (default 0.9)
#   --sims N            simulations (default 2000)
#   --name NAME         run name (default run_<sims>_ww<W>_<draftpredict commit>)
#   --tune-workers N    parallel Optuna workers (default 5)
#   --tune-trials N     Optuna trial budget (default 400; scaled down after the flag ablation)
#   --skip-tune         keep vendor/dp_tune's current snapshot (must match the commit)
#   --skip-train        keep the current draft / stats models for this weight
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
TUNE_TRIALS=400
SKIP_TUNE=""
SKIP_TRAIN=""
while [ $# -gt 0 ]; do
	case "$1" in
		--worlds-weight) WW="$2"; shift 2 ;;
		--sims) SIMS="$2"; shift 2 ;;
		--name) NAME="$2"; shift 2 ;;
		--tune-workers) TUNE_WORKERS="$2"; shift 2 ;;
		--tune-trials) TUNE_TRIALS="$2"; shift 2 ;;
		--skip-tune) SKIP_TUNE=1; shift ;;
		--skip-train) SKIP_TRAIN=1; shift ;;
		-h|--help) sed -n '2,27p' "$0"; exit 0 ;;
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

# 1. copy
LOGS="$ROOT/.cache/pipeline/_pending"
mkdir -p "$LOGS"
if [ -d vendor/dp_tune/.git ]; then
	step "update tuning copy" copy tune/update_dp_tune.sh
else
	step "create tuning copy" copy tune/setup_dp_tune.sh
fi
COMMIT=$(git -C vendor/dp_tune rev-parse --short upstream)
NAME="${NAME:-run_${SIMS}_ww$(echo "$WW" | tr -d .)_${COMMIT}}"
LOGS="$ROOT/.cache/pipeline/$NAME"
if [ -d "$LOGS" ]; then
	echo "logs for $NAME already exist; pick another --name" >&2
	exit 1
fi
mv "$ROOT/.cache/pipeline/_pending" "$LOGS"
echo "run $NAME: draftpredict $COMMIT, worlds weight $WW"

# 2. sync
step "sync vendor/dp to $COMMIT" sync "$PY" -m pickems.sync_model --ref "$COMMIT"

# 3. tune
if [ -z "$SKIP_TUNE" ]; then
	step "Optuna retune at w=$WW" tune \
		bash -c 'cd vendor/dp_tune && DP_WORLDS_WEIGHT="$1" scripts/tune_optuna_parallel.sh "$2" "$3" --warm-start --snapshot' _ "$WW" "$TUNE_WORKERS" "$TUNE_TRIALS"
fi
"$PY" - "$WW" "$COMMIT" <<'EOF'
import json, subprocess, sys
w, commit = float(sys.argv[1]), sys.argv[2]
snap = json.load(open("vendor/dp_tune/optuna_best_params.json"))
full = subprocess.run(["git", "-C", "vendor/dp_tune", "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()
if snap.get("worlds_weight") == w and snap.get("base_commit") == full:
	print(f"win model params: Worlds-retuned v{snap['version']} (weighted CV {snap['best_value']:.5f}, {snap['n_trials']} trials)")
else:
	print(f"win model params: draftpredict's own v{snap.get('version')} -- no retuned snapshot for w={w:g} on {commit} "
	      "(the retune didn't beat its warm start, or was skipped)")
EOF

# 4. train
if [ -z "$SKIP_TRAIN" ]; then
	step "train draft model" train_draft "$PY" -u -W ignore scripts/train_draft.py --gbm --final --worlds-weight "$WW"
	step "train stats model" train_stats "$PY" -u -W ignore scripts/train_stats.py --worlds-weight "$WW"
fi

# 5. simulate
step "simulate $SIMS" simulate "$PY" -u -W ignore scripts/simulate.py --sims "$SIMS" --worlds-weight "$WW" --name "$NAME"

echo
echo "=== done: $NAME"
printf '  %s\n' "${TIMES[@]}"
echo "  dashboard: dashboard/out/runs/$NAME.html"
echo "  compare:   .venv/bin/python -W ignore scripts/compare_runs.py <other run> $NAME"

#!/bin/bash
# Bring vendor/dp_tune up to date with ~/draftpredict, one way.
#
#   1. fetch draftpredict's current HEAD into the copy's `upstream` branch
#      (a fetch only reads ~/draftpredict; committed work only, so commit
#      anything in draftpredict's working tree you want included)
#   2. archive a retuned snapshot (optuna_best_params.json) to
#      .cache/snapshots/ and restore upstream's copy, since params tuned on the
#      old code don't carry over: retune after updating
#   3. rebase the Worlds-weighting patch onto it and re-export
#      tune/dp_worlds_weight.patch (stops on a conflict for you to resolve)
#   4. re-clone changed season CSVs, reinstall the venv if draftpredict's
#      package versions changed
#
# Then sync vendor/dp to the same commit so simulations and the retune agree:
#   python -m pickems.sync_model --ref <commit printed at the end>
#
#   tune/update_dp_tune.sh
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."
PICKEMS="$PWD"
SRC="${DP_SRC:-$HOME/draftpredict}"
DEST="${DP_TUNE_DIR:-vendor/dp_tune}"
[ -d "$DEST/.git" ] || { echo "$DEST missing: run tune/setup_dp_tune.sh" >&2; exit 1; }

# Refuse while a tuning run is using the copy.
for pid in $(pgrep -f "scripts\.(tune_optuna|ablate_flags)" || true); do
	if lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | grep -q "^n$PICKEMS/$DEST\$"; then
		echo "a tuning process (pid $pid) is running in $DEST; wait for it or stop it first" >&2
		exit 1
	fi
done

if [ -n "$(git -C "$SRC" status --porcelain -- draftpredict scripts optuna_best_params.json live_margin_noise.json)" ]; then
	echo "note: ~/draftpredict has uncommitted changes; only its committed HEAD is taken"
fi

cd "$DEST"
OLD=$(git rev-parse --short upstream)
git fetch -q "$SRC" HEAD
NEW=$(git rev-parse --short FETCH_HEAD)

if ! git diff --quiet -- optuna_best_params.json; then
	mkdir -p .cache/snapshots
	ARCHIVE=".cache/snapshots/$(date +%Y%m%d_%H%M%S)_base${OLD}.json"
	cp optuna_best_params.json "$ARCHIVE"
	git checkout -q -- optuna_best_params.json
	echo "archived the retuned snapshot (tuned on $OLD) to $DEST/$ARCHIVE"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
	echo "$DEST has other uncommitted changes; commit or discard them first" >&2
	git status --short --untracked-files=no >&2
	exit 1
fi

if [ "$OLD" = "$NEW" ]; then
	echo "code already at draftpredict $NEW"
else
	git branch -f upstream FETCH_HEAD
	if ! git rebase -q upstream worlds-weight; then
		echo "the Worlds-weighting patch conflicts with draftpredict $NEW: resolve in $DEST, 'git rebase --continue', then rerun this script" >&2
		exit 1
	fi
	git format-patch -q upstream..worlds-weight --stdout > "$PICKEMS/tune/dp_worlds_weight.patch"
	echo "rebased the patch from draftpredict $OLD onto $NEW (tune/dp_worlds_weight.patch updated: commit it)"
fi

for f in "$SRC"/*_LoL_esports_match_data_from_OraclesElixir.csv; do
	name=$(basename "$f")
	if [ ! -f "$name" ] || [ "$(stat -f '%z %m' "$f")" != "$(stat -f '%z %m' "$name")" ]; then
		rm -f "$name"
		cp -c -p "$f" .
		echo "updated $name"
	fi
done

"$SRC/.venv/bin/python" -m pip freeze | grep -viE '^-e |@ |^psycopg2|^py-clob|^windows-curses' > .cache/requirements.new.txt
if ! cmp -s .cache/requirements.new.txt .cache/requirements.txt; then
	mv .cache/requirements.new.txt .cache/requirements.txt
	.venv/bin/python -m pip install -q -r .cache/requirements.txt
	echo "venv updated to draftpredict's package versions"
else
	rm .cache/requirements.new.txt
fi

echo "dp_tune at draftpredict $NEW. Next: python -m pickems.sync_model --ref $NEW, then retune."

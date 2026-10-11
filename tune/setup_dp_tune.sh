#!/bin/bash
# One-way copy of ~/draftpredict for Worlds-weighted Optuna retunes.
#
# vendor/dp_tune is a standalone clone (gitignored here, not a submodule): its
# origin remote is removed and a pre-push hook refuses every push, so nothing
# done in it can reach ~/draftpredict or GitHub. tune/dp_worlds_weight.patch
# adds DP_WORLDS_WEIGHT to its fits and CV scores and moves its worker logs
# out of the /tmp directory the original repo's runs share. Season CSVs and the
# prior Optuna study (read by the flag ablation) are APFS copy-on-write clones;
# the venv is its own, with the original's package versions.
#
#   tune/setup_dp_tune.sh            # creates vendor/dp_tune (refuses if it exists)
#   tune/update_dp_tune.sh           # later: move it to draftpredict's latest commit
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$HOME/draftpredict"
DEST="vendor/dp_tune"
if [ -e "$DEST" ]; then
	echo "$DEST already exists; remove it first to rebuild" >&2
	exit 1
fi
git clone -q --no-hardlinks "$SRC" "$DEST"
cd "$DEST"
git remote remove origin
printf '#!/bin/sh\necho "dp_tune is a one-way copy of ~/draftpredict: pushing is disabled" >&2\nexit 1\n' > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
# `upstream` marks the draftpredict commit the copy is based on; the patch
# lives on `worlds-weight` above it (tune/update_dp_tune.sh moves both).
git branch -f upstream HEAD
git checkout -q -b worlds-weight
git -c user.name=pickems -c user.email=pickems@localhost am -q ../../tune/dp_worlds_weight.patch
for f in "$SRC"/*_LoL_esports_match_data_from_OraclesElixir.csv "$SRC"/optuna_joint_study.db*; do
	cp -c -p "$f" .
done
mkdir -p .cache
"$SRC/.venv/bin/python" -m pip freeze | grep -viE '^-e |@ |^psycopg2|^py-clob|^windows-curses' > .cache/requirements.txt
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r .cache/requirements.txt
echo "ready: $(git log --oneline -1) on $(git log --oneline -1 HEAD~1)"

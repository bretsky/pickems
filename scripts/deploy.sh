#!/usr/bin/env bash
# Stage the public site (main page + the run it was rendered from) and deploy to Cloudflare Pages.
#   scripts/deploy.sh                      # stage and deploy production
#   scripts/deploy.sh --stage              # stage only (prints the staging dir)
#   scripts/deploy.sh --snapshot [LABEL]   # freeze the current forecast as a snapshot, then redeploy production
# The run (and its buff-off pair, if any) is whatever report.json's meta.run_name names, i.e. the last
# scripts/render.py output; RUN=<run name> stages an earlier run from its runs/<run>.html instead.
#
# Snapshots: before rendering a forecast that includes a new day's results, run --snapshot so the
# forecast that preceded that day stays up. It goes to the Pages branch alias <LABEL>.pickems.pages.dev
# (LABEL defaults to pre-<first scheduled day after the run's training data>), is added to
# dashboard/versions.json (commit it), and production is redeployed so its version picker lists it and
# functions/v/[[path]].js serves it at <site>/v/<LABEL>/. Production deploys must run from the repo root
# (this script does) so wrangler picks up functions/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PROJECT=${PAGES_PROJECT:-pickems}
SITE=${SITE_DIR:-/tmp/pickems-site}
RUNOPT=${RUN:+--run $RUN}

if [ "${1:-}" = "--snapshot" ]; then
	LABEL=${2:-$($PY scripts/stage_site.py label $RUNOPT)}
	[[ "$LABEL" =~ ^[a-z0-9-]{1,28}$ ]] || { echo "label must be lowercase letters, digits and dashes (<=28): $LABEL" >&2; exit 1; }
	SNAP="$SITE-$LABEL"
	$PY scripts/stage_site.py stage "$SNAP" $RUNOPT --snapshot "$LABEL"
	npx wrangler pages deploy "$SNAP" --project-name "$PROJECT" --branch "$LABEL" --commit-dirty=true
	$PY scripts/stage_site.py record "$LABEL" $RUNOPT
	echo "snapshot deployed to https://$LABEL.$PROJECT.pages.dev/ -- commit dashboard/versions.json"
	# Production (deployed from the current git branch, as always) picks up the new versions.json (always from report.json's run, not RUN).
	RUNOPT=
fi

$PY scripts/stage_site.py stage "$SITE" $RUNOPT
[ "${1:-}" = "--stage" ] && { echo "$SITE"; exit 0; }
npx wrangler pages deploy "$SITE" --project-name "$PROJECT"

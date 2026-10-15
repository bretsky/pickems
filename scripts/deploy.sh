#!/usr/bin/env bash
# Stage the public site (main page + the run it was rendered from) and deploy to Cloudflare Pages.
#   scripts/deploy.sh            # stage and deploy
#   scripts/deploy.sh --stage    # stage only (prints the staging dir)
# The run (and its buff-off pair, if any) is whatever report.json's meta.run_name names, i.e. the last scripts/render.py output.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dashboard/out
RUN=$(python3 -c "import json,sys; print(json.load(open('$OUT/report.json'))['meta']['run_name'])")
[ -d "$OUT/runs/$RUN" ] || { echo "no $OUT/runs/$RUN (re-run scripts/render.py)" >&2; exit 1; }

SITE=${SITE_DIR:-/tmp/pickems-site}
rm -rf "$SITE"
mkdir -p "$SITE/runs"
cp "$OUT/index.html" "$OUT/report.json" "$SITE/"
cp -r "$OUT/runs/$RUN" "$SITE/runs/"
[ -f "$OUT/runs/$RUN.html" ] && cp "$OUT/runs/$RUN.html" "$SITE/runs/"
# The page's Worlds buff toggle links to the paired run (buff-off twin): ship it too.
PAIR=$(python3 -c "import json; print(json.load(open('$OUT/report.json'))['meta'].get('pair') or '')")
if [ -n "$PAIR" ]; then
	[ -d "$OUT/runs/$PAIR" ] || { echo "no $OUT/runs/$PAIR (re-run scripts/render.py for the pair)" >&2; exit 1; }
	cp -r "$OUT/runs/$PAIR" "$SITE/runs/"
	[ -f "$OUT/runs/$PAIR.html" ] && cp "$OUT/runs/$PAIR.html" "$SITE/runs/"
fi
echo "staged run $RUN${PAIR:+ + pair $PAIR} in $SITE ($(du -sh "$SITE" | cut -f1), $(find "$SITE" -type f | wc -l | tr -d ' ') files)"

[ "${1:-}" = "--stage" ] && exit 0
npx wrangler pages deploy "$SITE" --project-name "${PAGES_PROJECT:-pickems}"

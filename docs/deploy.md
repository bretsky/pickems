# Deploying the dashboard

`scripts/deploy.sh` stages the rendered page plus its run (and buff-off pair) and deploys it to
Cloudflare Pages (pickems.brettselby.xyz). Each forecast that preceded a day's games stays up as
a frozen snapshot: before rendering a run that includes new results, freeze the live one first:

```bash
scripts/deploy.sh --snapshot          # -> /v/pre-<next game day>/, then redeploys production
.venv/bin/python scripts/render.py ...  # the new run
scripts/deploy.sh
```

A snapshot is its own Pages deployment on a branch alias (one per day, no file-count limit on the
live site), served on the site's own domain at `/v/<label>/` by the Pages Function
`functions/v/[[path]].js` (it proxies only labels listed in `versions.json`). `dashboard/versions.json`
lists them (commit it after each snapshot); production serves it at `/versions.json` and every
page's "Forecast" picker reads it. Test the function locally with
`npx wrangler pages dev <staged dir> --compatibility-date=2026-05-01` from the repo root. The label defaults to
`pre-<first scheduled day after the run's training data>`; pass one to override, and
`RUN=<run name>` snapshots an earlier run from its `runs/<run>.html` (re-render it with `--no-index` first so it has the picker), and `TITLE="..."` names it in the picker. Version links are host-relative, so the picker and `/v/` work the same on pickems.pages.dev.

`scripts/deploy.sh`'s header lists every option.

## After the final

The **How we did** tab (see the README) comes from a run made once the final is in the data. For
Worlds 2026:

```bash
TITLE="Before knockouts" scripts/deploy.sh --snapshot before-knockouts   # freeze the last live forecast
scripts/full_pipeline.sh --skip-tune --skip-train --skip-h2h --sims 100 --name run_100_ww09_5e1a832_final
scripts/deploy.sh
```

The 2026 final run used `--skip-sync` with vendor/dp reset by hand to the commit behind the live
forecasts (draftpredict 5e1a832, patch d986011). That keeps its played-game prices (Result likelihood)
on the published model. draftpredict 2be099f needs torch in this venv (draft_nn) and a walker check
first. Its rebased patch is kept in vendor/dp as branch `worlds-weight-2be099f`.

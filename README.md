# pickems

Monte Carlo simulator of the 2026 League of Legends World Championship for
Riot's Pick'Em (bracket + Swiss picks) and Crystal Ball (stat questions).
Every game is simulated: First Selection, a procedural 20-action draft on patch
26.20, draftpredict's win probability on that exact draft, then a full box
score (game length, K/D/A per player, multikills, first blood, objectives).

## Run

```bash
poetry install
python -m pickems.sync_model                       # pull the latest draftpredict model + Oracle CSVs
.venv/bin/python scripts/simulate.py --sims 2000   # ~30 min on 9 workers
```

The dashboard lands in `dashboard/out/index.html` (self-contained) with the raw
numbers in `dashboard/out/report.json`. The header shows the date of the
newest game in the training history and the most recent
game in the training history that involves a Worlds team (a quick check of how
fresh the data is); runs made before these were recorded fall back to the current
history. To view it with a local server, use the
`dashboard` entry in `.claude/launch.json`.

## Walk-forward ratings

Every simulated game is fed back into draftpredict's trackers the way the next
Oracle's Elixir rebuild would feed that game (`pickems/sim/walk.py`), so later
games in a simulated tournament are predicted from team / player Elo, champion
power and the rest as they stand after the earlier results. It follows
draftpredict's day rules: within a day the Elo reads chain through live-quality
(win/loss) updates, and the full Oracle update (KDA-blended player score, exact
margin of victory) lands at the next day's open. The calendar is `schedule` in
`data/manual/worlds2026.yaml` (Riot's dates; the per-day split of each stage is
an assumption modelled on Worlds 2025). Each tournament writes to
copy-on-write layers, so it costs only its own updates.

Held at their pre-Worlds state: the lane gold-diff mastery tracker (needs gold
difference at 15 minutes, which isn't simulated), the adaptive lane multipliers
and the overpopularity slopes (the predictor keeps their outputs, not the
running sums an update needs; both move very little per game).

`scripts/check_walk.py` checks the walker against draftpredict itself: it builds
the predictor as of one day, walks a week of real Oracle games through
`OracleWalk`, and compares every tracker (and each intraday Elo read) with a
predictor built as of the week's end.

Refit the learned pieces when the data moves (new Oracle day, more 26.20 games):

```bash
.venv/bin/python scripts/train_draft.py --gbm --final   # draft model (~10 min); prints held-out accuracy
.venv/bin/python scripts/analysis/patch_shift.py         # patch -> pro presence response
.venv/bin/python -c "from pickems.history import load_games; from pickems.stats_model import fit, save; save(fit(load_games()))"
```

## Worlds-weighted training

`--worlds-weight w` (0 to 1) reweights every fitted model: Worlds games keep
weight 1, every other game gets `1 - w`. At 0 everything trains exactly as
above; at 1 only Worlds games count. It applies to the draft model, the stats
model and draftpredict's final logistic fit (its Elo / champion / player
trackers are features and are built as usual, as are this project's draft
trackers). Worlds is ~2.4% of the draft training games, so w = 0.9 already
makes it ~20% of the weight.

Weighted draft / stats models are saved beside the defaults (`*_ww<w>.pkl`).
The win model's weighting lives in `vendor/dp` itself (see below): a process
started with `PICKEMS_WORLDS_WEIGHT=w` (which `simulate.py --worlds-weight w`
sets) loads draftpredict with `DP_WORLDS_WEIGHT=w`, so its final fit is weighted,
it uses that weight's retuned params if a retune was adopted, and its predictor
is cached per weight in `vendor/dp/.cache`. The first weighted run builds it (~5 min).

```bash
.venv/bin/python scripts/train_draft.py --gbm --final --reuse-dataset --worlds-weight 0.9
.venv/bin/python scripts/train_stats.py --worlds-weight 0.9
.venv/bin/python scripts/simulate.py --sims 2000 --worlds-weight 0.9 --name run_2000_ww09
.venv/bin/python -W ignore scripts/eval_winprob.py 0 0.5 0.9 1   # win-model walk-forward CV per weight
```

`scripts/h2h.py` follows `PICKEMS_WORLDS_WEIGHT` in the environment.

Held-out results so far: the draft model gets slightly worse at 0.9 and clearly
worse at 1 (also on Jun-Oct 2026 international games). The win model's
walk-forward log loss on 912 Worlds games is 0.6091 at w=0, 0.6087 at 0.9,
0.6083 at 0.95 and 0.6155 at 1, so the gains are within noise.

### Worlds-weighted Optuna retune

Retunes run inside `vendor/dp` with draftpredict's own runner. With
`DP_WORLDS_WEIGHT=w`, every fit and every walk-forward score weights Worlds
games 1 and the rest `1 - w`, and `--snapshot` adopts the winner into
`optuna_best_params_ww<w>.json` (tagged with the weight and the draftpredict
commit), never into upstream's `optuna_best_params.json`, so unweighted runs
are unaffected.

```bash
cd vendor/dp && DP_WORLDS_WEIGHT=0.9 scripts/tune_optuna_parallel.sh 5 2400 --warm-start --snapshot
```

`scripts/full_pipeline.sh` runs the whole flow (sync, retune with the live
Optuna plot open, train both models, simulate); `--help` lists its options.

## draftpredict isolation

`pickems/sync_model.py` is the only thing that touches `~/draftpredict`, and it
only reads. `vendor/dp` is a git clone of it, not a submodule: its `origin`
remote is removed and a pre-push hook refuses every push, so nothing done in it
can reach `~/draftpredict` or GitHub. Its `upstream` branch is the draftpredict
commit it's based on (committed work only); `worlds-weight`, checked out, adds
`tune/dp_worlds_weight.patch`: the Worlds weighting above (a no-op at weight 0),
per-weight snapshots and predictor caches, and worker logs moved from the
`/tmp/tune_optuna_workers` dir the original repo's runs share into its own `.cache`.

Each sync fetches draftpredict's HEAD (`--ref <commit>` to pin), rebases the
patch onto it and re-exports `tune/dp_worlds_weight.patch` (commit it), archives
retuned snapshots made on the old code to `vendor/dp/.cache/snapshots/`,
copy-on-write clones changed season CSVs, and keeps `vendor/dp/.venv` (used by
the tuning scripts) on draftpredict's package versions. It refuses while a
retune is running there. The dashboard header shows the commit and params used.

## Knockout picks and model comparisons

```bash
.venv/bin/python -W ignore scripts/h2h.py --run <run name>                  # Bo5 odds for every pair of the 8 knockout teams (~8 min)
.venv/bin/python -W ignore scripts/compare_runs.py <old run> <new run>      # what moved between two runs, with Monte Carlo z-scores
```

Until the real quarterfinal draw exists, `scripts/h2h.py` uses the eight teams the
run predicts out of Swiss and the dashboard shows a placeholder draw that follows
the draw rules. Once it's drawn, fill `knockouts.draw` in
`data/manual/worlds2026.yaml` (bracket order), rerun `scripts/h2h.py`, then
re-render the dashboard. Knockout point values in `knockouts.pickem_points` are
assumed (5 / 5 / 10) until Riot publishes them.

The dashboard's single-run mode samples 400 of the simulations (plus enough of
each champion's wins that every "given champion" view has 25 to browse); each
carries its own brackets and its own answer to every Crystal Ball question.
Click any series in single-run mode to see its games, laid out like a broadcast
post-game screen: result bar, team totals (kills, gold, and from runs made after
this change towers, dragons, barons, grubs, heralds, Atakhan, elders and first
blood), pre-game win odds, the draft in order with champion icons, and each lane's
players face to face with damage bars. Runs made before per-game logging existed
show brackets only; re-simulate to get game details. The game logs add roughly
50 KB per sampled run, so a full dashboard is around 20 MB.

In single-run mode the Swiss picks panel also shows that run's answer key (who
went 3-0, 0-3 and 3-1/3-2), and every Crystal Ball card shows how likely that
run's outcome was plus the numbers behind it (presence counts, W-L, leaderboards,
the best single-game performances, every champion picked, ...), with chips that
open the exact game. In most-likely mode the cards add per-option averages, the
distribution behind each yes/no threshold, a pick-chance grid for every champion
and a region grid.

Single-run mode works on every tab. Title odds become that run's final
standings with the pre-sim chance of each team's finish; stat leaders become the
run's own leaderboards (each pick with its pre-sim chance of leading) and event
stats placed on the simulated distribution; the 26.20 meta table's simulated
columns show that run's drafts; player game logs open each game.

**Given champion** (the selector in the mode bar): every aggregate, from Swiss
picks and Crystal Ball to stat leaders, Elo and title odds, switches to only the
simulations a chosen team wins (any team with a title; under 30 gets a
small-sample warning), and single-run mode
browses only those runs. Knockout picks come from the head-to-head odds and don't
change.

Images: champion icons from Riot's Data Dragon, team logos and player photos from
Leaguepedia. Fetch the URLs once (and again when rosters change) before
rendering; the page links them, so viewing needs internet, and falls back to
initials without it. Fix a wrong guess in `data/manual/images.yaml`.

```bash
.venv/bin/python -m pickems.images    # writes data/derived/images.json
```

## Pieces

| Module | What it does |
|---|---|
| `history.py` | Oracle rows -> one record per game with the full ordered draft (bans/picks in draft order, roles, First Selection side), series grouping and fearless detection |
| `draft/` | Draft model: time-decayed trackers (meta, team habits, player pools, champion-role win rates, ally/enemy co-picks, lane matchups) -> per-action features -> conditional logit + grouped-softmax LightGBM |
| `soloq.py`, `meta.py` | u.gg per-patch solo-queue stats (13.13 onward) and the 26.20 presence projection |
| `stats_model.py` | Box-score sampler fitted on major-league games |
| `sim/` | Draft simulator, tournament engine (Play-In double elim, Swiss, knockouts), parallel runner, per-sim summaries |
| `report.py`, `dashboard.py` | Aggregation into title odds, Swiss pick optimizer, Crystal Ball leaderboards; HTML dashboard |
| `data/manual/worlds2026.yaml` | Format, seeds, rosters, Swiss pools, Play-In draw (from Liquipedia) |

## Validation (held-out Jun-Oct 2026 major leagues)

- Draft model, full candidate set: bans top-1 22.3% / top-5 62.0% (log loss 2.74);
  picks (champion + role) top-1 25.9% / top-5 63.1% (log loss 2.72).
- Stats model: simulated vs real game length 32.0 vs 32.1 min, kills per player
  2.87 vs 3.00, pentakill rate matches.

## Patch 26.20 findings (`scripts/analysis/patch_shift.py`)

Across 40 regular patches and the 2023-2025 Worlds patches:

- Pre-patch pro presence predicts on-patch presence with correlation 0.89 on a
  regular patch but only 0.73 at Worlds (13.19: 0.72, 14.18: 0.56, 15.20: 0.90),
  and the average per-champion shift is about twice as large.
- At Worlds, a +1pp solo-queue win-rate change moves pro presence about 2.4x more
  than on a regular patch (0.32 vs 0.14 logit per pp).
- Solo-queue movement does not predict pro win-rate changes, so the patch
  projection moves draft presence only; champion strength stays with draftpredict.

The simulator applies the solo-queue-driven multiplier on top of the draft
model's own meta trackers, raised to `META_ALPHA = 2.6` so simulated shifts
match the fitted Worlds response (`scripts/calibrate_meta.py`).

## Still open

- The "Your Pick'Ems" tab answers the 18 Worlds 2026 Crystal Ball questions and
  the 6-pick Play-In bracket (`report.pickem_answers`). Finals MVP is a vote, so
  its role answer comes from a KDA + damage heuristic and is the weakest one.
- Leaguepedia needs `LP_USERNAME` / `LP_PASSWORD` in `pickems/.env` (same
  account as python-sharp); Liquipedia is used until then.
- Rosters are the listed starters; edit the YAML for substitutions.

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

The dashboard lands in `dashboard/out/index.html` with the raw numbers in
`dashboard/out/report.json`. View it through the local server (the `dashboard`
entry in `.claude/launch.json`, or the command below, then open
http://localhost:8765/): single-run mode fetches each simulation's data from
`dashboard/out/runs/<run name>/`, which browsers block when the page is opened as
a file. Everything else works from the file too.

After editing `dashboard/template.html`, rebuild from an existing run without re-simulating
(defaults to the newest `run_*.pkl`; or pass a run name):

```bash
.venv/bin/python scripts/render.py [run name]
```

```bash
.venv/bin/python scripts/serve.py   # serves dashboard/out like the deployed site: Forecast picker, /v/<label>/ snapshots
```

## Publish

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
 The header shows the date of the
newest game in the training history and the most recent
game in the training history that involves a Worlds team (a quick check of how
fresh the data is); runs made before these were recorded fall back to the current
history.

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

## Worlds buff / nerf

Every team's past Worlds results are compared with its odds, and the gap
shifts its side of the win model's logit in every simulated game
(`pickems/worlds_buff.py`). Odds are a plain Elo walked over the whole history;
the shift is a random-effects posterior mean, residual / (variance + 1/tau^2),
with tau (how much more teams' Worlds results vary than chance allows) fitted
by marginal likelihood (tau = 0.47 today). No significance gate and no cap:
teams with little evidence get ~0 on their own. Simulations apply a fifth of
that estimate (`PICKEMS_WORLDS_BUFF`, default 0.2). The full estimates for the
2026 field: T1 +0.94 logit (43-13 vs 26.3 expected, z = 4.6), Anyone's Legend
+0.26, CTBC Flying Oyster -0.40, G2 -0.27, Cloud9 -0.26, Team Secret Whales
-0.22, the rest within +-0.15; applied, T1 +0.19 and CFO -0.08. Only Worlds proper counts: the regional qualifiers Oracle's Elixir
files under `WLDs` (see Series reversal) are relabelled `WLDs Qual` when the
history loads (`pickems.history.WORLDS_FROM`). They had diluted the spread:
with them, tau was 0.39 and T1 +0.71.
`PICKEMS_WORLDS_BUFF` scales it (default 0.2, 0 = off, 1 = the full estimate),
`PICKEMS_WORLDS_TAU` fixes tau; the applied shifts are stored in each run as
`worlds_buffs` and the scale as `worlds_buff_scale`.

The dashboard's **Worlds buff On / Off** toggle (mode bar, every tab) switches
between two runs of the same event: `<name>` with the buff and `<name>_nobuff`
with `PICKEMS_WORLDS_BUFF=0`, each with its own head-to-head file.
`scripts/full_pipeline.sh` simulates, runs h2h for and renders both (index.html
is the buffed one; the twin is `runs/<name>_nobuff.html`); the trained models
are the same for both, since the buff only shifts win odds at simulation time.
By hand: `scripts/render.py <twin> --pair <name> --no-index`, then
`scripts/render.py <name> --pair <twin>`.

`scripts/backtest_worlds_buff.py` builds the shifts from Worlds before year Y
and scores them on Y's games (2024 and 2025; no Worlds 2026 game has been
played yet). Gain over the bare Elo rises with tau up to about 0.4-0.5 (pooled
+22 millinats per game at the fitted tau, +-17; +14 at tau = 0.2, +-6), and the
fitted tau lands there.

## Form variance

Equally strong teams can differ in how spread out their results are
(`pickems/variance.py`). Each simulated tournament draws a persistent logit
"form" for every team and each of its five players, added to the model's logit in
every game they play. It has to persist: independent per-game noise would leave
every series distribution unchanged. A single game's probability is unchanged
too (the logit is stretched by sqrt(1 + pi s^2 / 8) first, so the form-averaged
probability equals the model's). Sizes: the pool sd is the excess variance of
team results per event over a plain Elo's Bernoulli noise (0.35 logit, times
`PICKEMS_FORM_SCALE`, default 0.5, since Elo is cruder than the win model; 0
turns the feature off). A team's own multiplier comes from its events' excess
variance (heavily shrunk: it is barely stable between events), a player's from
the variance of his 15-minute gold difference z-scored within league and role
(shrunk; LPL lacks the stat). Half the variance goes to the five players, half
to the team. Per-team and per-player sds are stored in each run as `form_sds`.

## Series reversal

Real series reverse slightly: tied 1-1 or 2-2, the last game's winner takes
the next game 47.1% of the time (major leagues + internationals since 2023,
n = 1859), and draftpredict's walk-forward odds on those games agree (48.6%).
Simulated series ran the other way (52.6%), with too many sweeps (knockout
Bo5 3-0/3-1/3-2 43/34/23 vs 34/38/29 at Worlds 2018-25). The intraday Elo
walk lifts a game's winner about +0.24 logit on the same draft (player and
region Elo), as it does on real data, but the procedural drafter doesn't
reproduce the loser's draft-side edge (First Selection, adapting) that offsets
it in real series. So every game after a series' first gets a logit shift
toward the previous game's loser, `SERIES_REVERSAL` = 0.2 in
`pickems/sim/tournament.py` (`PICKEMS_SERIES_REVERSAL` overrides, 0 = off),
fitted with `scripts/calibrate_reversal.py` (300 sims per value, same seeds):

| reversal | tied game | knockout Bo5 | Swiss Bo3 |
|---|---|---|---|
| 0 | 52.6% | 43/34/23 | 64/36 |
| 0.1 | 49.7% | 41/33/25 | 63/37 |
| 0.2 | 48.3% | 39/34/27 | 61/39 |
| 0.3 | 43.9% | 38/35/27 | 59/41 |

At 0.2 every split is within one standard error of Worlds 2018-25: knockouts
34/38/29 (56 series), play-in Bo5 38/38/25 (24), Swiss Bo3 56/44 (59, 2023-25).
Those counts use only each year's Worlds dates: Oracle's Elixir also files
some regional qualifiers under `WLDs` (LCK 2022; LPL and LCK 2023 and 2024;
LPL 2025 and 2026), which end weeks before Worlds starts (`WORLDS_FROM` in
`pickems/history.py`).

## Worlds-weighted training

`--worlds-weight w` (0 to 1) reweights every fitted model: Worlds games keep
weight 1, every other game gets `1 - w`. At 0 everything trains exactly as
above; at 1 only Worlds games count. It applies to the draft model, the stats
model and draftpredict's final logistic fit (its Elo / champion / player
trackers are features and are built as usual, as are this project's draft
trackers). Worlds is ~2% of the major-league games since 2024, so w = 0.9
already makes it ~16% of the weight. Worlds means Worlds proper: regional
qualifiers that Oracle's Elixir files under `WLDs` are relabelled `WLDs Qual`
(trained on like any other major league; see Worlds buff / nerf), and
vendor/dp's weighting applies the same dates (`WORLDS_FROM` in
`draftpredict/baseline.py`, in `tune/dp_worlds_weight.patch`). The adopted
w = 0.9 retune was run before this, with the qualifiers weighted as Worlds.

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
Optuna plot open, train both models, simulate with and without the Worlds buff,
h2h for each, render); `--help` lists its options.

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
.venv/bin/python -W ignore scripts/h2h.py --run <run name>                  # Bo5 odds for every pair of the 8 knockout teams (~8 min), .cache/sims/h2h_<run name>.pkl
.venv/bin/python -W ignore scripts/compare_runs.py <old run> <new run>      # what moved between two runs, with Monte Carlo z-scores
```

Until the real quarterfinal draw exists, `scripts/h2h.py` uses the eight teams the
run predicts out of Swiss and the dashboard shows a placeholder draw that follows
the draw rules. Once it's drawn, fill `knockouts.draw` in
`data/manual/worlds2026.yaml` (bracket order), rerun `scripts/h2h.py`, then
re-render the dashboard. Each run reads its own `h2h_<run name>.pkl` (runs made
before that read the shared `h2h_knockout.pkl`); set `PICKEMS_WORLDS_BUFF` to the
run's buff scale (0 for a `_nobuff` twin). Knockout point values in `knockouts.pickem_points` are
assumed (5 / 5 / 10) until Riot publishes them.

The dashboard's single-run mode can show any of the simulations; each carries its
own brackets and its own answer to every Crystal Ball question. Each one is a
~50 KB file in `dashboard/out/runs/<run name>/<i>.json` (about 100 MB for 2000
sims), loaded when you open it, so the page itself stays around 3 MB. The
`runs/<run name>.html` copy reads the same files.
Click any series in single-run mode to see its games, laid out like a broadcast
post-game screen: result bar, team totals (kills, gold, and from runs made after
this change towers, dragons, barons, grubs, heralds, Atakhan, elders and first
blood), pre-game win odds, the draft in order with champion icons, and each lane's
players face to face with damage bars. Runs made before per-game logging existed
show brackets only; re-simulate to get game details.

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

## Polymarket: Most Played Champion

Prices Polymarket's "Most Played Champion at Worlds <year>" market (most picks over
the whole event; ties on win rate, then bans, then name) off the sims, and sizes
orders by fractional Kelly on top of the positions already held. The plain sims are
overconfident about pick totals, so the recommendation sizes against two corrected
models at once and keeps only what grows the bankroll under both.

### Setup (once)

```bash
poetry install
export POLYMARKET_WALLET=0x...      # proxy wallet whose positions count as held (or pass --positions held.yaml)
scripts/calibrate_year.sh 2025      # backtest calibration, several hours, unattended; already run for 2025
```

`calibrate_year.sh <year>` re-runs a finished Worlds as if from several points of the
event and scores each run against what happened:

1. `make_asof.py <year> <first day>`: `vendor/dp_asof_<year>`, the data before Worlds
   (`PICKEMS_ASOF=<year>` points the whole pipeline at it), then the patch fit, draft
   and stats models trained on it (`.cache/asof_<year>/`).
2. `backtest_progress.sh`: for the pre-Worlds point and each fraction of the event's
   games played (default 15, 25, 40, 55, 70, 85%; `backtest_cuts.py` turns a fraction
   into a cut after that day's games), a data copy cut there (`make_asof.py --mid`,
   `PICKEMS_ASOF=<year>-<mmdd>[-<hhmm>]`, reusing the pre-Worlds models), so the games
   before the cut are replayed as a live run replays them. Each cut is simulated
   plain and with perturbed champion strength (`simulate_perturbed.py`, sigma 0, 0.4,
   0.6, 0.9; 400 sims each, ~12 min a run).
3. `calibrate_picks.py --summary .cache/calibration.jsonl`: per run, the share of
   games played, interval coverage of champion pick totals, and the fitted logit
   scale a of P(top k by picks) (a < 1: overconfident; 1: calibrated). See them with
   `.venv/bin/python scripts/calibrate_picks.py --table .cache/calibration.jsonl`.

It skips whatever already exists, so rerunning it with more fractions or sigmas only
adds those. Only fearless-draft Worlds (2025 on) are used: earlier pick totals were far
more concentrated.

### During Worlds

```bash
python -m pickems.sync_model                              # after copying the new Oracle's Elixir CSV into ~/draftpredict
scripts/full_pipeline.sh --skip-tune --skip-train         # the plain run (run_*.pkl)
.venv/bin/python -W ignore scripts/recommend_orders.py    # [--bankroll 1000] [--kelly 0.25] [--extend] [--no-sim]
```

`recommend_orders.py`:

1. Takes the newest plain run and its fraction of games played (played / the sims'
   mean total).
2. From `.cache/calibration.jsonl`, interpolates at that fraction (per backtest year,
   then averaged over years) the plain run's a, used as a temper p^a on its
   P(most picked), and the sigma at which perturbed runs reach a = 1 (capped at the
   largest sigma tried). If the cuts stop short of the run's fraction it says so;
   `--extend` backtests a cut there first (~1 h).
3. Simulates a 1000-sim perturbed run at that sigma on the same data (~30 min), or
   reuses it: it is named after the plain run, so a new plain run or a new sigma makes
   a new one. `--no-sim` skips this, sizing off the tempered run alone (champions the
   sims never made most picked stay at 0, so long-shot No bets go unchecked).
4. Sizes takers (walking the books) and resting limit bids with
   `bet_robustness.py` over the two models. Place the **robust sh** column: takers up
   to the limit price, limit bids at it. The sizing table shows each sizing's
   expected log growth if either model is right and the worst / best outcome.

Re-price against fresh books any time with `--no-sim` (reuses the perturbed run).

| Script | What it does |
|---|---|
| `most_picked.py` | P(most picked) for one run and the market against it (`--temp`, `--shrink`, `--wallet`) |
| `bet_robustness.py` | Several models at once (`run@temp`): each alone, mixture and robust (max-min) Kelly |
| `recommend_orders.py` | The above from the calibration, one command |
| `simulate_perturbed.py` | A run with per-sim random champion strength (`--sigma-pick`), named `perturb_*` (no dashboard) |
| `calibrate_year.sh`, `backtest_progress.sh`, `backtest_cuts.py`, `make_asof.py --mid` | The backtest sweep |
| `calibrate_picks.py` | Scores backtest runs' champion totals; `--table` shows every year and cut |

### Next year (Worlds 2027)

1. After Worlds 2026 ends and its full CSV is synced, add it to the calibration:
   `scripts/calibrate_year.sh 2026` (uses `data/manual/worlds2026.yaml`; its rows join
   2025's and are averaged at each fraction).
2. Add `data/manual/worlds2027.yaml` (with `oracle_patch`). The newest config sets
   `pickems.sync_model.EVENT_YEAR`, which the live pipeline, its config, the patch fit's
   past Worlds and the Polymarket slug (`most-played-champion-at-worlds-2027`; override
   with `PICKEMS_POLYMARKET_SLUG`) follow.
3. Add 2027 to `WORLDS_FROM` in `pickems/history.py` (regional qualifiers filed under
   WLDs before that date are relabelled).
4. The rest of the live pipeline still has dated training windows (`2024-01-01` in
   `draft/model.py`, `stats_model.py`, `variance.py`, `history.worlds_share`): move them
   on with the season.

Caveats: one backtest year so far (2025), at 400 sims a run, so read a and sigma as a
direction and rough size; draftpredict's tuned parameters were fitted on data through
2026, a small leak into every backtest.

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
  regular patch but only 0.73 at Worlds (13.19: 0.72, 14.18: 0.56, 15.20: 0.91),
  and the average per-champion shift is about twice as large.
- At Worlds, a +1pp solo-queue win-rate change moves pro presence about 2.4x more
  than on a regular patch (0.32 vs 0.15 logit per pp).
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

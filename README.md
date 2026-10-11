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
numbers in `dashboard/out/report.json`. To view it with a local server, use the
`dashboard` entry in `.claude/launch.json`.

Refit the learned pieces when the data moves (new Oracle day, more 26.20 games):

```bash
.venv/bin/python scripts/train_draft.py --gbm --final   # draft model (~10 min); prints held-out accuracy
.venv/bin/python scripts/analysis/patch_shift.py         # patch -> pro presence response
.venv/bin/python -c "from pickems.history import load_games; from pickems.stats_model import fit, save; save(fit(load_games()))"
```

## draftpredict isolation

`pickems/sync_model.py` is the only thing that touches `~/draftpredict`, and it
only reads: code and adopted params are copied into `vendor/dp/` (working tree
by default, `--ref <commit>` to pin), CSVs are APFS copy-on-write clones. The
predictor is built and cached inside `vendor/dp/.cache`. Rerun the sync after
improving draftpredict and the next simulation uses the new model; the
dashboard header shows the commit and params version used.

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

"""Run N full Worlds simulations in parallel and write summaries.
CLI wrapper: scripts/simulate.py.

Inputs are prepared once in the parent (history, draft trackers, projected
26.20 presence, resolved teams); every worker loads draftpredict's cached
predictor plus the draft and stats models, then runs its share of sims.
Output: .cache/sims/<run>.pkl (list of per-tournament summaries + Index).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pickems.draft.model import load as load_draft_model, model_path as draft_model_path
from pickems.draft.trackers import DraftState, build_state, days
from pickems.history import ROLES, load_games
from pickems.meta import project
from pickems.sim.played import played_series
from pickems.sim.summary import Index, summarize
from pickems.stats_model import load as load_stats, model_path as stats_model_path
from pickems.sync_model import CACHE_DIR, DEST as DP_ROOT
from pickems.teams import latest_game, load_config, resolve_teams
from pickems.variance import form_sds
from pickems.worlds_buff import buff_scale, worlds_buffs

OUT = CACHE_DIR / "sims"
_W: dict = {}


# Exponent on the solo-queue multiplier. The draft model's response to its
# presence feature is less than proportional, so the multiplier is amplified
# until the simulated patch shift matches the projected one (scripts/calibrate_meta.py).
META_ALPHA = float(__import__("os").environ.get("PICKEMS_META_ALPHA", "2.6"))

# Which trained models to load: 0 = the default fits, w > 0 = the draft / stats
# models trained with Worlds games up-weighted (train_draft.py / train_stats.py
# --worlds-weight w) and draftpredict's win model at the same weight (vendor/dp
# reads it at import; see pickems.model). Read from the environment so spawned
# workers inherit it.
def worlds_weight() -> float:
	return float(__import__("os").environ.get("PICKEMS_WORLDS_WEIGHT", "0"))


def model_paths() -> tuple[Path, Path]:
	paths = draft_model_path(worlds_weight()), stats_model_path(worlds_weight())
	missing = [str(p) for p in paths if not p.exists()]
	if missing:
		raise FileNotFoundError(f"no model trained for worlds_weight={worlds_weight():g}: {missing} "
		                        "(run scripts/train_draft.py --gbm --final and scripts/train_stats.py with --worlds-weight)")
	return paths


def ensure_models() -> dict | None:
	"""Fail early on missing draft / stats models, and build a stale win model
	once here rather than in every worker. Returns the Worlds-retuned
	draftpredict params it uses, if any."""
	from pickems.model import get_predictor, tuned_params  # noqa: PLC0415

	model_paths()
	get_predictor(worlds_weight=worlds_weight())
	return tuned_params(worlds_weight())


def _clock(sec: float) -> str:
	m, s = divmod(int(round(sec)), 60)
	return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m{s:02d}s"


def progress_line(done: int, total: int, elapsed: float, unit: str = "sim") -> str:
	"""`  100/2000  1m35s  (0.95s/sim, ETA 30m05s)`: ETA from the average rate so far."""
	rate = elapsed / done
	return f"  {done}/{total}  {_clock(elapsed)}  ({rate:.2f}s/{unit}, ETA {_clock(rate * (total - done))})"


def prepare(cfg: dict) -> dict:
	games = load_games()
	state = build_state(games)
	teams = resolve_teams(games, cfg)
	proj = project(games, str(cfg["oracle_patch"]))
	t = days(games.date.max())
	# Apply the projection as a bounded multiplier on the trackers' own presence
	# (the draft model's training-time quantity), so champions nobody plays
	# can't be inflated by the projection's floor.
	from pickems.draft.features import Featurizer  # noqa: PLC0415

	# The Worlds games already played are replayed into the event meta
	# (MetaTracker, worth PRIOR_GAMES against this prior), so the prior's
	# trackers leave them out rather than count them twice.
	played = played_series(games, teams, cfg)
	replayed = {g.gameid for s in played for g in s.games}
	base = DraftState(state.champions)
	for g in games[~games.gameid.isin(replayed)].itertuples():
		base.update(g)
	tracked = np.exp(Featurizer(base).meta(t, None)[0])
	ratio = np.ones(state.n)
	for c, row in proj.iterrows():
		if c in state.cix:
			ratio[state.cix[c]] = row.soloq_mult ** META_ALPHA
	presence = tracked * ratio
	buffs = worlds_buffs(games, int(games.date.max().year))
	form = form_sds(games, teams)
	# Starters first, then the non-starters who have played (Team.subs, set by played_series), so a starter's index never moves.
	ix = Index(state.champions, [(code, r, name) for code, tm in teams.items() for r, name in zip(ROLES, tm.player_names)]
	           + [(code, r, name) for code, tm in teams.items() for r, name, _ in tm.subs], list(teams))
	if played:
		print("already played: " + ", ".join(s.score() for s in played))
	return {"cfg": cfg, "state": state, "teams": teams, "presence": presence, "t": t, "ix": ix, "projection": proj, "worlds_buffs": buffs, "form_sds": form,
	        "played": played, "latest_game": latest_game(games, teams), "data_through": games.date.max().isoformat()}


def elo_scale() -> dict:
	"""Factor per rating kind that puts draftpredict's ratings on the standard Elo scale. The win model is a logistic regression
	on the rating gaps (elo_diff for teams, player_elo_diff_<role> for players), so a gap of d points moves the log-odds by
	coef * d where standard Elo has d * ln(10) / 400; scaling ratings about the base by coef / (ln(10) / 400) makes them agree.
	The player gaps also pass through draftpredict's adaptive lane multipliers before the model sees them (compute_features), at
	the level of a Worlds game: cross-region, international; the Bo1 Swiss opener (game 1, not fearless) for contexts keyed on those."""
	from pickems.model import get_predictor  # noqa: PLC0415

	pred = get_predictor(worlds_weight=worlds_weight())
	coef, unit = dict(zip(pred.feature_cols, pred.model.coef_[0])), math.log(10) / 400
	mult = dict.fromkeys(ROLES, 1.0)
	lm = pred.lane_multiplier_state
	if lm is not None:
		from draftpredict.context_splits import context_level  # noqa: PLC0415 -- vendored module
		from draftpredict.lane_multiplier import MULT_COLS  # noqa: PLC0415

		level = "global" if lm.scope == "global" else context_level(lm.scope, league_blue=None, league_red=None, season=pred.current_season,
		                                                            is_fearless=False, game_number=1, is_bridge=True)
		m = dict(zip(MULT_COLS, lm.multipliers(level)))
		mult = {r: float(m.get(f"player_elo_diff_{r}", 1.0)) for r in ROLES}
	return {"team": float(coef["elo_diff"] / unit), "player": {r: float(coef[f"player_elo_diff_{r}"] * mult[r] / unit) for r in ROLES}}


def predictor_before_worlds(played: list):
	"""draftpredict's predictor as of the day of the first Worlds game already played (~5 min), i.e. without any of them. None when nothing
	has been played yet."""
	if not played:
		return None
	from pickems.model import dp, dp_cwd  # noqa: PLC0415

	day = min(g.date for s in played for g in s.games).normalize()
	print(f"building predictor as of {day.date()} ...", flush=True)
	with dp_cwd():
		return dp().build_predictor(as_of=day)


def elo_before_worlds(teams: dict, ix: Index, played: list, pre=None) -> dict | None:
	"""Team and player Elo before the first Worlds game already played, in ix
	order. The sims start from draftpredict's cached predictor, which has
	walked those games, so their elo start is "now"; this reads the predictor
	as of that first game's day (pre, built if not given) for the real change
	so far. None when nothing has been played yet (start is then pre-Worlds)."""
	pre = pre if pre is not None else predictor_before_worlds(played)
	if pre is None:
		return None
	from pickems.model import dp, get_predictor  # noqa: PLC0415
	from pickems.sim.tournament import blended_parts  # noqa: PLC0415

	predict, full = dp(), get_predictor(worlds_weight=worlds_weight())
	day = min(g.date for s in played for g in s.games).normalize()
	tid = {c: predict._resolve_team(full, teams[c].team_key) for c in ix.teams}
	team = [blended_parts(pre, pre.team_tracker.get(tid[c]), tid[c], pre.team_weight) for c in ix.teams]
	player = [blended_parts(pre, pre.player_tracker.get(predict._resolve_player(full, teams[c].key_of(r, n))), tid[c], pre.player_league_blend_weight)
	          for c, r, n in ix.players]
	return {"as_of": day.isoformat(), "team": np.array([sum(p) for p in team], np.float32), "player": np.array([sum(p) for p in player], np.float32),
	        "team_raw": np.array([p[0] for p in team], np.float32), "player_raw": np.array([p[0] for p in player], np.float32)}


def score_played(payload: dict, pre) -> list[dict]:
	"""The win model's pre-game probability for every real Worlds game: the bracket is replayed once with the walk starting from `pre`
	(the predictor before any of those games), each real game priced on its real lineups and then fed to the walk, so a game is
	predicted from the ratings the games before it left (the way the sims predict the ones still to come). Worlds buff and series
	reversal apply as in the sims; the per-team form draw doesn't (it would only add noise to a price)."""
	from pickems.model import dp  # noqa: PLC0415
	from pickems.sim.draft import DraftSimulator  # noqa: PLC0415
	from pickems.sim.tournament import Tournament, WinModel  # noqa: PLC0415

	cfg = payload["cfg"]
	rng = np.random.default_rng(0)
	draft_path, stats_path = model_paths()
	draft_model = load_draft_model(draft_path)
	win = WinModel(dp(), pre, float(cfg["oracle_patch"]), cfg["league"], rng, payload["worlds_buffs"], None, payload["teams"])
	drafter = DraftSimulator(draft_model, payload["state"], payload["t"], rng)
	tour = Tournament(cfg, payload["teams"], drafter, win, load_stats(stats_path), rng, payload["presence"], played=payload["played"], score_played=True)
	tour.run()
	return tour.scored


def _init(payload_path: str, seed: int, perturb=None) -> None:
	from pickems.model import dp, get_predictor  # noqa: PLC0415 -- per-process draftpredict import

	with open(payload_path, "rb") as f:
		_W.update(pickle.load(f))
	_W["perturb"] = perturb
	_W["dp"], _W["pred"] = dp(), get_predictor(worlds_weight=worlds_weight())
	draft_path, stats_path = model_paths()
	_W["draft_model"] = load_draft_model(draft_path)
	_W["draft_model"].top_k = 150
	_W["stats"] = load_stats(stats_path)


def _run(seed: int) -> dict:
	from pickems.sim.draft import DraftSimulator  # noqa: PLC0415
	from pickems.sim.tournament import Tournament, WinModel  # noqa: PLC0415

	rng = np.random.default_rng(seed)
	cfg = _W["cfg"]
	model, presence = _W["draft_model"], _W["presence"]
	if _W.get("perturb") is not None:  # pickems.sim.perturb: this sim's draw of champion strength
		from pickems.sim.perturb import PerturbedModel  # noqa: PLC0415

		offset, mult = _W["perturb"].draw(seed, len(presence))
		model, presence = PerturbedModel(model, offset), presence * mult
	drafter = DraftSimulator(model, _W["state"], _W["t"], rng)
	win = WinModel(_W["dp"], _W["pred"], float(cfg["oracle_patch"]), cfg["league"], rng, _W["worlds_buffs"], _W["form_sds"], _W["teams"])
	tour = Tournament(cfg, _W["teams"], drafter, win, _W["stats"], rng, presence, played=_W["played"]).run()
	return summarize(tour, _W["ix"])


def run(sims: int, workers: int, seed: int = 0, name: str | None = None, perturb=None) -> Path:
	"""perturb: a pickems.sim.perturb.Perturb to draw champion strength per sim (None: the plain sims)."""
	cfg = load_config()
	tuned_params = ensure_models()
	t0 = time.time()
	payload = prepare(cfg)
	OUT.mkdir(parents=True, exist_ok=True)
	# a perturbed run keeps its own payload, so it can't clobber a plain run's mid-start
	payload_path = OUT / ("payload.pkl" if perturb is None else f"payload_{name}.pkl")
	with open(payload_path, "wb") as f:
		pickle.dump({k: v for k, v in payload.items() if k not in ("projection", "latest_game", "data_through")}, f)
	print(f"prepared in {time.time() - t0:.0f}s; running {sims} sims on {workers} workers")
	seeds = [seed * 1_000_003 + i for i in range(sims)]
	results, real = [], None
	t1 = time.time()
	ctx = mp.get_context("spawn")
	with ctx.Pool(workers, initializer=_init, initargs=(str(payload_path), seed, perturb)) as pool:
		for i, r in enumerate(pool.imap_unordered(_run, seeds, chunksize=4)):
			real = r.pop("real", None) or real
			results.append(r)
			if (i + 1) % max(sims // 20, 1) == 0:
				print(progress_line(i + 1, sims, time.time() - t1), flush=True)
	# the pre-Worlds predictor (~8 min) only feeds the dashboard's Elo and played-game odds,
	# which perturbed runs don't have
	pre = predictor_before_worlds(payload["played"]) if perturb is None else None
	elo_pre = elo_before_worlds(payload["teams"], payload["ix"], payload["played"], pre) if perturb is None else None
	scored = score_played(payload, pre) if pre is not None else None
	manifest = json.loads((DP_ROOT / "MANIFEST.json").read_text())
	name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
	out = {"ix": payload["ix"], "results": results, "projection": payload["projection"], "manifest": manifest,
	       "run_at": datetime.now(timezone.utc).isoformat(), "sims": sims, "cfg": cfg, "worlds_weight": worlds_weight(), "worlds_buffs": payload["worlds_buffs"], "worlds_buff_scale": buff_scale(), "form_sds": payload["form_sds"], "tuned_params": tuned_params,
	       "latest_game": payload["latest_game"], "data_through": payload["data_through"], "played": [s.score() for s in payload["played"]], "real": real, "elo_pre": elo_pre, "scored": scored, "elo_scale": elo_scale(),
	       "teams": {k: {"name": v.name, "region": v.region, "seed": v.seed, "players": v.player_names} for k, v in payload["teams"].items()}}
	if perturb is not None:
		out["perturb"] = vars(perturb)
		payload_path.unlink()
	with open(OUT / f"{name}.pkl", "wb") as f:
		pickle.dump(out, f)
	print(f"done: {OUT / (name + '.pkl')}")
	return OUT / f"{name}.pkl"

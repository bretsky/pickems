"""Run N full Worlds simulations in parallel and write summaries.
CLI wrapper: scripts/simulate.py.

Inputs are prepared once in the parent (history, draft trackers, projected
26.20 presence, resolved teams); every worker loads draftpredict's cached
predictor plus the draft and stats models, then runs its share of sims.
Output: .cache/sims/<run>.pkl (list of per-tournament summaries + Index).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pickems.draft.model import load as load_draft_model, model_path as draft_model_path
from pickems.draft.trackers import build_state, days
from pickems.history import ROLES, load_games
from pickems.meta import project
from pickems.sim.summary import Index, summarize
from pickems.stats_model import load as load_stats, model_path as stats_model_path
from pickems.sync_model import DEST as DP_ROOT, ROOT
from pickems.teams import latest_game, load_config, resolve_teams
from pickems.variance import form_sds
from pickems.worlds_buff import worlds_buffs

OUT = ROOT / ".cache" / "sims"
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

	tracked = np.exp(Featurizer(state).meta(t, None)[0])
	ratio = np.ones(state.n)
	for c, row in proj.iterrows():
		if c in state.cix:
			ratio[state.cix[c]] = row.soloq_mult ** META_ALPHA
	presence = tracked * ratio
	buffs = worlds_buffs(games, int(games.date.max().year))
	form = form_sds(games, teams)
	ix = Index(state.champions, [(code, r, name) for code, tm in teams.items() for r, name in zip(ROLES, tm.player_names)], list(teams))
	return {"cfg": cfg, "state": state, "teams": teams, "presence": presence, "t": t, "ix": ix, "projection": proj, "worlds_buffs": buffs, "form_sds": form,
	        "latest_game": latest_game(games, teams), "data_through": games.date.max().isoformat()}


def _init(payload_path: str, seed: int) -> None:
	from pickems.model import dp, get_predictor  # noqa: PLC0415 -- per-process draftpredict import

	with open(payload_path, "rb") as f:
		_W.update(pickle.load(f))
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
	drafter = DraftSimulator(_W["draft_model"], _W["state"], _W["t"], rng)
	win = WinModel(_W["dp"], _W["pred"], float(cfg["oracle_patch"]), cfg["league"], rng, _W["worlds_buffs"], _W["form_sds"], _W["teams"])
	tour = Tournament(cfg, _W["teams"], drafter, win, _W["stats"], rng, _W["presence"]).run()
	return summarize(tour, _W["ix"])


def run(sims: int, workers: int, seed: int = 0, name: str | None = None) -> Path:
	cfg = load_config()
	tuned_params = ensure_models()
	t0 = time.time()
	payload = prepare(cfg)
	OUT.mkdir(parents=True, exist_ok=True)
	payload_path = OUT / "payload.pkl"
	with open(payload_path, "wb") as f:
		pickle.dump({k: v for k, v in payload.items() if k not in ("projection", "latest_game", "data_through")}, f)
	print(f"prepared in {time.time() - t0:.0f}s; running {sims} sims on {workers} workers")
	seeds = [seed * 1_000_003 + i for i in range(sims)]
	results = []
	t1 = time.time()
	ctx = mp.get_context("spawn")
	with ctx.Pool(workers, initializer=_init, initargs=(str(payload_path), seed)) as pool:
		for i, r in enumerate(pool.imap_unordered(_run, seeds, chunksize=4)):
			results.append(r)
			if (i + 1) % max(sims // 20, 1) == 0:
				print(progress_line(i + 1, sims, time.time() - t1), flush=True)
	manifest = json.loads((DP_ROOT / "MANIFEST.json").read_text())
	name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
	out = {"ix": payload["ix"], "results": results, "projection": payload["projection"], "manifest": manifest,
	       "run_at": datetime.now(timezone.utc).isoformat(), "sims": sims, "cfg": cfg, "worlds_weight": worlds_weight(), "worlds_buffs": payload["worlds_buffs"], "form_sds": payload["form_sds"], "tuned_params": tuned_params,
	       "latest_game": payload["latest_game"], "data_through": payload["data_through"],
	       "teams": {k: {"name": v.name, "region": v.region, "seed": v.seed, "players": v.player_names} for k, v in payload["teams"].items()}}
	with open(OUT / f"{name}.pkl", "wb") as f:
		pickle.dump(out, f)
	print(f"done: {OUT / (name + '.pkl')}")
	return OUT / f"{name}.pkl"

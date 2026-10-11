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

from pickems.draft.model import load as load_draft_model
from pickems.draft.trackers import build_state, days
from pickems.history import ROLES, load_games
from pickems.meta import project
from pickems.sim.summary import Index, summarize
from pickems.stats_model import load as load_stats
from pickems.sync_model import DEST as DP_ROOT, ROOT
from pickems.teams import load_config, resolve_teams

OUT = ROOT / ".cache" / "sims"
_W: dict = {}


def prepare(cfg: dict) -> dict:
	games = load_games()
	state = build_state(games)
	teams = resolve_teams(games, cfg)
	proj = project(games, str(cfg["oracle_patch"]))
	presence = np.full(state.n, 0.005)
	for c, v in proj.projected.items():
		if c in state.cix:
			presence[state.cix[c]] = v
	t = days(games.date.max())
	ix = Index(state.champions, [(code, r, name) for code, tm in teams.items() for r, name in zip(ROLES, tm.player_names)], list(teams))
	return {"cfg": cfg, "state": state, "teams": teams, "presence": presence, "t": t, "ix": ix, "projection": proj}


def _init(payload_path: str, seed: int) -> None:
	from pickems.model import dp, get_predictor  # noqa: PLC0415 -- per-process draftpredict import

	with open(payload_path, "rb") as f:
		_W.update(pickle.load(f))
	_W["dp"], _W["pred"] = dp(), get_predictor()
	_W["draft_model"] = load_draft_model()
	_W["draft_model"].top_k = 150
	_W["stats"] = load_stats()


def _run(seed: int) -> dict:
	from pickems.sim.draft import DraftSimulator  # noqa: PLC0415
	from pickems.sim.tournament import Tournament, WinModel  # noqa: PLC0415

	rng = np.random.default_rng(seed)
	cfg = _W["cfg"]
	drafter = DraftSimulator(_W["draft_model"], _W["state"], _W["t"], rng)
	win = WinModel(_W["dp"], _W["pred"], float(cfg["oracle_patch"]), cfg["league"])
	tour = Tournament(cfg, _W["teams"], drafter, win, _W["stats"], rng, _W["presence"]).run()
	return summarize(tour, _W["ix"])


def run(sims: int, workers: int, seed: int = 0, name: str | None = None) -> Path:
	cfg = load_config()
	t0 = time.time()
	payload = prepare(cfg)
	OUT.mkdir(parents=True, exist_ok=True)
	payload_path = OUT / "payload.pkl"
	with open(payload_path, "wb") as f:
		pickle.dump({k: v for k, v in payload.items() if k != "projection"}, f)
	print(f"prepared in {time.time() - t0:.0f}s; running {sims} sims on {workers} workers")
	seeds = [seed * 1_000_003 + i for i in range(sims)]
	results = []
	t1 = time.time()
	ctx = mp.get_context("spawn")
	with ctx.Pool(workers, initializer=_init, initargs=(str(payload_path), seed)) as pool:
		for i, r in enumerate(pool.imap_unordered(_run, seeds, chunksize=4)):
			results.append(r)
			if (i + 1) % max(sims // 20, 1) == 0:
				el = time.time() - t1
				print(f"  {i + 1}/{sims}  {el:.0f}s  ({el / (i + 1):.2f}s/sim)", flush=True)
	manifest = json.loads((DP_ROOT / "MANIFEST.json").read_text())
	name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
	out = {"ix": payload["ix"], "results": results, "projection": payload["projection"], "manifest": manifest,
	       "run_at": datetime.now(timezone.utc).isoformat(), "sims": sims, "cfg": cfg,
	       "teams": {k: {"name": v.name, "region": v.region, "seed": v.seed, "players": v.player_names} for k, v in payload["teams"].items()}}
	with open(OUT / f"{name}.pkl", "wb") as f:
		pickle.dump(out, f)
	print(f"done: {OUT / (name + '.pkl')}")
	return OUT / f"{name}.pkl"

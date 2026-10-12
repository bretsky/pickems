"""Head-to-head series odds: every pair of a team list plays N fresh Bo5
(fearless) series through the same draft / win / stats machinery as the full
tournament. Used for knockout picks once (or before) the bracket is drawn.

Output: .cache/sims/h2h_<name>.pkl with {"teams", "best_of", "series",
"p": {a: {b: P(a wins series vs b)}}, "game_p": {...}}.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import pickle
import time
from datetime import datetime, timezone

import numpy as np

from pickems.sim import runner
from pickems.teams import load_config

H2H_DIR = runner.OUT


def _series(task: tuple[str, str, int, int]) -> tuple[str, str, int, int]:
	from pickems.sim.draft import DraftSimulator  # noqa: PLC0415
	from pickems.sim.tournament import Schedule, Tournament, WinModel  # noqa: PLC0415

	a, b, best_of, seed = task
	W = runner._W
	rng = np.random.default_rng(seed)
	cfg = W["cfg"]
	drafter = DraftSimulator(W["draft_model"], W["state"], W["t"], rng)
	win = WinModel(W["dp"], W["pred"], float(cfg["oracle_patch"]), cfg["league"], rng, W["worlds_buffs"])
	tour = Tournament(cfg, W["teams"], drafter, win, W["stats"], rng, W["presence"])
	s = tour.play_series("knockout", "H2H", W["teams"][a], W["teams"][b], best_of, True, None, day=Schedule(cfg["schedule"]).first("knockout"))
	return a, b, s.wins_a, s.wins_b


def run(teams: list[str], series: int = 400, best_of: int = 5, workers: int = 8, seed: int = 7, name: str = "knockout") -> dict:
	cfg = load_config()
	runner.ensure_models()
	payload = runner.prepare(cfg)
	path = runner.OUT / "payload.pkl"
	with open(path, "wb") as f:
		pickle.dump({k: v for k, v in payload.items() if k != "projection"}, f)
	pairs = list(itertools.combinations(teams, 2))
	# Alternate who is listed first so neither side's position matters.
	tasks = [((a, b) if i % 2 == 0 else (b, a)) + (best_of, seed * 10_000_019 + j * 1009 + i) for j, (a, b) in enumerate(pairs) for i in range(series)]
	wins = {(a, b): 0 for a in teams for b in teams if a != b}
	games = {(a, b): 0 for a in teams for b in teams if a != b}
	t0 = time.time()
	print(f"{len(pairs)} pairs x {series} Bo{best_of} series on {workers} workers")
	with mp.get_context("spawn").Pool(workers, initializer=runner._init, initargs=(str(path), seed)) as pool:
		for i, (a, b, wa, wb) in enumerate(pool.imap_unordered(_series, tasks, chunksize=8)):
			winner, loser = (a, b) if wa > wb else (b, a)
			wins[(winner, loser)] += 1
			games[(a, b)] += wa
			games[(b, a)] += wb
			if (i + 1) % max(len(tasks) // 10, 1) == 0:
				print(runner.progress_line(i + 1, len(tasks), time.time() - t0, "series"), flush=True)
	p = {a: {b: wins[(a, b)] / series for b in teams if b != a} for a in teams}
	game_p = {a: {b: games[(a, b)] / max(games[(a, b)] + games[(b, a)], 1) for b in teams if b != a} for a in teams}
	out = {"teams": teams, "best_of": best_of, "series": series, "p": p, "game_p": game_p,
	       "run_at": datetime.now(timezone.utc).isoformat(), "model": (runner.DP_ROOT / "MANIFEST.json").read_text()}
	with open(H2H_DIR / f"h2h_{name}.pkl", "wb") as f:
		pickle.dump(out, f)
	print(f"done: h2h_{name}.pkl in {time.time() - t0:.0f}s")
	return out

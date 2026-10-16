"""Add pre-Worlds Elo (runner.elo_before_worlds) to a run made before it was recorded.

  python scripts/backfill_elo_pre.py run_2000_ww09_...     # rewrites .cache/sims/<run>.pkl; then scripts/render.py

Uses the run's own Worlds weight. Only the first played game's day matters, so
history synced since the run is fine as long as that day is the same.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="+", help="run names or .pkl paths")
	args = ap.parse_args()
	from pickems.sim.runner import OUT  # noqa: PLC0415

	paths = [Path(r) if Path(r).exists() else OUT / (r.removesuffix(".pkl") + ".pkl") for r in args.run]
	runs = {p: pickle.loads(p.read_bytes()) for p in paths}
	weights = {r.get("worlds_weight", 0.0) for r in runs.values()}
	if len(weights) > 1:
		raise SystemExit(f"runs mix Worlds weights {sorted(weights)}; backfill them one weight at a time")
	os.environ["PICKEMS_WORLDS_WEIGHT"] = str(weights.pop())  # before anything imports draftpredict

	from pickems.history import load_games  # noqa: PLC0415
	from pickems.sim.played import played_series  # noqa: PLC0415
	from pickems.sim.runner import elo_before_worlds  # noqa: PLC0415
	from pickems.teams import resolve_teams  # noqa: PLC0415

	games, cache = load_games(), {}
	for p, run in runs.items():
		if not run.get("played"):
			print(f"{p.name}: nothing played, pre-Worlds = start; skipped")
			continue
		teams = resolve_teams(games, run["cfg"])
		played = played_series(games, teams, run["cfg"])
		day = min(g.date for s in played for g in s.games).normalize()
		key = (day, tuple(run["ix"].teams), tuple(run["ix"].players))
		if key not in cache:
			cache[key] = elo_before_worlds(teams, run["ix"], played)
		run["elo_pre"] = cache[key]
		p.write_bytes(pickle.dumps(run))
		print(f"{p.name}: pre-Worlds Elo as of {day.date()}")


if __name__ == "__main__":
	main()

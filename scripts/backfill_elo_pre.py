"""Add pre-Worlds Elo (runner.elo_before_worlds) and the standard-Elo scale (runner.elo_scale) to a run made before they were recorded.

  python scripts/backfill_elo_pre.py run_2000_ww09_...     # rewrites .cache/sims/<run>.pkl; then scripts/render.py

Uses the run's own Worlds weight. Runs from before ratings were region-blended need --own. Only the first played game's day matters, so
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
	ap.add_argument("--own", action="store_true", help="the runs were made before ratings were region-blended (fc07dee): their start / end are the team's / player's own rating, so store pre the same way")
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
	from pickems.sim.runner import elo_scale  # noqa: PLC0415

	scale = elo_scale()
	for p, run in runs.items():
		run["elo_scale"] = scale
		if not run.get("played"):
			p.write_bytes(pickle.dumps(run))
			print(f"{p.name}: elo scale added; nothing played, pre-Worlds = start")
			continue
		teams = resolve_teams(games, run["cfg"])
		played = played_series(games, teams, run["cfg"])
		day = min(g.date for s in played for g in s.games).normalize()
		key = (day, tuple(run["ix"].teams), tuple(run["ix"].players))
		if key not in cache:
			cache[key] = elo_before_worlds(teams, run["ix"], played)
		pre = cache[key]
		run["elo_pre"] = {**pre, "team": pre["team_raw"], "player": pre["player_raw"]} if args.own else pre
		p.write_bytes(pickle.dumps(run))
		print(f"{p.name}: pre-Worlds Elo as of {day.date()}")


if __name__ == "__main__":
	main()

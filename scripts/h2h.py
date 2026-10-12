"""Simulate head-to-head Bo5 series for the knockout teams.

Uses knockouts.draw from data/manual/worlds2026.yaml when it is filled in
(the real quarterfinal draw); otherwise the 8 teams the latest simulation run
predicts out of Swiss (pickems.knockout.most_likely_records). Writes
.cache/sims/h2h_<run>.pkl, which the dashboard of that run reads; the Worlds
buff follows PICKEMS_WORLDS_BUFF, so match it to the run's.

  .venv/bin/python -W ignore scripts/h2h.py [--run run_2000_v18] [--series 400]
"""

import argparse
import multiprocessing as mp
import pickle

from pickems.knockout import most_likely_records
from pickems.report import team_outcomes
from pickems.sim import h2h, runner
from pickems.teams import load_config


def knockout_teams(run_name: str) -> list[str]:
	draw = (load_config().get("knockouts") or {}).get("draw")
	if draw:
		return [t for pair in draw for t in pair]
	run = pickle.load(open(runner.OUT / f"{run_name}.pkl", "rb"))
	rec = most_likely_records(team_outcomes(run["results"], run["ix"].teams))
	return [t for t, r in rec.items() if r.startswith("3-")]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--run", default="run_2000_v18")
	ap.add_argument("--series", type=int, default=400)
	ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
	args = ap.parse_args()
	teams = knockout_teams(args.run)
	print("teams:", teams)
	h2h.run(teams, args.series, workers=args.workers, name=args.run)


if __name__ == "__main__":
	main()

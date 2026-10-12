"""Pick SERIES_REVERSAL (pickems.sim.tournament): in a series tied 1-1 or 2-2,
how often the last game's winner takes the next one, simulated vs real
(major leagues + internationals since 2023). Also prints Bo5 / Bo3 score
splits next to real Worlds 2018-25 (knockout 34/38/29, play-in 38/38/25, Swiss
Bo3 56/44 from Oracle's Elixir, Worlds dates only: see the README).

  .venv/bin/python -W ignore scripts/calibrate_reversal.py 0 0.1 0.2 0.3 --sims 300

Each value runs its own simulation (.cache/sims/calib_rev_<value>.pkl) with the
PICKEMS_WORLDS_WEIGHT in the environment; the dashboard is not rendered.
"""

import argparse
import os
import pickle
from collections import Counter

import numpy as np

from pickems.history import MAJOR_LEAGUES, load_games
from pickems.sim import runner


def tied_next(seqs) -> tuple[float, int]:
	"""Share of tied-series games (1-1, 2-2) won by the last game's winner.
	seqs: per series, whether team A won each game, in order."""
	hits = [s[i + 1] == s[i] for s in seqs for i in range(1, len(s) - 1) if 2 * sum(s[: i + 1]) == i + 1]
	return float(np.mean(hits)), len(hits)


def real_seqs() -> list[list[bool]]:
	g = load_games()
	g = g[g.league.isin(MAJOR_LEAGUES)].sort_values("date", kind="stable")
	out = []
	for _, s in g.groupby("series_id"):
		a = s.blue_team.iloc[0]
		out.append([(b == a) == bool(w) for b, w in zip(s.blue_team, s.blue_win)])
	return out


def sim_stats(path) -> tuple[list[list[bool]], Counter]:
	run = pickle.load(open(path, "rb"))
	seqs, scores = [], Counter()
	for r in run["results"]:
		for (stage, _, a, _, wa, wb), games in zip(r["series"], r["series_games"]):
			seqs.append([(g["blue"] if g["blue_win"] else g["red"]) == a for g in games])
			scores[(stage, f"{max(wa, wb)}-{min(wa, wb)}")] += 1
	return seqs, scores


def split(scores: Counter, stage: str, need: int) -> str:
	c = {k[1]: v for k, v in scores.items() if k[0] == stage and k[1].startswith(str(need))}
	n = sum(c.values())
	return "/".join(f"{c.get(f'{need}-{lo}', 0) / n:.0%}" for lo in range(need)) if n else "-"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("values", nargs="*", default=["0", "0.1", "0.2", "0.3"])
	ap.add_argument("--sims", type=int, default=300)
	ap.add_argument("--workers", type=int, default=8)
	args = ap.parse_args()
	rate, n = tied_next(real_seqs())
	print(f"real: last-game winner takes the tied game {rate:.3f} (n={n}, se {np.sqrt(rate * (1 - rate) / n):.3f}); "
	      "Worlds 2018-25 knockout Bo5 3-0/3-1/3-2 34%/38%/29%, play-in Bo5 38%/38%/25%, Swiss Bo3 2-0/2-1 56%/44%", flush=True)
	for v in args.values:
		os.environ["PICKEMS_SERIES_REVERSAL"] = v
		path = runner.run(args.sims, args.workers, seed=0, name=f"calib_rev_{v}")
		seqs, scores = sim_stats(path)
		rate, n = tied_next(seqs)
		print(f"reversal {v}: tied game {rate:.3f} (n={n}, se {np.sqrt(rate * (1 - rate) / n):.3f}); "
		      f"knockout Bo5 {split(scores, 'knockout', 3)}, play-in Bo5 {split(scores, 'play_in', 3)}, Swiss Bo3 {split(scores, 'swiss', 2)}", flush=True)


if __name__ == "__main__":
	main()

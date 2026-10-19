"""P(champion is the most picked over the whole tournament), per champion.

Not the same as highest mean pick count: a champion with a fat-tailed pick total can
lead more often than one with a higher average. Counted per simulation (ties split),
over all stages, picks only (bans excluded; use compare_runs Q1 for presence).
Ties on picks go to the higher win rate, then to more bans; only a tie on all three
is split evenly.

  .venv/bin/python -W ignore scripts/most_picked.py [run name] [--top 20]
"""

import argparse
import pickle

import numpy as np

from pickems.report import _stage_sum
from pickems.sim.runner import OUT
from pickems.sim.summary import CHAMP_STATS


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="?", help="run name (default: newest run_*.pkl)")
	ap.add_argument("--top", type=int, default=20)
	args = ap.parse_args()
	if args.run is None:
		path = max(OUT.glob("run_*.pkl"), key=lambda p: p.stat().st_mtime)
	else:
		path = OUT / (args.run.removesuffix(".pkl") + ".pkl")
	run = pickle.load(open(path, "rb"))
	res, champs = run["results"], run["ix"].champions

	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	stats = np.stack([_stage_sum(r, "all", "champ") for r in res]).astype(float)  # (sims, stat, champ)
	picks, bans, wins = stats[:, C["picks"]], stats[:, C["bans"]], stats[:, C["wins"]]
	with np.errstate(divide="ignore", invalid="ignore"):
		winrate = np.nan_to_num(wins / picks, nan=-1.0)

	# narrow the field key by key: picks, then win rate, then bans
	alive = np.ones_like(picks, bool)
	tied_on_picks = None
	for key in (picks, winrate, bans):
		v = np.where(alive, key, -np.inf)
		alive = v == v.max(axis=1, keepdims=True)
		if tied_on_picks is None:
			tied_on_picks = (alive.sum(axis=1) > 1).mean()
	still_tied = (alive.sum(axis=1) > 1).mean()
	p_top = (alive / alive.sum(axis=1, keepdims=True)).mean(axis=0)
	mean = picks.mean(axis=0)
	print(f"{path.name}: {len(res)} sims, P(tie on picks) = {tied_on_picks:.1%}, still tied after win rate and bans = {still_tied:.2%}")
	print(f"{'champion':<16}{'P(most picked)':>15}{'mean picks':>12}{'mean rank':>11}")
	rank = (-picks).argsort(axis=1).argsort(axis=1).mean(axis=0) + 1
	for i in np.argsort(-p_top)[: args.top]:
		if p_top[i] == 0:
			break
		print(f"{champs[i]:<16}{p_top[i]:>15.1%}{mean[i]:>12.1f}{rank[i]:>11.1f}")
	print(f"{'(sum)':<16}{p_top.sum():>15.1%}")


if __name__ == "__main__":
	main()

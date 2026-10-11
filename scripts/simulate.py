"""Simulate the full Worlds N times and build the dashboard.

  python -m pickems.sync_model                 # pull latest draftpredict model + data
  python scripts/simulate.py --sims 2000       # writes dashboard/out/index.html
"""

import argparse
import multiprocessing as mp
import os

from pickems.dashboard import render
from pickems.sim.runner import run


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--sims", type=int, default=200)
	ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--name", default=None)
	ap.add_argument("--worlds-weight", type=float, default=None, help="use models trained with this --worlds-weight (default 0)")
	args = ap.parse_args()
	if args.worlds_weight is not None:
		os.environ["PICKEMS_WORLDS_WEIGHT"] = str(args.worlds_weight)
	path = run(args.sims, args.workers, args.seed, args.name)
	print(f"dashboard: {render(path)}")


if __name__ == "__main__":
	main()

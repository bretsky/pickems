"""Simulate the full Worlds N times and build the dashboard.

  python -m pickems.sync_model                 # pull latest draftpredict model + data
  python scripts/simulate.py --sims 2000       # then open dashboard/out/index.html
"""

import argparse
import multiprocessing as mp

from pickems.sim.runner import run


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--sims", type=int, default=200)
	ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--name", default=None)
	args = ap.parse_args()
	run(args.sims, args.workers, args.seed, args.name)


if __name__ == "__main__":
	main()

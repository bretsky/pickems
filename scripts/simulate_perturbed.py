"""Simulate Worlds with parameter uncertainty on champion strength (pickems.sim.perturb).

Same sims as scripts/simulate.py, except each one first draws a per-champion logit
offset on the draft model's pick/ban scores and a multiplier on the projected meta.
Writes .cache/sims/perturb_<...>.pkl (no dashboard): the name never starts with
run_, so render, deploy, score_played and most_picked's default never pick it up.

  .venv/bin/python -W ignore scripts/simulate_perturbed.py --sims 2000 --worlds-weight 0.9 --sigma-pick 0.3
  PICKEMS_WORLDS_BUFF=0 ... --name perturb_..._nobuff        # buff off, as the pipeline's twin
  PICKEMS_ASOF=2025 ... --sims 500 --sigma-pick 0.3          # backtest; score with scripts/calibrate_picks.py

Then: scripts/most_picked.py perturb_<...>
"""

import argparse
import multiprocessing as mp
import os
from datetime import datetime

from pickems.sim.perturb import Perturb
from pickems.sim.runner import run


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--sims", type=int, default=200)
	ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--sigma-pick", type=float, default=0.3, help="sd of the per-champion logit offset on pick/ban scores")
	ap.add_argument("--sigma-presence", type=float, default=0.0, help="sd of the log multiplier on projected presence")
	ap.add_argument("--name", default=None, help="must start with perturb_ (default perturb_<sims>_p<sigma>_m<sigma>_<time>)")
	ap.add_argument("--worlds-weight", type=float, default=None, help="use models trained with this --worlds-weight (default 0)")
	args = ap.parse_args()
	if args.worlds_weight is not None:
		os.environ["PICKEMS_WORLDS_WEIGHT"] = str(args.worlds_weight)
	name = args.name or f"perturb_{args.sims}_p{args.sigma_pick:g}_m{args.sigma_presence:g}_{datetime.now():%Y%m%d-%H%M}"
	if not name.startswith("perturb_"):
		ap.error("--name must start with perturb_, so plain-run tooling never mistakes it for a run_")

	run(args.sims, args.workers, args.seed, name, Perturb(args.sigma_pick, args.sigma_presence))


if __name__ == "__main__":
	main()

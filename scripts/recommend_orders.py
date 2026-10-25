"""Recommended orders for the Polymarket "Most Played Champion at Worlds 2026" market,
with the model uncertainty the backtests measured.

  .venv/bin/python -W ignore scripts/recommend_orders.py [--run run_...] [--sims 1000] [--no-sim]
      [--bankroll 1000] [--kelly 0.25] [--wallet 0x... | --positions held.yaml]

1. Takes the plain run (newest run_*, buff on, or --run) and the stage of the event
   its data reaches (the last round in its schedule on or before its last game).
2. Looks up the past-Worlds backtest cut at the same stage in .cache/calibration.jsonl
   (scripts/backtest_progress.sh), else the latest earlier one, for
     - a: the plain run's fitted logit scale there, used as --temp on this run, and
     - sigma: the perturbation where the perturbed runs' fitted a crosses 1
       (linear between the sigmas tried, rounded to 0.1).
3. Reuses or simulates a perturbed run at that sigma on the same data (--sims, about
   1.5 s a sim on this machine plus ~6 min to finish; --no-sim skips it).
4. Sizes the orders with scripts/bet_robustness.py over those two models, the plain
   run tempered to a and the perturbed run. The robust sizes, which grow the bankroll
   under both, are the recommendation: the "robust sh" column of the order tables.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bet_robustness  # noqa: E402

from pickems import most_picked as mp  # noqa: E402
from pickems.sim.runner import OUT  # noqa: E402
from pickems.sync_model import ROOT  # noqa: E402
from pickems.teams import load_config  # noqa: E402

SUMMARY = ROOT / ".cache" / "calibration.jsonl"
STAGES = ["pre", "play_in", "R1", "R2", "R3", "R4", "R5", "QF", "SF", "Final"]


def stage(cfg, through):
	"""The last round of the schedule on or before day `through` (YYYY-MM-DD)."""
	done = [e for e in cfg["schedule"] if str(e["date"]) <= through]
	if not done:
		return "pre"
	e = done[-1]
	return "play_in" if e["stage"] == "play_in" else e["round"].split()[0]


def calibration(at):
	"""(backtest key, stage, a for the plain run, sigma where a crosses 1, rows) from the cut nearest `at`."""
	rows = {(r["asof"], r["run"]): r for r in map(json.loads, open(SUMMARY))}.values()
	by_key = {}
	for r in rows:
		cfg = load_config(ROOT / "data" / "manual" / f"worlds{r['asof'][:4]}.yaml")
		by_key.setdefault(r["asof"], (stage(cfg, r["data_through"]), []))[1].append(r)
	want = STAGES.index(at)
	usable = [(STAGES.index(s), k) for k, (s, rs) in by_key.items() if STAGES.index(s) <= want and any(r["sigma_pick"] == 0 for r in rs)]
	if not usable:
		raise SystemExit(f"no backtest cut at or before {at} in {SUMMARY}: run scripts/backtest_progress.sh")
	_, key = max(usable)
	st, rs = by_key[key]
	rs = sorted(rs, key=lambda r: r["sigma_pick"])
	a_plain = next(r["a"] for r in rs if r["sigma_pick"] == 0)
	s, a = np.array([r["sigma_pick"] for r in rs]), np.array([r["a"] for r in rs])
	if a.max() < 1:
		sigma = s.max()  # never reaches calibrated: the largest tried
	elif a[0] >= 1:
		sigma = 0.0
	else:
		i = int(np.argmax(a >= 1))
		sigma = s[i - 1] + (1 - a[i - 1]) * (s[i] - s[i - 1]) / (a[i] - a[i - 1])
	return key, st, a_plain, round(float(sigma), 1), rs


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--run", help="plain run (default: newest run_*.pkl, buff on)")
	ap.add_argument("--sims", type=int, default=1000, help="sims for the perturbed run if it has to be made")
	ap.add_argument("--no-sim", action="store_true", help="don't simulate: use the tempered plain run alone if no perturbed run exists")
	ap.add_argument("--bankroll", type=float, default=1000.0)
	ap.add_argument("--kelly", type=float, default=0.25)
	ap.add_argument("--min-edge", type=float, default=0.02)
	ap.add_argument("--wallet", default=os.environ.get("POLYMARKET_WALLET"))
	ap.add_argument("--positions")
	args = ap.parse_args()

	path = mp.run_path(args.run)
	run = pickle.load(open(path, "rb"))
	through = run["data_through"][:10]
	at = stage(run["cfg"], through)
	key, st, a, sigma, rs = calibration(at)
	print(f"plain run {path.stem}: data through {through}, stage {at}")
	print(f"backtest {key} (stage {st}): " + ", ".join(f"sigma {r['sigma_pick']:g} -> a {r['a']:.2f}" for r in rs)
		+ f"  =>  temper the plain run by {a:.2f}, perturb at sigma {sigma:g}")

	models = [f"{path.stem}@{a:.2f}"]
	if sigma > 0:
		name = f"perturb_{args.sims}_{path.stem.removeprefix('run_')}_p{sigma:g}"
		if not (OUT / f"{name}.pkl").exists() and not args.no_sim:
			print(f"simulating {name} ({args.sims} sims, roughly {args.sims * 1.5 / 60 + 6:.0f} min)...")
			env = os.environ | {"PICKEMS_WORLDS_BUFF": str(run["worlds_buff_scale"])}
			subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "simulate_perturbed.py"), "--sims", str(args.sims),
				"--sigma-pick", f"{sigma:g}", "--name", name, "--worlds-weight", f"{run['worlds_weight']:g}"], env=env, check=True)
		if (OUT / f"{name}.pkl").exists():
			pert = pickle.load(open(OUT / f"{name}.pkl", "rb"))
			if pert["data_through"] != run["data_through"]:
				print(f"warning: {name} has data through {pert['data_through'][:16]}, the plain run {run['data_through'][:16]}")
			del pert
			models.append(name)
		else:
			print("no perturbed run: sizing off the tempered plain run alone (zeros stay zero, so long-shot No bets are untested)")
	del run

	argv = models + ["--bankroll", str(args.bankroll), "--kelly", str(args.kelly), "--min-edge", str(args.min_edge)]
	if args.positions:
		argv += ["--positions", args.positions]
	elif args.wallet:
		argv += ["--wallet", args.wallet]
	print()
	bet_robustness.main(argv)
	print("\nrecommended: the robust sh column (limits rest at the limit price; takers up to it)")


if __name__ == "__main__":
	main()

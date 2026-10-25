"""Recommended orders for the Polymarket "Most Played Champion at Worlds <year>" market,
with the model uncertainty the backtests measured.

  .venv/bin/python -W ignore scripts/recommend_orders.py [--run run_...] [--sims 1000] [--no-sim]
      [--bankroll 1000] [--kelly 0.25] [--wallet 0x... | --positions held.yaml]

1. Takes the plain run (newest run_*, buff on, or --run) and how far through the event
   its data is: games already played over the sims' mean total.
2. Reads the past-Worlds backtest cuts in .cache/calibration.jsonl
   (scripts/backtest_progress.sh), each at its own fraction of games played, and
   interpolates between them at this run's fraction (clamped past the last cut;
   --extend first backtests a cut there):
     - a: the plain run's fitted logit scale, used as --temp on this run, and
     - sigma: the perturbation where the perturbed runs' fitted a crosses 1
       (linear between the sigmas tried, capped at the largest, rounded to 0.1).
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
from pickems.sync_model import DEST, EVENT_YEAR, ROOT, SOURCE  # noqa: E402

SUMMARY = ROOT / ".cache" / "calibration.jsonl"


def sigma_at_one(rs):
	"""Sigma where the perturbed runs' fitted a crosses 1, linear between the sigmas tried;
	(sigma, capped), capped when even the largest tried stays below 1."""
	s, a = np.array([r["sigma_pick"] for r in rs]), np.array([r["a"] for r in rs])
	if a.max() < 1:
		return float(s.max()), True
	if a[0] >= 1:
		return 0.0, False
	i = int(np.argmax(a >= 1))
	return float(s[i - 1] + (1 - a[i - 1]) * (s[i] - s[i - 1]) / (a[i] - a[i - 1])), False


def points():
	"""Per backtest year, the cuts scored so far: (fraction of games played, plain run's a,
	sigma where a crosses 1, capped, key), by fraction."""
	rows = {(r["asof"], r["run"]): r for r in map(json.loads, open(SUMMARY))}.values()
	by_key = {}
	for r in rows:
		by_key.setdefault(r["asof"], []).append(r)
	out = {}
	for key, rs in by_key.items():
		# one row per sigma: runs repeated at a sigma (different names or sims) are averaged
		by_s = {}
		for r in rs:
			by_s.setdefault(r["sigma_pick"], []).append(r)
		rs = [{**g[0], "a": float(np.mean([r["a"] for r in g]))} for s, g in sorted(by_s.items())]
		if rs[0]["sigma_pick"] != 0 or len(rs) < 2:
			continue  # needs the plain run and at least one perturbed one
		frac = rs[0].get("frac", rs[0]["played"] / rs[0]["games"])
		out.setdefault(key[:4], []).append((frac, rs[0]["a"], *sigma_at_one(rs), key))
	return {y: sorted(p) for y, p in out.items()}


def calibration(frac):
	"""(a, sigma, notes) at this fraction of games played: each year's cuts interpolated
	linearly in the fraction (clamped at the ends), then averaged over years."""
	pts = points()
	if not pts:
		raise SystemExit(f"no scored backtest cuts in {SUMMARY}: run scripts/backtest_progress.sh")
	a, s, notes = [], [], []
	for year, p in sorted(pts.items()):
		f = [x[0] for x in p]
		a.append(np.interp(frac, f, [x[1] for x in p]))
		s.append(np.interp(frac, f, [x[2] for x in p]))
		near = sorted(p, key=lambda x: abs(x[0] - frac))[:2]
		notes.append(f"{year}: " + "; ".join(f"{x[4]} ({x[0]:.0%} played) a {x[1]:.2f}, sigma {x[2]:.2g}" + (" (capped)" if x[3] else "")
			for x in sorted(near)))
		if frac > f[-1] + 0.05:
			notes.append(f"  {year}'s cuts stop at {f[-1]:.0%} played: extend with "
				f"scripts/backtest_progress.sh {year} {frac:.2f} (or --extend)")
	return float(np.mean(a)), round(float(np.mean(s)), 1), notes, pts


def freshness(run):
	"""Warnings when the run is behind the data: a newer CSV synced since it ran, or one
	waiting in ~/draftpredict that hasn't been synced."""
	from datetime import datetime, timezone
	name = f"{EVENT_YEAR}_LoL_esports_match_data_from_OraclesElixir.csv"
	ran = datetime.fromisoformat(run["run_at"]).timestamp()
	out = []
	if (SOURCE / name).exists() and (DEST / name).exists() and (SOURCE / name).stat().st_mtime > (DEST / name).stat().st_mtime:
		out.append(f"~/draftpredict has a newer {name}: python -m pickems.sync_model, then scripts/full_pipeline.sh --skip-tune --skip-train")
	elif (DEST / name).exists() and (DEST / name).stat().st_mtime > ran:
		out.append(f"{name} was synced after this run: scripts/full_pipeline.sh --skip-tune --skip-train for a run on it")
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--run", help="plain run (default: newest run_*.pkl, buff on)")
	ap.add_argument("--sims", type=int, default=1000, help="sims for the perturbed run if it has to be made")
	ap.add_argument("--no-sim", action="store_true", help="don't simulate: use the tempered plain run alone if no perturbed run exists")
	ap.add_argument("--extend", action="store_true", help="first backtest a cut at this run's progress if the scored cuts stop short of it")
	ap.add_argument("--temp", type=float, help="use this temper instead of the calibration's (with --sigma)")
	ap.add_argument("--sigma", type=float, help="use this perturbation sigma instead of the calibration's (with --temp)")
	ap.add_argument("--bankroll", type=float, default=1000.0)
	ap.add_argument("--kelly", type=float, default=0.25)
	ap.add_argument("--min-edge", type=float, default=0.02)
	ap.add_argument("--wallet", default=os.environ.get("POLYMARKET_WALLET"))
	ap.add_argument("--positions")
	args = ap.parse_args()

	path = mp.run_path(args.run)
	run = pickle.load(open(path, "rb"))
	for w in freshness(run):
		print(f"warning: {w}")
	if not args.wallet and not args.positions:
		print("warning: no --wallet / $POLYMARKET_WALLET or --positions: sizing as if nothing is held")
	played, total = mp.progress(run)
	frac = played / total
	if (args.temp is None) != (args.sigma is None):
		ap.error("--temp and --sigma go together")
	if args.temp is not None:
		a, sigma, notes, pts = args.temp, args.sigma, ["skipped: --temp and --sigma given"], {}
	else:
		a, sigma, notes, pts = calibration(frac)
	if args.extend:
		for year, p in pts.items():
			if frac > p[-1][0] + 0.05:
				subprocess.run([str(ROOT / "scripts" / "backtest_progress.sh"), year, f"{frac:.2f}"], check=True)
				a, sigma, notes, pts = calibration(frac)
	print(f"plain run {path.stem}: data through {run['data_through'][:16]}, {played} of ~{total:.0f} games played ({frac:.0%})")
	print("backtest cuts nearest: " + "\n  ".join(notes))
	print(f"=> temper the plain run by {a:.2f}, perturb at sigma {sigma:g}")

	models = [f"{path.stem}@{a:.2f}"]
	if sigma > 0:
		name = f"perturb_{args.sims}_{path.stem.removeprefix('run_')}_p{sigma:g}"
		if not (OUT / f"{name}.pkl").exists() and not args.no_sim:
			print(f"simulating {name} ({args.sims} sims, roughly {args.sims * 1.5 / 60 + 1:.0f} min)...")
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

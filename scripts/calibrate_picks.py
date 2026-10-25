"""How well calibrated are a backtest's champion pick totals? (PICKEMS_ASOF=<year>[-<mmdd>])

  PICKEMS_ASOF=2025 .venv/bin/python -W ignore scripts/calibrate_picks.py bt2025_1000 [perturb_... ...]
  PICKEMS_ASOF=2025-1016 ... perturb_bt_400_p0 --summary .cache/calibration.jsonl   # cut after Swiss R1
  .venv/bin/python scripts/calibrate_picks.py --table .cache/calibration.jsonl        # every run scored so far

A mid-event cut (scripts/make_asof.py --mid) scores the whole tournament's totals,
the games played before the cut included, which is what the market resolves on:
it says how far to trust a run made at that point of the event.

Scores each run's simulated pick total per champion against the real Worlds, over
every champion either side has picked:
  - spread: z = (real - sim mean) / sim sd per champion. A calibrated sim has
    mean z^2 ~ 1; sqrt(mean z^2) is how many times wider the spread should be.
    The robust version uses median |z| / 0.674 instead.
  - coverage: share of champions whose real total lands in the sim's central 50 /
    80 / 95% interval (randomised PIT, so ties at integer counts don't bias it).
  - top-k: P(champion finishes in the top k by picks, tie-breaks as the market),
    log loss, and the logit scale a that fits best, p' = sigmoid(a * logit p).
    a < 1 means overconfident. For small p, sigmoid(a * logit p) ~ p^a, so the
    pooled a is the --temp to give scripts/most_picked.py.
  - most picked: what the run gave the real winner. One event, so it shows no
    calibration by itself; it is there as a sanity check.

Champions that are close in a top-k race share outcomes (exactly k are in), so the
effective sample is smaller than the champion count: read a as a direction and a
rough size, and compare runs (plain vs perturbed sigmas) on the same year.
"""

import argparse
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from pickems.report import _stage_sum
from pickems.sim.summary import CHAMP_STATS
from pickems.sync_model import ASOF, ASOF_YEAR, CACHE_DIR, ROOT

ROLES = ["top", "jng", "mid", "bot", "sup"]
SRC = ROOT / "vendor" / "dp"  # the full, uncut data
KS = (1, 3, 5, 10)


def real_totals(year: int, start: str, champs: list[str]) -> np.ndarray:
	"""(stat, champ) real picks / bans / wins at that Worlds, in CHAMP_STATS order for those three."""
	df = pd.read_csv(SRC / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
	df = df[(df.league == "WLDs") & (df.date >= start)]
	ply, team = df[df.position.isin(ROLES)], df[df.position == "team"]
	picks = Counter(ply.champion.dropna())
	wins = Counter(ply[ply.result == 1].champion.dropna())
	bans = Counter(c for col in [f"ban{i}" for i in range(1, 6)] for c in team[col].dropna())
	unknown = sorted(set(picks) - set(champs))
	if unknown:
		print(f"warning: real champions missing from the sim index: {', '.join(unknown)}")
	print(f"real: {team.gameid.nunique()} games from {df.date.min()[:10]} to {df.date.max()[:10]}")
	return np.array([[d.get(c, 0) for c in champs] for d in (picks, bans, wins)], float)


def ranks(picks, bans, wins, champs):
	"""Rank of each champion (0 = most picked) per row: picks, then win rate, then bans, then name."""
	with np.errstate(divide="ignore", invalid="ignore"):
		wr = np.nan_to_num(wins / picks, nan=-1.0)
	alpha = np.broadcast_to(np.argsort(np.argsort(champs)), picks.shape)
	order = np.lexsort((alpha, -bans, -wr, -picks), axis=-1)
	return np.argsort(order, axis=-1)


def fit_scale(p, y):
	"""Logit scale a maximising the Bernoulli likelihood of y under sigmoid(a * logit p)."""
	L = logit(p)

	def nll(a):
		q = np.clip(expit(a * L), 1e-12, 1 - 1e-12)
		return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum()

	return minimize_scalar(nll, bounds=(0.05, 4), method="bounded").x, nll(1.0) / len(y)


def score(name, real):
	"""Print the scores for one run; returns (real totals, summary row)."""
	run = pickle.load(open(CACHE_DIR / "sims" / f"{name}.pkl", "rb"))
	champs = list(run["ix"].champions)
	if real is None:
		start = min(str(e["date"]) for e in run["cfg"]["schedule"])
		real = real_totals(ASOF_YEAR, start, champs)
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	stats = np.stack([_stage_sum(r, "all", "champ") for r in run["results"]]).astype(float)
	picks, bans, wins = stats[:, C["picks"]], stats[:, C["bans"]], stats[:, C["wins"]]
	n = len(picks)
	rp, rb, rw = real
	pert = run.get("perturb")
	# games already fixed: every sim replays them ("T1 3-1 GEN" is 4 games)
	played = sum(sum(map(int, s.split()[-2].split("-"))) for s in run.get("played", []))
	total = rp.sum() / 10
	print(f"\n== {name}: {n} sims" + (f", perturb {pert}" if pert else "") + f", data through {run['data_through'][:10]}"
		f", {played} of {total:.0f} games played")

	# pick totals, over champions anyone picked (real or in at least 1% of sims)
	keep = (rp > 0) | ((picks > 0).mean(axis=0) >= 0.01)
	x, X = rp[keep], picks[:, keep]
	mu, sd = X.mean(axis=0), X.std(axis=0)
	z = (x - mu) / np.maximum(sd, 0.5)
	rng = np.random.default_rng(0)
	below, at = (X < x).mean(axis=0), (X == x).mean(axis=0)
	pit = below + rng.random(len(x)) * at
	cov = {c: np.mean(np.abs(pit - 0.5) <= c / 2) for c in (0.5, 0.8, 0.95)}
	print(f"pick totals, {keep.sum()} champions: mean z^2 {np.mean(z ** 2):.2f} -> spread x{np.sqrt(np.mean(z ** 2)):.2f}"
		f" (robust x{np.median(np.abs(z)) / 0.674:.2f}); bias (real - sim) {np.mean(x - mu):+.2f} picks")
	print("  coverage: " + ", ".join(f"{c:.0%} interval {v:.0%}" for c, v in cov.items())
		+ "   PIT deciles: " + " ".join(str(v) for v in np.histogram(pit, bins=10, range=(0, 1))[0]))
	worst = np.argsort(-np.abs(z))[:6]
	names = np.array(champs)[keep]
	print("  furthest out: " + ", ".join(f"{names[i]} {x[i]:.0f} vs {mu[i]:.1f}±{sd[i]:.1f}" for i in worst))

	# top-k membership
	R, real_rank = ranks(picks, bans, wins, champs), ranks(rp[None], rb[None], rw[None], champs)[0]
	eps = 0.5 / n
	P, Y, A = [], [], {}
	print(f"  {'top-k':<8}{'log loss':>10}{'fit a':>8}   real top k: model P")
	for k in KS:
		p = np.clip((R < k).mean(axis=0), eps, 1 - eps)
		y = (real_rank < k).astype(float)
		a, ll = fit_scale(p, y)
		P.append(p), Y.append(y)
		A[k] = a
		top = np.flatnonzero(y)[np.argsort(real_rank[y > 0])]
		print(f"  {k:<8}{ll:>10.4f}{a:>8.2f}   " + ", ".join(f"{champs[i]} {p[i]:.0%}" for i in top[:5]) + (" ..." if k > 5 else ""))
	a, _ = fit_scale(np.concatenate(P), np.concatenate(Y))
	print(f"  pooled logit scale a = {a:.2f} (most_picked --temp {a:.2f})")
	win = int(np.argmin(real_rank))
	p1 = (R == 0).mean(axis=0)
	print(f"most picked: {champs[win]} ({rp[win]:.0f} picks); run gave it {p1[win]:.1%}, "
		f"rank {1 + (p1 > p1[win]).sum()} (favourite {champs[int(np.argmax(p1))]} {p1.max():.1%})")
	row = {"asof": ASOF, "run": name, "sims": n, "sigma_pick": (pert or {}).get("sigma_pick", 0.0),
	       "data_through": run["data_through"][:16], "played": played, "games": round(total), "frac": played / total,
	       "spread": float(np.sqrt(np.mean(z ** 2))), **{f"cov{round(c * 100)}": float(v) for c, v in cov.items()},
	       "a": float(a), **{f"a{k}": float(v) for k, v in A.items()},
	       "winner": champs[win], "p_winner": float(p1[win]), "rank_winner": int(1 + (p1 > p1[win]).sum()),
	       "favourite": champs[int(np.argmax(p1))], "p_favourite": float(p1.max())}
	return real, row


def table(path):
	"""Every summary row so far, by year and then by how much of the event was played."""
	rows = {(r["asof"], r["run"]): r for r in map(json.loads, open(path))}  # a rescore replaces the old row
	rows = sorted(rows.values(), key=lambda r: (r["asof"][:4], r["played"], r["sigma_pick"]))
	print(f"{'asof':<14}{'run':<26}{'played':>9}{'sigma':>7}{'spread':>8}{'cov 50/80/95':>15}{'a':>6}{'a1':>6}"
		f"{'winner':>22}{'favourite':>22}")
	for r in rows:
		print(f"{r['asof']:<14}{r['run'][:25]:<26}{r['played']:>4}/{r['games']:<4}{r['sigma_pick']:>7g}{r['spread']:>8.2f}"
			f"{r['cov50']:>7.0%}{r['cov80']:>4.0%}{r['cov95']:>4.0%}{r['a']:>6.2f}{r['a1']:>6.2f}"
			f"{r['winner'][:10]:>12} {r['p_winner']:>5.1%} #{r['rank_winner']:<2}{r['favourite'][:10]:>12} {r['p_favourite']:>5.1%}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("runs", nargs="*", help="run names under .cache/asof_<key>/sims")
	ap.add_argument("--summary", help="append one JSON row per run to this file")
	ap.add_argument("--table", help="print the rows in this summary file and stop")
	args = ap.parse_args()
	if args.table:
		return table(args.table)
	if not ASOF or not args.runs:
		ap.error("set PICKEMS_ASOF=<year>[-<mmdd>] and name runs: this scores a backtest against that year's real Worlds")
	real = None
	for name in args.runs:
		real, row = score(name.removesuffix(".pkl"), real)
		if args.summary:
			with open(args.summary, "a") as f:
				f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
	main()

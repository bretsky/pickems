"""Which predictions moved between two simulation runs (e.g. after a new
draftpredict model)? Only quantities both runs recorded are compared, and each
change is scored against Monte Carlo noise: z = delta / SE of the difference
of two independent proportions.

  .venv/bin/python -W ignore scripts/compare_runs.py run_2000 run_2000_v18
"""

import json
import pickle
import sys

import numpy as np

from pickems.report import _leader_probs, _player_index, _stage_sum, team_outcomes
from pickems.sim.runner import OUT
from pickems.sim.summary import CHAMP_STATS, PLAYER_STATS, STAGES


def load(name):
	return pickle.load(open(OUT / f"{name}.pkl", "rb"))


def probs(run) -> dict[str, float]:
	"""Flat {label: probability} of everything comparable across runs."""
	res, ix, teams = run["results"], run["ix"], run["teams"]
	out = {}
	for o in team_outcomes(res, ix.teams):
		t = o["team"]
		for k in ("champion", "final", "semis", "quarters", "play_in_win", "swiss_3_0", "swiss_0_3"):
			out[f"{t}: {k.replace('_', ' ')}"] = o[k]
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	P = {k: i for i, k in enumerate(PLAYER_STATS)}
	champ = np.stack([_stage_sum(r, "all", "champ") for r in res]).astype(float)
	pl = np.stack([_stage_sum(r, "all", "player") for r in res]).astype(float)
	games = np.array([sum(r["stages"][s]["n_games"] for s in STAGES) for r in res], float)
	picks, bans = champ[:, C["picks"]], champ[:, C["bans"]]
	with np.errstate(divide="ignore", invalid="ignore"):
		qs = {
			"Q1 highest presence": ((picks + bans) / games[:, None], None, ix.champions),
			"Q2 highest win rate": (np.nan_to_num(champ[:, C["wins"]] / picks, nan=-1), picks >= 5, ix.champions),
			"Q3 most deaths/game": (np.nan_to_num(champ[:, C["deaths"]] / picks, nan=-1), picks >= 5, ix.champions),
			"Q4 highest KDA": (np.nan_to_num((pl[:, P["kills"]] + pl[:, P["assists"]]) / np.maximum(pl[:, P["deaths"]], 1), nan=-1), pl[:, P["games"]] >= 5,
			                   [f"{n} ({t})" for t, _, n in ix.players]),
		}
	for q, (v, el, names) in qs.items():
		lp = _leader_probs(v, el)
		for i in np.flatnonzero(lp > 0):
			out[f"{q}: {names[i]}"] = float(lp[i])
	k = pl[:, P["kills"]]
	out["Q7 Peyz out-kills Gumayusi"] = float((k[:, _player_index(ix, "Peyz")] > k[:, _player_index(ix, "Gumayusi")]).mean())
	out["Q8 Faker reaches 700 kills"] = float((k[:, _player_index(ix, "Faker")] >= 84).mean())
	out["Q9 Teemo picked"] = float((picks[:, ix.champions.index("Teemo")] > 0).mean())
	out["Q16 LCS team in knockouts"] = float(np.mean([any(teams[t]["region"] == "LCS" for t in r["quarterfinalists"]) for r in res]))
	for n in (2, 3, 4, 5):
		out[f"Q15 {n} regions in knockouts"] = float(np.mean([len({teams[t]['region'] for t in r["quarterfinalists"]}) == n for r in res]))
	for i, c in enumerate(ix.champions):
		out[f"meta: {c} presence (mean)"] = float(((picks[:, i] + bans[:, i]) / games).mean())
	return out


def main():
	a_name, b_name = sys.argv[1:3]
	A, B = load(a_name), load(b_name)
	pa, pb = probs(A), probs(B)
	na, nb = A["sims"], B["sims"]
	rows = []
	for key in sorted(set(pa) | set(pb)):
		x, y = pa.get(key, 0.0), pb.get(key, 0.0)
		if key.startswith("meta:"):
			z = None  # a mean rate, not a proportion of sims
		else:
			se = np.sqrt(x * (1 - x) / na + y * (1 - y) / nb)
			z = (y - x) / se if se > 0 else 0.0
		rows.append({"what": key, "before": x, "after": y, "delta": y - x, "z": z})
	rows.sort(key=lambda r: -abs(r["delta"]))
	ma, mb = A["manifest"], B["manifest"]
	print(f"{a_name}: params v{ma.get('params_version')} ({ma['commit'][:7]})  ->  {b_name}: params v{mb.get('params_version')} ({mb['commit'][:7]})\n")
	print(f"{'prediction':52s} {'before':>7s} {'after':>7s} {'delta':>7s}  z")
	for r in rows[:40]:
		z = "" if r["z"] is None else f"{r['z']:+.1f}{' *' if abs(r['z']) >= 2 else ''}"
		print(f"{r['what'][:52]:52s} {r['before']:7.1%} {r['after']:7.1%} {r['delta']:+7.1%}  {z}")
	(OUT / f"compare_{a_name}_{b_name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
	main()

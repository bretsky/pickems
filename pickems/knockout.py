"""Knockout-stage picks: who advances from Swiss with which record, the
quarterfinal draw (real one from the config once known, otherwise a
placeholder that follows the draw rules), and the bracket that maximizes
expected Pick'Em points given head-to-head Bo5 odds (pickems.sim.h2h).

Bracket order: QF1 and QF2 feed SF1, QF3 and QF4 feed SF2.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

RECORD_SLOTS = [("3-0", 2), ("3-1", 3), ("3-2", 3), ("2-3", 3), ("1-3", 3), ("0-3", 2)]
DEFAULT_POINTS = {"QF": 5, "SF": 5, "Final": 10}


def most_likely_records(outcomes: list[dict]) -> dict[str, str]:
	"""Assign each Swiss team one final record so every record bucket holds its
	real number of teams, maximizing the summed probability of the assignment."""
	teams = [o for o in outcomes if o["swiss"] > 0.5]
	slots = [rec for rec, n in RECORD_SLOTS for _ in range(n)]
	key = {"3-0": "swiss_3_0", "3-1": "swiss_3_1", "3-2": "swiss_3_2", "2-3": "swiss_2_3", "1-3": "swiss_1_3", "0-3": "swiss_0_3"}
	cost = np.array([[-(o[key[s]] / max(o["swiss"], 1e-9)) for s in slots] for o in teams])
	rows, cols = linear_sum_assignment(cost)
	return {teams[r]["team"]: slots[c] for r, c in zip(rows, cols)}


def placeholder_draw(records: dict[str, str], regions: dict[str, str], strength: dict[str, float]) -> list[list[str]]:
	"""Follows the draw rules: each 3-0 team meets a 3-2 team, 3-0 teams in
	opposite halves, everyone else paired among themselves. Among valid draws,
	prefers no same-region quarterfinal, then the one pairing the strongest
	3-0 team with the weakest 3-2 team."""
	t30 = sorted([t for t, r in records.items() if r == "3-0"], key=lambda t: -strength[t])
	t32 = [t for t, r in records.items() if r == "3-2"]
	t31 = [t for t, r in records.items() if r == "3-1"]
	best = None
	for opp in itertools.permutations(t32, len(t30)):
		rest = t31 + [t for t in t32 if t not in opp]
		for perm in itertools.permutations(rest):
			others = [perm[i:i + 2] for i in range(0, len(perm), 2)]
			qfs = [[a, b] for a, b in zip(t30, opp)] + [list(x) for x in others]
			same = sum(regions[a] == regions[b] for a, b in qfs)
			mismatch = -sum(strength[a] - strength[b] for a, b in zip(t30, opp))
			key = (same, mismatch)
			if best is None or key < best[0]:
				best = (key, qfs)
	qfs = best[1]
	return [qfs[0], qfs[2], qfs[1], qfs[3]] if len(t30) == 2 else qfs


def slot_probs(draw: list[list[str]], p: dict) -> dict[str, dict[str, float]]:
	"""P(team wins each bracket slot) from pairwise series odds."""
	qf = [{a: p[a][b], b: p[b][a]} for a, b in draw]

	def merge(x, y):
		return {t: px * sum(py * p[t][u] for u, py in y.items()) for t, px in x.items()} | {u: py * sum(px * p[u][t] for t, px in x.items()) for u, py in y.items()}

	sf = [merge(qf[0], qf[1]), merge(qf[2], qf[3])]
	final = merge(sf[0], sf[1])
	return {"QF1": qf[0], "QF2": qf[1], "QF3": qf[2], "QF4": qf[3], "SF1": sf[0], "SF2": sf[1], "Final": final}


def best_bracket(draw: list[list[str]], p: dict, points: dict | None = None) -> dict:
	"""Enumerate all 128 consistent brackets; expected points use each slot's
	win probability (a pick scores if that team wins that match)."""
	pts = points or DEFAULT_POINTS
	sp = slot_probs(draw, p)
	best = []
	for q in itertools.product(*draw):
		for s1 in (q[0], q[1]):
			for s2 in (q[2], q[3]):
				for f in (s1, s2):
					picks = {"QF1": q[0], "QF2": q[1], "QF3": q[2], "QF4": q[3], "SF1": s1, "SF2": s2, "Final": f}
					ev = sum(pts[k.rstrip("1234")] * sp[k][v] for k, v in picks.items())
					best.append((ev, picks))
	best.sort(key=lambda x: -x[0])
	ev, picks = best[0]
	return {"draw": draw, "picks": picks, "expected_points": ev, "max_points": 4 * pts["QF"] + 2 * pts["SF"] + pts["Final"],
	        "pick_prob": {k: sp[k][v] for k, v in picks.items()}, "slot_probs": sp, "points": pts,
	        "champion_odds": dict(sorted(sp["Final"].items(), key=lambda kv: -kv[1]))}

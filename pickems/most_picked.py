"""P(champion is the most picked over the whole tournament) from a run, and that as
outcome probabilities for the Polymarket market (scripts/most_picked.py,
scripts/bet_robustness.py).
"""

import pickle

import numpy as np
import yaml

from pickems import polymarket as pm
from pickems.report import _stage_sum
from pickems.sim.runner import OUT
from pickems.sim.summary import CHAMP_STATS


def run_path(name=None):
	"""A run's pickle: by name, or the newest run_*.pkl with the buff on."""
	if name is None:
		return max((p for p in OUT.glob("run_*.pkl") if not p.stem.endswith("_nobuff")), key=lambda p: p.stat().st_mtime)
	return OUT / (name.removesuffix(".pkl") + ".pkl")


def load_run(name=None):
	path = run_path(name)
	run = pickle.load(open(path, "rb"))
	return path, run["results"], list(run["ix"].champions)


def p_most_picked(res, champs):
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	stats = np.stack([_stage_sum(r, "all", "champ") for r in res]).astype(float)  # (sims, stat, champ)
	picks, bans, wins = stats[:, C["picks"]], stats[:, C["bans"]], stats[:, C["wins"]]
	with np.errstate(divide="ignore", invalid="ignore"):
		winrate = np.nan_to_num(wins / picks, nan=-1.0)

	# narrow the field key by key: picks, then win rate, then bans, then name
	alive = np.ones_like(picks, bool)
	tied_on_picks = None
	for key in (picks, winrate, bans):
		v = np.where(alive, key, -np.inf)
		alive = v == v.max(axis=1, keepdims=True)
		if tied_on_picks is None:
			tied_on_picks = (alive.sum(axis=1) > 1).mean()
	still_tied = (alive.sum(axis=1) > 1).mean()
	alpha = np.argsort(np.argsort(champs))
	first = np.where(alive, alpha, len(champs)).argmin(axis=1)
	p_top = np.bincount(first, minlength=len(champs)) / len(res)
	rank = (-picks).argsort(axis=1).argsort(axis=1).mean(axis=0) + 1
	return p_top, picks.mean(axis=0), rank, tied_on_picks, still_tied


def outcome_probs(sim, n, markets, temp=1.0, shrink=0.0):
	"""(states, probs) over the market's outcomes from sim = {champion: P(most picked)} over n sims:
	listed champions the sim knows, plus Other for everyone unlisted."""
	listed = {m.name for m in markets if m.name != pm.OTHER}
	states = sorted(listed & sim.keys()) + [pm.OTHER]
	raw = np.array([sim[s] for s in states[:-1]] + [sum(v for c, v in sim.items() if c not in listed)])
	if temp != 1:  # p^t, from scripts/calibrate_picks.py; can't lift what the sims never saw (perturbed runs can)
		raw = raw ** temp / (raw ** temp).sum()
	# Jeffreys smoothing so no state the sim never saw gets probability exactly 0
	probs = (raw * n + 0.5) / (n + 0.5 * len(states))
	if shrink:
		# market prior: best bid/ask mids rescaled to fill what the model leaves outside Other
		by_name = {m.name: m for m in markets}
		mid = np.array([((by_name[s].best_bid or 0) + (by_name[s].best_ask or 1)) / 2 for s in states[:-1]])
		prior = np.append(mid / mid.sum() * (1 - probs[-1]), probs[-1])
		probs = (1 - shrink) * probs + shrink * prior
	return states, probs


def market_q(states, probs, markets):
	"""{market name: P(Yes)}, 0 for listed names the sim doesn't know."""
	q = dict(zip(states, probs))
	for m in markets:
		q.setdefault(m.name, 0.0)
	return q


def thresholds(q, n, min_edge):
	"""Edge a bet needs per share: min_edge, or 2 sim standard errors if larger."""
	return {k: max(min_edge, 2 * np.sqrt(v * (1 - v) / n)) for k, v in q.items()}


def holdings(wallet, positions, ev, markets, q):
	"""Held positions as Bets: from a --positions YAML, else the wallet's, else none."""
	by_name = {m.name: m for m in markets}
	if positions:
		rows = yaml.safe_load(open(positions)) or []
		missing = [r["champion"] for r in rows if r["champion"] not in by_name]
		if missing:
			raise SystemExit(f"--positions: no market for {', '.join(missing)}")
		# YAML reads a bare YES / NO as a boolean
		side = lambda v: ("YES" if v else "NO") if isinstance(v, bool) else str(v).upper()
		return [pm.held_bet(by_name[r["champion"]], side(r["side"]), float(r["shares"]), float(r["avg"]), q) for r in rows]
	if wallet:
		return pm.positions(wallet, ev["id"], markets, q)
	return []

"""Do the Polymarket "Most Played Champion" bets survive model uncertainty?

Prices the market off several models at once, each a run with an optional power
on its P(most picked) (name@t, as most_picked --temp t), for example the plain run,
the plain run tempered to the backtest's fitted a, and perturbed runs at a few
sigmas (scripts/simulate_perturbed.py):

  .venv/bin/python -W ignore scripts/bet_robustness.py run_2000_... run_2000_...@0.7 perturb_2000_..._p0.6
      [--weights 1 1 2] [--bankroll 1000] [--kelly 0.25] [--min-edge 0.02] [--wallet 0x... | --positions held.yaml]

Shows, for the champions that matter, what each model and their weighted mixture
give, and the market. Candidate orders (takers walking the book, resting limit bids)
come from the mixture. They are then sized four ways, on top of held positions:
  - by each model alone (how much the answer swings with the model),
  - by the mixture (the Bayesian answer if the weights are your belief in each),
  - robust: the most that grows wealth under every model (max-min Kelly),
and each sizing's expected log growth is scored under every model, so the cost of
sizing by the wrong one is in view. The orders table gives each order's edge under
every model next to its mixture and robust size.
"""

import argparse
import os
from dataclasses import replace

import numpy as np

from pickems import most_picked as mp
from pickems import polymarket as pm


def parse_model(spec):
	name, _, t = spec.partition("@")
	return name, float(t) if t else 1.0


def sized(scheme, takers, limits, owned, P, w, states, args):
	"""Copies of the candidate bets sized by one scheme: held, then takers, then limits.
	Also returns the least favourable model weights (robust), or None."""
	tk, lm = [replace(b) for b in takers], [replace(b) for b in limits]
	held = [replace(b) for b in owned]
	for b in held:
		b.full = b.shares / (args.kelly * args.bankroll)
	lam = None
	for bets, under in ((tk, held), (lm, held + tk)):
		if scheme == "robust":
			_, lam = pm.robust_kelly(bets, P, states, args.kelly, args.bankroll, held=under)
		else:
			probs = w @ P if scheme == "mixture" else P[scheme]
			pm.kelly(bets, probs, states, args.kelly, args.bankroll, held=under)
	return held, tk, lm, lam


def outcome(bets, P, states, bankroll):
	"""Per model: expected log growth of the bankroll at the actual (fractional) stakes, and
	the worst-state and best-state return."""
	if not bets:
		return np.zeros(len(P)), 0.0, 0.0
	_, R = pm._returns(bets, states)
	W = 1 + np.array([b.shares for b in bets]) @ R / bankroll
	return P @ np.log(np.maximum(W, 1e-12)), W.min() - 1, W.max() - 1


def main(argv=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("models", nargs="+", help="run names, each optionally @temp (e.g. run_2000_...@0.7)")
	ap.add_argument("--weights", type=float, nargs="+", help="weight per model in the mixture (default equal)")
	ap.add_argument("--bankroll", type=float, default=1000.0)
	ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
	ap.add_argument("--min-edge", type=float, default=0.02, help="min mixture edge per share, in probability")
	ap.add_argument("--wallet", default=os.environ.get("POLYMARKET_WALLET"),
		help="Polymarket wallet (proxy) address whose positions count as held (default $POLYMARKET_WALLET)")
	ap.add_argument("--positions", help="YAML list of held positions: {champion, side: YES|NO, shares, avg}, instead of --wallet")
	ap.add_argument("--show", type=float, default=0.01, help="list champions any model gives at least this")
	args = ap.parse_args(argv)
	w = np.array(args.weights or [1.0] * len(args.models), float)
	if len(w) != len(args.models):
		ap.error("--weights needs one weight per model")
	w = w / w.sum()

	ev, markets = pm.fetch()
	labels, sims, ns, loaded = [], [], [], {}
	for spec in args.models:
		name, t = parse_model(spec)
		if name not in loaded:  # name@t reuses the run
			path, res, champs = mp.load_run(name)
			loaded[name] = path, res, champs, mp.p_most_picked(res, champs)[0]
		path, res, champs, p_top = loaded[name]
		labels.append(f"M{len(labels) + 1}")
		sims.append((dict(zip(champs, p_top)), len(res), t))
		ns.append(len(res))
		print(f"{labels[-1]}: {path.stem}" + (f" ^{t:g}" if t != 1 else "") + f", {len(res)} sims, weight {w[len(labels) - 1]:.2f}")
	# one set of states for every model: champions any of them knows
	known = set().union(*(s.keys() for s, _, _ in sims))
	rows = []
	for sim, n, t in sims:
		states, probs = mp.outcome_probs({c: sim.get(c, 0.0) for c in known}, n, markets, t)
		rows.append(probs)
	P = np.array(rows)
	mix = w @ P
	q = mp.market_q(states, mix, markets)
	n_eff = min(ns)
	thr = mp.thresholds(q, n_eff, args.min_edge)
	owned = mp.holdings(args.wallet, args.positions, ev, markets, q)
	by_name = {m.name: m for m in markets}

	# what each model says, beside the market
	col = lambda s: "".join(f"{s:>8}" if isinstance(s, str) else f"{s:>8.1%}" for s in s)
	held_names = {b.market.name for b in owned}
	print(f"\n{'champion':<16}" + col(labels + ["mix"]) + f"{'bid':>8}{'ask':>8}{'spread':>9}")
	for k in np.argsort(-mix):
		name = states[k]
		if P[:, k].max() < args.show and name not in held_names:
			continue
		m = by_name.get(name)
		bid, ask = (m.best_bid, m.best_ask) if m and m.tradeable else (None, None)
		print(f"{name:<16}" + col(list(P[:, k]) + [mix[k]]) + f"{mp_fmt(bid):>8}{mp_fmt(ask):>8}"
			f"{P[:, k].max() / max(P[:, k].min(), 1e-9):>8.1f}x")

	takers = pm.split_closing(pm.taker_bets(markets, q, thr), owned)
	limits = pm.split_closing(pm.limit_bets(markets, q, thr), owned)
	schemes = list(range(len(P))) + ["mixture", "robust"]
	names = [*labels, "mixture", "robust"]
	result = {s: sized(s, takers, limits, owned, P, w, states, args) for s in schemes}

	print(f"\nsizing at {args.kelly:g} Kelly on ${args.bankroll:,.0f} cash" + (f" + {len(owned)} held positions" if owned else "")
		+ "; expected log growth of the bankroll (%) if each model is right, takers + limits all filled")
	print(f"  {'sized by':<10}{'stake':>9}" + "".join(f"{l:>8}" for l in labels) + f"{'mix':>8}{'worst':>8}{'best':>8}")
	for s, label in zip(schemes, names):
		held, tk, lm, _ = result[s]
		bets = held + tk + lm
		g, lo, hi = outcome(bets, P, states, args.bankroll)
		stake = sum(b.shares * b.cost for b in tk + lm)
		print(f"  {label:<10}{stake:>9.2f}" + "".join(f"{x:>+8.2%}" for x in g) + f"{w @ g:>+8.2%}{lo:>+8.1%}{hi:>+8.1%}")
	print("  (worst / best: the bankroll's return in the worst and best outcome, under any model)")
	lam = result["robust"][3]
	if lam is not None:
		print("  robust = Kelly under the least favourable mix of the models: "
			+ ", ".join(f"{l} {x:.0%}" for l, x in zip(labels, lam)) + " (limits on top of the takers)")

	# the orders, with their edge under every model
	_, mtk, mlm, _ = result["mixture"]
	_, rtk, rlm, _ = result["robust"]
	held_shares = {(b.market.name, b.side): b.shares for b in owned}
	for title, mb, rb in (("take now", mtk, rtk), ("rest as limit bids", mlm, rlm)):
		orders = {}
		for b, r in zip(mb, rb):
			o = orders.setdefault((b.market.name, b.side), {"price": 0.0, "mix": 0.0, "rob": 0.0, "cost": 0.0, "edges": None})
			o["price"] = max(o["price"], b.price)
			o["mix"] += b.shares
			o["rob"] += r.shares
			if o["edges"] is None:  # edge at the best level, per model
				yes = P[:, states.index(b.market.name)] if b.market.name in states else np.zeros(len(P))
				o["edges"] = (yes if b.side == "YES" else 1 - yes) - b.cost
		live = {k: o for k, o in orders.items() if max(o["mix"], o["rob"]) * o["price"] >= 0.5}
		print(f"\n{title}" + ("" if live else ": none"))
		if not live:
			continue
		print(f"  {'side':<5}{'champion':<16}{'limit':>7}" + "".join(f"{'edge ' + l:>9}" for l in labels)
			+ f"{'mix sh':>9}{'robust sh':>11}")
		for (name, side), o in sorted(live.items()):
			other = "NO" if side == "YES" else "YES"
			note = f"  (= sell {other})" if held_shares.get((name, other)) else ""
			print(f"  {side:<5}{name:<16}{o['price']:>7.3f}" + "".join(f"{e:>+9.3f}" for e in o["edges"])
				+ f"{o['mix']:>9.1f}{o['rob']:>11.1f}{note}")


def mp_fmt(x):
	return "-" if x is None else f"{x:.3f}"


if __name__ == "__main__":
	main()

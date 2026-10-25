"""P(champion is the most picked over the whole tournament), per champion.

Not the same as highest mean pick count: a champion with a fat-tailed pick total can
lead more often than one with a higher average. Counted per simulation, over all
stages, picks only (bans excluded; use compare_runs Q1 for presence). Ties follow
the Polymarket rules: higher win rate, then more bans, then alphabetical.

Then prices the Polymarket "Most Played Champion at Worlds <year>" market against it:
taker orders worth placing now (walking the book's depth), then resting limit bids at
the highest price that keeps the edge, sized by fractional Kelly jointly across
outcomes (limits on top of the takers, as if all fill) and on top of any positions
already held (--wallet or --positions; --bankroll is the cash beside them). A bet
needs an edge per share, after taker fees, of at least max(--min-edge, 2 sim
standard errors).

  .venv/bin/python -W ignore scripts/most_picked.py [run name] [--top 20]
      [--bankroll 1000] [--kelly 0.25] [--min-edge 0.02] [--temp 1] [--shrink 0]
      [--wallet 0x... | --positions held.yaml] [--all] [--no-market]
"""

import argparse
import os

import numpy as np

from pickems import most_picked as mp
from pickems import polymarket as pm


def fmt(x, spec=".3f"):
	return "-" if x is None else format(x, spec)


def market_report(p_top, champs, n, args):
	ev, markets = pm.fetch()
	listed = {m.name for m in markets if m.name != pm.OTHER}
	sim = dict(zip(champs, p_top))
	unknown = sorted(n for n in listed if n not in sim and not n.startswith("Champion "))
	if unknown:
		print(f"warning: market names not in the sim, treated as P = 0: {', '.join(unknown)}")

	states, probs = mp.outcome_probs(sim, n, markets, args.temp, args.shrink)
	q = mp.market_q(states, probs, markets)
	thr = mp.thresholds(q, n, args.min_edge)
	owned = mp.holdings(args.wallet, args.positions, ev, markets, q)

	live = [m for m in markets if m.tradeable]
	print(f"\n{ev['title']} (polymarket.com/event/{pm.SLUG}): {len(live)} tradeable markets, "
		f"volume ${float(ev.get('volume') or 0):,.0f}")
	print(f"{'champion':<16}{'model':>7}{'bid':>7}{'ask':>7}{'last':>7}{'YES edge':>10}{'NO edge':>9}")
	for m in sorted(live, key=lambda m: -q[m.name]):
		p = q[m.name]
		ye = p - m.best_ask - m.fee(m.best_ask) if m.best_ask else None
		ne = (1 - p) - (1 - m.best_bid) - m.fee(1 - m.best_bid) if m.best_bid else None
		if p < 0.005 and (ye is None or ye < 0) and (ne is None or ne < thr[m.name]) and not args.all:
			continue
		print(f"{m.name:<16}{p:>7.1%}{fmt(m.best_bid):>7}{fmt(m.best_ask):>7}{fmt(m.last):>7}"
			f"{fmt(ye, '+.3f'):>10}{fmt(ne, '+.3f'):>9}")
	print(f"{'Other':<16}{q[pm.OTHER]:>7.1%}  (not tradeable)")

	# held positions are sunk payoffs; takers are sized on top of them, then the limits on
	# top of both as if every limit fills. At the Kelly fraction f a holding of h shares
	# stands for h / f in the full-Kelly problem.
	for b in owned:
		b.full = b.shares / (args.kelly * args.bankroll)
	takers = pm.split_closing(pm.taker_bets(markets, q, thr), owned)
	limits = pm.split_closing(pm.limit_bets(markets, q, thr), owned)
	g_held = pm.kelly([], probs, states, args.kelly, args.bankroll, held=owned)
	g_take = pm.kelly(takers, probs, states, args.kelly, args.bankroll, held=owned)
	g_all = pm.kelly(limits, probs, states, args.kelly, args.bankroll, held=owned + takers)
	print(f"\nsizing: {args.kelly:g} Kelly on ${args.bankroll:,.0f} cash"
		+ (f", model tempered p^{args.temp:g}" if args.temp != 1 else "")
		+ (f", model shrunk {args.shrink:.0%} toward market mids" if args.shrink else "")
		+ f"; full-Kelly log growth {g_held:+.4f} as held, {g_take:+.4f} with takers, {g_all:+.4f} if all limits fill too")
	if owned:
		print(f"\nheld ({args.wallet or args.positions})")
		print(f"  {'side':<5}{'champion':<16}{'shares':>9}{'avg':>7}{'model':>8}{'bid':>7}{'EV vs avg':>11}")
		for b in sorted(owned, key=lambda b: (b.market.name, b.side)):
			bid = b.market.best_bid if b.side == "YES" else (1 - b.market.best_ask if b.market.best_ask else None)
			print(f"  {b.side:<5}{b.market.name:<16}{b.shares:>9.1f}{b.price:>7.3f}{b.prob:>8.1%}{fmt(bid):>7}"
				f"{b.shares * (b.prob - b.price):>+11.2f}")
	held_shares = {(b.market.name, b.side): b.shares for b in owned}

	def show(title, bets, verb):
		# one order per market and side: book levels merge into "up to" the worst price
		orders = {}
		for b in bets:
			if b.shares > 0:
				orders.setdefault((b.market.name, b.side), []).append(b)
		rows = []
		for (name, side), bs in sorted(orders.items()):
			shares = sum(b.shares for b in bs)
			stake = sum(b.shares * b.cost for b in bs)
			if stake >= 0.5:
				rows.append((name, side, max(b.price for b in bs), bs[0].prob, shares, stake,
					sum(b.shares * b.edge for b in bs), bs[0].market.min_size))
		print(f"\n{title}" + ("" if rows else ": none"))
		if not rows:
			return
		print(f"  {'':<4}{'side':<5}{'champion':<16}{'limit':>7}{'avg':>7}{'model':>8}{'shares':>9}{'stake':>9}{'EV':>8}")
		for name, side, limit, prob, shares, stake, ev_, min_size in rows:
			small = "  (< min size)" if shares < min_size else ""
			# buying the other side of a holding is selling it: the pair merges back into $1
			other = "NO" if side == "YES" else "YES"
			if held_shares.get((name, other)):
				small += f"  (= sell {min(shares, held_shares[(name, other)]):.1f} {other} {'down to' if verb == 'buy' else 'at'} {1 - limit:.3f})"
			print(f"  {verb:<4}{side:<5}{name:<16}{limit:>7.3f}{stake / shares:>7.3f}{prob:>8.1%}"
				f"{shares:>9.1f}{stake:>9.2f}{ev_:>+8.2f}{small}")
		print(f"  total stake ${sum(r[5] for r in rows):,.2f}, EV ${sum(r[6] for r in rows):+,.2f}")

	show("take now (limit = worst book level to take; avg includes taker fee)", takers, "buy")
	show("rest as limit bids, at the highest price that keeps the edge (no maker fee)", limits, "bid")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="?", help="run name (default: newest run_*.pkl, buff on)")
	ap.add_argument("--top", type=int, default=20)
	ap.add_argument("--no-market", action="store_true", help="skip the Polymarket comparison")
	ap.add_argument("--bankroll", type=float, default=1000.0)
	ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
	ap.add_argument("--min-edge", type=float, default=0.02, help="min edge per share, in probability")
	ap.add_argument("--temp", type=float, default=1.0, help="power on the model's P(most picked), < 1 flattens (calibrate_picks.py)")
	ap.add_argument("--shrink", type=float, default=0.0, help="weight on market mids to hedge model error")
	ap.add_argument("--wallet", default=os.environ.get("POLYMARKET_WALLET"),
		help="Polymarket wallet (proxy) address whose positions count as held (default $POLYMARKET_WALLET)")
	ap.add_argument("--positions", help="YAML list of held positions: {champion, side: YES|NO, shares, avg}, instead of --wallet")
	ap.add_argument("--all", action="store_true", help="list every tradeable market, not just live ones")
	args = ap.parse_args()
	path, res, champs = mp.load_run(args.run)

	p_top, mean, rank, tied_on_picks, still_tied = mp.p_most_picked(res, champs)
	print(f"{path.name}: {len(res)} sims, P(tie on picks) = {tied_on_picks:.1%}, still tied after win rate and bans = {still_tied:.2%}")
	print(f"{'champion':<16}{'P(most picked)':>15}{'mean picks':>12}{'mean rank':>11}")
	for i in np.argsort(-p_top)[: args.top]:
		if p_top[i] == 0:
			break
		print(f"{champs[i]:<16}{p_top[i]:>15.1%}{mean[i]:>12.1f}{rank[i]:>11.1f}")
	print(f"{'(sum)':<16}{p_top.sum():>15.1%}")

	if not args.no_market:
		market_report(p_top, champs, len(res), args)


if __name__ == "__main__":
	main()

"""Polymarket "Most Played Champion at Worlds 2026" market: books, fees and Kelly sizing.

The event is a neg-risk group of binary markets, one per listed champion plus "Other"
(any unlisted champion). Each market's Yes book is also its No book: buying No at q
is hitting a Yes bid at 1 - q. Takers pay rate * (p * (1 - p))**exponent per share
(the market's feeSchedule); makers pay nothing.

Sizing maximises expected log wealth jointly over every outcome, so Yes bets on
mutually exclusive champions and No bets that all lose on the same champion are
sized together rather than each as if it were the only bet.
"""

import json
from dataclasses import dataclass, field, replace

import numpy as np
import requests
from scipy.optimize import minimize

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
SLUG = "most-played-champion-at-worlds-2026"
OTHER = "Other"


@dataclass
class Market:
	name: str
	yes_token: str
	no_token: str
	tradeable: bool
	tick: float
	min_size: float
	fee_rate: float
	fee_exp: float
	last: float | None
	bids: list[tuple[float, float]] = field(default_factory=list)  # Yes bids (price, size), best first
	asks: list[tuple[float, float]] = field(default_factory=list)  # Yes asks (price, size), best first

	@property
	def best_bid(self):
		return self.bids[0][0] if self.bids else None

	@property
	def best_ask(self):
		return self.asks[0][0] if self.asks else None

	def fee(self, price):
		"""Taker fee per share bought at price."""
		return self.fee_rate * (price * (1 - price)) ** self.fee_exp


def fetch(slug=SLUG):
	ev = requests.get(f"{GAMMA}/events", params={"slug": slug}, timeout=30).json()[0]
	markets = []
	for m in ev["markets"]:
		if m.get("closed"):
			continue
		fs = m.get("feeSchedule") or {}
		on = m.get("feesEnabled", False)
		markets.append(Market(
			name=m["groupItemTitle"],
			yes_token=json.loads(m["clobTokenIds"])[0],
			no_token=json.loads(m["clobTokenIds"])[1],
			tradeable=bool(m.get("acceptingOrders")) and bool(m.get("active")),
			tick=float(m.get("orderPriceMinTickSize") or 0.001),
			min_size=float(m.get("orderMinSize") or 5),
			fee_rate=float(fs.get("rate", 0)) if on else 0.0,
			fee_exp=float(fs.get("exponent", 1)),
			last=float(m["lastTradePrice"]) if m.get("lastTradePrice") is not None else None,
		))
	live = [m for m in markets if m.tradeable]
	books = requests.post(f"{CLOB}/books", json=[{"token_id": m.yes_token} for m in live], timeout=30).json()
	by_token = {b["asset_id"]: b for b in books}
	for m in live:
		b = by_token.get(m.yes_token, {})
		m.bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
		m.asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
	return ev, markets


def positions(wallet, event_id, markets, q):
	"""A wallet's open positions in the event, as held Bets (sizes in shares)."""
	rows = requests.get(f"{DATA}/positions", params={"user": wallet, "eventId": event_id, "sizeThreshold": 0, "limit": 500}, timeout=30).json()
	side_of = {m.yes_token: (m, "YES") for m in markets} | {m.no_token: (m, "NO") for m in markets}
	out = []
	for r in rows:
		if r["asset"] not in side_of or float(r["size"]) <= 0:
			continue
		m, side = side_of[r["asset"]]
		out.append(held_bet(m, side, float(r["size"]), float(r["avgPrice"]), q))
	return out


def held_bet(m, side, shares, avg, q):
	p = q.get(m.name, 0.0)
	return Bet(m, side, avg, 0.0, shares, False, p if side == "YES" else 1 - p, shares=shares, held=True)


@dataclass
class Bet:
	market: Market
	side: str  # "YES" or "NO"
	price: float  # price paid per share of side (before fee)
	fee: float  # per share
	cap: float  # max shares (book depth for takers, inf for resting limits)
	maker: bool
	prob: float  # model P(side pays)
	full: float = 0.0  # full-Kelly shares per unit bankroll
	shares: float = 0.0
	held: bool = False  # already owned: price is the average paid, and sunk
	closes: bool = False  # bought against a holding of the other side: each share merges with it into $1 now

	@property
	def cost(self):
		return 0.0 if self.held else self.price + self.fee

	@property
	def cash(self):
		"""Cash out per share now: a closing share pays for itself and returns the rest."""
		return self.cost - 1 if self.closes else self.cost

	@property
	def edge(self):
		return self.prob - self.cost


def taker_bets(markets, q, thr):
	"""Every book level whose edge clears thr[name], as a bet capped at its depth."""
	out = []
	for m in markets:
		if not m.tradeable:
			continue
		p = q[m.name]
		for a, size in m.asks:
			b = Bet(m, "YES", a, m.fee(a), size, False, p)
			if b.edge < thr[m.name]:
				break
			out.append(b)
		for bid, size in m.bids:
			b = Bet(m, "NO", 1 - bid, m.fee(1 - bid), size, False, 1 - p)
			if b.edge < thr[m.name]:
				break
			out.append(b)
	return out


def limit_bets(markets, q, thr):
	"""One resting bid per market and side at the highest price that still clears thr[name]
	and stays below the opposite best (so it rests rather than takes), if that beats the
	current best bid. Bidding lower gains edge and loses fill chance.

	At most one side per market, the one with the larger edge: bids on both sides that
	sum below 1 would be sized as a riskless pair, but the side that fills is the one
	the market moved against."""
	out = []
	for m in markets:
		if not m.tradeable:
			continue
		p = q[m.name]
		sides = []
		# a No bid at x rests against Yes bids, so it must stay below 1 - best Yes bid;
		# it must also beat the side's best bid, or it only queues behind it
		for side, prob, own, opp in (
			("YES", p, m.best_bid, m.best_ask),
			("NO", 1 - p, 1 - m.best_ask if m.best_ask else None, 1 - m.best_bid if m.best_bid else None),
		):
			price = np.floor((prob - thr[m.name]) / m.tick + 1e-9) * m.tick
			if opp is not None:
				price = min(price, opp - m.tick)
			price = round(price, 6)
			if price < m.tick or price <= (own or 0) or prob - price < thr[m.name]:
				continue
			sides.append(Bet(m, side, price, 0.0, np.inf, True, prob))
		if sides:
			out.append(max(sides, key=lambda b: b.edge))
	return out


def split_closing(bets, held):
	"""Split bets on the other side of a holding into the shares that close it (cash
	coming in) and the rest, filling the holding from the best price down."""
	left = {(b.market.name, b.side): b.shares for b in held}
	out = []
	for b in sorted(bets, key=lambda b: (b.market.name, b.side, b.price)):
		other = (b.market.name, "NO" if b.side == "YES" else "YES")
		n = min(b.cap, left.get(other, 0.0))
		if n > 0:
			left[other] -= n
			out.append(replace(b, cap=n, closes=True))
		if b.cap > n:
			out.append(replace(b, cap=b.cap - n))
	return out


def _returns(bs, states):
	"""(cash out now, net payoff per share in each state) for bets bs."""
	cost = np.array([b.cost for b in bs]).reshape(-1)
	cash = np.array([b.cash for b in bs]).reshape(-1)
	pays = np.array([[(b.market.name == st) == (b.side == "YES") for st in states] for b in bs], float)
	return cash, (pays - cost[:, None]).reshape(len(bs), len(states))


def kelly(bets, probs, states, frac=1.0, bankroll=1.0, held=()):
	"""Set bet.shares to frac times the jointly log-optimal holding, on top of held bets
	already sized by an earlier call.

	probs[k] is P(state k); states[k] names the market that resolves Yes in state k.
	Wealth in state k is 1 - sum(s_j cost_j) + sum(s_j pays_jk), in bankroll units
	at full Kelly. Cash and book depth bind on the scaled holding, so at frac < 1 the
	full-Kelly problem may spend up to 1/frac of the bankroll.
	"""
	probs = np.asarray(probs, float)
	hc, hR = _returns(held, states)
	h = np.array([b.full for b in held])
	W0, cash = 1 + h @ hR, 1 - frac * (h @ hc)
	if not bets:
		return float(probs @ np.log(W0))
	cash_out, R = _returns(bets, states)
	keep = probs > 0
	R, pk, W0 = R[:, keep], probs[keep], W0[keep]

	def neg(s):
		W = W0 + s @ R
		if (W <= 0).any():
			return 1e9
		return -(pk * np.log(W)).sum()

	def grad(s):
		W = np.maximum(W0 + s @ R, 1e-12)
		return -(R * (pk / W)).sum(axis=1)

	res = minimize(neg, np.zeros(len(bets)), jac=grad, method="SLSQP",
		bounds=[(0, b.cap / bankroll / frac) for b in bets],
		constraints=[{"type": "ineq", "fun": lambda s: cash - frac * (s @ cash_out), "jac": lambda s: -frac * cash_out}],
		options={"maxiter": 1000, "ftol": 1e-12})
	s = np.clip(res.x, 0, None)
	for b, x in zip(bets, s):
		b.full = x
		b.shares = frac * x * bankroll
	return -neg(s)  # full-Kelly expected log growth


def growth(bets, probs, states, held=()):
	"""Full-Kelly expected log growth of the sized bets (their .full) and holdings under probs,
	which may be one distribution over states or one per row."""
	bs = list(held) + list(bets)
	_, R = _returns(bs, states)
	W = 1 + np.array([b.full for b in bs]) @ R
	with np.errstate(divide="ignore"):
		return np.atleast_2d(probs) @ np.log(np.maximum(W, 0))


def robust_kelly(bets, prob_sets, states, frac=1.0, bankroll=1.0, held=(), iters=200):
	"""kelly() for the worst of several models: maximise the smallest expected log growth
	over the rows of prob_sets (each a distribution over states), so a stake only goes in
	as far as it grows wealth under every model.

	Each model's growth is concave in the stakes, so by the minimax theorem this is
	kelly() under the least favourable mixture of the models. That mixture is found by
	exponentiated gradient on the weights (the gradient is each model's growth at the
	current stakes); the best stakes seen are kept. Sets bet.shares as kelly() does;
	returns (full-Kelly growth under each model, the least favourable weights)."""
	P = np.atleast_2d(np.asarray(prob_sets, float))
	lam = np.full(len(P), 1 / len(P))
	best = (-np.inf, None, lam)
	for i in range(iters if len(P) > 1 else 1):
		kelly(bets, lam @ P, states, frac, bankroll, held)
		g = growth(bets, P, states, held)
		if g.min() > best[0]:
			best = (g.min(), [b.full for b in bets], lam)
		spread = g.max() - g.min()
		if spread < 1e-7:
			break
		lam = lam * np.exp(-(g - g.mean()) / spread / np.sqrt(i + 1))
		lam /= lam.sum()
	for b, x in zip(bets, best[1] or []):
		b.full = x
		b.shares = frac * x * bankroll
	return growth(bets, P, states, held), best[2]

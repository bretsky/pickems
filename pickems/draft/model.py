"""Conditional-logit draft model: P(action = candidate) = softmax(X @ w) over
available candidates, with separate weights for bans and picks.

Training walks the draft history in date order. Each action's features are read
from the trackers as of before that game; the game is folded in afterwards, so
nothing leaks. Negatives are sampled uniformly from the available candidates
(consistent for MNL, McFadden 1978); evaluation scores against the full set.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from pickems.draft.features import BAN_FEATURES, PICK_FEATURES, DraftContext, Featurizer, NR
from pickems.draft.trackers import DraftState, champion_list, days
from pickems.history import MAJOR_LEAGUES, ROLES
from pickems.sync_model import ROOT

MODEL_PATH = ROOT / ".cache" / "draft_model.pkl"


def contexts_for_game(g, state: DraftState):
	"""Yield (kind, ctx, chosen_index) for every action of a historical game."""
	t = days(g.date)
	side_team = {"Blue": g.blue_team, "Red": g.red_team}
	side_players = {"Blue": list(g.blue_players), "Red": list(g.red_players)}
	picks = {"Blue": {}, "Red": {}}
	unavailable = np.zeros(state.n, dtype=bool)
	if g.fearless:
		for c in g.prior_picks:
			if c in state.cix:
				unavailable[state.cix[c]] = True
	n_pick = {"Blue": 0, "Red": 0}
	n_ban = {"Blue": 0, "Red": 0}
	for slot, (act, side, champ, role) in enumerate(g.actions):
		other = "Red" if side == "Blue" else "Blue"
		c = state.cix.get(champ) if champ is not None else None
		ctx = DraftContext(
			t=t, team=side_team[side], opp=side_team[other], players=side_players[side], opp_players=side_players[other],
			own_picks=dict(picks[side]), opp_picks=dict(picks[other]), unavailable=unavailable.copy(), slot=slot,
			own_pick_ordinal=n_pick[side], own_ban_ordinal=n_ban[side],
		)
		if c is not None and not unavailable[c]:
			if act == "ban":
				yield "ban", ctx, c
			else:
				yield "pick", ctx, c * NR + ROLES.index(role)
		if c is not None:
			unavailable[c] = True
			if act == "pick":
				picks[side][ROLES.index(role)] = c
		if act == "pick":
			n_pick[side] += 1
		else:
			n_ban[side] += 1


@dataclass
class Groups:
	X: np.ndarray        # (rows, F) float32
	start: np.ndarray    # group start offsets; chosen row is always first in its group
	game_ix: np.ndarray  # group -> game row (for splitting)


def build_dataset(games: pd.DataFrame, train_from: str = "2024-01-01", n_neg: int = 40, seed: int = 0):
	"""Walk all games; emit sampled candidate groups for major-league games
	on/after train_from. Returns (state at end, ban Groups, pick Groups, full-eval
	examples for the last 15% of groups are recomputed separately)."""
	rng = np.random.default_rng(seed)
	state = DraftState(champion_list(games))
	feat = Featurizer(state)
	start_t = pd.Timestamp(train_from)
	out = {"ban": ([], [], []), "pick": ([], [], [])}
	for gi, g in enumerate(games.itertuples()):
		if g.date >= start_t and g.league in MAJOR_LEAGUES:
			for kind, ctx, chosen in contexts_for_game(g, state):
				X, mask = feat.ban(ctx) if kind == "ban" else feat.pick(ctx)
				avail = np.flatnonzero(mask)
				avail = avail[avail != chosen]
				neg = rng.choice(avail, size=min(n_neg, len(avail)), replace=False)
				rows, starts, gix = out[kind]
				rows.append(X[np.concatenate([[chosen], neg])].astype(np.float32))
				gix.append(gi)
		state.update(g)
	res = {}
	for kind, (rows, _, gix) in out.items():
		sizes = np.array([len(r) for r in rows])
		res[kind] = Groups(np.concatenate(rows), np.concatenate([[0], np.cumsum(sizes)[:-1]]), np.array(gix))
	return state, res["ban"], res["pick"]


def _nll(w: np.ndarray, X: np.ndarray, start: np.ndarray, l2: float):
	s = X @ w
	# segment logsumexp
	m = np.maximum.reduceat(s, start)
	sizes = np.diff(np.append(start, len(s)))
	e = np.exp(s - np.repeat(m, sizes))
	z = np.add.reduceat(e, start)
	lse = m + np.log(z)
	nll = -(s[start] - lse).sum() + 0.5 * l2 * (w @ w)
	p = e / np.repeat(z, sizes)
	grad_rows = p[:, None] * X
	g = np.add.reduceat(grad_rows, start).sum(axis=0) - X[start].sum(axis=0) + l2 * w
	return nll, g


def fit(groups: Groups, mask_groups: np.ndarray, l2: float = 1.0) -> np.ndarray:
	sel = np.flatnonzero(mask_groups)
	starts = groups.start
	ends = np.append(starts[1:], len(groups.X))
	idx = np.concatenate([np.arange(starts[i], ends[i]) for i in sel])
	X = groups.X[idx].astype(np.float64)
	sizes = ends[sel] - starts[sel]
	start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
	w0 = np.zeros(X.shape[1])
	r = minimize(_nll, w0, args=(X, start, l2), jac=True, method="L-BFGS-B", options={"maxiter": 500})
	return r.x


@dataclass
class DraftModel:
	w_ban: np.ndarray
	w_pick: np.ndarray
	temperature: float = 1.0

	def ban_probs(self, feat: Featurizer, ctx: DraftContext) -> np.ndarray:
		X, mask = feat.ban(ctx)
		return _softmax(X @ self.w_ban / self.temperature, mask)

	def pick_probs(self, feat: Featurizer, ctx: DraftContext) -> np.ndarray:
		X, mask = feat.pick(ctx)
		return _softmax(X @ self.w_pick / self.temperature, mask)


def _softmax(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
	s = np.where(mask, s, -np.inf)
	s = s - s.max()
	e = np.exp(s)
	return e / e.sum()


def evaluate(model: DraftModel, games: pd.DataFrame, state_builder, test_mask_fn) -> dict:
	"""Full-candidate log loss / top-k on test games, walking a fresh state."""
	state = state_builder()
	feat = Featurizer(state)
	stats = {k: {"nll": 0.0, "n": 0, "top1": 0, "top5": 0} for k in ("ban", "pick")}
	for g in games.itertuples():
		if test_mask_fn(g):
			for kind, ctx, chosen in contexts_for_game(g, state):
				p = model.ban_probs(feat, ctx) if kind == "ban" else model.pick_probs(feat, ctx)
				st = stats[kind]
				st["nll"] -= np.log(max(p[chosen], 1e-12))
				st["n"] += 1
				order = np.argsort(-p)
				st["top1"] += int(order[0] == chosen)
				st["top5"] += int(chosen in order[:5])
		state.update(g)
	return {k: {"nll": v["nll"] / max(v["n"], 1), "top1": v["top1"] / max(v["n"], 1), "top5": v["top5"] / max(v["n"], 1), "n": v["n"]} for k, v in stats.items()}


def save(model: DraftModel, path=MODEL_PATH) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "wb") as f:
		pickle.dump(model, f)


def load(path=MODEL_PATH) -> DraftModel:
	with open(path, "rb") as f:
		return pickle.load(f)

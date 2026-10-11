"""Per-action candidate features for the draft model.

A ban action scores every champion; a pick action scores every
(champion, open role) pair, flattened as champ * 5 + role. Both return a
(n_candidates, n_features) matrix plus an availability mask; the conditional
logit in model.py turns scores into a distribution over the available rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pickems.draft.trackers import DraftState
from pickems.history import ROLES

NR = len(ROLES)
EPS = 1e-9


@dataclass
class DraftContext:
	"""An in-progress draft from the acting team's point of view."""

	t: float                       # days since trackers.EPOCH
	team: str                      # acting team key
	opp: str
	players: list[str]             # acting team's players by role
	opp_players: list[str]
	own_picks: dict[int, int]      # role -> champ index
	opp_picks: dict[int, int]
	unavailable: np.ndarray        # bool per champ: banned, picked, or fearless-excluded
	slot: int                      # 0..19 index into DRAFT_ORDER
	own_pick_ordinal: int = 0      # how many picks the acting team already made
	own_ban_ordinal: int = 0
	presence_override: np.ndarray | None = None  # projected meta presence (e.g. 26.20 prior)
	strength_override: np.ndarray | None = None  # per (champ, role) logit strength shift


BAN_FEATURES = [
	"log_presence", "log_ban_rate", "opp_comfort", "opp_share", "opp_team_picks", "own_comfort",
	"own_team_bans", "banned_vs_opp", "best_role_strength", "p2_x_opp_comfort", "p2_x_presence", "opp_role_relevance",
	"presence_trend", "opp_ally_lift", "counters_own", "lane_counter_own",
]
PICK_FEATURES_BASE = [
	"log_presence", "log_pick_rate", "log_role_prob", "player_games", "player_share", "team_picks",
	"role_strength", "player_wr", "early_x_flex", "deny_opp", "early_x_presence", "lane_opp_known_x_player_games",
	"presence_trend", "ally_lift", "enemy_lift", "lane_matchup_wr", "lane_matchup_freq",
]
PICK_FEATURES = PICK_FEATURES_BASE + [f"role_{r}_ord{k}" for r in ROLES for k in range(5)]


class Featurizer:
	def __init__(self, state: DraftState, wr_k: float = 20.0, pwr_k: float = 8.0):
		self.s = state
		self.wr_k, self.pwr_k = wr_k, pwr_k

	# --- shared per-champion reads -------------------------------------------------
	def meta(self, t: float, override: np.ndarray | None):
		s = self.s
		G = s.g_games.get(None, t)
		gp, gb = s.g_pick.get(None, t), s.g_ban.get(None, t)
		presence = (gp + gb + 0.05) / (G + 1.0)
		if override is not None:
			presence = override
		pick_rate = (gp + 0.05) / (G + 1.0)
		ban_rate = (gb + 0.05) / (G + 1.0)
		return np.log(np.clip(presence, 1e-4, 1.5)), np.log(pick_rate), np.log(ban_rate)

	def trend(self, t: float) -> np.ndarray:
		s = self.s
		fast = (s.g_pick_fast.get(None, t) + s.g_ban_fast.get(None, t) + 0.05) / (s.g_games_fast.get(None, t) + 1.0)
		slow = (s.g_pick.get(None, t) + s.g_ban.get(None, t) + 0.05) / (s.g_games.get(None, t) + 1.0)
		return np.log(np.clip(fast, 1e-4, 1.5)) - np.log(np.clip(slow, 1e-4, 1.5))

	def _pair_norm(self, which: str, t: float):
		"""Row-normalized co-pick matrix base: colsum share of each champion."""
		key = (which, t)
		cache = self.__dict__.setdefault("_pair_cache", {})
		if key not in cache:
			cache.clear()
			tr = self.s.ally if which == "ally" else self.s.enemy
			M = np.stack([tr.get(i, t) for i in range(self.s.n)])
			col = M.sum(axis=0)
			cache[key] = (M, M.sum(axis=1), col / max(col.sum(), EPS))
		return cache[key]

	def lift(self, which: str, anchors: list[int], t: float) -> np.ndarray:
		"""Sum over anchor champions a of log((co[a,c]+.5)/(total[a]*base[c]+.5))."""
		if not anchors:
			return np.zeros(self.s.n)
		M, tot, base = self._pair_norm(which, t)
		out = np.zeros(self.s.n)
		for a in anchors:
			out += np.log((M[a] + 0.5) / (tot[a] * base + 0.5))
		return out

	def lane(self, role: int, enemy: int, t: float, k: float = 10.0):
		g = self.s.lane_games.get((role, enemy), t)
		w = self.s.lane_wins.get((role, enemy), t)
		p = (w + 0.5 * k) / (g + k)
		return np.log(p / (1 - p)), np.log1p(g)

	def role_prob(self, t: float) -> np.ndarray:
		"""(n, 5) P(role | champ), lightly smoothed."""
		m = np.stack([self.s.role.get(r, t) for r in ROLES], axis=1) + 0.02
		return m / m.sum(axis=1, keepdims=True)

	def role_strength(self, t: float, override: np.ndarray | None) -> np.ndarray:
		"""(n, 5) shrunk logit win rate per champion-role."""
		g = np.stack([self.s.cr_games.get(r, t) for r in ROLES], axis=1)
		w = np.stack([self.s.cr_wins.get(r, t) for r in ROLES], axis=1)
		p = (w + 0.5 * self.wr_k) / (g + self.wr_k)
		out = np.log(p / (1 - p))
		return out + override if override is not None else out

	def player(self, pid: str, t: float):
		g = self.s.p_games.get(pid, t)
		w = self.s.p_wins.get(pid, t)
		tot = self.s.p_total.get(pid, t)
		p = (w + 0.5 * self.pwr_k) / (g + self.pwr_k)
		return g, g / (tot + 1.0), np.log(p / (1 - p))

	# --- bans --------------------------------------------------------------------
	def ban(self, ctx: DraftContext) -> tuple[np.ndarray, np.ndarray]:
		s, t = self.s, ctx.t
		lp, _, lbr = self.meta(t, ctx.presence_override)
		rp = self.role_prob(t)
		strength = self.role_strength(t, ctx.strength_override)
		opp_open = [r for r in range(NR) if r not in ctx.opp_picks]
		own_open = [r for r in range(NR) if r not in ctx.own_picks]
		opp_comfort = np.zeros(s.n)
		opp_share = np.zeros(s.n)
		for r in opp_open:
			g, share, _ = self.player(ctx.opp_players[r], t)
			opp_comfort += g * rp[:, r]
			opp_share = np.maximum(opp_share, share * rp[:, r])
		own_comfort = np.zeros(s.n)
		for r in own_open:
			g, _, _ = self.player(ctx.players[r], t)
			own_comfort += g * rp[:, r]
		phase2 = float(ctx.slot >= 12)
		# Champions that beat our already-locked laners in lane (bannable counters
		# whose role the opponent still has open).
		lane_counter = np.zeros(s.n)
		for r, c in ctx.own_picks.items():
			if r in ctx.opp_picks:
				continue
			wr, _ = self.lane(r, c, t)
			lane_counter += wr * rp[:, r]
		X = np.column_stack([
			lp, lbr, np.log1p(opp_comfort), opp_share, np.log1p(s.t_pick.get(ctx.opp, t)), np.log1p(own_comfort),
			np.log1p(s.t_ban.get(ctx.team, t)), np.log1p(s.t_banned_vs.get(ctx.opp, t)),
			(strength * rp).sum(axis=1), phase2 * np.log1p(opp_comfort), phase2 * lp,
			np.log(rp[:, opp_open].sum(axis=1) + 0.01) if opp_open else np.zeros(s.n),
			self.trend(t), self.lift("ally", list(ctx.opp_picks.values()), t),
			self.lift("enemy", list(ctx.own_picks.values()), t), lane_counter,
		])
		return X, ~ctx.unavailable

	# --- picks -------------------------------------------------------------------
	def pick(self, ctx: DraftContext) -> tuple[np.ndarray, np.ndarray]:
		s, t, n = self.s, ctx.t, self.s.n
		lp, lpr, _ = self.meta(t, ctx.presence_override)
		rp = self.role_prob(t)
		strength = self.role_strength(t, ctx.strength_override)
		early = float(ctx.own_pick_ordinal <= 1)
		flex = (rp > 0.15).sum(axis=1).astype(float)
		team_picks = np.log1p(s.t_pick.get(ctx.team, t))
		opp_open = [r for r in range(NR) if r not in ctx.opp_picks]
		deny = np.zeros(n)
		for r in opp_open:
			g, _, _ = self.player(ctx.opp_players[r], t)
			deny += g * rp[:, r]
		deny = np.log1p(deny)
		trend = self.trend(t)
		ally_lift = self.lift("ally", list(ctx.own_picks.values()), t)
		enemy_lift = self.lift("enemy", list(ctx.opp_picks.values()), t)
		F = len(PICK_FEATURES)
		X = np.zeros((n, NR, F))
		mask = np.zeros((n, NR), dtype=bool)
		for r in range(NR):
			if r in ctx.own_picks:
				continue
			g, share, pwr = self.player(ctx.players[r], t)
			lane_known = float(r in ctx.opp_picks)
			lwr, lfreq = self.lane(r, ctx.opp_picks[r], t) if lane_known else (np.zeros(n), np.zeros(n))
			cols = [
				lp, lpr, np.log(rp[:, r]), np.log1p(g), share, team_picks, strength[:, r], pwr,
				early * flex, deny, early * lp, lane_known * np.log1p(g),
				trend, ally_lift, enemy_lift, lwr, lfreq,
			]
			X[:, r, : len(cols)] = np.column_stack(cols)
			X[:, r, len(PICK_FEATURES_BASE) + r * 5 + min(ctx.own_pick_ordinal, 4)] = 1.0
			mask[:, r] = ~ctx.unavailable
		return X.reshape(n * NR, F), mask.reshape(n * NR)

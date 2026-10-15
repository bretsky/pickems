"""Time-decayed counters walked chronologically over the draft history.

Each counter stores value * 2**(t / half_life) against a fixed epoch, so a read
is one multiply and nothing has to be decayed in place as time passes. Values
are per-champion numpy vectors keyed by team / player / role.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pickems.history import INTERNATIONAL, MAJOR_LEAGUES, ROLES

EPOCH = pd.Timestamp("2020-01-01")


def days(ts: pd.Timestamp) -> float:
	return (ts - EPOCH).total_seconds() / 86400.0


class Decayed:
	"""key -> per-champion vector, exponentially decayed with `half_life` days."""

	def __init__(self, n: int, half_life: float):
		self.n, self.hl = n, half_life
		self.data: dict = {}

	def add(self, key, idx, t: float, w: float = 1.0) -> None:
		v = self.data.get(key)
		if v is None:
			v = self.data[key] = np.zeros(self.n)
		v[idx] += w * 2.0 ** (t / self.hl)

	def get(self, key, t: float) -> np.ndarray:
		v = self.data.get(key)
		return np.zeros(self.n) if v is None else v * 2.0 ** (-t / self.hl)


class DecayedScalar:
	def __init__(self, half_life: float):
		self.hl = half_life
		self.data: dict = {}

	def add(self, key, t: float, w: float = 1.0) -> None:
		self.data[key] = self.data.get(key, 0.0) + w * 2.0 ** (t / self.hl)

	def get(self, key, t: float) -> float:
		return self.data.get(key, 0.0) * 2.0 ** (-t / self.hl)


class DraftState:
	"""Everything the draft features read, as of some moment in time."""

	def __init__(self, champions: list[str], meta_hl: float = 21.0, team_hl: float = 75.0, player_hl: float = 150.0, wr_hl: float = 60.0):
		self.champions = champions
		self.cix = {c: i for i, c in enumerate(champions)}
		n = len(champions)
		self.n = n
		# Global meta (major leagues only): picks / bans / games.
		self.g_pick = Decayed(n, meta_hl)
		self.g_ban = Decayed(n, meta_hl)
		self.g_games = DecayedScalar(meta_hl)
		# Games in which each champion could have been picked: in fearless play,
		# anything either team already picked earlier in the series is excluded.
		self.g_avail = Decayed(n, meta_hl)
		# Role distribution per champion (slow).
		self.role = Decayed(n, 240.0)
		# Champion-role results (major leagues): games, wins.
		self.cr_games = Decayed(n, wr_hl)
		self.cr_wins = Decayed(n, wr_hl)
		# Team habits.
		self.t_pick = Decayed(n, team_hl)
		self.t_ban = Decayed(n, team_hl)
		self.t_banned_vs = Decayed(n, team_hl)  # what opponents ban against this team
		self.t_games = DecayedScalar(team_hl)
		# Player pools (all leagues).
		self.p_games = Decayed(n, player_hl)
		self.p_wins = Decayed(n, player_hl)
		self.p_total = DecayedScalar(player_hl)
		# Short-memory meta, for trend (presence now vs the meta_hl window).
		self.g_pick_fast = Decayed(n, meta_hl / 3)
		self.g_ban_fast = Decayed(n, meta_hl / 3)
		self.g_games_fast = DecayedScalar(meta_hl / 3)
		self.g_avail_fast = Decayed(n, meta_hl / 3)
		# Champion pairs (major leagues): ally co-picks, enemy co-picks, and lane
		# matchup results keyed by (role, enemy champ) -> wins of each champ vs it.
		self.ally = Decayed(n, 90.0)
		self.enemy = Decayed(n, 90.0)
		self.lane_games = Decayed(n, 180.0)
		self.lane_wins = Decayed(n, 180.0)
		self.last_t = 0.0

	def ensure(self, champ: str) -> int:
		if champ not in self.cix:
			raise KeyError(champ)
		return self.cix[champ]

	def update(self, g) -> None:
		"""Fold one finished game in (call after reading its features)."""
		t = days(g.date)
		self.last_t = t
		major = g.league in MAJOR_LEAGUES
		for side, opp in (("blue", "red"), ("red", "blue")):
			team = getattr(g, f"{side}_team")
			won = g.blue_win if side == "blue" else 1 - g.blue_win
			self.t_games.add(team, t)
			for r, (champ, player) in enumerate(zip(getattr(g, f"{side}_champs"), getattr(g, f"{side}_players"))):
				c = self.cix.get(champ)
				if c is None:
					continue
				self.t_pick.add(team, c, t)
				self.p_games.add(player, c, t)
				self.p_wins.add(player, c, t, won)
				self.role.add(ROLES[r], c, t)
				if major:
					self.g_pick.add(None, c, t)
					self.g_pick_fast.add(None, c, t)
					self.cr_games.add(ROLES[r], c, t)
					self.cr_wins.add(ROLES[r], c, t, won)
			for player in getattr(g, f"{side}_players"):
				self.p_total.add(player, t)
		side_team = {"Blue": g.blue_team, "Red": g.red_team}
		for act, side, champ, _ in g.actions:
			if act != "ban" or champ is None or champ not in self.cix:
				continue
			c = self.cix[champ]
			team = side_team[side]
			opp = side_team["Red" if side == "Blue" else "Blue"]
			self.t_ban.add(team, c, t)
			self.t_banned_vs.add(opp, c, t)
			if major:
				self.g_ban.add(None, c, t)
				self.g_ban_fast.add(None, c, t)
		if major:
			self.g_games.add(None, t)
			self.g_games_fast.add(None, t)
			locked = [self.cix[c] for c in g.prior_picks if c in self.cix] if g.fearless else []
			for avail in (self.g_avail, self.g_avail_fast):
				avail.add(None, slice(None), t)
				if locked:
					avail.add(None, locked, t, -1.0)
			self._update_pairs(g, t)

	def _update_pairs(self, g, t: float) -> None:
		blue = [self.cix.get(c) for c in g.blue_champs]
		red = [self.cix.get(c) for c in g.red_champs]
		if None in blue or None in red:
			return
		for own, opp, won in ((blue, red, g.blue_win), (red, blue, 1 - g.blue_win)):
			own_arr, opp_arr = np.array(own), np.array(opp)
			for r, c in enumerate(own):
				others = own_arr[own_arr != c]
				self.ally.add(c, others, t)
				self.enemy.add(c, opp_arr, t)
				self.lane_games.add((r, opp[r]), c, t)
				self.lane_wins.add((r, opp[r]), c, t, won)


def build_state(games: pd.DataFrame) -> DraftState:
	"""Trackers folded over every game in `games` (the sim's starting point)."""
	state = DraftState(champion_list(games))
	for g in games.itertuples():
		state.update(g)
	return state


def champion_list(games: pd.DataFrame) -> list[str]:
	seen = set()
	for acts in games.actions:
		for _, _, c, _ in acts:
			if c is not None:
				seen.add(c)
	return sorted(seen)


__all__ = ["DraftState", "champion_list", "days", "INTERNATIONAL"]

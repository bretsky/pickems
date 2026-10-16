"""Worlds series already played (Oracle's Elixir), so the simulation replays
them as they happened instead of drawing them.

Every Worlds game in the training history (league WLDs, the event's year; the
regional qualifiers Oracle also files under WLDs are relabelled by
pickems.history.load_raw) becomes a GameRecord with its real draft and box
score. The predictor already holds these games (draftpredict's Oracle walk
read them), so Tournament.play_series takes them as given: nothing is fed to
the walk, only the live-only series state (game number, momentum, First
Selection) moves on, so a series still in progress is finished by simulation
from where it stands.

Series are queued per pair of teams in play order: a tournament's next series
between two teams takes the next real one, which is how rematches (Play-In
lower bracket, Swiss vs knockouts) line up.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import pandas as pd

from pickems.history import ROLES, WORLDS
from pickems.sim.draft import Draft
from pickems.teams import Team

OBJ = ["dragons", "elders", "barons", "heralds", "void_grubs", "atakhans", "towers", "inhibitors"]  # stats_model.OBJ_COLS


@dataclass
class PlayedGame:
	draft: Draft
	blue_win: bool
	stats: dict      # StatsModel.sample's shape
	date: pd.Timestamp
	gameid: str


@dataclass
class PlayedSeries:
	a: str           # team codes, in the order of the series' first game (blue, red)
	b: str
	games: list[PlayedGame]

	@property
	def pair(self) -> frozenset:
		return frozenset((self.a, self.b))

	def score(self) -> str:
		wins = {self.a: 0, self.b: 0}
		for g in self.games:
			wins[(g.draft.blue if g.blue_win else g.draft.red).code] += 1
		return f"{self.a} {wins[self.a]}-{wins[self.b]} {self.b}"


def _n(x) -> float:
	return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _players(stats: dict, team_kills: int) -> list[dict]:
	col = lambda c, r: _n(stats.get(c, [None] * 5)[r])
	return [{"kills": int(col("kills", r)), "deaths": int(col("deaths", r)), "assists": int(col("assists", r)), "cs": int(col("total cs", r)),
	         "damage": col("damagetochampions", r), "gold": col("totalgold", r), "firstbloodkill": int(col("firstbloodkill", r)),
	         "firstbloodvictim": int(col("firstbloodvictim", r)), "doublekills": int(col("doublekills", r)), "triplekills": int(col("triplekills", r)),
	         "quadrakills": int(col("quadrakills", r)), "pentakills": int(col("pentakills", r)), "team_kills": team_kills}
	        for r in range(len(ROLES))]


def played_game(g, blue: Team, red: Team) -> PlayedGame:
	"""One history row (pickems.history.build_games) as the simulator's game."""
	d = Draft(blue, red, g.first_pick_side, [x for x in g.actions if x[2] is not None], list(g.blue_champs), list(g.red_champs))
	won, lost = ("blue", "red") if g.blue_win else ("red", "blue")
	wt, lt = getattr(g, f"{won}_team_stats"), getattr(g, f"{lost}_team_stats")
	wk, lk = int(_n(wt["kills"])), int(_n(lt["kills"]))
	stats = {
		"length": float(g.gamelength) / 60, "win_kills": wk, "lose_kills": lk,
		"win_players": _players(getattr(g, f"{won}_player_stats"), wk), "lose_players": _players(getattr(g, f"{lost}_player_stats"), lk),
		"win_obj": {k: _n(wt.get(k)) for k in OBJ}, "lose_obj": {k: _n(lt.get(k)) for k in OBJ},
	}
	if wk + lk > 0:
		stats["firstblood_winner"] = _n(wt.get("firstblood")) == 1
	return PlayedGame(d, bool(g.blue_win), stats, g.date, g.gameid)


def played_series(games: pd.DataFrame, teams: dict[str, Team], cfg: dict) -> list[PlayedSeries]:
	"""Every series of this Worlds in the history so far, in play order."""
	year = min(pd.Timestamp(str(e["date"])) for e in cfg["schedule"]).year
	w = games[(games.league == WORLDS) & (games.date.dt.year == year)]
	by_key = {t.team_key: t for t in teams.values()}
	out = []
	for _, grp in w.groupby("series_id", sort=False):
		grp = grp.sort_values(["date", "game_in_series"])
		first = grp.iloc[0]
		missing = [n for k, n in ((first.blue_team, first.blue_name), (first.red_team, first.red_name)) if k not in by_key]
		if missing:
			raise KeyError(f"Worlds game {first.gameid} has teams not in the config: {missing}")
		a, b = by_key[first.blue_team], by_key[first.red_team]
		out.append(PlayedSeries(a.code, b.code, [played_game(g, by_key[g.blue_team], by_key[g.red_team]) for g in grp.itertuples()]))
	return sorted(out, key=lambda s: s.games[0].date)


def queue(series: list[PlayedSeries]) -> dict[frozenset, deque]:
	q: dict[frozenset, deque] = {}
	for s in series:
		q.setdefault(s.pair, deque()).append(s)
	return q

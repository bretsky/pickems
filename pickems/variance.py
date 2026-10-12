"""Team / player form variance: two sides with the same average strength can have
different spreads of outcomes.

Each simulated tournament draws a persistent logit "form" for every team and
every player (hot or cold for the whole event), added to the model's logit in
every game they play:

	shift(team) = form_team + sum over its five players of form_player

Independent per-game noise would change nothing (a coin with the marginal win
probability is the same coin); persistence is what makes a streaky team's
tournament different from a steady team's. A single game's win probability is
left unchanged: before the form is added the model's logit is stretched by
sqrt(1 + pi s^2 / 8) (s^2 = both sides' form variance), the probit-style
correction that makes the form-averaged probability equal the model's.

Sizes, all in logit units:
  * pool sd: the excess variance of team results at one event (a split) beyond
    Bernoulli noise, against a plain Elo walked over the history: for a team
    with V = sum p(1-p) over its games and R = wins - sum p,
    sigma^2 = (R^2 - V) / V^2 per team-event, averaged. Elo is cruder than the
    win model, so part of that excess is just Elo being wrong, and PICKEMS_FORM_SCALE
    (default 0.5) scales the sd down.
  * team multiplier: the team's own mean sigma^2 over its events relative to the
    pool, shrunk toward 1 (TEAM_PRIOR events of pseudo-evidence; it is weakly
    stable between events, so heavily).
  * player multiplier: variance of the player's 15-minute gold difference
    (z-scored within league and role) over his recent games relative to 1,
    shrunk toward 1 (PLAYER_PRIOR games). Leagues without that stat (LPL mostly)
    leave players at 1.
  * PLAYER_SHARE of the pool variance goes to the five players (split equally,
    each scaled by its multiplier), the rest to the team.

PICKEMS_FORM_SCALE=0 turns it off.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pickems.history import ROLES
from pickems.teams import Team

ELO_K = 24.0
MIN_EVENT_GAMES = 15
TEAM_PRIOR = 8.0       # events
PLAYER_PRIOR = 40.0    # games
PLAYER_SINCE = "2024-01-01"
PLAYER_SHARE = 0.5
MULT_RANGE = (0.25, 4.0)


def form_scale() -> float:
	return float(os.environ.get("PICKEMS_FORM_SCALE", "0.5"))


def team_event_excess(games: pd.DataFrame) -> pd.DataFrame:
	"""One row per team-event (league, year, split) with >= MIN_EVENT_GAMES games:
	team_key, excess = (R^2 - V) / V^2, the moment estimate of the persistent
	form variance."""
	elo: dict[str, float] = defaultdict(lambda: 1500.0)
	rows = []
	for g in games.itertuples():
		p = 1 / (1 + 10 ** ((elo[g.red_team] - elo[g.blue_team]) / 400))
		ev = f"{g.league}|{g.year}|{g.split}"
		rows.append((g.blue_team, ev, g.blue_win - p, p * (1 - p)))
		rows.append((g.red_team, ev, p - g.blue_win, p * (1 - p)))
		d = ELO_K * (g.blue_win - p)
		elo[g.blue_team] += d
		elo[g.red_team] -= d
	df = pd.DataFrame(rows, columns=["team_key", "event", "res", "var"])
	a = df.groupby(["team_key", "event"]).agg(R=("res", "sum"), V=("var", "sum"), n=("res", "size")).reset_index()
	a = a[a.n >= MIN_EVENT_GAMES]
	a["excess"] = (a.R ** 2 - a.V) / a.V ** 2
	return a[["team_key", "event", "excess"]]


def player_gold_z(games: pd.DataFrame, since: str = PLAYER_SINCE) -> pd.DataFrame:
	"""player_key, z: 15-minute gold difference per game, divided by the sd of
	its (league, role)."""
	rows = []
	for g in games[games.date >= since].itertuples():
		for side in ("blue", "red"):
			for r, k, v in zip(ROLES, getattr(g, f"{side}_players"), getattr(g, f"{side}_player_stats")["golddiffat15"]):
				if v == v and v != 0:
					rows.append((g.league, r, k, float(v)))
	df = pd.DataFrame(rows, columns=["league", "role", "player_key", "gd"])
	sd = df.groupby(["league", "role"]).gd.transform(lambda s: np.sqrt((s ** 2).mean()))
	df["z"] = df.gd / sd
	return df[["player_key", "z"]]


def _shrunk(excess_mean: float, n: float, prior: float) -> float:
	return float(np.clip((n * excess_mean + prior * 1.0) / (n + prior), *MULT_RANGE))


def form_sds(games: pd.DataFrame, teams: dict[str, Team], scale: float | None = None) -> dict:
	"""{"pool_sd", "team": {team_key: sd}, "player": {player_key: sd}} for the
	given teams' squads (empty dicts when scale is 0)."""
	scale = form_scale() if scale is None else scale
	if not scale:
		return {"pool_sd": 0.0, "team": {}, "player": {}}
	ev = team_event_excess(games)
	pool_var = max(float(ev.excess.mean()), 1e-4)
	by_team = ev.groupby("team_key").excess.agg(["mean", "size"])
	z = player_gold_z(games)
	by_player = z.groupby("player_key").z.agg(lambda s: float((s ** 2).mean())).to_frame("var").join(z.groupby("player_key").size().rename("n"))
	base = (scale ** 2) * pool_var
	team_sd, player_sd = {}, {}
	for t in teams.values():
		m = _shrunk(by_team.loc[t.team_key, "mean"] / pool_var, by_team.loc[t.team_key, "size"], TEAM_PRIOR) if t.team_key in by_team.index else 1.0
		team_sd[t.team_key] = float(np.sqrt(base * (1 - PLAYER_SHARE) * m))
		for k in t.players:
			m = _shrunk(by_player.loc[k, "var"], by_player.loc[k, "n"], PLAYER_PRIOR) if k in by_player.index else 1.0
			player_sd[k] = float(np.sqrt(base * PLAYER_SHARE / len(ROLES) * m))
	return {"pool_sd": float(scale * np.sqrt(pool_var)), "team": team_sd, "player": player_sd}


class Form:
	"""One tournament's drawn form: shift(team) and the variance of a pairing's
	shift difference."""

	def __init__(self, sds: dict, teams: dict[str, Team], rng: np.random.Generator):
		self.var: dict[str, float] = {}
		self.shift: dict[str, float] = {}
		for code in sorted(teams):
			t = teams[code]
			sds_t = [sds["team"].get(t.team_key, 0.0)] + [sds["player"].get(k, 0.0) for k in t.players]
			self.shift[t.team_key] = float(sum(rng.normal(0.0, s) for s in sds_t))
			self.var[t.team_key] = float(sum(s * s for s in sds_t))

	def apply(self, logit: float, blue: str, red: str) -> float:
		"""The logit with the game's form: marginal-preserving stretch, then the drawn shifts."""
		s2 = self.var.get(blue, 0.0) + self.var.get(red, 0.0)
		return logit * np.sqrt(1 + np.pi * s2 / 8) + self.shift.get(blue, 0.0) - self.shift.get(red, 0.0)

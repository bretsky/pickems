"""Worlds buff / nerf: a win-probability shift for teams whose past Worlds results beat or missed their odds.

Each past Worlds game (before the simulated year) is scored against a plain
Elo walked over every game in the history: the team's residual is its result
minus that pre-game win probability. That is a random-effects model: a team's
true Worlds effect is delta ~ N(0, tau^2), its estimate from n games is the
summed residual over the summed p(1-p) (variance 1/V), and the posterior mean
is residual / (V + 1/tau^2). tau is not chosen: it is fitted, by maximizing
the marginal likelihood of every team's estimate, so it is exactly how much
more teams' Worlds results vary than chance alone allows. Every team is
shrunk by that rule -- no significance gate, so a team with little evidence
gets ~0 on its own and nothing falls off a cliff at a threshold. The shift is
added to the team's side of the win model's logit in every simulated game
(WinModel.p_blue). scripts/backtest_worlds_buff.py checks the whole scheme out
of sample.

PICKEMS_WORLDS_BUFF scales it (default 0.2: a fifth of the posterior mean;
0 turns it off, 1 is the full estimate);
PICKEMS_WORLDS_TAU fixes tau instead of fitting it.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pickems.history import WORLDS

ELO_K = 24.0
TAU_MAX = 1.0


def buff_scale() -> float:
	return float(os.environ.get("PICKEMS_WORLDS_BUFF", "0.2"))


def worlds_games(games: pd.DataFrame) -> pd.DataFrame:
	"""One row per Worlds game: year, blue/red team_key + name, blue_win and the
	pre-game Elo win probability p_blue (Elo walked over the whole history)."""
	elo: dict[str, float] = defaultdict(lambda: 1500.0)
	rows = []
	for g in games.itertuples():
		p = 1 / (1 + 10 ** ((elo[g.red_team] - elo[g.blue_team]) / 400))
		if g.league == WORLDS:
			rows.append((g.year, g.blue_team, g.red_team, g.blue_name, g.red_name, g.blue_win, p))
		d = ELO_K * (g.blue_win - p)
		elo[g.blue_team] += d
		elo[g.red_team] -= d
	return pd.DataFrame(rows, columns=["year", "blue_team", "red_team", "blue_name", "red_name", "blue_win", "p_blue"])


def team_rows(wg: pd.DataFrame) -> pd.DataFrame:
	"""Per-team view of worlds_games: team_key, name, win, p."""
	blue = pd.DataFrame({"team_key": wg.blue_team, "name": wg.blue_name, "win": wg.blue_win, "p": wg.p_blue})
	red = pd.DataFrame({"team_key": wg.red_team, "name": wg.red_name, "win": 1 - wg.blue_win, "p": 1 - wg.p_blue})
	return pd.concat([blue, red], ignore_index=True)


def team_evidence(rows: pd.DataFrame) -> pd.DataFrame:
	"""Per team: games, wins, expected wins, residual (res) and information (v = sum p(1-p))."""
	rows = rows.assign(res=rows.win - rows.p, v=rows.p * (1 - rows.p))
	return rows.groupby("team_key").agg(name=("name", "last"), games=("win", "size"), wins=("win", "sum"), expected=("p", "sum"),
	                                    res=("res", "sum"), v=("v", "sum"))


def fit_tau(ev: pd.DataFrame) -> float:
	"""Marginal-likelihood tau: each team's estimate res/v ~ N(0, tau^2 + 1/v)."""
	if ev.empty:
		return 0.0
	d, s2 = (ev.res / ev.v).values, (1 / ev.v).values
	nll = lambda t: 0.5 * np.sum(np.log(t * t + s2) + d ** 2 / (t * t + s2))
	t = minimize_scalar(nll, bounds=(1e-4, TAU_MAX), method="bounded").x
	return 0.0 if nll(1e-4) - nll(t) < 1e-3 else float(t)  # no detectable spread: no shift


def shifts(ev: pd.DataFrame, tau: float) -> pd.Series:
	"""Posterior-mean logit shift per team at prior sd tau."""
	return ev.res / (ev.v + 1 / tau ** 2) if tau else ev.res * 0.0


def worlds_buffs(games: pd.DataFrame, before_year: int, scale: float | None = None, tau: float | None = None) -> dict[str, dict]:
	"""team_key -> {name, games, wins, expected, z, tau, logit} for every team with Worlds history before `before_year`."""
	scale = buff_scale() if scale is None else scale
	if not scale:
		return {}
	wg = worlds_games(games)
	ev = team_evidence(team_rows(wg[wg.year < before_year]))
	if tau is None:
		tau = float(os.environ["PICKEMS_WORLDS_TAU"]) if "PICKEMS_WORLDS_TAU" in os.environ else fit_tau(ev)
	sh = shifts(ev, tau)
	return {k: {"name": r["name"], "games": int(r.games), "wins": int(r.wins), "expected": float(r.expected), "z": float(r.res / np.sqrt(r.v)),
	            "tau": tau, "logit": float(scale * sh[k])} for k, r in ev.iterrows()}

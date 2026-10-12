"""Worlds buff / nerf: a win-probability shift for teams whose past Worlds results beat or missed their odds.

Each past Worlds game (before the simulated year) is scored against a plain
Elo walked over every game in the history: the team's residual is its result
minus that pre-game win probability. A team whose residuals add up to a
significant over- or underperformance (|z| >= MIN_Z, a one-sided ~95% bar)
gets a logit bonus (or penalty), ridge-shrunk toward 0 so a short run of
upsets earns little, and capped at +-CAP. The shift is added to the team's side
of the win model's logit in every simulated game (WinModel.p_blue); teams
inside the bar get nothing.

PICKEMS_WORLDS_BUFF scales it (default 1; 0 turns it off).
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pickems.history import WORLDS

ELO_K = 24.0
MIN_Z = 1.64
RIDGE = 11.0   # prior sd 0.3 logit: delta = residual / (variance + 1 / 0.3^2)
CAP = 0.5      # max |logit| shift


def buff_scale() -> float:
	return float(os.environ.get("PICKEMS_WORLDS_BUFF", "1"))


def worlds_residuals(games: pd.DataFrame, before_year: int) -> pd.DataFrame:
	"""One row per team per Worlds game before `before_year`: team_key, name,
	win, p (pre-game Elo win probability)."""
	elo: dict[str, float] = defaultdict(lambda: 1500.0)
	rows = []
	for g in games.itertuples():
		p = 1 / (1 + 10 ** ((elo[g.red_team] - elo[g.blue_team]) / 400))
		if g.league == WORLDS and g.year < before_year:
			rows.append((g.blue_team, g.blue_name, g.blue_win, p))
			rows.append((g.red_team, g.red_name, 1 - g.blue_win, 1 - p))
		d = ELO_K * (g.blue_win - p)
		elo[g.blue_team] += d
		elo[g.red_team] -= d
	return pd.DataFrame(rows, columns=["team_key", "name", "win", "p"])


def worlds_buffs(games: pd.DataFrame, before_year: int, scale: float | None = None) -> dict[str, dict]:
	"""team_key -> {name, games, wins, expected, z, logit} for every team that clears the bar (negative logit = nerf)."""
	scale = buff_scale() if scale is None else scale
	if not scale:
		return {}
	df = worlds_residuals(games, before_year)
	df["res"], df["var"] = df.win - df.p, df.p * (1 - df.p)
	out = {}
	for key, grp in df.groupby("team_key"):
		res, var = grp.res.sum(), grp["var"].sum()
		z = res / np.sqrt(var)
		if abs(z) < MIN_Z:
			continue
		out[key] = {"name": grp.name.iloc[-1], "games": len(grp), "wins": int(grp.win.sum()), "expected": float(grp.p.sum()),
		            "z": float(z), "logit": float(scale * np.clip(res / (var + RIDGE), -CAP, CAP))}
	return out

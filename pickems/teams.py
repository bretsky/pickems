"""Tournament config (data/manual/*.yaml) -> resolved teams with Oracle ids.

Oracle's Elixir keys teams/players by teamid/playerid; the config uses display
names. Each name resolves to the id of its most recent appearance (players
preferring rows with that team), which is also how draftpredict resolves names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from pickems.history import ROLES
from pickems.sync_model import ASOF, ROOT

CONFIG = ROOT / "data" / "manual" / f"worlds{ASOF or 2026}.yaml"


@dataclass
class Team:
	code: str
	name: str          # Oracle teamname
	region: str
	seed: int
	team_key: str
	players: list[str]       # Oracle player keys by role
	player_names: list[str]
	# Non-starters: (role, name, Oracle key). The config's `subs` (listed so they get a rating even before they play), then
	# those who played a Worlds game so far, which pickems.sim.played.played_series adds.
	subs: list[tuple[str, str, str]] = field(default_factory=list)

	def key_of(self, role: str, name: str) -> str:
		"""Oracle key of the starter or sub called `name` in `role`."""
		if self.player_names[ROLES.index(role)] == name:
			return self.players[ROLES.index(role)]
		return next(k for r, n, k in self.subs if r == role and n == name)


def load_config(path: Path = CONFIG) -> dict:
	return yaml.safe_load(path.read_text())


def resolve_teams(games: pd.DataFrame, cfg: dict) -> dict[str, Team]:
	team_key_by_name: dict[str, str] = {}
	player_key: dict[tuple[str, str], str] = {}  # (playername, teamname) -> key
	player_any: dict[str, str] = {}
	for g in games.itertuples():  # chronological: later rows overwrite
		for side in ("blue", "red"):
			tname, tkey = getattr(g, f"{side}_name"), getattr(g, f"{side}_team")
			team_key_by_name[tname] = tkey
			for pname, pkey in zip(getattr(g, f"{side}_player_names"), getattr(g, f"{side}_players")):
				player_key[(pname, tname)] = pkey
				player_any[pname] = pkey
	out = {}
	for code, t in cfg["teams"].items():
		name = t["name"]
		if name not in team_key_by_name:
			raise KeyError(f"{code}: team {name!r} not in Oracle data")
		missing = []

		def key_of(pname: str) -> str:
			key = player_key.get((pname, name)) or player_any.get(pname)
			if key is None:
				missing.append(pname)
				key = f"unknown:{pname}"
			return key

		players = [key_of(str(p)) for p in t["players"]]
		subs = [(r, str(p), key_of(str(p))) for r, p in t.get("subs", [])]
		if missing:
			print(f"warning: {code} players with no Oracle history: {missing}")
		out[code] = Team(code, name, t["region"], t["seed"], team_key_by_name[name], players, [str(p) for p in t["players"]], subs)
	return out


def latest_game(games: pd.DataFrame, teams: dict[str, Team]) -> dict | None:
	"""The most recent game in the training history with a Worlds team on either side."""
	code_by_key = {t.team_key: code for code, t in teams.items()}
	hit = games[games.blue_team.isin(code_by_key) | games.red_team.isin(code_by_key)]
	if hit.empty:
		return None
	g = hit.loc[hit.date.idxmax()]
	return {"date": g.date.isoformat(), "league": g.league, "game": int(g.game),
	        "blue": g.blue_name, "red": g.red_name, "blue_code": code_by_key.get(g.blue_team),
	        "red_code": code_by_key.get(g.red_team), "blue_win": bool(g.blue_win)}

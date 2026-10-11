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
from pickems.sync_model import ROOT

CONFIG = ROOT / "data" / "manual" / "worlds2026.yaml"


@dataclass
class Team:
	code: str
	name: str          # Oracle teamname
	region: str
	seed: int
	team_key: str
	players: list[str]       # Oracle player keys by role
	player_names: list[str]


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
		players, missing = [], []
		for pname in t["players"]:
			pname = str(pname)
			key = player_key.get((pname, name)) or player_any.get(pname)
			if key is None:
				missing.append(pname)
				key = f"unknown:{pname}"
			players.append(key)
		if missing:
			print(f"warning: {code} players with no Oracle history: {missing}")
		out[code] = Team(code, name, t["region"], t["seed"], team_key_by_name[name], players, [str(p) for p in t["players"]])
	return out

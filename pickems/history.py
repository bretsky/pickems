"""Oracle's Elixir -> one row per game with the full draft and per-player stats.

Draft reconstruction: Oracle's team rows carry ban1-5 / pick1-5 in each team's
own draft order, and `firstPick` marks which side picked first (since 2026's
"First Selection" rule this is often red). The standard tournament draft is

  bans:  A1 B1 A2 B2 A3 B3        picks: A1 B1 B2 A2 A3 B3
  bans:  B4 A4 B5 A5              picks: B4 A4 A5 B5

with A = the first-pick team. Each pick's role comes from the player rows.

Series: consecutive games between the same two teams (Oracle's `game` counter,
restarting at 1) form one series; `prior_picks` holds every champion either
team picked earlier in it, which is what fearless excludes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pickems.sync_model import CACHE_DIR, DEST as DP_ROOT, EVENT_YEAR, ROOT

ROLES = ["top", "jng", "mid", "bot", "sup"]
CACHE = CACHE_DIR / "history.pkl"
CACHE_VERSION = 2  # bump when build_games' output changes

# (action, side) where side "A" = first-pick team, "B" = the other.
DRAFT_ORDER = [
	("ban", "A"), ("ban", "B"), ("ban", "A"), ("ban", "B"), ("ban", "A"), ("ban", "B"),
	("pick", "A"), ("pick", "B"), ("pick", "B"), ("pick", "A"), ("pick", "A"), ("pick", "B"),
	("ban", "B"), ("ban", "A"), ("ban", "B"), ("ban", "A"),
	("pick", "B"), ("pick", "A"), ("pick", "A"), ("pick", "B"),
]
# Index into each team's own ban1-5 / pick1-5 for each DRAFT_ORDER slot.
_ORDINALS = []
_seen: dict[tuple[str, str], int] = {}
for _a, _s in DRAFT_ORDER:
	_seen[(_a, _s)] = _seen.get((_a, _s), 0) + 1
	_ORDINALS.append(_seen[(_a, _s)])

# Top-flight leagues + internationals: what the draft model trains on. Every
# league still feeds the trackers (player pools of e.g. LCP rookies come from
# wherever they played).
MAJOR_LEAGUES = {
	"LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "CBLOL", "LCP", "PCS", "VCS", "LJL", "LLA",
	"MSI", "WLDs", "FST", "EWC", "Asia Master", "AM", "WLDs Qual",
}
INTERNATIONAL = {"MSI", "WLDs", "FST", "EWC"}
WORLDS = "WLDs"
# Oracle's Elixir also files some regional qualifiers under WLDs (LCK 2022;
# LPL and LCK 2023 and 2024; LPL 2025 and 2026), weeks before Worlds starts.
# load_raw relabels a WLDs game before its year's date here as "WLDs Qual"
# (still a major league, not Worlds or international). Any day between the
# qualifier's end and Worlds' first day works; a year with a config
# (data/manual/worlds<year>.yaml) and no entry here uses that config's first
# scheduled day. vendor/dp's Worlds weighting keeps its own copy
# (draftpredict.baseline.WORLDS_FROM).
WORLDS_FROM = {2022: "2022-09-20", 2023: "2023-10-01", 2024: "2024-09-20", 2025: "2025-10-07", 2026: "2026-10-05"}
for _cfg in (ROOT / "data" / "manual").glob("worlds[0-9][0-9][0-9][0-9].yaml"):
	WORLDS_FROM.setdefault(int(_cfg.stem[6:]), min(str(e["date"]) for e in yaml.safe_load(_cfg.read_text())["schedule"]))


def worlds_weights(leagues, worlds_weight: float) -> np.ndarray:
	"""Per-game training weight: Worlds games 1, everything else 1 - worlds_weight.
	0 trains as before (every game equal); 1 trains on Worlds games only."""
	if not 0.0 <= worlds_weight <= 1.0:
		raise ValueError(f"worlds_weight must be in [0, 1], got {worlds_weight}")
	return np.where(np.asarray(leagues) == WORLDS, 1.0, 1.0 - worlds_weight)


def worlds_share(games: pd.DataFrame, worlds_weight: float, since: str = "2024-01-01") -> float:
	"""Worlds games' share of total training weight over the major-league games
	since `since` (what the draft and stats models train on)."""
	g = games[(games.date >= since) & games.league.isin(MAJOR_LEAGUES)]
	w = worlds_weights(g.league.values, worlds_weight)
	return float(w[g.league.values == WORLDS].sum() / w.sum())


def weighted_path(path: Path, worlds_weight: float) -> Path:
	"""Where a model trained with this worlds_weight lives (0 keeps the original file)."""
	return path if worlds_weight == 0 else path.with_name(f"{path.stem}_ww{worlds_weight:g}{path.suffix}")

STAT_COLS = [
	"kills", "deaths", "assists", "doublekills", "triplekills", "quadrakills", "pentakills",
	"firstbloodkill", "firstbloodassist", "firstbloodvictim", "damagetochampions", "visionscore",
	"totalgold", "total cs", "golddiffat15",
]
TEAM_STAT_COLS = [
	"kills", "deaths", "firstblood", "dragons", "elders", "barons", "heralds", "void_grubs", "atakhans",
	"towers", "inhibitors", "firstdragon", "firstbaron", "firsttower", "firstherald",
]


def oracle_patch(p) -> float | None:
	try:
		return round(float(p), 2)
	except (TypeError, ValueError):
		return None


def load_raw(years: list[int]) -> pd.DataFrame:
	frames = []
	for y in years:
		df = pd.read_csv(DP_ROOT / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
		df["league"] = df["league"].replace({"NA LCS": "LCS", "EU LCS": "LEC", "OGN": "LCK"})
		frames.append(df)
	df = pd.concat(frames, ignore_index=True)
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	start = pd.to_datetime(df["date"].dt.year.map(WORLDS_FROM))
	df.loc[(df["league"] == WORLDS) & (df["date"] < start), "league"] = "WLDs Qual"
	return df


def _key(df: pd.DataFrame, id_col: str, name_col: str, fallback: str) -> pd.Series:
	return df[id_col].fillna(df[name_col]).fillna(fallback)


def build_games(years: list[int]) -> pd.DataFrame:
	"""One row per game with full draft. Games lacking a complete draft are dropped."""
	raw = load_raw(years)
	team = raw[raw.position == "team"].copy()
	team["team_key"] = _key(team, "teamid", "teamname", "noid:" + team["league"].astype(str) + ":" + team["teamname"].astype(str))
	ply = raw[raw.position.isin(ROLES)].copy()
	ply["player_key"] = _key(ply, "playerid", "playername", "unnamed")

	stat_cols = [c for c in STAT_COLS if c in ply]
	ply["role_ix"] = ply["position"].map({r: i for i, r in enumerate(ROLES)})
	ply = ply.sort_values(["gameid", "side", "role_ix"])
	# (gameid, side) -> per-role lists, only for complete 5-player sides.
	lineup: dict[tuple, dict] = {}
	for (gid, side), grp in ply.groupby(["gameid", "side"], sort=False):
		if len(grp) != 5 or grp["role_ix"].tolist() != [0, 1, 2, 3, 4]:
			continue
		lineup[(gid, side)] = {
			"champs": grp["champion"].tolist(), "players": grp["player_key"].tolist(), "names": grp["playername"].tolist(),
			"stats": {c: grp[c].tolist() for c in stat_cols},
		}

	rows = []
	team_by_game = {gid: grp for gid, grp in team.groupby("gameid", sort=False)}
	for gid, grp in team_by_game.items():
		if len(grp) != 2 or set(grp["side"]) != {"Blue", "Red"}:
			continue
		blue = grp[grp.side == "Blue"].iloc[0]
		red = grp[grp.side == "Red"].iloc[0]
		if (gid, "Blue") not in lineup or (gid, "Red") not in lineup:
			continue
		fp = "Red" if red.get("firstPick") == 1 else "Blue"
		sides = {"A": fp, "B": "Red" if fp == "Blue" else "Blue"}
		rec = {"Blue": blue, "Red": red}
		roles_of = {s: dict(zip(lineup[(gid, s)]["champs"], ROLES)) for s in ("Blue", "Red")}
		actions = []
		ok = True
		for (act, s), n in zip(DRAFT_ORDER, _ORDINALS):
			side = sides[s]
			champ = rec[side].get(f"{act}{n}")
			if act == "pick" and (pd.isna(champ) or champ not in roles_of[side]):
				ok = False
				break
			actions.append((act, side, None if pd.isna(champ) else champ, roles_of[side].get(champ) if act == "pick" else None))
		if not ok:
			continue
		row = {
			"gameid": gid, "date": blue["date"], "league": blue["league"], "year": blue["year"], "split": blue.get("split"),
			"playoffs": blue.get("playoffs"), "patch": oracle_patch(blue["patch"]), "game": blue.get("game"),
			"first_pick_side": fp, "blue_team": blue["team_key"], "red_team": red["team_key"],
			"blue_name": blue["teamname"], "red_name": red["teamname"], "blue_win": int(blue["result"] == 1),
			"gamelength": blue["gamelength"], "actions": actions,
		}
		for s, tr in (("blue", blue), ("red", red)):
			lu = lineup[(gid, s.capitalize())]
			row[f"{s}_players"] = lu["players"]
			row[f"{s}_player_names"] = lu["names"]
			row[f"{s}_champs"] = lu["champs"]
			row[f"{s}_team_stats"] = {c: tr.get(c) for c in TEAM_STAT_COLS}
			row[f"{s}_player_stats"] = lu["stats"]
		rows.append(row)
	games = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
	_assign_series(games)
	return games


def _assign_series(games: pd.DataFrame) -> None:
	"""series_id, game_in_series and prior_picks (champions either team already
	picked earlier in the same series)."""
	series_id = np.empty(len(games), dtype=object)
	game_n = np.zeros(len(games), dtype=int)
	prior = [frozenset()] * len(games)
	open_series: dict[frozenset, dict] = {}
	next_id = 0
	for i, g in enumerate(games.itertuples()):
		pair = frozenset((g.blue_team, g.red_team))
		cur = open_series.get(pair)
		new = cur is None or (g.date - cur["last"]) > pd.Timedelta(hours=14) or (pd.notna(g.game) and g.game == 1 and cur["n"] >= 1)
		if new:
			cur = {"id": next_id, "n": 0, "last": g.date, "picked": set()}
			next_id += 1
			open_series[pair] = cur
		cur["n"] += 1
		cur["last"] = g.date
		series_id[i] = cur["id"]
		game_n[i] = cur["n"]
		prior[i] = frozenset(cur["picked"])
		cur["picked"].update(g.blue_champs)
		cur["picked"].update(g.red_champs)
	games["series_id"] = series_id
	games["game_in_series"] = game_n
	games["prior_picks"] = prior
	_flag_fearless(games)


def _flag_fearless(games: pd.DataFrame) -> None:
	"""A (league, year, split) counts as fearless when later games of its
	series essentially never repeat an earlier pick (by either team)."""
	later = games[games.game_in_series > 1]
	repeat = later.apply(lambda g: len(g.prior_picks & set(g.blue_champs + g.red_champs)) > 0, axis=1)
	key = later.league.astype(str) + "|" + later.year.astype(str) + "|" + later.split.fillna("").astype(str)
	rate = repeat.groupby(key).mean()
	fearless_keys = set(rate[rate < 0.05].index)
	all_key = games.league.astype(str) + "|" + games.year.astype(str) + "|" + games.split.fillna("").astype(str)
	games["fearless"] = all_key.isin(fearless_keys)


def load_games(years: list[int] | None = None, rebuild: bool = False) -> pd.DataFrame:
	years = years or [y for y in range(2023, EVENT_YEAR + 1) if (DP_ROOT / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv").exists()]
	src_mtime = max((DP_ROOT / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv").stat().st_mtime for y in years)
	if CACHE.exists() and not rebuild:
		with open(CACHE, "rb") as f:
			cached = pickle.load(f)
		if cached["years"] == years and cached["src_mtime"] == src_mtime and cached.get("version") == CACHE_VERSION:
			return cached["games"]
	games = build_games(years)
	CACHE.parent.mkdir(parents=True, exist_ok=True)
	with open(CACHE, "wb") as f:
		pickle.dump({"years": years, "src_mtime": src_mtime, "version": CACHE_VERSION, "games": games}, f)
	return games

"""Per-patch solo-queue champion stats from u.gg's public stats CDN.

u.gg keeps champion_ranking files for every patch back to at least 13.x
(lolalytics only serves the current season), which is what the patch-shift
analysis needs. Patch strings use Riot's client numbering, which matches
Oracle's Elixir (16.20 == "26.20").

Each file: [ {role: [[champ_id, matchups, wins, matches, ...], ...]},
             {champ_id: bans, ..., "total_matches": N}, timestamp, total ]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from pickems.sync_model import ROOT

RAW = ROOT / "data" / "raw" / "ugg"
URL = "https://stats2.u.gg/lol/1.5/champion_ranking/world/{patch}/ranked_solo_5x5/{rank}/1.5.0.json"
ROLE_MAP = {"top": "top", "jungle": "jng", "mid": "mid", "adc": "bot", "supp": "sup"}
HEADERS = {"User-Agent": "Mozilla/5.0 (pickems research)"}


def champion_names() -> dict[int, str]:
	"""Riot champion key -> display name as Oracle's Elixir spells it."""
	path = RAW / "champion_ddragon.json"
	if not path.exists():
		v = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=30).json()[0]
		path.write_bytes(requests.get(f"https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json", timeout=30).content)
	data = json.loads(path.read_text())["data"]
	names = {int(c["key"]): c["name"] for c in data.values()}
	# Oracle's Elixir spellings that differ from Data Dragon display names.
	fix = {"Nunu & Willump": "Nunu", "Wukong": "Wukong", "Renata Glasc": "Renata Glasc"}
	return {k: fix.get(v, v) for k, v in names.items()}


def patch_key(patch) -> str:
	"""'16.20' -> '16_20', '16.5' -> '16_5' (u.gg doesn't zero-pad). Floats are
	read the Oracle way: 15.2 means 15.20 and 15.02 means 15.2."""
	if isinstance(patch, str):
		major, minor = patch.replace("_", ".").split(".")
		return f"{int(major)}_{int(minor)}"
	major = int(patch)
	return f"{major}_{round((patch - major) * 100)}"


def all_patches(first: str = "13.1", last: str = "16.20") -> list[str]:
	out = []
	for major in range(int(first.split(".")[0]), int(last.split(".")[0]) + 1):
		for minor in range(1, 25):
			key = f"{major}_{minor}"
			out.append(key)
			if key == patch_key(last):
				return out
	return out


def fetch(patch: str, rank: str = "emerald_plus", refresh: bool = False) -> Path | None:
	"""patch like '16.20'. Cached under data/raw/ugg; None if u.gg has no file."""
	RAW.mkdir(parents=True, exist_ok=True)
	key = patch_key(patch)
	path = RAW / f"rank_{key}_{rank}.json"
	if path.exists() and not refresh:
		return path
	r = requests.get(URL.format(patch=key, rank=rank), headers=HEADERS, timeout=60)
	if r.status_code != 200:
		return None
	path.write_bytes(r.content)
	time.sleep(0.3)
	return path


def parse(path: Path, names: dict[int, str]) -> pd.DataFrame:
	d = json.loads(path.read_text())
	roles, bans = d[0], d[1]
	total = float(bans.get("total_matches") or d[3])
	rows = []
	for role, entries in roles.items():
		for e in entries:
			cid = int(e[0])
			rows.append({"champ": names.get(cid, str(cid)), "role": ROLE_MAP.get(role, role), "wins": e[2], "games": e[3]})
	df = pd.DataFrame(rows)
	df["total_matches"] = total
	ban = {names.get(int(k), k): v for k, v in bans.items() if k != "total_matches"}
	df["bans"] = df.champ.map(ban).fillna(0.0)
	return df


def load_patches(patches: list[str], rank: str = "emerald_plus") -> pd.DataFrame:
	names = champion_names()
	frames = []
	for p in patches:
		path = fetch(p, rank)
		if path is None:
			continue
		df = parse(path, names)
		df["patch"] = p
		frames.append(df)
	out = pd.concat(frames, ignore_index=True)
	out["wr"] = out.wins / out.games.clip(lower=1)
	# A game has 10 picks; pick rate = share of games the champion appears in (this role).
	out["pick_rate"] = out.games / out.total_matches
	out["ban_rate"] = out.bans / out.total_matches
	return out

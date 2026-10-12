"""Image URLs for the dashboard: champion icons (Riot's Data Dragon), team logos
and player photos (Leaguepedia). Fetched once with `python -m pickems.images`
into data/derived/images.json; the dashboard links the URLs (nothing is
embedded) and falls back to text when a URL is missing or fails to load.

Leaguepedia's cargo tables need credentials, so this uses the plain MediaWiki
API: team logos are the "<Team>logo square.png" files, player photos the newest
"<TEAM>_<Player>_<year>_..." file on the player's page. `overrides` in
data/manual/images.yaml pins a page or file where those guesses go wrong.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from pickems.sync_model import ROOT
from pickems.teams import load_config

PATH = ROOT / "data" / "derived" / "images.json"
OVERRIDES = ROOT / "data" / "manual" / "images.yaml"
LP_API = "https://lol.fandom.com/api.php"
DDRAGON = "https://ddragon.leagueoflegends.com"
UA = {"User-Agent": "pickems-dashboard/0.1 (personal Worlds Pick'Em simulator)"}


def _get(url: str, params: dict | None = None) -> dict:
	if params:
		url += "?" + urllib.parse.urlencode(params)
	with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
		return json.load(r)


def _norm(s: str) -> str:
	return re.sub(r"[^a-z0-9]", "", s.lower())


def champion_icons() -> dict:
	"""{"base": url prefix, "ids": {normalized name: Data Dragon id}} for the newest patch
	(normalized = lowercase letters and digits only, so "Kai'Sa" -> "kaisa")."""
	ver = _get(f"{DDRAGON}/api/versions.json")[0]
	data = _get(f"{DDRAGON}/cdn/{ver}/data/en_US/champion.json")["data"]
	ids = {_norm(c["name"]): cid for cid, c in data.items()}
	ids["nunuwillump"] = "Nunu"
	return {"version": ver, "base": f"{DDRAGON}/cdn/{ver}/img/champion/", "ids": ids}


def _imageinfo(files: list[str], width: int) -> dict[str, str]:
	out = {}
	for i in range(0, len(files), 50):
		chunk = files[i:i + 50]
		d = _get(LP_API, {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url", "iiurlwidth": width,
		                  "titles": "|".join(f"File:{f}" for f in chunk)})
		norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
		for p in d["query"]["pages"].values():
			if p.get("imageinfo"):
				title = norm.get(p["title"], p["title"])
				out[title.removeprefix("File:")] = p["imageinfo"][0]["thumburl"]
		time.sleep(0.5)
	return out


SPLIT_RANK = [("split_1", 1), ("spring", 1), ("msi", 2), ("split_2", 3), ("summer", 3), ("split_3", 4), ("worlds", 5), ("wc", 5)]


def _photo_score(f: str, player: str, team: str) -> tuple | None:
	low = f.lower().replace(" ", "_")
	if f"_{player.lower().replace(' ', '_')}_" not in f"_{low}" or not low.endswith((".png", ".jpg")):
		return None
	years = re.findall(r"20\d\d", low)
	if not years:
		return None
	rank = max((r for k, r in SPLIT_RANK if k in low), default=0)
	return (int(years[-1]), low.startswith(team.lower() + "_"), rank)


def _page_photo(page: str, player: str, team: str) -> tuple | None:
	try:
		d = _get(LP_API, {"action": "parse", "format": "json", "prop": "images", "redirects": 1, "page": page})
	except Exception:  # noqa: BLE001 -- a missing photo is fine
		return None
	time.sleep(0.5)
	cands = [(s, f) for f in d.get("parse", {}).get("images", []) if (s := _photo_score(f, player, team))]
	return max(cands) if cands else None


def player_photos(teams: dict, pages: dict[str, str]) -> dict[str, str]:
	"""{"TEAM|role index": file name} -- newest photo on each player's page. When the
	name page has none (a disambiguation page), tries every "<name> (...)" page and
	keeps the one whose newest photo is most recent (current team preferred)."""
	files = {}
	for code, t in teams.items():
		for r, p in enumerate(t["players"]):
			best = _page_photo(pages.get(p, p), p, code)
			if best is None and p not in pages:
				d = _get(LP_API, {"action": "query", "format": "json", "list": "prefixsearch", "pslimit": 30, "pssearch": f"{p} ("})
				titles = [x["title"] for x in d["query"]["prefixsearch"] if x["title"].startswith(f"{p} (") and "/" not in x["title"]]
				found = [b for tt in titles if (b := _page_photo(tt, p, code))]
				best = max(found, key=lambda b: (b[0][1], b[0])) if found else None
			if best:
				files[f"{code}|{r}"] = best[1]
	return files


def fetch() -> dict:
	cfg = load_config()
	teams = cfg["teams"]
	ov = yaml.safe_load(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
	out: dict = {"champions": champion_icons()}
	logo_files = {code: ov.get("team_logos", {}).get(code, f'{t["name"]}logo square.png') for code, t in teams.items()}
	urls = _imageinfo(sorted(set(logo_files.values())), 96)
	out["teams"] = {code: urls[f] for code, f in logo_files.items() if f in urls}
	photos = player_photos(teams, ov.get("player_pages", {}))
	photos.update(ov.get("player_files", {}))
	urls = _imageinfo(sorted(set(photos.values())), 120)
	out["players"] = {k: urls[f] for k, f in photos.items() if f in urls}
	PATH.parent.mkdir(parents=True, exist_ok=True)
	PATH.write_text(json.dumps(out, indent=1))
	return out


def load() -> dict:
	return json.loads(PATH.read_text()) if PATH.exists() else {}


if __name__ == "__main__":
	res = fetch()
	teams = load_config()["teams"]
	missing_t = [c for c in teams if c not in res["teams"]]
	missing_p = [p for c, t in teams.items() for r, p in enumerate(t["players"]) if f"{c}|{r}" not in res["players"]]
	print(f"champions {len(res['champions']['ids'])}, team logos {len(res['teams'])}/{len(teams)}, player photos {len(res['players'])}")
	print("missing logos:", missing_t)
	print("missing photos:", missing_p)

"""Render a simulation run into dashboard/out/index.html (the aggregate report
JSON is embedded in the page; each simulation's single-run data is a separate
file in dashboard/out/runs/<run name>/, fetched on demand, so view the page over
http: the `dashboard` entry in .claude/launch.json). Every render also keeps a
copy as dashboard/out/runs/<run name>.html, so earlier runs stay viewable.

pair: another run's name to link from the page's Worlds buff toggle (the same
event simulated with the buff on vs off; the pipeline renders both). index:
False writes only the runs/<run name>.html copy, leaving index.html and
report.json to the main run."""

from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path

from pickems.report import build, run_champ_pick_chance, run_files
from pickems.sync_model import ROOT

TEMPLATE = ROOT / "dashboard" / "template.html"
OUT = ROOT / "dashboard" / "out" / "index.html"
VERSIONS = ROOT / "dashboard" / "versions.json"


def _clean(o):
	if isinstance(o, float):
		return None if math.isnan(o) or math.isinf(o) else round(o, 5)
	if isinstance(o, dict):
		return {str(k): _clean(v) for k, v in o.items()}
	if isinstance(o, (list, tuple)):
		return [_clean(v) for v in o]
	if hasattr(o, "item"):
		return _clean(o.item())
	return o


def champ_prior(sims: Path) -> dict | None:
	"""Chance each champion is picked at least once as the earliest snapshot (versions.json,
	before any Worlds games) forecast it, so the page can show which real picks were least
	expected. Computed from that run's .pkl in sims/ once, then cached next to it."""
	try:
		first = min(json.loads(VERSIONS.read_text())["snapshots"], key=lambda s: s.get("data_through") or "")
	except (OSError, ValueError, KeyError):
		return None
	cache = sims / f"champ_prior_{first['run_name']}.json"
	if not cache.exists():
		path = sims / f"{first['run_name']}.pkl"
		if not path.exists():
			return None
		p = run_champ_pick_chance(pickle.loads(path.read_bytes()))
		cache.write_text(json.dumps({"label": first.get("title") or first["label"], "p": p}))
	return json.loads(cache.read_text())


def render(run_path: Path, out: Path = OUT, pair: str | None = None, index: bool = True) -> Path:
	run = pickle.loads(run_path.read_bytes())
	run["name"] = run_path.stem
	if "latest_game" not in run:  # runs made before it was recorded: use the history as it is now
		from pickems.history import load_games  # noqa: PLC0415
		from pickems.teams import latest_game, resolve_teams  # noqa: PLC0415

		games = load_games()
		run["latest_game"] = latest_game(games, resolve_teams(games, run["cfg"]))
		run["data_through"] = games.date.max().isoformat()
	if "played" not in run["projection"]:  # runs made before the Worlds-game blend: add it, as of the run's data
		import pandas as pd  # noqa: PLC0415

		from pickems.history import load_games  # noqa: PLC0415
		from pickems.meta import blend_played, worlds_games  # noqa: PLC0415

		as_of = pd.Timestamp(run["data_through"]) + pd.Timedelta(seconds=1)
		run["projection"] = blend_played(run["projection"], worlds_games(load_games(), str(run["cfg"]["oracle_patch"]), as_of))
	report = build(run)
	report["patch_mix"] = run["projection"].attrs.get("pre_patches", {})
	report["patch_games"] = {k: run["projection"].attrs.get(k) for k in ("target_games", "prior_games")}
	report["meta"]["run_name"] = run_path.stem
	report["champ_prior"] = champ_prior(run_path.parent)
	report["meta"]["pair"] = pair
	# Every simulation's single-run data goes to runs/<run name>/<i>.json, shared by
	# index.html and the runs/<run name>.html copy; the page fetches one at a time
	# (so it has to be served over http, e.g. the `dashboard` launch config).
	rundir = out.parent / "runs" / run_path.stem
	shutil.rmtree(rundir, ignore_errors=True)
	rundir.mkdir(parents=True)
	for i, payload in run_files(run):
		(rundir / f"{i}.json").write_text(json.dumps(_clean(payload), separators=(",", ":")))
	payload = json.dumps(_clean(report), separators=(",", ":")).replace("</", "<\\/")
	page = TEMPLATE.read_text().replace("__REPORT__", payload)
	copy = out.parent / "runs" / f"{run_path.stem}.html"
	copy.write_text(page)
	if not index:
		return copy
	out.write_text(page)
	(out.parent / "report.json").write_text(json.dumps(_clean(report), indent=1))
	return out

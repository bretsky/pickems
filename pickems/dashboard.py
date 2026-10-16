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

from pickems.report import build, run_files
from pickems.sync_model import ROOT

TEMPLATE = ROOT / "dashboard" / "template.html"
OUT = ROOT / "dashboard" / "out" / "index.html"


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


def render(run_path: Path, out: Path = OUT, pair: str | None = None, index: bool = True) -> Path:
	run = pickle.loads(run_path.read_bytes())
	run["name"] = run_path.stem
	if "latest_game" not in run:  # runs made before it was recorded: use the history as it is now
		from pickems.history import load_games  # noqa: PLC0415
		from pickems.teams import latest_game, resolve_teams  # noqa: PLC0415

		games = load_games()
		run["latest_game"] = latest_game(games, resolve_teams(games, run["cfg"]))
		run["data_through"] = games.date.max().isoformat()
	report = build(run)
	report["patch_mix"] = run["projection"].attrs.get("pre_patches", {})
	report["patch_games"] = {k: run["projection"].attrs.get(k) for k in ("target_games", "prior_games")}
	report["meta"]["run_name"] = run_path.stem
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

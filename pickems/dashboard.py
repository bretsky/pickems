"""Render a simulation run into dashboard/out/index.html (self-contained:
the report JSON is embedded in the page)."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

from pickems.report import build
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


def render(run_path: Path, out: Path = OUT) -> Path:
	run = pickle.loads(run_path.read_bytes())
	if "latest_game" not in run:  # runs made before it was recorded: use the history as it is now
		from pickems.history import load_games  # noqa: PLC0415
		from pickems.teams import latest_game, resolve_teams  # noqa: PLC0415

		games = load_games()
		run["latest_game"] = latest_game(games, resolve_teams(games, run["cfg"]))
		run["data_through"] = games.date.max().isoformat()
	report = build(run)
	report["patch_mix"] = run["projection"].attrs.get("pre_patches", {})
	payload = json.dumps(_clean(report), separators=(",", ":")).replace("</", "<\\/")
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(TEMPLATE.read_text().replace("__REPORT__", payload))
	(out.parent / "report.json").write_text(json.dumps(_clean(report), indent=1))
	return out

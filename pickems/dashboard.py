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
	report = build(run)
	report["patch_mix"] = run["projection"].attrs.get("pre_patches", {})
	payload = json.dumps(_clean(report), separators=(",", ":")).replace("</", "<\\/")
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(TEMPLATE.read_text().replace("__REPORT__", payload))
	(out.parent / "report.json").write_text(json.dumps(_clean(report), indent=1))
	return out

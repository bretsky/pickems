"""Stage the public site and keep its list of daily snapshots (used by scripts/deploy.sh).

The live site is always the newest forecast. Each snapshot is the forecast as it stood before a
day's games, deployed once to its own Cloudflare Pages branch alias (<label>.pickems.pages.dev)
and never touched again; functions/v/[[path]].js serves it at <site>/v/<label>/. dashboard/versions.json lists them; production serves it at
/versions.json (CORS-open, so snapshots on other hosts can read it) and every page builds its
version picker from it.

  stage_site.py label [--run NAME]                      # default snapshot label for a run
  stage_site.py stage DIR [--run NAME] [--snapshot L]   # stage a run (production or snapshot L)
  stage_site.py record LABEL [--run NAME] [--title T]   # add / replace a snapshot in versions.json
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "out"
VERSIONS = ROOT / "dashboard" / "versions.json"
FAVICON = ROOT / "dashboard" / "favicon.svg"  # a file, not a data: URI, so search results can show it
SCHEDULE = max((ROOT / "data" / "manual").glob("worlds[0-9][0-9][0-9][0-9].yaml"))  # the newest Worlds (as pickems.sync_model.EVENT_YEAR)
HOME = os.environ.get("PICKEMS_HOME", "https://pickems.brettselby.xyz/")  # override to test staging locally
HEADERS = "/versions.json\n  Access-Control-Allow-Origin: *\n  Cache-Control: no-cache\n"
NOT_FOUND = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Not found | Pick'Em Lab</title>
<style>
:root { color-scheme: light dark; --bg: #f3f5f8; --ink: #141822; --muted: #5d6676; --accent: #0b7285 }
@media (prefers-color-scheme: dark) { :root { --bg: #0f131a; --ink: #e7eaf0; --muted: #98a2b3; --accent: #3bc9db } }
body { background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; margin: 0; padding: 15vh 16px; text-align: center }
h1 { font-size: 1.6rem; margin: 0 0 8px } p { color: var(--muted) } a { color: var(--accent); font-weight: 600 }
</style>
<h1>Page not found</h1>
<p>There's nothing at this address.</p>
<p><a href="/">Go to the Worlds 2026 Pick'Em Lab</a></p>
"""


def run_meta(run):
	"""meta of the named run (from its runs/<run>.html), or of report.json's run when run is None."""
	if run is None:
		return json.loads((OUT / "report.json").read_text())["meta"]
	page = (OUT / "runs" / f"{run}.html").read_text()
	m = re.search(r'<script id="report" type="application/json">(.*?)</script>', page, re.S)
	return json.loads(m.group(1))["meta"]


def default_label(meta):
	"""pre-<first scheduled day after the training data ends>: the day whose games this forecast precedes.
	data_through is UTC, and an evening's games in the Americas end after midnight UTC, so the day is
	read 12 hours earlier (games anywhere finish before noon UTC of the next day)."""
	through = (dt.datetime.fromisoformat(meta["data_through"].removesuffix("Z")) - dt.timedelta(hours=12)).date()
	days = sorted({dt.date.fromisoformat(str(d["date"])) for d in yaml.safe_load(SCHEDULE.read_text())["schedule"]})
	nxt = next((d for d in days if d > through), None)
	if nxt is None:
		sys.exit(f"no scheduled day after {through}; pass a label")
	return f"pre-{nxt.isoformat()}"


def load_versions():
	return json.loads(VERSIONS.read_text()) if VERSIONS.exists() else {"latest": "/", "snapshots": []}


def inject(page, snapshot, main=False):
	"""Tell the page where the version list lives and, on a snapshot, which one it is. Pages on the live
	site or under its /v/ proxy read the list from their own host (custom domain or pages.dev); the
	absolute URL is the fallback for a snapshot opened on its own branch alias.
	For search engines only the live main page is indexable, canonical on the custom domain (so
	pickems.pages.dev isn't a duplicate); snapshots and the runs/ pages are noindex."""
	tags = f'<meta name="pickems-versions" content="{HOME}versions.json">'
	if snapshot:
		tags += f'\n<meta name="pickems-snapshot" content="{html.escape(snapshot)}">'
	if main and not snapshot:
		tags += f'\n<link rel="canonical" href="{HOME}">\n<meta property="og:url" content="{HOME}">'
	else:
		tags += '\n<meta name="robots" content="noindex">'
	head = '<meta charset="utf-8">'
	assert head in page, "template lost its charset meta"
	return page.replace(head, head + "\n" + tags, 1)


def stage(site, run, snapshot):
	meta = run_meta(run)
	name = meta["run_name"]
	src = OUT / "index.html" if run is None else OUT / "runs" / f"{run}.html"
	if not (OUT / "runs" / name).is_dir():
		sys.exit(f"no {OUT / 'runs' / name} (re-run scripts/render.py)")
	pair = meta.get("pair")
	if pair and not (OUT / "runs" / pair).is_dir():
		sys.exit(f"no {OUT / 'runs' / pair} (re-run scripts/render.py for the pair)")

	if site.exists():
		shutil.rmtree(site)
	(site / "runs").mkdir(parents=True)
	(site / "index.html").write_text(inject(src.read_text(), snapshot, main=True))
	(site / "404.html").write_text(NOT_FOUND)  # without one, Pages answers every unknown path with index.html and a 200
	shutil.copy(FAVICON, site / "favicon.svg")
	for r in filter(None, [name, pair]):
		shutil.copytree(OUT / "runs" / r, site / "runs" / r)
		page = OUT / "runs" / f"{r}.html"
		if page.exists():
			(site / "runs" / page.name).write_text(inject(page.read_text(), snapshot))
	if snapshot is None:
		# report.json is the raw numbers of the live forecast; versions.json only lives on production.
		if run is None:
			shutil.copy(OUT / "report.json", site / "report.json")
		(site / "versions.json").write_text(json.dumps(load_versions(), indent=1) + "\n")
		(site / "_headers").write_text(HEADERS)
		(site / "_redirects").write_text("/favicon.ico /favicon.svg 301\n")  # crawlers and old browsers ask for /favicon.ico
		(site / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {HOME}sitemap.xml\n")
		lastmod = f"<lastmod>{meta['run_at'][:10]}</lastmod>" if meta.get("run_at") else ""
		(site / "sitemap.xml").write_text(
			'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
			f"<url><loc>{HOME}</loc>{lastmod}</url>\n</urlset>\n")
	files = sum(1 for p in site.rglob("*") if p.is_file())
	print(f"staged {snapshot or 'production'}: run {name}{' + pair ' + pair if pair else ''} in {site} ({files} files)", file=sys.stderr)


def record(label, run, title=None):
	meta = run_meta(run)
	v = load_versions()
	v["snapshots"] = [s for s in v["snapshots"] if s["label"] != label]
	m = re.fullmatch(r"pre-(\d{4}-\d\d-\d\d)", label)
	before = m.group(1) if m else None
	v["snapshots"].append({
		"label": label, "title": title, "before": before, "url": f"/v/{label}/", "run_name": meta["run_name"],
		"data_through": meta.get("data_through"), "run_at": meta.get("run_at"),
	})
	v["snapshots"].sort(key=lambda s: s.get("run_at") or "", reverse=True)  # newest first
	VERSIONS.write_text(json.dumps(v, indent=1) + "\n")
	print(f"recorded {label} in {VERSIONS.relative_to(ROOT)}", file=sys.stderr)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = ap.add_subparsers(dest="cmd", required=True)
	p = sub.add_parser("label")
	p.add_argument("--run")
	p = sub.add_parser("stage")
	p.add_argument("site", type=Path)
	p.add_argument("--run")
	p.add_argument("--snapshot")
	p = sub.add_parser("record")
	p.add_argument("label")
	p.add_argument("--run")
	p.add_argument("--title", help="picker name (default: 'Before <day> games' for pre-<date> labels, else the label)")
	a = ap.parse_args()
	if a.cmd == "label":
		print(default_label(run_meta(a.run)))
	elif a.cmd == "stage":
		stage(a.site, a.run, a.snapshot)
	else:
		record(a.label, a.run, a.title)


if __name__ == "__main__":
	main()

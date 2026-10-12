"""Rebuild the dashboard from an existing simulation run (no re-simulating).

  python scripts/render.py                       # newest .cache/sims/run_*.pkl
  python scripts/render.py run_2000_ww09_...     # a run by name (or path to a .pkl)
  python scripts/render.py A_nobuff --pair A --no-index   # buff-off twin: runs/A_nobuff.html only
  python scripts/render.py A --pair A_nobuff      # main page, its toggle links the twin
"""

import argparse
from pathlib import Path

from pickems.dashboard import render
from pickems.sim.runner import OUT


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="?", help="run name or .pkl path (default: newest run_*.pkl)")
	ap.add_argument("--pair", help="run the page's Worlds buff toggle links to (same event, buff on vs off)")
	ap.add_argument("--no-index", action="store_true", help="write only dashboard/out/runs/<run>.html, not index.html")
	args = ap.parse_args()
	if args.run is None:
		runs = sorted(OUT.glob("run_*.pkl"), key=lambda p: p.stat().st_mtime)
		if not runs:
			raise SystemExit(f"no run_*.pkl in {OUT}")
		path = runs[-1]
	else:
		path = Path(args.run)
		if not path.exists():
			path = OUT / (args.run.removesuffix(".pkl") + ".pkl")
	print(f"rendering {path.name}")
	print(f"dashboard: {render(path, pair=args.pair, index=not args.no_index)}")


if __name__ == "__main__":
	main()

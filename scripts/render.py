"""Rebuild the dashboard from an existing simulation run (no re-simulating).

  python scripts/render.py                       # newest .cache/sims/run_*.pkl
  python scripts/render.py run_2000_ww09_...     # a run by name (or path to a .pkl)
"""

import argparse
from pathlib import Path

from pickems.dashboard import render
from pickems.sim.runner import OUT


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="?", help="run name or .pkl path (default: newest run_*.pkl)")
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
	print(f"dashboard: {render(path)}")


if __name__ == "__main__":
	main()

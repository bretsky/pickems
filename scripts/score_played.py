"""Add the pre-game win probabilities of the Worlds games already played to an existing run (no re-simulating).

  python scripts/score_played.py [run name]      # newest run_*.pkl by default; then scripts/render.py

Reads the history as it is now, so the games priced are the ones played so far, which may be more than the run replayed."""

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run", nargs="?", help="run name or .pkl path (default: newest run_*.pkl)")
	args = ap.parse_args()
	from pickems.sim.runner import OUT  # noqa: PLC0415

	if args.run is None:
		runs = sorted((p for p in OUT.glob("run_*.pkl") if not p.stem.endswith("_nobuff")), key=lambda p: p.stat().st_mtime)
		path = runs[-1]
	else:
		path = Path(args.run)
		if not path.exists():
			path = OUT / (args.run.removesuffix(".pkl") + ".pkl")
	run = pickle.loads(path.read_bytes())
	os.environ["PICKEMS_WORLDS_WEIGHT"] = str(run.get("worlds_weight", 0.0))
	os.environ["PICKEMS_WORLDS_BUFF"] = str(run.get("worlds_buff_scale", 0.0))
	from pickems.sim.runner import predictor_before_worlds, prepare, score_played  # noqa: PLC0415

	payload = prepare(run["cfg"])
	pre = predictor_before_worlds(payload["played"])
	run["scored"] = score_played(payload, pre) if pre is not None else None
	path.write_bytes(pickle.dumps(run))
	print(f"{path.name}: priced {len(run['scored'] or [])} games")


if __name__ == "__main__":
	main()

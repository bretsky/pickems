"""Build vendor/dp_asof_<year>: a copy of vendor/dp whose Oracle's Elixir data stops
before a Worlds starts, for backtesting (PICKEMS_ASOF=<year> points the pipeline at it).

  python scripts/make_asof.py 2025 2025-10-14     # keeps games dated before 2025-10-14
  python scripts/make_asof.py 2025 2025-10-16 --mid    # same Worlds, cut after Swiss R1

--mid builds vendor/dp_asof_<year>-<mmdd> (PICKEMS_ASOF=<year>-<mmdd>) for the same
Worlds cut part-way through: the games played before the cut are in the data, so
the sims replay them as a live run does. It reuses the pre-Worlds backtest's draft
and stats models (cloned from .cache/asof_<year>, so run backtest_pipeline.sh for
the year first): they were trained without any Worlds game, as the live run's are
trained before its Worlds, and nothing after the cut leaks in.

It is a git worktree of vendor/dp (so the code is exactly the current patched
draftpredict), with the season CSVs before the cutoff year cloned, the cutoff
year's CSV filtered, later years removed (and dropped from elo.SEASON_FILES), and the untracked support files
copied. Tuned params in optuna_best_params.json were fitted on all data
(through 2026): a small leak the backtest inherits.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "vendor" / "dp"


def main(year: int, cutoff: str, mid: bool = False) -> None:
	key = f"{year}-{cutoff[5:7]}{cutoff[8:10]}" if mid else str(year)
	dest = ROOT / "vendor" / f"dp_asof_{key}"
	if not dest.exists():
		subprocess.run(["git", "-C", str(SRC), "worktree", "add", "--detach", str(dest), "HEAD"], check=True)
	for f in ("MANIFEST.json", "optuna_best_params.json", "live_margin_noise.json", "auto_draft_config.json"):
		if (SRC / f).exists():
			subprocess.run(["cp", "-c", str(SRC / f), str(dest / f)], check=True)
	for f in sorted(SRC.glob("optuna_best_params_ww*.json")):
		subprocess.run(["cp", "-c", str(f), str(dest / f.name)], check=True)
	for csv in sorted(SRC.glob("*_LoL_esports_match_data_from_OraclesElixir.csv")):
		y = int(csv.name[:4])
		out = dest / csv.name
		if y < year:
			subprocess.run(["cp", "-c", str(csv), str(out)], check=True)
		elif y == year:
			df = pd.read_csv(csv, low_memory=False)
			keep = df[pd.to_datetime(df["date"], errors="coerce") < pd.Timestamp(cutoff)]
			keep.to_csv(out, index=False)
			print(f"{csv.name}: {len(keep)} of {len(df)} rows kept")
		else:
			out.unlink(missing_ok=True)
	# draftpredict reads every file in elo.SEASON_FILES (an empty season breaks its
	# loaders), so later seasons are removed from that list in this copy only.
	elo = dest / "draftpredict" / "elo.py"
	text = elo.read_text()
	for y in range(year + 1, 2100):
		text = re.sub(rf'\t"{y}_LoL_esports_match_data_from_OraclesElixir.csv",\n', "", text)
	elo.write_text(text)
	if mid:
		src, out = ROOT / ".cache" / f"asof_{year}", ROOT / ".cache" / f"asof_{key}"
		models = sorted(src.glob("draft_model*.pkl")) + sorted(src.glob("stats_model*.pkl"))
		if not models:
			sys.exit(f"no models in {src}: run scripts/backtest_pipeline.sh {year} <pre-Worlds cut> first")
		(out / "sims").mkdir(parents=True, exist_ok=True)
		for f in models:
			subprocess.run(["cp", "-c", str(f), str(out / f.name)], check=True)
	print(f"ready: {dest} (PICKEMS_ASOF={key})")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("year", type=int)
	ap.add_argument("cutoff", help="keep games dated before this day (YYYY-MM-DD)")
	ap.add_argument("--mid", action="store_true", help="a cut part-way through that Worlds, reusing its pre-Worlds models")
	args = ap.parse_args()
	main(args.year, args.cutoff, args.mid)

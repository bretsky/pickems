"""Cut times for a past Worlds at given fractions of its games played, for
scripts/backtest_progress.sh (which calls this for any cut given as a fraction).

  .venv/bin/python scripts/backtest_cuts.py 2025 0.25 0.5     # prints one YYYY-MM-DDTHH:MM per fraction

Each cut lands after the session (games less than 6 h apart) in which that share of
the event's real games is reached, one hour after its last game, in the data's UTC
times, so a cut never splits a day's play.
"""

import sys

import pandas as pd

from pickems.history import WORLDS_FROM
from pickems.sync_model import ROOT


def cuts(year: int, fracs: list[float]) -> list[str]:
	csv = ROOT / "vendor" / "dp" / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
	df = pd.read_csv(csv, low_memory=False, usecols=["gameid", "league", "date", "position"])
	df = df[(df.league == "WLDs") & (df.date >= WORLDS_FROM[year]) & (df.position == "team")]
	t = pd.to_datetime(df.groupby("gameid").date.first()).sort_values().reset_index(drop=True)
	out = []
	for f in fracs:
		i = min(max(int(-(-f * len(t) // 1)) - 1, 0), len(t) - 1)  # the game that reaches f
		while i + 1 < len(t) and t[i + 1] - t[i] < pd.Timedelta(hours=6):
			i += 1
		out.append(f"{t[i] + pd.Timedelta(hours=1):%Y-%m-%dT%H:%M}")
	return out


if __name__ == "__main__":
	print("\n".join(cuts(int(sys.argv[1]), [float(x) for x in sys.argv[2:]])))

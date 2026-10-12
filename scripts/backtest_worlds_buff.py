"""Does the Worlds buff predict anything out of sample?

For each test year Y (Worlds 2024, 2025 and the 2026 games played so far), the
buff is built from Worlds before Y only and then scored on Y's Worlds games:
log loss of the pre-game Elo probability with every team's shift added to its
side of the logit, against the Elo alone. Reported for fixed tau across a grid,
for the tau fitted on the training Worlds (what the sim uses), and per year.
Positive "gain" = better than no buff, in thousandths of a nat per game; the
standard error is over games (pooled across years, games in a year are not
independent of each other, so treat it as a rough guide).

  .venv/bin/python scripts/backtest_worlds_buff.py
"""

import numpy as np

from pickems.history import load_games
from pickems.worlds_buff import fit_tau, shifts, team_evidence, team_rows, worlds_games

GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]


def logloss(p, y):
	p = np.clip(p, 1e-6, 1 - 1e-6)
	return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scored(wg, year, tau):
	"""Per-game log loss on `year` with the shifts built from Worlds before it (tau None: fitted)."""
	ev = team_evidence(team_rows(wg[wg.year < year]))
	tau = fit_tau(ev) if tau is None else tau
	sh = shifts(ev, tau)
	test = wg[wg.year == year]
	shift = test.blue_team.map(sh).fillna(0.0).values - test.red_team.map(sh).fillna(0.0).values
	logit = np.log(test.p_blue / (1 - test.p_blue)) + shift
	y = test.blue_win.values
	return logloss(test.p_blue.values, y), logloss(1 / (1 + np.exp(-logit)), y), tau


def main():
	wg = worlds_games(load_games())
	years = [y for y in sorted(wg.year.unique()) if y > wg.year.min()]
	print("test games:", {int(y): int((wg.year == y).sum()) for y in years})
	rows = [("fitted", None)] + [(f"tau={t:g}", t) for t in GRID]
	print(f"\n{'scheme':>10} " + " ".join(f"{'tau ' + str(int(y)):>9}" for y in years) + f" {'pooled gain':>12} {'+-se':>6}")
	for name, tau in rows:
		base, new, taus = [], [], []
		for y in years:
			b, n, t = scored(wg, y, tau)
			base.append(b), new.append(n), taus.append(t)
		d = np.concatenate(base) - np.concatenate(new)
		print(f"{name:>10} " + " ".join(f"{t:9.2f}" for t in taus) + f" {1000 * d.mean():12.2f} {1000 * d.std() / np.sqrt(len(d)):6.2f}")
	print("\nper-year gain (x1000 per game):")
	for name, tau in rows:
		print(f"{name:>10} " + " ".join(f"{int(y)}: {1000 * (lambda b, n, t: (b - n).mean())(*scored(wg, y, tau)):7.2f}" for y in years))


if __name__ == "__main__":
	main()

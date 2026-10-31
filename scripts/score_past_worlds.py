"""Result likelihood (report.likelihood_section) of a whole past Worlds, to compare with the live one's.

  python scripts/score_past_worlds.py 2024 2025 [2026]     # PICKEMS_WORLDS_WEIGHT / _BUFF / PICKEMS_WIN_TEMP as the live run (0.9 / 0.2 / 1 by default)

Each year's headline numbers (no series) are merged into data/derived/past_worlds_likelihood.json, which report.build hands the
page for the result-likelihood panel's comparison tooltips.

Like runner.score_played, but without the bracket: every real game of that year's Worlds is priced in play order by the win model
walked from draftpredict's predictor as of the event's first day (worlds buff from earlier Worlds only, series reversal on),
then fed to the walk. Teams and lineups are read off the games (most common starter per role), each team's region is its most
played domestic league that year, and the patch is the one the event was played on. Tuned params were fitted on all data, a
small leak a backtest also has."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PICKEMS_WORLDS_WEIGHT", "0.9")
os.environ.setdefault("PICKEMS_WORLDS_BUFF", "0.2")

import numpy as np  # noqa: E402

INTL = {"WLDs", "WLDs Qual", "MSI", "EWC", "AC", "FST"}


def config(games, year: int) -> dict:
	from pickems.history import ROLES, WORLDS  # noqa: PLC0415

	w = games[(games.league == WORLDS) & (games.date.dt.year == year)].sort_values("date")
	dom = games[(games.date.dt.year == year) & ~games.league.isin(INTL)]
	teams = {}
	for key in dict.fromkeys([*w.blue_team, *w.red_team]):
		name = w[w.blue_team == key].blue_name.tolist() + w[w.red_team == key].red_name.tolist()
		starters = Counter()
		for side in ("blue", "red"):
			for names in w[w[f"{side}_team"] == key][f"{side}_player_names"]:
				starters.update(enumerate(names))
		players = [max((n for (r, n) in starters if r == i), key=lambda n: starters[(i, n)]) for i in range(len(ROLES))]
		leagues = Counter(dom[dom.blue_team == key].league.tolist() + dom[dom.red_team == key].league.tolist())
		teams[name[-1]] = {"name": name[-1], "region": leagues.most_common(1)[0][0] if leagues else "WLDs", "seed": 1, "players": players}
	return {"teams": teams, "schedule": [{"date": str(w.date.min().date())}], "oracle_patch": str(w.patch.mode()[0]), "league": WORLDS}


def score(year: int) -> dict:
	from pickems.history import ROLES, load_games  # noqa: PLC0415
	from pickems.model import dp  # noqa: PLC0415
	from pickems.report import likelihood_section  # noqa: PLC0415
	from pickems.sim.played import played_series  # noqa: PLC0415
	from pickems.sim.runner import predictor_before_worlds  # noqa: PLC0415
	from pickems.sim.tournament import WinModel  # noqa: PLC0415
	from pickems.teams import resolve_teams  # noqa: PLC0415
	from pickems.worlds_buff import worlds_buffs  # noqa: PLC0415

	games = load_games()
	cfg = config(games, year)
	teams = resolve_teams(games, cfg)
	played = played_series(games, teams, cfg)
	fearless = dict(zip(games.gameid, games.fearless))
	win = WinModel(dp(), predictor_before_worlds(played), float(cfg["oracle_patch"]), cfg["league"], np.random.default_rng(0),
	               worlds_buffs(games, year), None, teams)
	win.reset()
	scored = []
	for s in played:
		win.start_day(s.games[0].date.normalize())
		win.start_series(teams[s.a], teams[s.b])
		for n, g in enumerate(s.games, 1):  # Tournament._score
			d, st = g.draft, g.stats
			won, lost = (d.blue, d.red) if g.blue_win else (d.red, d.blue)
			names = {won.code: st["win_players"], lost.code: st["lose_players"]}
			lineup = lambda t: replace(t, players=[t.key_of(r, p["name"]) for r, p in zip(ROLES, names[t.code])], player_names=[p["name"] for p in names[t.code]])
			d = replace(d, blue=lineup(d.blue), red=lineup(d.red))
			p = win.p_blue(d, bool(fearless[g.gameid]) and len(s.games) > 1)  # Bo1s aren't fearless
			scored.append({"gameid": g.gameid, "stage": "", "round": str(s.games[0].date.date()), "game": n, "blue": d.blue.code, "red": d.red.code,
			               "p_blue": float(p), "blue_win": g.blue_win})
			win.record(d, g.blue_win, st)
	return likelihood_section(scored)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("years", type=int, nargs="+")
	ap.add_argument("--out", type=Path, help="also write {year: section} with its series as JSON")
	args = ap.parse_args()
	out = {}
	for y in args.years:
		L = out[y] = score(y)
		print(f"{y}: {L['n']} games  surprise {L['pctile']:.1%}  z {L['z']:+.2f}  geo {L['geo']:.1%} vs {L['expected_geo']:.1%}  "
		      f"favorites {L['favorites']}/{L['n']} vs {L['expected_favorites']:.1f}  log10 p {L['logp'] / np.log(10):.1f}", flush=True)
		for x in L["series"][:5]:
			print(f"    {x['round']}  {x['a']} {x['wa']}-{x['wb']} {x['b']}  avg winner chance {x['geo']:.0%}")
	if args.out:
		args.out.write_text(json.dumps(out, default=float, indent=1))
	from pickems.report import PAST_LIKELIHOOD  # noqa: PLC0415
	from pickems.sim.tournament import win_temp  # noqa: PLC0415

	past = json.loads(PAST_LIKELIHOOD.read_text()) if PAST_LIKELIHOOD.exists() else {}
	past.update({str(y): {**{k: v for k, v in L.items() if k != "series"}, "win_temp": win_temp()} for y, L in out.items()})
	PAST_LIKELIHOOD.parent.mkdir(parents=True, exist_ok=True)
	PAST_LIKELIHOOD.write_text(json.dumps(dict(sorted(past.items())), indent=1))


if __name__ == "__main__":
	main()

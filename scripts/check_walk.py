"""Check pickems.sim.walk against draftpredict's own Oracle walk.

Builds draftpredict's predictor as of day D, feeds the real Oracle's Elixir
games from D up to D2 through OracleWalk, and compares every tracker with a
predictor built as of D2 (which walked those games itself). Also checks the
intraday reads: team and player Elo as OracleWalk's reader sees them before
each game vs the pre-game values draftpredict's walk recorded.

  python scripts/check_walk.py                       # Worlds 2025 Play-In + first Swiss days
  python scripts/check_walk.py --start 2026-06-26 --end 2026-07-03

Trackers OracleWalk doesn't update (lane gold-diff mastery, lane multipliers,
overpopularity slopes) are reported but not counted as failures.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from collections import ChainMap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PICKEMS_WORLDS_WEIGHT", "0")

from pickems.model import dp, dp_cwd  # noqa: E402
from pickems.sim.walk import ROLES, OracleWalk, WalkGame, walk_config  # noqa: E402

TOL = 1e-9


def _plain(v):
	if isinstance(v, ChainMap):
		return dict(v)
	return v


def _diff_dicts(name: str, got: dict, want: dict) -> list[str]:
	got, want = _plain(got), _plain(want)
	out = []
	for k in set(got) | set(want):
		a, b = got.get(k), want.get(k)
		if isinstance(a, (int, float)) and isinstance(b, (int, float)):
			if not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
				out.append(f"{name}[{k!r}]: walked {a!r} vs built {b!r}")
		elif a != b:
			out.append(f"{name}[{k!r}]: walked {a!r} vs built {b!r}")
	return out


def _diff_obj(name: str, got, want) -> list[str]:
	out = []
	for attr, b in vars(want).items():
		a = getattr(got, attr)
		if isinstance(b, (dict, ChainMap)) and b and all(hasattr(v, "__dict__") for v in _plain(b).values()):
			for k in b:
				out += _diff_obj(f"{name}.{attr}[{k!r}]", a[k], b[k])
		elif isinstance(b, (dict, ChainMap)):
			out += _diff_dicts(f"{name}.{attr}", a, b)
		elif isinstance(b, np.ndarray):
			if a.shape != b.shape or not np.allclose(a, b, rtol=TOL, atol=TOL):
				out.append(f"{name}.{attr}: arrays differ")
		elif isinstance(b, float):
			if not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
				out.append(f"{name}.{attr}: walked {a!r} vs built {b!r}")
		elif a != b:
			out.append(f"{name}.{attr}: walked {a!r} vs built {b!r}")
	return out


WALKED = ["team_tracker", "league_tracker", "player_tracker", "blue_side_tracker", "clutch_tracker", "champion_tracker", "player_champion_tracker",
          "series_streak_tracker", "series_depth_tracker", "player_series_depth_tracker", "game_length_baseline", "ally_trackers", "counter_trackers"]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--start", default="2025-10-14")
	ap.add_argument("--end", default="2025-10-20", help="exclusive")
	ap.add_argument("--cache", type=Path, default=None, help="pickle the two builds here and reuse them")
	args = ap.parse_args()
	D, D2 = pd.Timestamp(args.start), pd.Timestamp(args.end)

	predict = dp()
	from draftpredict.elo import compute_elo_features, merge_league_elo, SEASON_FILES  # noqa: PLC0415
	from draftpredict.margin_of_victory import compute_game_length_z, compute_gold_margin_d, live_margin_view  # noqa: PLC0415
	from draftpredict.player_elo import compute_player_elo_features  # noqa: PLC0415
	from draftpredict.tuned_config import feature_config_from_params, load_snapshot  # noqa: PLC0415

	cache = args.cache / f"check_walk_{D.date()}_{D2.date()}.pkl" if args.cache else None
	if cache and cache.exists():
		start, built, games, player_games = pickle.loads(cache.read_bytes())
	else:
		with dp_cwd():
			print(f"building predictor as of {D.date()} ...", flush=True)
			start = predict.build_predictor(as_of=D)
			print(f"building predictor as of {D2.date()} ...", flush=True)
			built = predict.build_predictor(as_of=D2)
			games = predict.load_team_games(SEASON_FILES)
			player_games = predict.load_player_games(SEASON_FILES)
		if cache:
			cache.parent.mkdir(parents=True, exist_ok=True)
			cache.write_bytes(pickle.dumps((start, built, games, player_games)))
	games = games[games.date < D2].reset_index(drop=True)
	player_games = player_games[player_games.gameid.isin(games.gameid)].reset_index(drop=True)

	# draftpredict's own pre-game team / player Elo for the window (intraday reads).
	config = feature_config_from_params(load_snapshot()["best_params"])
	elo_features = compute_elo_features(games, **config["elo_kwargs"])
	pg = merge_league_elo(player_games, elo_features).merge(games[["gameid", "split_blue", "game"]], on="gameid", how="left")
	player_features = compute_player_elo_features(pg, **config["player_elo_kwargs"])
	team_pre = elo_features.set_index("gameid")
	player_pre = player_features.set_index(["gameid", "position"])

	z, _ = compute_game_length_z(games)
	live_z, live_gold = live_margin_view(games, z, compute_gold_margin_d(games))
	live = dict(zip(games.gameid, zip(live_z, live_gold)))

	window = games[games.date >= D]
	roles = {gid: grp.set_index("position") for gid, grp in player_games[player_games.date >= D].groupby("gameid")}
	print(f"walking {len(window)} games over {window.date.dt.normalize().nunique()} days", flush=True)

	def walk_game(row) -> WalkGame:
		pr = roles[row.gameid]
		col = lambda c: [pr.at[r, c] for r in ROLES]
		num = lambda v: None if pd.isna(v) else float(v)
		bans = [getattr(row, f"ban{i}_{s}", None) for s in ("blue", "red") for i in range(1, 6)]
		return WalkGame(
			blue_team=row.teamid_blue, red_team=row.teamid_red, blue_players=col("player_key_blue"), red_players=col("player_key_red"),
			blue_champs=col("champion_blue"), red_champs=col("champion_red"), bans=[b for b in bans if isinstance(b, str)],
			result_blue=float(row.result_blue), league=row.league, patch=float(row.patch), game=row.game, playoffs=int(row.playoffs),
			primary_blue=row.primary_blue, primary_red=row.primary_red, is_bridge=bool(row.is_bridge), is_minor_bridge=bool(row.is_minor_bridge),
			length_s=num(getattr(row, "gamelength_blue", None)), gold_blue=num(getattr(row, "totalgold_blue", None)), gold_red=num(getattr(row, "totalgold_red", None)),
			blue_kda=list(zip(col("kills_blue"), col("deaths_blue"), col("assists_blue"))),
			red_kda=list(zip(col("kills_red"), col("deaths_red"), col("assists_red"))),
			live=live[row.gameid],
		)

	walk = OracleWalk(predict, start, np.random.default_rng(0), walk_config())
	read_errs = []
	for day, day_games in window.groupby(window.date.dt.normalize(), sort=True):
		walk.start_day(day)
		pre = {}
		for row in day_games.itertuples(index=False):  # team walk: (date, game) order
			g = walk_game(row)
			want = team_pre.loc[row.gameid]
			for side, t in (("blue", g.blue_team), ("red", g.red_team)):
				got = walk.reader.team_tracker.get(t)
				if not math.isclose(got, want[f"team_elo_{side}"], rel_tol=TOL):
					read_errs.append(f"{day.date()} {row.gameid} team_elo_{side}: reader {got} vs walk {want[f'team_elo_{side}']}")
			pre[(row.date, row.gameid)] = (g, walk.record_team(g))
		for (_, gid) in sorted(pre):  # player walks: (date, gameid) order
			g, p = pre[(_, gid)]
			for r, kb, kr in zip(ROLES, g.blue_players, g.red_players):
				want = player_pre.loc[(gid, r)]
				for side, k in (("blue", kb), ("red", kr)):
					got = walk.reader.player_tracker.get(k)
					games_got = walk.reader.player_tracker.games_played.get(k, 0)
					if not math.isclose(got, want[f"player_elo_{side}"], rel_tol=TOL) or games_got != want[f"player_games_{side}"]:
						read_errs.append(f"{day.date()} {gid} {r} player_{side}: reader {got}/{games_got}g vs walk {want[f'player_elo_{side}']}/{want[f'player_games_{side}']}g")
			walk.record_players(g, p)
	walk.start_day(D2)  # flush the last day's overpopularity batch

	errs = []
	for name in WALKED:
		got, want = getattr(walk.pred, name), getattr(built, name)
		errs += _diff_obj(name, got, want) if hasattr(want, "__dict__") and not isinstance(want, dict) else [
			e for k in want for e in _diff_obj(f"{name}[{k!r}]", got[k], want[k])]
	op_got, op_want = walk.pred.champion_overpopularity_tracker, built.champion_overpopularity_tracker
	errs += _diff_dicts("overpopularity.champion_index", op_got.champion_index, op_want.champion_index)
	for attr in ("counts", "games"):
		a, b = getattr(op_got, attr), getattr(op_want, attr)
		if a.shape != b.shape or not np.allclose(a, b, rtol=TOL, atol=TOL):
			errs.append(f"overpopularity.{attr} differs")

	frozen = []
	if built.player_champion_lane_tracker is not None:
		frozen.append(f"lane gold-diff tracker: {len(_diff_obj('lane', walk.pred.player_champion_lane_tracker, built.player_champion_lane_tracker))} entries moved in the build")
	if built.lane_multiplier_state is not None:
		d = max(float(np.abs(walk.pred.lane_multiplier_state.s_by_level[k] - v).max()) for k, v in built.lane_multiplier_state.s_by_level.items() if k in walk.pred.lane_multiplier_state.s_by_level)
		frozen.append(f"lane multipliers: max |change in log-multiplier| {d:.4f}")
	ds = [abs(op_got.power_slope[r] - op_want.power_slope[r]) for r in ROLES]
	frozen.append(f"overpopularity slopes: max change {max(ds):.2e}")

	print(f"\nintraday reads: {len(read_errs)} mismatches")
	for e in read_errs[:20]:
		print("  " + e)
	print(f"trackers: {len(errs)} mismatches")
	for e in errs[:40]:
		print("  " + e)
	print("held fixed (not walked): " + "; ".join(frozen))
	sys.exit(1 if errs or read_errs else 0)


if __name__ == "__main__":
	main()

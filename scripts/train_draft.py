"""Fit the draft model and report held-out accuracy.

Train: major-league games 2024-01-01 .. cutoff. Test: everything after cutoff
(full-candidate log loss, top-1/top-5), plus a uniform baseline for scale.
"""

import argparse
import time

import numpy as np
import pandas as pd

from pickems.draft.features import BAN_FEATURES, PICK_FEATURES
from pickems.draft.model import DraftModel, build_dataset, evaluate, fit, save
from pickems.draft.trackers import DraftState, champion_list
from pickems.history import MAJOR_LEAGUES, load_games


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cutoff", default="2026-06-01")
	ap.add_argument("--final", action="store_true", help="refit on all data after evaluating and save")
	ap.add_argument("--l2", type=float, default=1.0)
	args = ap.parse_args()
	games = load_games()
	t0 = time.time()
	state, bans, picks = build_dataset(games)
	print(f"dataset built in {time.time()-t0:.0f}s: {len(bans.start)} ban groups, {len(picks.start)} pick groups")
	cutoff = pd.Timestamp(args.cutoff)
	dates = games.date.values
	train_b = dates[bans.game_ix] < np.datetime64(cutoff)
	train_p = dates[picks.game_ix] < np.datetime64(cutoff)
	w_ban = fit(bans, train_b, args.l2)
	w_pick = fit(picks, train_p, args.l2)
	model = DraftModel(w_ban, w_pick)
	for name, w in (("BAN", dict(zip(BAN_FEATURES, w_ban))), ("PICK", dict(zip(PICK_FEATURES, w_pick)))):
		print(name, {k: round(v, 3) for k, v in w.items()})
	test = lambda g: g.date >= cutoff and g.league in MAJOR_LEAGUES
	res = evaluate(model, games, lambda: DraftState(champion_list(games)), test)
	print("held-out:", res)
	if args.final:
		w_ban = fit(bans, np.ones(len(bans.start), bool), args.l2)
		w_pick = fit(picks, np.ones(len(picks.start), bool), args.l2)
		save(DraftModel(w_ban, w_pick))
		print("saved final model")


if __name__ == "__main__":
	main()

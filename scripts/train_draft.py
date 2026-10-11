"""Fit the draft model and report held-out accuracy.

Train: major-league games 2024-01-01 .. cutoff. Test: everything after cutoff
(full-candidate log loss, top-1/top-5), plus a uniform baseline for scale.
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from pickems.draft.features import BAN_FEATURES, PICK_FEATURES
from pickems.draft.model import DraftModel, LinearScorer, build_dataset, evaluate, fit, fit_gbm, save
from pickems.sync_model import ROOT

DATASET_CACHE = ROOT / ".cache" / "draft_dataset.pkl"
from pickems.draft.trackers import DraftState, champion_list
from pickems.history import MAJOR_LEAGUES, load_games


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cutoff", default="2026-06-01")
	ap.add_argument("--final", action="store_true", help="refit on all data after evaluating and save")
	ap.add_argument("--l2", type=float, default=1.0)
	ap.add_argument("--gbm", action="store_true", help="add boosted trees on top of the linear scores")
	ap.add_argument("--rounds", type=int, default=300)
	ap.add_argument("--reuse-dataset", action="store_true")
	args = ap.parse_args()
	games = load_games()
	t0 = time.time()
	if args.reuse_dataset and DATASET_CACHE.exists():
		bans, picks = pickle.load(open(DATASET_CACHE, "rb"))
	else:
		_, bans, picks = build_dataset(games)
		pickle.dump((bans, picks), open(DATASET_CACHE, "wb"))
	print(f"dataset ready in {time.time()-t0:.0f}s: {len(bans.start)} ban groups, {len(picks.start)} pick groups")
	cutoff = pd.Timestamp(args.cutoff)
	dates = games.date.values
	train_b = dates[bans.game_ix] < np.datetime64(cutoff)
	train_p = dates[picks.game_ix] < np.datetime64(cutoff)
	w_ban = fit(bans, train_b, args.l2)
	w_pick = fit(picks, train_p, args.l2)
	model = DraftModel(LinearScorer(w_ban), LinearScorer(w_pick))
	if args.gbm:
		model = DraftModel(fit_gbm(bans, train_b, LinearScorer(w_ban), args.rounds), fit_gbm(picks, train_p, LinearScorer(w_pick), args.rounds))
	for name, w in (("BAN", dict(zip(BAN_FEATURES, w_ban))), ("PICK", dict(zip(PICK_FEATURES, w_pick)))):
		print(name, {k: round(v, 3) for k, v in w.items()})
	test = lambda g: g.date >= cutoff and g.league in MAJOR_LEAGUES
	res = evaluate(model, games, lambda: DraftState(champion_list(games)), test)
	print("held-out:", res)
	if args.final:
		all_b, all_p = np.ones(len(bans.start), bool), np.ones(len(picks.start), bool)
		lb, lp = LinearScorer(fit(bans, all_b, args.l2)), LinearScorer(fit(picks, all_p, args.l2))
		final = DraftModel(fit_gbm(bans, all_b, lb, args.rounds), fit_gbm(picks, all_p, lp, args.rounds)) if args.gbm else DraftModel(lb, lp)
		save(final)
		print("saved final model")


if __name__ == "__main__":
	main()

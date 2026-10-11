"""Fit the draft model and report held-out accuracy.

Train: major-league games 2024-01-01 .. cutoff. Test: everything after cutoff
(full-candidate log loss, top-1/top-5), also broken out for international events.

--worlds-weight w in [0, 1] reweights training: Worlds games keep weight 1,
every other game gets 1 - w (0 = all games equal, 1 = Worlds games only). A
model trained with w > 0 is saved next to the default one as draft_model_ww<w>.pkl.
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from pickems.draft.features import BAN_FEATURES, PICK_FEATURES
from pickems.draft.model import DraftModel, LinearScorer, build_dataset, evaluate, fit, fit_gbm, model_path, save
from pickems.sync_model import ROOT

DATASET_CACHE = ROOT / ".cache" / "draft_dataset.pkl"
from pickems.draft.trackers import DraftState, champion_list
from pickems.history import INTERNATIONAL, MAJOR_LEAGUES, load_games, worlds_weights


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cutoff", default="2026-06-01")
	ap.add_argument("--final", action="store_true", help="refit on all data after evaluating and save")
	ap.add_argument("--l2", type=float, default=1.0)
	ap.add_argument("--gbm", action="store_true", help="add boosted trees on top of the linear scores")
	ap.add_argument("--rounds", type=int, default=300)
	ap.add_argument("--reuse-dataset", action="store_true")
	ap.add_argument("--worlds-weight", type=float, default=0.0, help="0 = every game equal, 1 = Worlds games only")
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
	leagues = games.league.values
	gw_b, gw_p = worlds_weights(leagues[bans.game_ix], args.worlds_weight), worlds_weights(leagues[picks.game_ix], args.worlds_weight)
	if args.worlds_weight:
		print(f"worlds_weight {args.worlds_weight:g}: Worlds share of training weight {gw_b[train_b & (gw_b == 1)].sum() / gw_b[train_b].sum():.1%}")
	w_ban = fit(bans, train_b, args.l2, gw_b)
	w_pick = fit(picks, train_p, args.l2, gw_p)
	model = DraftModel(LinearScorer(w_ban), LinearScorer(w_pick))
	if args.gbm:
		model = DraftModel(fit_gbm(bans, train_b, LinearScorer(w_ban), args.rounds, weights=gw_b), fit_gbm(picks, train_p, LinearScorer(w_pick), args.rounds, weights=gw_p))
	for name, w in (("BAN", dict(zip(BAN_FEATURES, w_ban))), ("PICK", dict(zip(PICK_FEATURES, w_pick)))):
		print(name, {k: round(v, 3) for k, v in w.items()})
	test = lambda g: g.date >= cutoff and g.league in MAJOR_LEAGUES
	res = evaluate(model, games, lambda: DraftState(champion_list(games)), test, {"international": lambda g: g.league in INTERNATIONAL})
	intl = res.pop("international")
	print("held-out:", res)
	print("held-out international:", intl)
	if args.final:
		all_b, all_p = np.ones(len(bans.start), bool), np.ones(len(picks.start), bool)
		lb, lp = LinearScorer(fit(bans, all_b, args.l2, gw_b)), LinearScorer(fit(picks, all_p, args.l2, gw_p))
		final = (DraftModel(fit_gbm(bans, all_b, lb, args.rounds, weights=gw_b), fit_gbm(picks, all_p, lp, args.rounds, weights=gw_p))
		         if args.gbm else DraftModel(lb, lp))
		save(final, model_path(args.worlds_weight))
		print(f"saved final model: {model_path(args.worlds_weight)}")


if __name__ == "__main__":
	main()

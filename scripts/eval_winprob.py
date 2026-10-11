"""Walk-forward CV of draftpredict's win model under each --worlds-weight.

Same harness as draftpredict's own walk_forward_logloss (expanding window,
fold boundaries from build_fold_boundaries, scaled logistic with C=1), with the
training rows weighted by worlds_weight, scored separately on all games,
international events and Worlds. Builds the feature dataset once (~1-2 min).

  .venv/bin/python -W ignore scripts/eval_winprob.py 0 0.5 0.9 1
"""

import argparse

import numpy as np
from sklearn.metrics import log_loss

from pickems.history import INTERNATIONAL, WORLDS, worlds_weights
from pickems.model import dp, dp_cwd, fit_weighted_logistic_raw_space


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("weights", nargs="*", type=float, default=[0.0, 0.5, 0.9, 1.0])
	args = ap.parse_args()
	predict = dp()
	from draftpredict.baseline import N_FOLDS, build_baseline_dataset, build_fold_boundaries  # noqa: PLC0415
	from draftpredict.data import load_player_games, load_team_games  # noqa: PLC0415
	from draftpredict.tuned_config import feature_config_from_params, load_snapshot  # noqa: PLC0415

	with dp_cwd():
		games = load_team_games(predict.SEASON_FILES)
		player_games = load_player_games(predict.SEASON_FILES)
		snapshot = load_snapshot()
		config = feature_config_from_params(snapshot["best_params"]) if snapshot else None
		df = build_baseline_dataset(games, player_games, **(config["baseline_kwargs"] if config else {}))
	feature_cols = config["feature_cols"] if config else predict.FEATURE_COLS
	league = df.gameid.map(dict(zip(games.gameid, games.league))).values
	boundaries = build_fold_boundaries(df, N_FOLDS)
	scopes = {"all": np.ones(len(df), bool), "international": np.isin(league, list(INTERNATIONAL)), "worlds": league == WORLDS}
	X, y = df[feature_cols].values, df.result_blue.values
	# Every weight is scored on the same folds: the first whose training window
	# already holds enough Worlds games for the Worlds-only (w = 1) fit.
	first = next(i for i in range(1, len(boundaries)) if scopes["worlds"][(df.date <= boundaries[i - 1]).values].sum() >= 100)
	in_cv = (df.date > boundaries[first - 1]).values
	print(f"{len(df)} games; scoring folds {first}-{len(boundaries) - 1} (from {boundaries[first - 1].date()}): "
	      + ", ".join(f"{k} {int((m & in_cv).sum())}" for k, m in scopes.items()))
	for ww in args.weights:
		w = worlds_weights(league, ww)
		p = np.full(len(df), np.nan)
		for i in range(first, len(boundaries)):
			train = (df.date <= boundaries[i - 1]).values
			test = ((df.date > boundaries[i - 1]) & (df.date <= boundaries[i])).values
			model = fit_weighted_logistic_raw_space(X[train], y[train], w[train])
			p[test] = model.predict_proba(X[test])[:, 1]
		scored = ~np.isnan(p)
		res = {k: log_loss(y[scored & m], p[scored & m], labels=[0, 1]) for k, m in scopes.items()}
		acc = {k: ((p[scored & m] > 0.5) == y[scored & m]).mean() for k, m in scopes.items()}
		print(f"w={ww:g}: " + "  ".join(f"{k} ll {res[k]:.4f} acc {acc[k]:.1%}" for k in scopes), flush=True)


if __name__ == "__main__":
	main()

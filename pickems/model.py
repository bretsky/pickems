"""Loads the vendored draftpredict model (see sync_model.py) in-process.

draftpredict resolves its season CSVs relative to the cwd and its params/cache
relative to its package root, so everything it touches lives under vendor/dp/.
The trained Predictor is cached by draftpredict's own predictor_cache, inside
vendor/dp/.cache -- never the original project's cache. Worlds-weighted
predictors (get_predictor(worlds_weight=w)) are cached in pickems' own .cache.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from pickems.sync_model import DEST as DP_ROOT


@contextlib.contextmanager
def dp_cwd():
	prev = os.getcwd()
	os.chdir(DP_ROOT)
	try:
		yield
	finally:
		os.chdir(prev)


def _ensure_path() -> None:
	if not (DP_ROOT / "draftpredict").exists():
		raise RuntimeError("vendor/dp is empty: run `python -m pickems.sync_model` first")
	if str(DP_ROOT) not in sys.path:
		sys.path.insert(0, str(DP_ROOT))


@lru_cache(maxsize=1)
def dp():
	"""The vendored scripts/predict.py module (build_predictor, compute_features, ...)."""
	_ensure_path()
	import scripts.predict as predict  # noqa: PLC0415 -- resolved from vendor/dp

	return predict


@lru_cache(maxsize=2)
def get_predictor(force_retrain: bool = False, worlds_weight: float = 0.0):
	"""worlds_weight 0: draftpredict's own predictor, from its own cache.
	w > 0: the same build with the final logistic fit's sample weights set to
	1 for Worlds games and 1 - w for the rest (see build_weighted_predictor)."""
	predict = dp()
	if worlds_weight:
		return _weighted_predictor(worlds_weight, force_retrain)
	from draftpredict.predictor_cache import get_predictor as cached  # noqa: PLC0415

	with dp_cwd():
		return cached(predict.build_predictor, force_retrain=force_retrain, quiet=True)


def _weighted_predictor(worlds_weight: float, force_retrain: bool):
	"""Cached under pickems' .cache (never vendor/dp's predictor.pkl), keyed by
	draftpredict's own fingerprint so a re-sync or data refresh invalidates it."""
	from draftpredict.predictor_cache import _fingerprint  # noqa: PLC0415

	from pickems.history import weighted_path  # noqa: PLC0415
	from pickems.sync_model import ROOT  # noqa: PLC0415

	path = weighted_path(ROOT / ".cache" / "predictor.pkl", worlds_weight)
	tuned = tuned_snapshot(worlds_weight)
	with dp_cwd():
		fp = (_fingerprint(), tuned)
	if path.exists() and not force_retrain:
		with open(path, "rb") as f:
			cached = pickle.load(f)
		if cached["fingerprint"] == fp:
			return cached["predictor"]
	predictor = build_weighted_predictor(worlds_weight)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "wb") as f:
		pickle.dump({"fingerprint": fp, "predictor": predictor}, f)
	return predictor


def fit_weighted_logistic_raw_space(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0):
	"""draftpredict.baseline.fit_logistic_raw_space with sample weights (scaler
	and logistic both weighted); zero-weight rows are dropped."""
	from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
	from sklearn.pipeline import make_pipeline  # noqa: PLC0415
	from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

	keep = w > 0
	X, y, w = X[keep], y[keep], w[keep]
	pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
	pipe.fit(X, y, standardscaler__sample_weight=w, logisticregression__sample_weight=w)
	scaler, lr = pipe[0], pipe[-1]
	coef_raw = lr.coef_ / scaler.scale_
	lr.intercept_ = lr.intercept_ - (coef_raw * scaler.mean_).sum(axis=1)
	lr.coef_ = coef_raw
	lr.feature_scale_ = scaler.scale_.copy()
	return lr


DP_TUNE = DP_ROOT.parent / "dp_tune"


def tuned_snapshot(worlds_weight: float) -> dict | None:
	"""The Optuna snapshot a Worlds-weighted retune in vendor/dp_tune adopted
	(tune/setup_dp_tune.sh; tune_optuna_parallel.sh --snapshot) when it was run
	at this worlds_weight, else None (vendor/dp's own snapshot is used). Its
	params only mean the same thing under the code they were tuned on, so
	vendor/dp must be synced from the draftpredict commit the snapshot records."""
	import json  # noqa: PLC0415

	path = DP_TUNE / "optuna_best_params.json"
	if not path.exists():
		return None
	snap = json.loads(path.read_text())
	if snap.get("worlds_weight") != worlds_weight:
		return None
	base = snap.get("base_commit", "")
	manifest = json.loads((DP_ROOT / "MANIFEST.json").read_text())
	if manifest["commit"] != base or manifest.get("dirty"):
		raise RuntimeError(f"vendor/dp_tune's snapshot was tuned on draftpredict {base[:7] or '(unknown)'}, but vendor/dp is synced from "
		                   f"{manifest['commit'][:7]}{' (dirty)' if manifest.get('dirty') else ''}: sync that commit "
		                   f"(python -m pickems.sync_model --ref {base[:7]}) or retune")
	return snap


def build_weighted_predictor(worlds_weight: float):
	"""draftpredict's build_predictor, unchanged except that its final logistic
	fit is weighted by worlds_weight. The swap is done on the in-memory
	scripts.predict module for the duration of the build; no file under
	vendor/dp (or ~/draftpredict) is touched. Trackers (Elo, champion power,
	...) are features and are built exactly as before. When vendor/dp_tune
	holds a snapshot retuned at this worlds_weight, its params replace
	vendor/dp's (tuned_snapshot)."""
	from pickems.history import worlds_weights  # noqa: PLC0415

	predict = dp()
	orig_build, orig_fit, orig_snapshot = predict.build_baseline_dataset, predict.fit_logistic_raw_space, predict.load_snapshot
	tuned = tuned_snapshot(worlds_weight)
	seen: dict = {}

	def build(games, *args, **kwargs):
		df = orig_build(games, *args, **kwargs)
		seen["league"] = dict(zip(games.gameid, games.league))
		seen["df"] = df
		return df

	def fit(X, y, C: float = 1.0):
		df = seen["df"]
		assert len(df) == len(X), "final fit no longer runs on build_baseline_dataset's rows"
		w = worlds_weights(df.gameid.map(seen["league"]).values, worlds_weight)
		return fit_weighted_logistic_raw_space(X, y, w, C)

	predict.build_baseline_dataset, predict.fit_logistic_raw_space = build, fit
	if tuned is not None:
		predict.load_snapshot = lambda *a, **k: tuned
	try:
		with dp_cwd():
			predictor = predict.build_predictor()
	finally:
		predict.build_baseline_dataset, predict.fit_logistic_raw_space, predict.load_snapshot = orig_build, orig_fit, orig_snapshot
	predictor.worlds_weight = worlds_weight
	predictor.tuned_params = None if tuned is None else {k: tuned.get(k) for k in ("version", "best_value", "n_trials", "saved_at")}
	return predictor

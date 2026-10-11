"""Loads the vendored draftpredict model (see sync_model.py) in-process.

draftpredict resolves its season CSVs relative to the cwd and its params/cache
relative to its package root, so everything it touches lives under vendor/dp/.
The trained Predictor is cached by draftpredict's own predictor_cache, inside
vendor/dp/.cache -- never the original project's cache.

Worlds weighting lives in vendor/dp itself (tune/dp_worlds_weight.patch): it
reads DP_WORLDS_WEIGHT once, at import, and then weights its final fit, uses
that weight's retuned params (optuna_best_params_ww<w>.json) when a retune has
been adopted, and caches the predictor per weight. So one process serves one
weight: dp() sets DP_WORLDS_WEIGHT from PICKEMS_WORLDS_WEIGHT before the import.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from functools import lru_cache

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
	os.environ["DP_WORLDS_WEIGHT"] = os.environ.get("PICKEMS_WORLDS_WEIGHT", "0")
	import scripts.predict as predict  # noqa: PLC0415 -- resolved from vendor/dp

	return predict


@lru_cache(maxsize=1)
def get_predictor(force_retrain: bool = False, worlds_weight: float = 0.0):
	"""draftpredict's predictor at this process's Worlds weight (0: exactly
	upstream's), from vendor/dp's own per-weight cache."""
	predict = dp()
	from draftpredict.baseline import WORLDS_WEIGHT  # noqa: PLC0415
	from draftpredict.predictor_cache import get_predictor as cached  # noqa: PLC0415

	if WORLDS_WEIGHT != worlds_weight:
		raise RuntimeError(f"draftpredict was loaded at worlds_weight {WORLDS_WEIGHT:g} in this process, not {worlds_weight:g} "
		                   "(set PICKEMS_WORLDS_WEIGHT before anything imports it)")
	with dp_cwd():
		return cached(predict.build_predictor, force_retrain=force_retrain, quiet=True)


def tuned_params(worlds_weight: float) -> dict | None:
	"""Summary of the Worlds-retuned params a weighted predictor uses (None when
	no retune at this weight has been adopted, so upstream's params are used)."""
	path = DP_ROOT / f"optuna_best_params_ww{worlds_weight:g}.json"
	if not worlds_weight or not path.exists():
		return None
	snap = json.loads(path.read_text())
	return {k: snap.get(k) for k in ("version", "best_value", "n_trials", "saved_at", "base_commit")}

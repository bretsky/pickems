"""Loads the vendored draftpredict model (see sync_model.py) in-process.

draftpredict resolves its season CSVs relative to the cwd and its params/cache
relative to its package root, so everything it touches lives under vendor/dp/.
The trained Predictor is cached by draftpredict's own predictor_cache, inside
vendor/dp/.cache -- never the original project's cache.
"""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=1)
def get_predictor(force_retrain: bool = False):
	predict = dp()
	from draftpredict.predictor_cache import get_predictor as cached  # noqa: PLC0415

	with dp_cwd():
		return cached(predict.build_predictor, force_retrain=force_retrain, quiet=True)

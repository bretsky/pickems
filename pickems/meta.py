"""Projected Worlds-patch pro presence per champion.

Rule fitted in scripts/analysis/patch_shift.py: binomial GLM on
  logit(presence on new patch) ~ logit(presence over the last 300 major games)
                               + solo-queue dWR, dlog pick, dlog ban, WR level
with dWR etc. measured on the new patch against the patches the pre-window was
played on. Worlds patches react to solo-queue movement much more than regular
ones; by default the coefficients are the midpoint of the regular-patch fit
and the past-Worlds refit (three events is too few to take the refit alone).

Patch deltas only move presence. Solo-queue movement did not predict pro win
rate changes on the new patch, so champion strength in the win model is left
to draftpredict.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pickems.history import MAJOR_LEAGUES
from pickems.soloq import load_patches, patch_key
from pickems.sync_model import ROOT

FIT_PATH = ROOT / "data" / "derived" / "patch_shift_fit.json"
TERMS = ["const", "logit_pre", "dwr_pp", "dlog_pick", "dlog_ban", "wr_level_pp"]


def coefficients(worlds_weight: float = 0.5) -> dict[str, float]:
	fit = json.loads(FIT_PATH.read_text())
	reg, wld = fit["with_soloq"]["params"], fit["worlds_refit"]["params"]
	return {k: (1 - worlds_weight) * reg[k] + worlds_weight * wld[k] for k in TERMS}


def _presence(games: pd.DataFrame) -> pd.Series:
	cnt: dict[str, int] = {}
	for acts in games.actions:
		for c in {c for _, _, c, _ in acts if c is not None}:
			cnt[c] = cnt.get(c, 0) + 1
	return pd.Series(cnt, dtype=float) / max(len(games), 1)


def _soloq_by_champ(patches: list[str]) -> pd.DataFrame:
	sq = load_patches(patches)
	g = sq.assign(wg=sq.wr * sq.games).groupby(["patch", "champ"]).agg(wg=("wg", "sum"), games=("games", "sum"), pick=("pick_rate", "sum"), ban=("ban_rate", "max"))
	g["wr"] = g.wg / g.games.clip(lower=1)
	return g[["wr", "pick", "ban", "games"]]


def project(games: pd.DataFrame, target_patch: str, as_of: pd.Timestamp | None = None, worlds_weight: float = 0.5) -> pd.DataFrame:
	"""Per-champion table: pre presence, solo-queue deltas, projected presence."""
	major = games[games.league.isin(MAJOR_LEAGUES)]
	if as_of is not None:
		major = major[major.date < as_of]
	pre_games = major[major.date >= major.date.max() - pd.Timedelta(days=90)].tail(300)
	pre = _presence(pre_games)
	mix = pre_games.patch.dropna().map(patch_key).value_counts(normalize=True)
	tkey = patch_key(target_patch)
	mix = mix[mix.index != tkey]
	sq = _soloq_by_champ([k.replace("_", ".") for k in [*mix.index, tkey]])
	sq.index = sq.index.set_levels([sq.index.levels[0].map(patch_key), sq.index.levels[1]])
	keys = [k for k in mix.index if k in sq.index.get_level_values(0)]
	w = mix[keys] / mix[keys].sum()
	prev = sum(sq.loc[k] * wt for k, wt in w.items())
	cur = sq.loc[tkey]
	b = coefficients(worlds_weight)
	# A patch only days old carries a uniform pick/ban-rate offset vs complete
	# patches (sample composition); only relative movement is signal, so center
	# the log deltas on their games-weighted mean.
	both = cur.index.intersection(prev.dropna().index)
	wts = cur.loc[both, "games"]
	lp_off = np.average(np.log(cur.loc[both, "pick"] + 1e-4) - np.log(prev.loc[both, "pick"] + 1e-4), weights=wts)
	lb_off = np.average(np.log(cur.loc[both, "ban"] + 1e-4) - np.log(prev.loc[both, "ban"] + 1e-4), weights=wts)
	champs = sorted(set(pre.index) | set(cur.index))
	rows = []
	for c in champs:
		p0 = float(np.clip(pre.get(c, 0.0), 0.005, 0.99))
		row = {"champ": c, "pre": pre.get(c, 0.0)}
		eta = b["const"] + b["logit_pre"] * np.log(p0 / (1 - p0))
		if c in cur.index and c in prev.index and not prev.loc[c].isna().any():
			cw, pw = cur.loc[c], prev.loc[c]
			dwr = (cw.wr - pw.wr) * 100
			dlp = np.log(cw.pick + 1e-4) - np.log(pw.pick + 1e-4) - lp_off
			dlb = np.log(cw.ban + 1e-4) - np.log(pw.ban + 1e-4) - lb_off
			sq_eta = b["dwr_pp"] * dwr + b["dlog_pick"] * dlp + b["dlog_ban"] * dlb + b["wr_level_pp"] * (cw.wr - 0.5) * 100
			eta += sq_eta
			row.update({"soloq_wr": cw.wr, "soloq_dwr_pp": dwr, "soloq_dlog_pick": dlp, "soloq_dlog_ban": dlb, "soloq_logit": sq_eta})
		row["projected"] = float(1 / (1 + np.exp(-eta)))
		rows.append(row)
	out = pd.DataFrame(rows).set_index("champ")
	# Patch-driven odds multiplier alone (no regression to the mean), centered on
	# the presence-weighted average so the overall meta mass is unchanged. This
	# is what the draft simulator applies on top of its own trackers.
	sq = out.soloq_logit.fillna(0.0)
	sq -= np.average(sq, weights=out.pre + 1e-3)
	out["soloq_mult"] = np.exp(sq).clip(1 / 3, 3)
	# The GLM's intercept shrinks everything slightly; keep the total presence mass
	# of the pre-window (picks+bans per game is fixed by the draft format).
	out["projected"] *= out.pre.sum() / out.projected.sum()
	out["projected"] = out.projected.clip(upper=0.995)
	out.attrs["pre_patches"] = dict(w.round(3))
	return out.sort_values("projected", ascending=False)

"""Projected Worlds-patch pro presence per champion.

Rule fitted in scripts/analysis/patch_shift.py: binomial GLM on
  logit(presence on new patch) ~ logit(presence over the last 300 major games)
                               + solo-queue dWR, dlog pick, dlog ban, WR level
with dWR etc. measured on the new patch against the patches the pre-window was
played on. Worlds patches react to solo-queue movement much more than regular
ones; by default the coefficients are the midpoint of the regular-patch fit
and the past-Worlds refit (three events is too few to take the refit alone).

Worlds games already played on the new patch replace the projection as they
come in: the pre window holds only games from earlier patches, and the final
figure is the projection worth PRIOR_GAMES games blended with presence observed
at Worlds on the new patch (other events on it are left out). The simulator blends the same way (pickems.sim.draft.MetaTracker).

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
from pickems.sync_model import ASOF, ASOF_YEAR, ROOT

FIT_PATH = ROOT / "data" / "derived" / (f"asof_{ASOF_YEAR}" if ASOF else "") / "patch_shift_fit.json"
# How many games of the new patch the projection is worth, here and in the
# simulator's event meta.
PRIOR_GAMES = 40.0
TERMS = ["const", "logit_pre", "dwr_pp", "dlog_pick", "dlog_ban", "wr_level_pp"]


def coefficients(worlds_weight: float = 0.5) -> dict[str, float]:
	fit = json.loads(FIT_PATH.read_text())
	reg, wld = fit["with_soloq"]["params"], fit["worlds_refit"]["params"]
	return {k: (1 - worlds_weight) * reg[k] + worlds_weight * wld[k] for k in TERMS}


def presence_counts(games: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
	"""Per champion: games it was picked or banned in, and games it was locked out
	of (fearless excludes champions already picked earlier in the series)."""
	cnt: dict[str, int] = {}
	locked: dict[str, int] = {}
	for acts, prior, fearless in zip(games.actions, games.prior_picks, games.fearless):
		for c in {c for _, _, c, _ in acts if c is not None}:
			cnt[c] = cnt.get(c, 0) + 1
		if fearless:
			for c in prior:
				locked[c] = locked.get(c, 0) + 1
	return cnt, locked


def presence_and_avail(games: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Per champion: share of games it was picked or banned in among the games it
	was available for, and that availability count."""
	cnt, locked = presence_counts(games)
	avail = pd.Series({c: max(len(games) - locked.get(c, 0), 1) for c in cnt}, dtype=float)
	return pd.Series(cnt, dtype=float) / avail, avail


def _presence(games: pd.DataFrame) -> pd.Series:
	return presence_and_avail(games)[0]


def _soloq_by_champ(patches: list[str]) -> pd.DataFrame:
	sq = load_patches(patches)
	g = sq.assign(wg=sq.wr * sq.games).groupby(["patch", "champ"]).agg(wg=("wg", "sum"), games=("games", "sum"), pick=("pick_rate", "sum"), ban=("ban_rate", "max"))
	g["wr"] = g.wg / g.games.clip(lower=1)
	return g[["wr", "pick", "ban", "games"]]


def project(games: pd.DataFrame, target_patch: str, as_of: pd.Timestamp | None = None, worlds_weight: float = 0.5) -> pd.DataFrame:
	"""Per-champion table: pre presence (earlier patches), solo-queue deltas,
	presence played on the target patch so far, projected presence."""
	major = games[games.league.isin(MAJOR_LEAGUES)]
	if as_of is not None:
		major = major[major.date < as_of]
	tkey = patch_key(target_patch)
	on_target = major.patch.map(patch_key, na_action="ignore") == tkey
	# Only Worlds itself counts as the new patch's evidence: other events on the
	# patch (qualifiers, regional leftovers) draft a different meta.
	new_games, major = worlds_games(major, target_patch), major[~on_target]
	pre_games = major[major.date >= major.date.max() - pd.Timedelta(days=90)].tail(300)
	pre = _presence(pre_games)
	pre_raw = pd.Series(presence_counts(pre_games)[0], dtype=float) / max(len(pre_games), 1)
	mix = pre_games.patch.dropna().map(patch_key).value_counts(normalize=True)
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
		row = {"champ": c, "pre": pre.get(c, 0.0), "pre_raw": pre_raw.get(c, 0.0)}
		eta = b["const"] + b["logit_pre"] * np.log(p0 / (1 - p0))
		if c in cur.index and c in prev.index and not prev.loc[c].isna().any():
			cw, pw = cur.loc[c], prev.loc[c]
			dwr = (cw.wr - pw.wr) * 100
			dlp = np.log(cw.pick + 1e-4) - np.log(pw.pick + 1e-4) - lp_off
			dlb = np.log(cw.ban + 1e-4) - np.log(pw.ban + 1e-4) - lb_off
			sq_eta = b["dwr_pp"] * dwr + b["dlog_pick"] * dlp + b["dlog_ban"] * dlb + b["wr_level_pp"] * (cw.wr - 0.5) * 100
			eta += sq_eta
			row.update({"soloq_wr": cw.wr, "soloq_pick": cw.pick, "soloq_ban": cw.ban, "soloq_dwr_pp": dwr, "soloq_dlog_pick": dlp, "soloq_dlog_ban": dlb, "soloq_logit": sq_eta})
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
	return blend_played(out, new_games)


def worlds_games(games: pd.DataFrame, target_patch: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
	"""Worlds games played on the target patch (before `as_of`)."""
	g = games[(games.league == "WLDs") & (games.patch.map(patch_key, na_action="ignore") == patch_key(target_patch))]
	return g if as_of is None else g[g.date < as_of]


def blend_played(out: pd.DataFrame, new_games: pd.DataFrame) -> pd.DataFrame:
	"""Blend what the target patch has shown so far into the projection, per
	champion over the games it was available for (fearless lockouts).

	`pre`, `played` and `projected` are on that availability basis, which is
	what feeds the draft model. The `*_raw` columns are the same quantities as
	Riot reports them (picks + bans over every game, no fearless adjustment),
	for display; `projected_raw` scales `projected` by the champion's
	availability share in the pre window."""
	out = out.copy()
	cnt, locked = presence_counts(new_games)
	n = len(new_games)
	avail = pd.Series({c: n - locked.get(c, 0) for c in out.index}, dtype=float)
	played = pd.Series(cnt, dtype=float).reindex(out.index, fill_value=0.0)
	if "pre_raw" in out:
		share = (out.pre_raw / out.pre.where(out.pre > 0)).fillna(1.0)
		out["played_raw"] = played / n if n else np.nan
		out["projected_raw"] = (out.projected * share * PRIOR_GAMES + played) / (PRIOR_GAMES + n)
	out["played"] = played / avail.clip(lower=1) if n else np.nan
	out["projected"] = (out.projected * PRIOR_GAMES + played) / (PRIOR_GAMES + avail)
	out.attrs["target_games"] = n
	out.attrs["prior_games"] = PRIOR_GAMES
	return out.sort_values("projected", ascending=False)

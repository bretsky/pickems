"""How much does a new patch (and the Worlds patch in particular) move pro
drafts, and how much of that move does solo queue predict?

For every pro patch P with enough major-league games (and for each Worlds,
using the Worlds games themselves as the target):
  y        = per-champion presence (picked or banned) in games on P
  pre      = presence over the last 300 major-league games (within 90 days) before P
  soloq    = u.gg Emerald+ win rate / pick rate / ban rate on P, minus the same
             stats averaged over the patches the pre-window was played on

Fits a binomial GLM  logit(presence_P) ~ logit(pre) + dWR + dlog(pick) + dlog(ban) + WR_P
on regular patches, then checks the same fit on the Worlds targets:
residual spread, coefficient stability, and how much the soloq terms add
over "same as before" — the extrapolation rule the simulator uses for 26.20.

Writes data/derived/patch_shift_{rows,fit}.json for the dashboard.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pickems.history import MAJOR_LEAGUES, load_games
from pickems.soloq import load_patches, patch_key
from pickems.sync_model import ASOF, ROOT

OUT = ROOT / "data" / "derived" / (f"asof_{ASOF}" if ASOF else "")
WORLDS_PATCH = {2023: 13.19, 2024: 14.18, 2025: 15.20}
MIN_GAMES = 120


def presence(games: pd.DataFrame) -> pd.Series:
	cnt: dict[str, int] = {}
	for acts in games.actions:
		for c in {c for _, _, c, _ in acts if c is not None}:
			cnt[c] = cnt.get(c, 0) + 1
	return pd.Series(cnt, dtype=float) / max(len(games), 1)


def soloq_table(patches: list[str]) -> pd.DataFrame:
	sq = load_patches(patches)
	# Per champion across roles: games-weighted WR, total pick / ban rate.
	agg = sq.groupby(["patch", "champ"]).apply(
		lambda d: pd.Series({"wr": np.average(d.wr, weights=d.games.clip(lower=1)), "pick": d.pick_rate.sum(), "ban": d.ban_rate.max(), "games": d.games.sum()}),
		include_groups=False,
	).reset_index()
	return agg


def prev_patch_key(key: str) -> str:
	major, minor = map(int, key.split("_"))
	return f"{major}_{minor - 1}" if minor > 1 else f"{major - 1}_24"


def build_rows(games: pd.DataFrame) -> pd.DataFrame:
	major = games[games.league.isin(MAJOR_LEAGUES)].copy()
	major["pkey"] = major.patch.map(lambda p: patch_key(p) if pd.notna(p) else None)
	targets = []
	for pkey, grp in major.groupby("pkey"):
		dom = grp[grp.league != "WLDs"]
		if len(dom) >= MIN_GAMES:
			targets.append(("regular", pkey, dom, dom.date.min()))
	for year, wp in WORLDS_PATCH.items():
		w = major[(major.league == "WLDs") & (major.year == year) & (major.pkey == patch_key(wp))]
		targets.append(("worlds", patch_key(wp), w, w.date.min()))
	keys = sorted(set(major.pkey.dropna()))
	sq = soloq_table([k.replace("_", ".") for k in keys])
	sq["pkey"] = sq.patch.map(patch_key)
	sq = sq.set_index(["pkey", "champ"])
	rows = []
	for kind, pkey, tgt, start in targets:
		pre_games = major[(major.date < start) & (major.date >= start - pd.Timedelta(days=90))].tail(300)
		if len(pre_games) < 60 or len(tgt) < 40:
			continue
		y, pre = presence(tgt), presence(pre_games)
		mix = pre_games.pkey.value_counts(normalize=True)
		mix = mix[[k for k in mix.index if k != pkey and k in sq.index.get_level_values(0)]]
		if mix.empty:
			continue
		mix = mix / mix.sum()
		prev = sum(sq.loc[k][["wr", "pick", "ban"]] * w for k, w in mix.items())
		champs = set(y.index) | set(pre.index)
		for c in champs:
			cur = sq.loc[(pkey, c)] if (pkey, c) in sq.index else None
			prv = prev.loc[c] if c in prev.index and not prev.loc[c].isna().any() else None
			if cur is None or prv is None:
				continue
			rows.append({
				"kind": kind, "patch": pkey, "champ": c, "n": len(tgt), "y": y.get(c, 0.0), "pre": pre.get(c, 0.0),
				"wr": cur.wr, "dwr": cur.wr - prv.wr, "pre_patches": ",".join(mix.index),
				"dlpick": np.log(cur.pick + 1e-4) - np.log(prv.pick + 1e-4),
				"dlban": np.log(cur.ban + 1e-4) - np.log(prv.ban + 1e-4),
			})
	return pd.DataFrame(rows)


def design(df: pd.DataFrame, soloq: bool) -> np.ndarray:
	pre = np.clip(df.pre, 0.5 / 100, 0.99)
	cols = [np.log(pre / (1 - pre))]
	if soloq:
		cols += [df.dwr * 100, df.dlpick, df.dlban, (df.wr - 0.5) * 100]
	return sm.add_constant(np.column_stack(cols))


def binom_dev(y, n, p):
	k = y * n
	p = np.clip(p, 1e-6, 1 - 1e-6)
	return -2 * (k * np.log(p) + (n - k) * np.log(1 - p)).sum()


def main():
	games = load_games()
	df = build_rows(games)
	OUT.mkdir(parents=True, exist_ok=True)
	df.to_json(OUT / "patch_shift_rows.json", orient="records")
	reg, wld = df[df.kind == "regular"], df[df.kind == "worlds"]
	print(f"{reg.patch.nunique()} regular patches, {len(reg)} rows; worlds rows {len(wld)}")
	res = {}
	for soloq in (False, True):
		X = design(reg, soloq)
		endog = np.column_stack([reg.y * reg.n, (1 - reg.y) * reg.n])
		fit = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
		name = "with_soloq" if soloq else "pre_only"
		pw = fit.predict(design(wld, soloq))
		pr = fit.predict(X)
		res[name] = {
			"params": dict(zip(["const", "logit_pre", "dwr_pp", "dlog_pick", "dlog_ban", "wr_level_pp"][: X.shape[1]], fit.params.round(4).tolist())),
			"bse": fit.bse.round(4).tolist(),
			"regular_dev_per_row": binom_dev(reg.y.values, reg.n.values, pr) / len(reg),
			"worlds_dev_per_row": binom_dev(wld.y.values, wld.n.values, pw) / len(wld),
		}
		print(name, json.dumps(res[name], indent=1))
		# Worlds-specific refit: does the relationship itself change at Worlds?
		if soloq and len(wld):
			Xw = design(wld, True)
			fw = sm.GLM(np.column_stack([wld.y * wld.n, (1 - wld.y) * wld.n]), Xw, family=sm.families.Binomial()).fit()
			res["worlds_refit"] = {"params": dict(zip(res[name]["params"].keys(), fw.params.round(4).tolist())), "bse": fw.bse.round(4).tolist()}
			print("worlds refit", res["worlds_refit"])
	# Raw stability: correlation of presence pre vs on-patch, regular vs worlds.
	for kind, d in (("regular", reg), ("worlds", wld)):
		per = d.groupby("patch").apply(lambda g: np.corrcoef(g.pre, g.y)[0, 1], include_groups=False)
		res[f"corr_{kind}"] = {"mean": float(per.mean()), "by_patch": per.round(3).to_dict() if kind == "worlds" else None}
		res[f"mean_abs_shift_{kind}"] = float((d.y - d.pre).abs().groupby(d.patch).mean().mean())
	print({k: v for k, v in res.items() if k.startswith(("corr", "mean_abs"))})
	(OUT / "patch_shift_fit.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
	main()

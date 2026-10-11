"""Game-stat simulator: given a draft and a winner, sample game length, team
kills, per-player kills/deaths/assists, multikills, first blood, and team
objectives.

Fitted on major-league + international games (2024-01 onward). Every
team / champion / player effect is a multiplicative ratio observed/expected,
shrunk towards 1 with a pseudo-count, so thin samples stay near the role or
league baseline.

  length     log-normal; mean = base + team tendencies + champion tendencies
  team kills negative binomial; mean = f(win/loss, length) * team aggression
             (for and against) * champion kill tendencies
  K/D/A      kills split by Dirichlet-multinomial shares (role base *
             champion * player); deaths, which spread far more evenly than
             a multinomial would, by a multivariate hypergeometric draw from
             an urn of deaths + death_urn balls in those shares;
             assists ~ Binomial over the teammates' kills with a
             role/champion/player participation rate
  multikills per player-game Poisson rates in the player's kills
  objectives resampled from the historical game nearest in (length, kills)

fit(worlds_weight=w) weights every game in every aggregate: Worlds games 1,
everything else 1 - w (0 = all games equal, 1 = Worlds games only).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pickems.history import MAJOR_LEAGUES, ROLES, weighted_path, worlds_weights
from pickems.sync_model import ROOT

PATH = ROOT / ".cache" / "stats_model.pkl"
OBJ_COLS = ["dragons", "elders", "barons", "heralds", "void_grubs", "atakhans", "towers", "inhibitors"]
MULTI = ["doublekills", "triplekills", "quadrakills", "pentakills"]


def _shrunk_ratio(obs: dict, exp: dict, k: float) -> dict:
	return {key: (obs[key] + k) / (exp[key] + k) for key in obs}


@dataclass
class StatsModel:
	len_mu: float
	len_sigma: float
	len_team: dict
	len_champ: dict
	k_base: dict           # (won) -> intercept, slope on log length
	k_alpha: dict          # won -> NB dispersion
	team_kills_for: dict
	team_kills_against: dict
	champ_kills_for: dict
	role_kill_share: np.ndarray
	role_death_share: np.ndarray
	champ_kill_mult: dict  # (champ, role) -> mult
	champ_death_mult: dict
	player_kill_mult: dict
	player_death_mult: dict
	kp_role: np.ndarray
	kp_mult: dict          # (champ, role) / player -> mult on participation odds
	share_conc: float      # Dirichlet concentration of kill shares
	death_urn: float       # extra balls in the death urn (smaller = deaths spread more evenly)
	# Damage to champions and CS: log-linear per role (see _fit_loglinear), plus
	# shrunk champion-role and player effects and a per-role residual sd.
	dmg_coef: np.ndarray   # (5, 3): const, log length, log1p(kills + assists)
	dmg_champ: dict
	dmg_player: dict
	dmg_sd: np.ndarray
	cs_coef: np.ndarray    # (5, 3): const, log length, won  (on log CS per minute)
	cs_champ: dict
	cs_player: dict
	cs_sd: np.ndarray
	multi_rates: dict      # name -> (a, b) for rate = exp(a + b*log1p(kills)) * kills
	fb_winner: float
	fb_role_kill: np.ndarray
	fb_role_victim: np.ndarray
	obj_keys: np.ndarray   # (n, 3) length, winner kills, loser kills
	obj_win: np.ndarray    # (n, len(OBJ_COLS))
	obj_lose: np.ndarray
	obj_w: np.ndarray | None = None  # per bank game sampling weight (None = uniform)

	# --- sampling ---------------------------------------------------------------
	def sample(self, rng: np.random.Generator, win_team: str, lose_team: str, win_champs: list[str], lose_champs: list[str],
	           win_players: list[str], lose_players: list[str], p_win: float) -> dict:
		"""p_win: the pre-game probability the eventual winner had (stomps shorter)."""
		mu = self.len_mu + self.len_team.get(win_team, 0) + self.len_team.get(lose_team, 0)
		mu += sum(self.len_champ.get(c, 0) for c in win_champs + lose_champs)
		mu += -0.10 * (p_win - 0.5)
		length = float(np.exp(rng.normal(mu, self.len_sigma)))
		length = min(max(length, 15.0), 65.0)
		ll = np.log(length)
		out = {"length": length}
		kills = {}
		for won, team, opp, champs in ((1, win_team, lose_team, win_champs), (0, lose_team, win_team, lose_champs)):
			a, b = self.k_base[won]
			m = np.exp(a + b * ll) * self.team_kills_for.get(team, 1.0) * self.team_kills_against.get(opp, 1.0)
			m *= np.prod([self.champ_kills_for.get(c, 1.0) for c in champs]) ** 0.5
			r = 1.0 / self.k_alpha[won]
			kills[won] = int(rng.negative_binomial(r, r / (r + m)))
		out["win_kills"], out["lose_kills"] = kills[1], kills[0]
		out["win_players"] = self._players(rng, kills[1], kills[0], win_champs, win_players, length, 1)
		out["lose_players"] = self._players(rng, kills[0], kills[1], lose_champs, lose_players, length, 0)
		# First blood (only if any kill happened).
		if kills[1] + kills[0] > 0:
			fb_win = rng.random() < self.fb_winner if kills[0] and kills[1] else kills[1] > 0
			killers = out["win_players"] if fb_win else out["lose_players"]
			victims = out["lose_players"] if fb_win else out["win_players"]
			wk = np.array([pl["kills"] for pl in killers], float) * self.fb_role_kill + 1e-9
			wv = np.array([pl["deaths"] for pl in victims], float) * self.fb_role_victim + 1e-9
			killers[rng.choice(5, p=wk / wk.sum())]["firstbloodkill"] = 1
			victims[rng.choice(5, p=wv / wv.sum())]["firstbloodvictim"] = 1
			out["firstblood_winner"] = bool(fb_win)
		# Objectives from the nearest historical game.
		key = np.array([ll, np.log1p(kills[1]), np.log1p(kills[0])])
		d = ((self.obj_keys - key) ** 2 * np.array([4.0, 1.0, 1.0])).sum(axis=1)
		near = np.argpartition(d, 25)[:25]
		if self.obj_w is None:
			j = rng.choice(near)
		else:
			j = rng.choice(near, p=self.obj_w[near] / self.obj_w[near].sum())
		out["win_obj"] = dict(zip(OBJ_COLS, self.obj_win[j].tolist()))
		out["lose_obj"] = dict(zip(OBJ_COLS, self.obj_lose[j].tolist()))
		return out

	def _players(self, rng, team_kills: int, opp_kills: int, champs: list[str], players: list[str], length: float, won: int) -> list[dict]:
		wk = np.array([self.role_kill_share[r] * self.champ_kill_mult.get((c, r), 1.0) * self.player_kill_mult.get(p, 1.0) for r, (c, p) in enumerate(zip(champs, players))])
		wd = np.array([self.role_death_share[r] * self.champ_death_mult.get((c, r), 1.0) * self.player_death_mult.get(p, 1.0) for r, (c, p) in enumerate(zip(champs, players))])
		k = rng.multinomial(team_kills, rng.dirichlet(wk / wk.sum() * self.share_conc)) if team_kills else np.zeros(5, int)
		dth = rng.multivariate_hypergeometric(_urn(wd / wd.sum(), opp_kills + self.death_urn), opp_kills) if opp_kills else np.zeros(5, int)
		out = []
		for r, (c, p) in enumerate(zip(champs, players)):
			kp = self.kp_role[r]
			odds = kp / (1 - kp) * self.kp_mult.get((c, r), 1.0) * self.kp_mult.get(p, 1.0)
			kp = odds / (1 + odds)
			assists = int(rng.binomial(max(team_kills - k[r], 0), kp))
			pl = {"kills": int(k[r]), "deaths": int(dth[r]), "assists": assists, "firstbloodkill": 0, "firstbloodvictim": 0, "team_kills": team_kills}
			ll = np.log(length)
			mu = self.dmg_coef[r] @ [1.0, ll, np.log1p(k[r] + assists)] + self.dmg_champ.get((c, r), 0.0) + self.dmg_player.get(p, 0.0)
			pl["damage"] = float(np.exp(rng.normal(mu, self.dmg_sd[r])))
			mu = self.cs_coef[r] @ [1.0, ll, won] + self.cs_champ.get((c, r), 0.0) + self.cs_player.get(p, 0.0)
			pl["cs"] = int(np.exp(rng.normal(mu, self.cs_sd[r])) * length)
			for name in MULTI:
				a, b = self.multi_rates[name]
				lam = np.exp(a + b * np.log1p(k[r])) * k[r] if k[r] else 0.0
				pl[name] = int(rng.poisson(lam))
			# Nested: a penta also counts as a quadra etc. in Oracle; enforce ordering.
			for hi, lo in (("pentakills", "quadrakills"), ("quadrakills", "triplekills"), ("triplekills", "doublekills")):
				pl[lo] = max(pl[lo], pl[hi])
			out.append(pl)
		return out


def _wmean(x, w) -> float:
	return float(np.sum(w * x) / np.sum(w))


def _wvar(x, w) -> float:
	return _wmean((x - _wmean(x, w)) ** 2, w)


def fit(games: pd.DataFrame, since: str = "2024-01-01", worlds_weight: float = 0.0) -> StatsModel:
	g = games[(games.date >= since) & games.league.isin(MAJOR_LEAGUES)].copy()
	g = g[g.gamelength.notna() & (g.gamelength > 600)]
	g["wt"] = worlds_weights(g.league.values, worlds_weight)
	g_all, g = g, g[g.wt > 0]
	W = g.wt.values
	L = np.log(g.gamelength.values / 60.0)
	L_by_game = dict(zip(g.gameid, L))
	# --- length
	mu = _wmean(L, W)
	res = L - mu
	team_res, team_n = defaultdict(float), defaultdict(float)
	champ_res, champ_n = defaultdict(float), defaultdict(float)
	for r, row in zip(res, g.itertuples()):
		for t in (row.blue_team, row.red_team):
			team_res[t] += row.wt * r / 2
			team_n[t] += row.wt
		for c in row.blue_champs + row.red_champs:
			champ_res[c] += row.wt * r / 10
			champ_n[c] += row.wt
	len_team = {t: team_res[t] / (team_n[t] + 30) for t in team_res}
	len_champ = {c: champ_res[c] / (champ_n[c] + 150) for c in champ_res}
	fitted = np.array([mu + len_team[r.blue_team] + len_team[r.red_team] + sum(len_champ[c] for c in r.blue_champs + r.red_champs) for r in g.itertuples()])
	len_sigma = float(np.sqrt(_wvar(L - fitted, W)))

	# --- team kills: per (won) log-linear in log length
	recs = []
	for row, ll in zip(g.itertuples(), L):
		bk, rk = row.blue_team_stats.get("kills"), row.red_team_stats.get("kills")
		if pd.isna(bk) or pd.isna(rk):
			continue
		bw = row.blue_win
		recs.append((1 if bw else 0, ll, bk, row.blue_team, row.red_team, row.blue_champs, row))
		recs.append((0 if bw else 1, ll, rk, row.red_team, row.blue_team, row.red_champs, row))
	rec_w = np.array([r[6].wt for r in recs])
	k_base = {}
	for won in (0, 1):
		sub = [(ll, k, row.wt) for w, ll, k, *_, row in recs if w == won]
		x = np.array([s[0] for s in sub])
		y = np.log(np.array([s[1] for s in sub]) + 0.5)
		b, a = np.polyfit(x, y, 1, w=np.sqrt([s[2] for s in sub]))
		k_base[won] = (float(a), float(b))
	expected = [np.exp(k_base[w][0] + k_base[w][1] * ll) for w, ll, *_ in recs]
	# Calibrate the log-scale fit's mean to the count scale.
	ratio = _wmean(np.array([r[2] for r in recs]), rec_w) / _wmean(np.array(expected), rec_w)
	k_base = {w: (a + np.log(ratio), b) for w, (a, b) in k_base.items()}
	expected = np.array(expected) * ratio
	obs_f, exp_f, obs_a, exp_a = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
	obs_c, exp_c = defaultdict(float), defaultdict(float)
	for (w, ll, k, team, opp, champs, row), e in zip(recs, expected):
		k, e = row.wt * k, row.wt * e
		obs_f[team] += k
		exp_f[team] += e
		obs_a[opp] += k
		exp_a[opp] += e
		for c in champs:
			obs_c[c] += k
			exp_c[c] += e
	team_for = _shrunk_ratio(obs_f, exp_f, 400.0)
	team_against = _shrunk_ratio(obs_a, exp_a, 400.0)
	champ_for = _shrunk_ratio(obs_c, exp_c, 600.0)
	obs_k = np.array([r[2] for r in recs])
	won_k = np.array([r[0] for r in recs])
	# Dispersion around the mean sample() actually uses (team and champion
	# multipliers included); measured around the bare length curve it would
	# absorb that heterogeneity and sampling would then count it twice.
	full = np.array([e * team_for[team] * team_against[opp] * np.prod([champ_for[c] for c in champs]) ** 0.5
	                 for (_, _, _, team, opp, champs, _), e in zip(recs, expected)])
	k_alpha = {}
	for w in (0, 1):
		o, e, ww = obs_k[won_k == w], full[won_k == w], rec_w[won_k == w]
		# NB: Var = m + alpha * m^2, per game
		k_alpha[w] = float(max(np.sum(ww * ((o - e) ** 2 - e)) / np.sum(ww * e ** 2), 0.01))

	# --- player shares
	ks, ds = np.zeros(5), np.zeros(5)
	cko, cke, cdo, cde = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
	pko, pke, pdo, pde = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
	kp_num, kp_den = np.zeros(5), np.zeros(5)
	rows_side = []
	for row in g.itertuples():
		for side in ("blue", "red"):
			st = getattr(row, f"{side}_player_stats")
			k = np.array(st.get("kills", [np.nan] * 5), float)
			d = np.array(st.get("deaths", [np.nan] * 5), float)
			a = np.array(st.get("assists", [np.nan] * 5), float)
			if np.isnan(k).any() or np.isnan(d).any() or np.isnan(a).any():
				continue
			rows_side.append((row, side, k, d, a))
			ks += row.wt * k
			ds += row.wt * d
			tk = k.sum()
			kp_num += row.wt * a
			kp_den += row.wt * np.maximum(tk - k, 0)
	role_k = ks / ks.sum()
	role_d = ds / ds.sum()
	kp_role = kp_num / kp_den
	kpo, kpe = defaultdict(float), defaultdict(float)
	for row, side, k, d, a in rows_side:
		champs = getattr(row, f"{side}_champs")
		players = getattr(row, f"{side}_players")
		tk, td = k.sum(), d.sum()
		wt = row.wt
		for r in range(5):
			ck = (champs[r], r)
			cko[ck] += wt * k[r]
			cke[ck] += wt * tk * role_k[r]
			cdo[ck] += wt * d[r]
			cde[ck] += wt * td * role_d[r]
			pko[players[r]] += wt * k[r]
			pke[players[r]] += wt * tk * role_k[r]
			pdo[players[r]] += wt * d[r]
			pde[players[r]] += wt * td * role_d[r]
			opp_k = max(tk - k[r], 0)
			kpo[ck] += wt * a[r]
			kpe[ck] += wt * opp_k * kp_role[r]
			kpo[players[r]] += wt * a[r]
			kpe[players[r]] += wt * opp_k * kp_role[r]
	champ_kill = _shrunk_ratio(cko, cke, 40.0)
	champ_death = _shrunk_ratio(cdo, cde, 40.0)
	player_kill = _shrunk_ratio(pko, pke, 60.0)
	player_death = _shrunk_ratio(pdo, pde, 60.0)
	kp_mult = _shrunk_ratio(kpo, kpe, 60.0)

	share_conc = _dirichlet_conc(rows_side, 2, role_k, champ_kill, player_kill)
	death_urn = _fit_death_urn(rows_side, role_d, champ_death, player_death)

	# --- multikills: rate per kill as log-linear in log1p(kills)
	multi_rates = {}
	for name in MULTI:
		# Drawn from every game, weight-0 ones included, so the all-games prior
		# below exists even when the weighted fit sees only Worlds.
		xs, ys, mw = [], [], []
		for row in g_all.itertuples():
			for side in ("blue", "red"):
				st = getattr(row, f"{side}_player_stats")
				k, m = st.get("kills"), st.get(name)
				if k is None or m is None:
					continue
				for r in range(5):
					if not pd.isna(k[r]) and k[r] > 0 and not pd.isna(m[r]):
						xs.append(np.log1p(k[r]))
						ys.append((m[r], k[r]))
						mw.append(row.wt)
		xs, mw = np.array(xs), np.array(mw)
		cnt = np.array([y[0] for y in ys])
		kk = np.array([y[1] for y in ys])
		# Weighted fits shrink toward the all-games fit: with Worlds-heavy weights
		# a rare event (zero Worlds pentakills) has no information on its own.
		prior, h0 = _poisson_rate_fit(xs, kk, cnt, np.ones_like(mw))
		a, b = prior if np.allclose(mw, 1.0) else _poisson_rate_fit(xs, kk, cnt, mw, prior, h0 * (MULTI_PRIOR_EVENTS / h0[0, 0]))[0]
		multi_rates[name] = (float(a), float(b))

	dmg_fit = _fit_loglinear(rows_side, L_by_game, "damagetochampions", lambda k, a, ll, won: [1.0, ll, np.log1p(k + a)], per_minute=False)
	cs_fit = _fit_loglinear(rows_side, L_by_game, "total cs", lambda k, a, ll, won: [1.0, ll, won], per_minute=True)

	# --- first blood
	fbw, fbn = 0, 0
	fbk, fbv = np.zeros(5), np.zeros(5)
	kk_tot, dd_tot = np.zeros(5), np.zeros(5)
	for row, side, k, d, a in rows_side:
		st = getattr(row, f"{side}_player_stats")
		fk, fv = st.get("firstbloodkill"), st.get("firstbloodvictim")
		won = row.blue_win if side == "blue" else 1 - row.blue_win
		if fk is not None and not any(pd.isna(fk)):
			fk = np.array(fk, float)
			fbk += row.wt * fk
			kk_tot += row.wt * k
			if fk.sum() > 0:
				fbw += row.wt * won
				fbn += row.wt
		if fv is not None and not any(pd.isna(fv)):
			fbv += row.wt * np.array(fv, float)
			dd_tot += row.wt * d
	fb_role_kill = (fbk / fbk.sum()) / (kk_tot / kk_tot.sum())
	fb_role_victim = (fbv / fbv.sum()) / (dd_tot / dd_tot.sum())

	# --- objectives bank
	keys, win_obj, lose_obj, obj_w = [], [], [], []
	for row, ll in zip(g.itertuples(), L):
		w, l = ("blue", "red") if row.blue_win else ("red", "blue")
		ws, ls = getattr(row, f"{w}_team_stats"), getattr(row, f"{l}_team_stats")
		wv = [ws.get(c) for c in OBJ_COLS]
		lv = [ls.get(c) for c in OBJ_COLS]
		if any(pd.isna(x) for x in wv + lv) or pd.isna(ws.get("kills")) or pd.isna(ls.get("kills")):
			continue
		keys.append((ll, np.log1p(ws["kills"]), np.log1p(ls["kills"])))
		win_obj.append(wv)
		lose_obj.append(lv)
		obj_w.append(row.wt)

	return StatsModel(
		len_mu=float(mu), len_sigma=len_sigma, len_team=len_team, len_champ=len_champ,
		k_base=k_base, k_alpha=k_alpha, team_kills_for=team_for, team_kills_against=team_against, champ_kills_for=champ_for,
		role_kill_share=role_k, role_death_share=role_d, champ_kill_mult=champ_kill, champ_death_mult=champ_death,
		player_kill_mult=player_kill, player_death_mult=player_death, kp_role=kp_role, kp_mult=kp_mult, share_conc=share_conc, death_urn=death_urn,
		dmg_coef=dmg_fit[0], dmg_champ=dmg_fit[1], dmg_player=dmg_fit[2], dmg_sd=dmg_fit[3],
		cs_coef=cs_fit[0], cs_champ=cs_fit[1], cs_player=cs_fit[2], cs_sd=cs_fit[3],
		multi_rates=multi_rates, fb_winner=fbw / max(fbn, 1), fb_role_kill=fb_role_kill, fb_role_victim=fb_role_victim,
		obj_keys=np.array(keys), obj_win=np.array(win_obj, float), obj_lose=np.array(lose_obj, float),
		obj_w=None if worlds_weight == 0 else np.array(obj_w),
	)


def _urn(p: np.ndarray, size: float) -> np.ndarray:
	"""Integer ball counts in proportions p, summing to round(size)."""
	x = p * round(size)
	n = np.floor(x).astype(np.int64)
	n[np.argsort(n - x)[: round(size) - n.sum()]] += 1
	return n


def _fit_death_urn(rows_side, role_share, champ_mult, player_mult) -> float:
	"""Urn excess c for deaths: a hypergeometric draw of n from n + c balls has
	the multinomial share variance times c / (n + c - 1). Solve for the c that
	matches the observed squared share deviations summed over team-games."""
	resid, mvar, ns, ws = [], [], [], []
	for row, side, k, d, a in rows_side:
		n = d.sum()
		if n < 5:
			continue
		champs, players = getattr(row, f"{side}_champs"), getattr(row, f"{side}_players")
		w = np.array([role_share[r] * champ_mult.get((champs[r], r), 1) * player_mult.get(players[r], 1) for r in range(5)])
		p = w / w.sum()
		resid.append(((d / n - p) ** 2).mean())
		mvar.append((p * (1 - p)).mean() / n)
		ns.append(n)
		ws.append(row.wt)
	resid, mvar, ns, ws = map(np.array, (resid, mvar, ns, ws))
	gap = lambda c: np.sum(ws * (resid - mvar * c / (ns + c - 1)))
	lo, hi = 1.0, 1e4
	if gap(hi) > 0:
		return hi
	for _ in range(60):
		mid = np.sqrt(lo * hi)
		lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
	return float(lo)


def _wmedian(x, w) -> float:
	o = np.argsort(x)
	c = np.cumsum(w[o])
	return float(x[o][np.searchsorted(c, c[-1] / 2)])


MULTI_PRIOR_EVENTS = 20.0  # multikill prior strength, in all-games expected events


def _poisson_rate_fit(xs, kk, cnt, w, prior=None, prec=None):
	"""Weighted Poisson GLM rate = exp(a + b x) * kk by Newton/IRLS. With a
	prior (mean, precision matrix) it is the MAP fit, finite even when the
	weighted counts are all zero. Returns ((a, b), Hessian of the likelihood)."""
	if prior is None:
		theta = np.array([np.log((w * cnt).sum() / (w * kk).sum() + 1e-9), 0.0])
		prior, prec = np.zeros(2), np.zeros((2, 2))
	else:
		theta = np.array(prior, float)
	X = np.column_stack([np.ones_like(xs), xs])
	for _ in range(100):
		lam = np.exp(X @ theta) * kk
		g = X.T @ (w * (cnt - lam)) - prec @ (theta - prior)
		h = X.T @ (X * (w * lam)[:, None])
		step = np.linalg.solve(h + prec, g)
		theta = theta + step
		if np.abs(step).max() < 1e-10:
			break
	return theta, h


def _dirichlet_conc(rows_side, col: int, role_share, champ_mult, player_mult) -> float:
	"""Dirichlet concentration from the spread of per-player shares of the team's
	kills (col 2) or deaths (col 3) around their expectation. Shares no more
	spread than a plain multinomial get a near-infinite concentration."""
	shares, exps, share_w, ns = [], [], [], []
	for rs in rows_side:
		row, side, x = rs[0], rs[1], rs[col]
		if x.sum() >= 10:
			champs = getattr(row, f"{side}_champs")
			players = getattr(row, f"{side}_players")
			w = np.array([role_share[r] * champ_mult.get((champs[r], r), 1) * player_mult.get(players[r], 1) for r in range(5)])
			shares.append(x / x.sum())
			exps.append(w / w.sum())
			share_w.append(row.wt)
			ns.append(x.sum())
	shares, exps, share_w = np.array(shares), np.array(exps), np.array(share_w)
	n_avg = _wmean(np.array(ns), share_w)
	resid_var = _wmean(((shares - exps) ** 2).mean(axis=1), share_w)
	multinom_var = _wmean((exps * (1 - exps)).mean(axis=1), share_w) / n_avg
	# Var of DirMult share = p(1-p)/n * (n + c)/(1 + c)
	excess = max(resid_var / multinom_var, 1.01)
	return float(max((n_avg - excess) / (excess - 1), 2.0))


def _fit_loglinear(rows_side, L_by_game, stat: str, design, per_minute: bool, k_champ: float = 30.0, k_player: float = 40.0):
	"""Per role: OLS of log(stat[/min]) on design(k, a, log length, won); then
	shrunk mean residuals by (champion, role) and by player; per-role residual
	scale. The scale is robust (MAD): role swaps and off-meta picks leave a long
	low tail that would otherwise inflate a symmetric log-normal's high tail."""
	recs = []
	for row, side, k, d, a in rows_side:
		vals = getattr(row, f"{side}_player_stats").get(stat)
		if vals is None:
			continue
		ll = L_by_game[row.gameid]
		won = int(row.blue_win if side == "blue" else 1 - row.blue_win)
		champs, players = getattr(row, f"{side}_champs"), getattr(row, f"{side}_players")
		for r in range(5):
			v = vals[r]
			if pd.isna(v) or v <= 0:
				continue
			y = np.log(v / np.exp(ll)) if per_minute else np.log(v)
			recs.append((r, design(k[r], a[r], ll, won), y, champs[r], players[r], row.wt))
	nf = len(recs[0][1])
	coef, sd = np.zeros((5, nf)), np.zeros(5)
	res = np.zeros(len(recs))
	wts = np.array([x[5] for x in recs])
	for r in range(5):
		idx = [i for i, x in enumerate(recs) if x[0] == r]
		X = np.array([recs[i][1] for i in idx])
		y = np.array([recs[i][2] for i in idx])
		sw = np.sqrt(wts[idx])
		coef[r] = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
		res[idx] = y - X @ coef[r]
	champ_sum, champ_n = defaultdict(float), defaultdict(float)
	for (r, _, _, c, _, w), e in zip(recs, res):
		champ_sum[(c, r)] += w * e
		champ_n[(c, r)] += w
	champ = {key: champ_sum[key] / (champ_n[key] + k_champ) for key in champ_sum}
	res2 = np.array([e - champ[(c, r)] for (r, _, _, c, _, _), e in zip(recs, res)])
	pl_sum, pl_n = defaultdict(float), defaultdict(float)
	for (r, _, _, _, p, w), e in zip(recs, res2):
		pl_sum[p] += w * e
		pl_n[p] += w
	player = {key: pl_sum[key] / (pl_n[key] + k_player) for key in pl_sum}
	res3 = np.array([e - player[p] for (r, _, _, _, p, _), e in zip(recs, res2)])
	roles = np.array([x[0] for x in recs])
	for r in range(5):
		e, w = res3[roles == r], wts[roles == r]
		sd[r] = 1.4826 * _wmedian(np.abs(e - _wmedian(e, w)), w)
	return coef, champ, player, sd


def model_path(worlds_weight: float = 0.0):
	return weighted_path(PATH, worlds_weight)


def save(m: StatsModel, path=PATH) -> None:
	with open(path, "wb") as f:
		pickle.dump(m, f)


def load(path=PATH) -> StatsModel:
	with open(path, "rb") as f:
		return pickle.load(f)

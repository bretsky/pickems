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
  K/D/A      kills and deaths split by Dirichlet-multinomial shares
             (role base * champion * player); assists ~ Binomial over the
             teammates' kills with a role/champion/player participation rate
  multikills per player-game Poisson rates in the player's kills
  objectives resampled from the historical game nearest in (length, kills)
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pickems.history import MAJOR_LEAGUES, ROLES
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
	share_conc: float      # Dirichlet concentration
	multi_rates: dict      # name -> (a, b) for rate = exp(a + b*log1p(kills)) * kills
	fb_winner: float
	fb_role_kill: np.ndarray
	fb_role_victim: np.ndarray
	obj_keys: np.ndarray   # (n, 3) length, winner kills, loser kills
	obj_win: np.ndarray    # (n, len(OBJ_COLS))
	obj_lose: np.ndarray

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
		out["win_players"] = self._players(rng, kills[1], kills[0], win_champs, win_players)
		out["lose_players"] = self._players(rng, kills[0], kills[1], lose_champs, lose_players)
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
		j = rng.choice(near)
		out["win_obj"] = dict(zip(OBJ_COLS, self.obj_win[j].tolist()))
		out["lose_obj"] = dict(zip(OBJ_COLS, self.obj_lose[j].tolist()))
		return out

	def _players(self, rng, team_kills: int, opp_kills: int, champs: list[str], players: list[str]) -> list[dict]:
		wk = np.array([self.role_kill_share[r] * self.champ_kill_mult.get((c, r), 1.0) * self.player_kill_mult.get(p, 1.0) for r, (c, p) in enumerate(zip(champs, players))])
		wd = np.array([self.role_death_share[r] * self.champ_death_mult.get((c, r), 1.0) * self.player_death_mult.get(p, 1.0) for r, (c, p) in enumerate(zip(champs, players))])
		k = rng.multinomial(team_kills, rng.dirichlet(wk / wk.sum() * self.share_conc)) if team_kills else np.zeros(5, int)
		dth = rng.multinomial(opp_kills, rng.dirichlet(wd / wd.sum() * self.share_conc)) if opp_kills else np.zeros(5, int)
		out = []
		for r, (c, p) in enumerate(zip(champs, players)):
			kp = self.kp_role[r]
			odds = kp / (1 - kp) * self.kp_mult.get((c, r), 1.0) * self.kp_mult.get(p, 1.0)
			kp = odds / (1 + odds)
			assists = int(rng.binomial(max(team_kills - k[r], 0), kp))
			pl = {"kills": int(k[r]), "deaths": int(dth[r]), "assists": assists, "firstbloodkill": 0, "firstbloodvictim": 0}
			for name in MULTI:
				a, b = self.multi_rates[name]
				lam = np.exp(a + b * np.log1p(k[r])) * k[r] if k[r] else 0.0
				pl[name] = int(rng.poisson(lam))
			# Nested: a penta also counts as a quadra etc. in Oracle; enforce ordering.
			for hi, lo in (("pentakills", "quadrakills"), ("quadrakills", "triplekills"), ("triplekills", "doublekills")):
				pl[lo] = max(pl[lo], pl[hi])
			out.append(pl)
		return out


def fit(games: pd.DataFrame, since: str = "2024-01-01") -> StatsModel:
	g = games[(games.date >= since) & games.league.isin(MAJOR_LEAGUES)].copy()
	g = g[g.gamelength.notna() & (g.gamelength > 600)]
	L = np.log(g.gamelength.values / 60.0)
	# --- length
	mu = L.mean()
	res = L - mu
	team_res, team_n = defaultdict(float), defaultdict(int)
	champ_res, champ_n = defaultdict(float), defaultdict(int)
	for r, row in zip(res, g.itertuples()):
		for t in (row.blue_team, row.red_team):
			team_res[t] += r / 2
			team_n[t] += 1
		for c in row.blue_champs + row.red_champs:
			champ_res[c] += r / 10
			champ_n[c] += 1
	len_team = {t: team_res[t] / (team_n[t] + 30) for t in team_res}
	len_champ = {c: champ_res[c] / (champ_n[c] + 150) for c in champ_res}
	fitted = np.array([mu + len_team[r.blue_team] + len_team[r.red_team] + sum(len_champ[c] for c in r.blue_champs + r.red_champs) for r in g.itertuples()])
	len_sigma = float(np.std(L - fitted))

	# --- team kills: per (won) log-linear in log length
	recs = []
	for row, ll in zip(g.itertuples(), L):
		bk, rk = row.blue_team_stats.get("kills"), row.red_team_stats.get("kills")
		if pd.isna(bk) or pd.isna(rk):
			continue
		bw = row.blue_win
		recs.append((1 if bw else 0, ll, bk, row.blue_team, row.red_team, row.blue_champs, row))
		recs.append((0 if bw else 1, ll, rk, row.red_team, row.blue_team, row.red_champs, row))
	k_base = {}
	for won in (0, 1):
		sub = [(ll, k) for w, ll, k, *_ in recs if w == won]
		x = np.array([s[0] for s in sub])
		y = np.log(np.array([s[1] for s in sub]) + 0.5)
		b, a = np.polyfit(x, y, 1)
		k_base[won] = (float(a), float(b))
	expected = [np.exp(k_base[w][0] + k_base[w][1] * ll) for w, ll, *_ in recs]
	# Calibrate the log-scale fit's mean to the count scale.
	ratio = np.mean([r[2] for r in recs]) / np.mean(expected)
	k_base = {w: (a + np.log(ratio), b) for w, (a, b) in k_base.items()}
	expected = np.array(expected) * ratio
	obs_f, exp_f, obs_a, exp_a = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
	obs_c, exp_c = defaultdict(float), defaultdict(float)
	for (w, ll, k, team, opp, champs, _), e in zip(recs, expected):
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
	k_alpha = {}
	for w in (0, 1):
		o, e = obs_k[won_k == w], expected[won_k == w]
		k_alpha[w] = float(max((np.var(o - e) - e.mean()) / e.mean() ** 2, 0.01))

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
			ks += k
			ds += d
			tk = k.sum()
			kp_num += a
			kp_den += np.maximum(tk - k, 0)
	role_k = ks / ks.sum()
	role_d = ds / ds.sum()
	kp_role = kp_num / kp_den
	kpo, kpe = defaultdict(float), defaultdict(float)
	for row, side, k, d, a in rows_side:
		champs = getattr(row, f"{side}_champs")
		players = getattr(row, f"{side}_players")
		tk, td = k.sum(), d.sum()
		for r in range(5):
			ck = (champs[r], r)
			cko[ck] += k[r]
			cke[ck] += tk * role_k[r]
			cdo[ck] += d[r]
			cde[ck] += td * role_d[r]
			pko[players[r]] += k[r]
			pke[players[r]] += tk * role_k[r]
			pdo[players[r]] += d[r]
			pde[players[r]] += td * role_d[r]
			opp_k = max(tk - k[r], 0)
			kpo[ck] += a[r]
			kpe[ck] += opp_k * kp_role[r]
			kpo[players[r]] += a[r]
			kpe[players[r]] += opp_k * kp_role[r]
	champ_kill = _shrunk_ratio(cko, cke, 40.0)
	champ_death = _shrunk_ratio(cdo, cde, 40.0)
	player_kill = _shrunk_ratio(pko, pke, 60.0)
	player_death = _shrunk_ratio(pdo, pde, 60.0)
	kp_mult = _shrunk_ratio(kpo, kpe, 60.0)

	# Dirichlet concentration from the spread of kill shares around expectation.
	shares, exps = [], []
	for row, side, k, d, a in rows_side:
		if k.sum() >= 10:
			champs = getattr(row, f"{side}_champs")
			players = getattr(row, f"{side}_players")
			w = np.array([role_k[r] * champ_kill.get((champs[r], r), 1) * player_kill.get(players[r], 1) for r in range(5)])
			shares.append(k / k.sum())
			exps.append(w / w.sum())
	shares, exps = np.array(shares), np.array(exps)
	n_avg = np.mean([k.sum() for _, _, k, _, _ in rows_side if k.sum() >= 10])
	resid_var = ((shares - exps) ** 2).mean()
	multinom_var = (exps * (1 - exps)).mean() / n_avg
	# Var of DirMult share = p(1-p)/n * (n + c)/(1 + c)
	excess = max(resid_var / multinom_var, 1.01)
	share_conc = float(max((n_avg - excess) / (excess - 1), 2.0))

	# --- multikills: rate per kill as log-linear in log1p(kills)
	multi_rates = {}
	for name in MULTI:
		xs, ys = [], []
		for row, side, k, d, a in rows_side:
			st = getattr(row, f"{side}_player_stats").get(name)
			if st is None:
				continue
			for r in range(5):
				if k[r] > 0 and not pd.isna(st[r]):
					xs.append(np.log1p(k[r]))
					ys.append((st[r], k[r]))
		xs = np.array(xs)
		cnt = np.array([y[0] for y in ys])
		kk = np.array([y[1] for y in ys])
		# Poisson GLM via a few IRLS steps on rate = exp(a + b x) * kills.
		a, b = np.log(cnt.sum() / kk.sum() + 1e-9), 0.0
		for _ in range(50):
			lam = np.exp(a + b * xs) * kk
			ga, gb = (cnt - lam).sum(), ((cnt - lam) * xs).sum()
			haa, hab, hbb = lam.sum(), (lam * xs).sum(), (lam * xs * xs).sum()
			det = haa * hbb - hab * hab
			a += (hbb * ga - hab * gb) / det
			b += (haa * gb - hab * ga) / det
		multi_rates[name] = (float(a), float(b))

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
			fbk += fk
			kk_tot += k
			if fk.sum() > 0:
				fbw += won
				fbn += 1
		if fv is not None and not any(pd.isna(fv)):
			fbv += np.array(fv, float)
			dd_tot += d
	fb_role_kill = (fbk / fbk.sum()) / (kk_tot / kk_tot.sum())
	fb_role_victim = (fbv / fbv.sum()) / (dd_tot / dd_tot.sum())

	# --- objectives bank
	keys, win_obj, lose_obj = [], [], []
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

	return StatsModel(
		len_mu=float(mu), len_sigma=len_sigma, len_team=len_team, len_champ=len_champ,
		k_base=k_base, k_alpha=k_alpha, team_kills_for=team_for, team_kills_against=team_against, champ_kills_for=champ_for,
		role_kill_share=role_k, role_death_share=role_d, champ_kill_mult=champ_kill, champ_death_mult=champ_death,
		player_kill_mult=player_kill, player_death_mult=player_death, kp_role=kp_role, kp_mult=kp_mult, share_conc=share_conc,
		multi_rates=multi_rates, fb_winner=fbw / max(fbn, 1), fb_role_kill=fb_role_kill, fb_role_victim=fb_role_victim,
		obj_keys=np.array(keys), obj_win=np.array(win_obj, float), obj_lose=np.array(lose_obj, float),
	)


def save(m: StatsModel, path=PATH) -> None:
	with open(path, "wb") as f:
		pickle.dump(m, f)


def load(path=PATH) -> StatsModel:
	with open(path, "rb") as f:
		return pickle.load(f)

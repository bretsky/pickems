"""Game-stat simulator: given a draft and a winner, sample game length, team
kills, per-player kills/deaths/assists, CS, gold, damage, multikills, first
blood, and team objectives.

Fitted on major-league + international games (2024-01 onward). Every
team / champion / player effect is a multiplicative ratio observed/expected,
shrunk towards 1 with a pseudo-count, so thin samples stay near the role or
league baseline.

  length,    mean length = base + team + champion tendencies; mean kills =
  team kills f(win/loss, length) * team aggression (for and against) *
             champion kill tendencies. Noise is a joint bootstrap of one
             historical game's (length residual, winner and loser kill
             residual on the variance-stabilising sqrt scale), so tails and
             the length-kills correlation are the observed ones rather than
             a parametric family's
  K/D/A      kills split by Dirichlet-multinomial shares (role base *
             champion * player); deaths, which spread far more evenly than
             a multinomial would, by a multivariate hypergeometric draw from
             an urn of a * deaths + c balls, in shares tilted away from
             players who took more of the kills; assists ~ Binomial over the
             teammates' kills with a role/champion/player participation rate
  CS         team total as a log-ratio to the minions that reach one lane by
             then (lane_minions) on length, win and kill margin, capped at
             the most a team has ever farmed; split by per-player log shares
             on the player's own K/D/A, renormalised so funnelling one player
             starves the others, then capped at the most each role has ever
             farmed (water-filling the excess to teammates)
  gold, dmg  log-linear per role on own K/D/A and CS (damage on gold), so a
             player's box score moves together
  box noise  the CS (team and share), gold and damage residuals of one
             historical team-game, drawn together: the within-team farm
             trade-off and the residuals' real shapes and cross-correlations
             carry over. Gold and damage residuals are stored standardised
             by a scale fitted on the prediction (a tank's damage swings more
             in log terms than a carry's) and rescaled at the sampled game's
  multikills per player-game Poisson rates in the player's kills past the
             multikill's threshold
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
MULTI_N = {name: i + 2 for i, name in enumerate(MULTI)}  # kills one multikill takes


def lane_minions(t):
	"""Minions that have reached one lane by minute t: a 6-minion wave every 30 s
	from 0:30 (no later than the real first spawn, so an upper bound), plus a
	cannon every 3rd wave before 15:00, every 2nd before 25:00, then every wave."""
	t = np.asarray(t, float)
	n = np.floor((t - 0.5) * 2 + 1).clip(0).astype(int)  # waves spawned by t
	cannons = np.minimum(n, 30) // 3 + (np.minimum(n, 50) - np.minimum(n, 30)) // 2 + np.maximum(n - 50, 0)
	return 6 * n + cannons


def _cap(x: np.ndarray, cap: np.ndarray) -> np.ndarray:
	"""Clip x at (per-entry) cap, handing the excess to the uncapped entries in
	proportion to their values (repeat until nothing is over); the total is
	preserved unless everyone is capped."""
	x = x.astype(float).copy()
	cap = np.broadcast_to(cap, x.shape)
	for _ in range(5):
		over = x > cap
		if not over.any():
			break
		excess = (x[over] - cap[over]).sum()
		x[over] = cap[over]
		free = ~over & (x < cap)
		if not free.any() or x[free].sum() <= 0:
			break
		x[free] += excess * x[free] / x[free].sum()
	return np.minimum(x, cap)


def _shrunk_ratio(obs: dict, exp: dict, k: float) -> dict:
	return {key: (obs[key] + k) / (exp[key] + k) for key in obs}


@dataclass
class StatsModel:
	len_mu: float
	len_team: dict
	len_champ: dict
	k_base: dict           # (won) -> intercept, slope on log length
	team_kills_for: dict
	team_kills_against: dict
	champ_kills_for: dict
	k_gamma: dict          # won -> exponents on the team (for * against) and champion-product multipliers
	bank: np.ndarray       # (n, 3): length log-residual, winner / loser sqrt(kills) - sqrt(mean)
	bank_p: np.ndarray     # sampling probability of each bank game
	role_kill_share: np.ndarray
	role_death_share: np.ndarray
	champ_kill_mult: dict  # (champ, role) -> mult
	champ_death_mult: dict
	player_kill_mult: dict
	player_death_mult: dict
	kp_role: np.ndarray
	kp_mult: dict          # (champ, role) / player -> mult on participation odds
	share_conc: float      # Dirichlet concentration of kill shares
	death_urn: tuple       # (a, c): urn of a * deaths + c balls (smaller = deaths spread more evenly)
	death_gamma: float     # exponent on the champion * player death multipliers
	death_slope: np.ndarray  # (5,) change in a player's death share per unit of excess kill share
	# Team CS: log(team CS / lane_minions) on const, log length, won, kill
	# margin, team deaths per minute.
	team_cs_coef: np.ndarray
	team_cs_team: dict
	team_cs_cap: float     # max team CS / lane_minions
	player_cs_cap: np.ndarray  # (5,) max player CS / lane_minions, by role
	box_bank: np.ndarray   # (n, 16): one team-game's team CS residual, then 5 each of CS share and standardised gold, damage residuals
	box_bank_p: np.ndarray
	# Per role, log-linear (see _fit_loglinear) with shrunk champion-role and
	# player effects; residuals come from box_bank.
	# Deaths enter linearly: every death is a respawn timer of lost farm.
	cs_coef: np.ndarray    # log CS share: const, log length, won, log1p k, deaths, log1p a
	cs_champ: dict
	cs_player: dict
	cs_het: np.ndarray     # (5, 3) residual scale = max(a + b * prediction, floor); unit for CS
	gold_coef: np.ndarray  # log gold: const, log length, won, log1p k, deaths, log1p a, log1p CS
	gold_champ: dict
	gold_player: dict
	gold_het: np.ndarray
	dmg_coef: np.ndarray   # log damage: const, log length, won, log1p(k + a), deaths, log gold
	dmg_champ: dict
	dmg_player: dict
	dmg_het: np.ndarray
	multi_rates: dict      # name -> (a, b) for rate = exp(a) * (kills - n + 1)^b, n = MULTI_N[name]
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
		eps, z_w, z_l = self.bank[rng.choice(len(self.bank), p=self.bank_p)]
		length = min(max(float(np.exp(mu + eps)), 15.0), 65.0)
		ll = np.log(length)
		out = {"length": length}
		kills = {}
		for won, team, opp, champs, z in ((1, win_team, lose_team, win_champs, z_w), (0, lose_team, win_team, lose_champs, z_l)):
			a, b = self.k_base[won]
			gt, gc = self.k_gamma[won]
			m = np.exp(a + b * ll) * (self.team_kills_for.get(team, 1.0) * self.team_kills_against.get(opp, 1.0)) ** gt
			m *= np.prod([self.champ_kills_for.get(c, 1.0) for c in champs]) ** gc
			x = max(np.sqrt(m) + z, 0.0) ** 2
			kills[won] = int(x) + int(rng.random() < x % 1)
		out["win_kills"], out["lose_kills"] = kills[1], kills[0]
		out["win_players"] = self._players(rng, kills[1], kills[0], win_team, win_champs, win_players, length, 1)
		out["lose_players"] = self._players(rng, kills[0], kills[1], lose_team, lose_champs, lose_players, length, 0)
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

	def _players(self, rng, team_kills: int, opp_kills: int, team: str, champs: list[str], players: list[str], length: float, won: int) -> list[dict]:
		wk = np.array([self.role_kill_share[r] * self.champ_kill_mult.get((c, r), 1.0) * self.player_kill_mult.get(p, 1.0) for r, (c, p) in enumerate(zip(champs, players))])
		wd = np.array([self.role_death_share[r] * (self.champ_death_mult.get((c, r), 1.0) * self.player_death_mult.get(p, 1.0)) ** self.death_gamma for r, (c, p) in enumerate(zip(champs, players))])
		pk = wk / wk.sum()
		k = rng.multinomial(team_kills, rng.dirichlet(pk * self.share_conc)) if team_kills else np.zeros(5, int)
		q = _death_shares(wd / wd.sum(), k, pk, self.death_slope)
		ua, uc = self.death_urn
		dth = rng.multivariate_hypergeometric(_urn(q, ua * opp_kills + uc), opp_kills) if opp_kills else np.zeros(5, int)
		ast = np.zeros(5, int)
		for r, (c, p) in enumerate(zip(champs, players)):
			kp = self.kp_role[r]
			odds = kp / (1 - kp) * self.kp_mult.get((c, r), 1.0) * self.kp_mult.get(p, 1.0)
			ast[r] = rng.binomial(max(team_kills - k[r], 0), odds / (1 + odds))
		ll = np.log(length)
		# CS: a bounded team total, then shares that follow each player's game.
		lm = float(lane_minions(length))
		y = self.team_cs_coef @ [1.0, ll, won, (team_kills - opp_kills) / (team_kills + opp_kills + 1), opp_kills / length] + self.team_cs_team.get(team, 0.0)
		noise = self.box_bank[rng.choice(len(self.box_bank), p=self.box_bank_p)]
		team_cs = min(np.exp(y + noise[0]), self.team_cs_cap) * lm
		mu = np.array([self.cs_coef[r] @ [1.0, ll, won, np.log1p(k[r]), dth[r], np.log1p(ast[r])] + self.cs_champ.get((c, r), 0.0) + self.cs_player.get(p, 0.0)
		               for r, (c, p) in enumerate(zip(champs, players))])
		share = np.exp(mu + noise[1:6] * _het_scale(self.cs_het, mu))
		cs = np.round(_cap(team_cs * share / share.sum(), self.player_cs_cap * lm)).astype(int)
		out = []
		for r, (c, p) in enumerate(zip(champs, players)):
			pl = {"kills": int(k[r]), "deaths": int(dth[r]), "assists": int(ast[r]), "cs": int(cs[r]),
			      "firstbloodkill": 0, "firstbloodvictim": 0, "team_kills": team_kills}
			lk, ld, la = np.log1p(k[r]), dth[r], np.log1p(ast[r])
			mu = self.gold_coef[r] @ [1.0, ll, won, lk, ld, la, np.log1p(cs[r])] + self.gold_champ.get((c, r), 0.0) + self.gold_player.get(p, 0.0)
			pl["gold"] = float(np.exp(mu + noise[6 + r] * _het_scale(self.gold_het[r], mu)))
			mu = self.dmg_coef[r] @ [1.0, ll, won, np.log1p(k[r] + ast[r]), ld, np.log(pl["gold"])] + self.dmg_champ.get((c, r), 0.0) + self.dmg_player.get(p, 0.0)
			pl["damage"] = float(np.exp(mu + noise[11 + r] * _het_scale(self.dmg_het[r], mu)))
			for name in MULTI:
				a, b = self.multi_rates[name]
				n = MULTI_N[name]
				# Impossible below n kills; at most k // n disjoint n-kill streaks.
				pl[name] = min(int(rng.poisson(np.exp(a + b * np.log(k[r] - n + 1)))), int(k[r]) // n) if k[r] >= n else 0
			# Nested: a penta also counts as a quadra etc. in Oracle; enforce ordering.
			for hi, lo in (("pentakills", "quadrakills"), ("quadrakills", "triplekills"), ("triplekills", "doublekills")):
				pl[lo] = max(pl[lo], pl[hi])
			out.append(pl)
		return out


def _het_scale(het: np.ndarray, mu):
	"""Residual scale at prediction mu: het is (a, b, floor) for one role, or
	(5, 3) with mu per role."""
	het = np.asarray(het)
	return np.maximum(het[..., 0] + het[..., 1] * mu, het[..., 2])


def _death_shares(p_death: np.ndarray, k: np.ndarray, p_kill: np.ndarray, slope: np.ndarray) -> np.ndarray:
	"""Expected death shares given how the team's kills were split: a player
	with more than their expected share of kills dies less."""
	tk = k.sum()
	x = k / tk - p_kill if tk else np.zeros(5)
	q = np.clip(p_death + slope * x, 0.01, None)
	return q / q.sum()


def _wmean(x, w) -> float:
	return float(np.sum(w * x) / np.sum(w))


def fit(games: pd.DataFrame, since: str = "2024-01-01", worlds_weight: float = 0.0) -> StatsModel:
	g = games[(games.date >= since) & games.league.isin(MAJOR_LEAGUES)].copy()
	g = g[g.gamelength.notna() & (g.gamelength > 600)]
	g["wt"] = worlds_weights(g.league.values, worlds_weight)
	g_all, g = g, g[g.wt > 0]
	W = g.wt.values
	L = np.log(g.gamelength.values / 60.0)
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
	eps_by_game = dict(zip(g.gameid, L - fitted))

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
	# How much of each tendency shows up in a game: the raw ratios are noisy
	# and overlap (teams pick their champions), so taken at face value they
	# overstate the spread of means. Poisson fit of kills on expected *
	# exp(c + gt * log team ratio + gc * log champion product), per win/loss,
	# exponents clipped to [0, 1] and c refit to keep the mean.
	X = np.column_stack([np.ones(len(recs)),
	                     [np.log(team_for[team] * team_against[opp]) for _, _, _, team, opp, _, _ in recs],
	                     [np.sum(np.log([champ_for[c] for c in champs])) for *_, champs, _ in recs]])
	obs_k = np.array([r[2] for r in recs], float)
	won_k = np.array([r[0] for r in recs])
	full, k_gamma = np.zeros(len(recs)), {}
	for w in (0, 1):
		i = won_k == w
		theta, _ = _poisson_fit(X[i], expected[i], obs_k[i], rec_w[i])
		# At most face value: thin (Worlds-heavy) fits otherwise read the
		# shrinkage as under-spread and amplify noisy ratios.
		theta[1:] = np.clip(theta[1:], 0.0, 1.0)
		theta[0] = np.log(np.sum(rec_w[i] * obs_k[i]) / np.sum(rec_w[i] * expected[i] * np.exp(X[i, 1:] @ theta[1:])))
		k_base[w] = (k_base[w][0] + theta[0], k_base[w][1])
		k_gamma[w] = (float(theta[1]), float(theta[2]))
		full[i] = expected[i] * np.exp(X[i] @ theta)
	# Bootstrap bank: each game's length residual and both sides' kill residuals
	# against that mean, on the sqrt scale where count noise is about the same
	# at every mean, so a residual transfers to a game with a different mean.
	bank, bank_w = [], []
	for i in range(0, len(recs), 2):  # recs come in (blue, red) pairs per game
		(wb, _, kb, *_, row), (_, _, kr, *_) = recs[i], recs[i + 1]
		(kw, ew), (kl, el) = ((kb, full[i]), (kr, full[i + 1])) if wb else ((kr, full[i + 1]), (kb, full[i]))
		bank.append((eps_by_game[row.gameid], np.sqrt(kw) - np.sqrt(ew), np.sqrt(kl) - np.sqrt(el)))
		bank_w.append(row.wt)
	bank, bank_w = np.array(bank), np.array(bank_w)

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
	death_slope, death_gamma, death_urn = _fit_deaths(rows_side, role_k, champ_kill, player_kill, role_d, champ_death, player_death)

	# --- multikills: rate log-linear in log(kills - n + 1), the kills past the
	# threshold. At k = n every kill must land in one streak; a smooth curve in
	# log1p(k) * k overpredicts there by ~1.5-2.5x and underpredicts mid-range.
	multi_rates = {}
	for name in MULTI:
		# Drawn from every game, weight-0 ones included, so the all-games prior
		# below exists even when the weighted fit sees only Worlds. Only players
		# with at least n kills: below that the count is structurally 0, and
		# fitting those zeros would bend the curve the sampler uses at k >= n.
		n = MULTI_N[name]
		xs, ys, mw = [], [], []
		for row in g_all.itertuples():
			for side in ("blue", "red"):
				st = getattr(row, f"{side}_player_stats")
				k, m = st.get("kills"), st.get(name)
				if k is None or m is None:
					continue
				for r in range(5):
					if not pd.isna(k[r]) and k[r] >= n and not pd.isna(m[r]):
						xs.append(np.log(k[r] - n + 1))
						ys.append(m[r])
						mw.append(row.wt)
		xs, mw = np.array(xs), np.array(mw)
		cnt, kk = np.array(ys), np.ones_like(mw)
		# Weighted fits shrink toward the all-games fit: with Worlds-heavy weights
		# a rare event (a single Worlds pentakill) has almost no information on its own.
		prior, h0 = _poisson_rate_fit(xs, kk, cnt, np.ones_like(mw))
		a, b = prior if np.allclose(mw, 1.0) else _poisson_rate_fit(xs, kk, cnt, mw, prior, h0 * (MULTI_PRIOR_EVENTS / h0[0, 0]))[0]
		multi_rates[name] = (float(a), float(b))

	# --- CS, gold, damage
	P = _player_table(rows_side)
	tm = P["role"] == 0  # one row per team-game
	lm = lane_minions(np.exp(P["ll"][tm]))
	cs_ok = P["team_cs"][tm] > 0
	Xt = np.column_stack([np.ones(tm.sum()), P["ll"][tm], P["won"][tm], P["kd"][tm], P["td"][tm] / np.exp(P["ll"][tm])])[cs_ok]
	yt = np.log(P["team_cs"][tm][cs_ok] / lm[cs_ok])
	wt_t, teams_t = P["wt"][tm][cs_ok], P["team"][tm][cs_ok]
	sw = np.sqrt(wt_t)
	team_cs_coef = np.linalg.lstsq(Xt * sw[:, None], yt * sw, rcond=None)[0]
	res = yt - Xt @ team_cs_coef
	t_sum, t_n = defaultdict(float), defaultdict(float)
	for t, e, w in zip(teams_t, res, wt_t):
		t_sum[t] += w * e
		t_n[t] += w
	team_cs_team = {t: t_sum[t] / (t_n[t] + 20.0) for t in t_sum}
	res = res - np.array([team_cs_team[t] for t in teams_t])
	team_cs_cap = float(np.exp(yt.max()))
	ok = (P["team_cs"] > 0) & ~np.isnan(P["cs"])  # all five or none of a team-game
	ratio = P["cs"][ok] / lane_minions(np.exp(P["ll"][ok]))
	player_cs_cap = np.array([ratio[P["role"][ok] == r].max() for r in range(5)])
	lk, ld, la = np.log1p(P["k"]), P["d"], np.log1p(P["a"])
	base = [np.ones(len(lk)), P["ll"], P["won"]]
	cs_fit = _fit_loglinear(P, ok, np.column_stack(base + [lk, ld, la]), np.log(np.maximum(P["cs"], 1) / P["team_cs"]), use_het=False)
	ok_g = ok & (P["gold"] > 0)
	gold_fit = _fit_loglinear(P, ok_g, np.column_stack(base + [lk, ld, la, np.log1p(P["cs"])]), np.log(P["gold"]))
	ok_d = ok_g & (P["dmg"] > 0)
	dmg_fit = _fit_loglinear(P, ok_d, np.column_stack(base + [np.log1p(P["k"] + P["a"]), ld, np.log(P["gold"])]), np.log(P["dmg"]))
	# Rows come five per team-game (role order), so team-game i is rows 5i..5i+4.
	# Team-games with a missing stat or any residual past 4 robust sd (role
	# swaps, mislabelled roles) stay out of the bank: transplanted onto a normal
	# game they would hand one player a line no normal game produces.
	box_bank = np.column_stack([res] + [f[4][ok].reshape(-1, 5) for f in (cs_fit, gold_fit, dmg_fit)])
	box_bank_w = P["wt"][ok][::5]
	scale = np.r_[1.4826 * _wmedian(np.abs(res - _wmedian(res, wt_t)), wt_t), cs_fit[3], gold_fit[3], dmg_fit[3]]
	keep = (np.abs(box_bank) <= 4 * scale).all(axis=1)
	box_bank, box_bank_w = box_bank[keep], box_bank_w[keep]

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
		len_mu=float(mu), len_team=len_team, len_champ=len_champ,
		k_base=k_base, k_gamma=k_gamma, team_kills_for=team_for, team_kills_against=team_against, champ_kills_for=champ_for,
		bank=bank, bank_p=bank_w / bank_w.sum(),
		role_kill_share=role_k, role_death_share=role_d, champ_kill_mult=champ_kill, champ_death_mult=champ_death,
		player_kill_mult=player_kill, player_death_mult=player_death, kp_role=kp_role, kp_mult=kp_mult, share_conc=share_conc,
		death_urn=death_urn, death_gamma=death_gamma, death_slope=death_slope,
		team_cs_coef=team_cs_coef, team_cs_team=team_cs_team, team_cs_cap=team_cs_cap, player_cs_cap=player_cs_cap,
		cs_coef=cs_fit[0], cs_champ=cs_fit[1], cs_player=cs_fit[2], cs_het=cs_fit[5], box_bank=box_bank, box_bank_p=box_bank_w / box_bank_w.sum(),
		gold_coef=gold_fit[0], gold_champ=gold_fit[1], gold_player=gold_fit[2], gold_het=gold_fit[5],
		dmg_coef=dmg_fit[0], dmg_champ=dmg_fit[1], dmg_player=dmg_fit[2], dmg_het=dmg_fit[5],
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


def _fit_deaths(rows_side, role_k, champ_kill, player_kill, role_d, champ_death, player_death) -> tuple[np.ndarray, float, tuple]:
	"""Per-role slope of death-share excess on kill-share excess (weighted by
	team deaths); then, by simulation on the observed team-games (their own
	kills and death totals), grid-fit the exponent on the champion * player
	death multipliers and the urn (a, c) of a * deaths + c balls so each
	player's death-count tail matches, per team death total.
	Moment-matching share variance is not enough: the multipliers overstate
	how unequal deaths are, and the tail is what shows."""
	recs = []
	for row, side, k, d, a in rows_side:
		if d.sum() < 5:
			continue
		champs, players = getattr(row, f"{side}_champs"), getattr(row, f"{side}_players")
		wk = np.array([role_k[r] * champ_kill.get((champs[r], r), 1) * player_kill.get(players[r], 1) for r in range(5)])
		mult = np.array([champ_death.get((champs[r], r), 1) * player_death.get(players[r], 1) for r in range(5)])
		recs.append((k, d, wk / wk.sum(), mult, row.wt))
	num, den = np.zeros(5), np.zeros(5)
	for k, d, pk, mult, w in recs:
		if k.sum():
			wd = role_d * mult
			x = k / k.sum() - pk
			num += w * d.sum() * x * (d / d.sum() - wd / wd.sum())
			den += w * d.sum() * x * x
	slope = num / den
	ws = np.array([r[4] for r in recs])
	n = np.array([r[1].sum() for r in recs]).astype(int)
	bucket = np.repeat(np.minimum(n // 5, 6), 5)
	wp = np.repeat(ws, 5)
	nb, nm = 7, 14
	tail = lambda deaths: np.bincount(bucket * nm + np.minimum(deaths, nm - 1), wp, minlength=nb * nm).reshape(nb, nm)[:, ::-1].cumsum(1)[:, ::-1]
	th = tail(np.concatenate([r[1] for r in recs]).astype(int))
	best = (np.inf, None)
	for gamma in (0.0, 0.25, 0.5):
		qs = [_death_shares((role_d * mult ** gamma) / (role_d * mult ** gamma).sum(), k, pk, slope) for k, _, pk, mult, _ in recs]
		for a in (1.0, 1.05, 1.1, 1.2, 1.35):
			for c in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0):
				rng = np.random.default_rng(1)
				sim = np.concatenate([rng.multivariate_hypergeometric(_urn(q, a * ni + c), ni) for q, ni in zip(qs, n)])
				ts = tail(sim)
				# Chi-square on each player's upper tail, P(deaths >= m), per death-total bucket.
				err = np.sum((ts - th) ** 2 / (th + ts + 1.0))
				if err < best[0]:
					best = (err, (gamma, a, c))
	gamma, a, c = best[1]
	return slope, float(gamma), (float(a), float(c))


def _wmedian(x, w) -> float:
	o = np.argsort(x)
	c = np.cumsum(w[o])
	return float(x[o][np.searchsorted(c, c[-1] / 2)])


MULTI_PRIOR_EVENTS = 20.0  # multikill prior strength, in all-games expected events


def _poisson_fit(X, offset, y, w):
	"""Weighted Poisson GLM y ~ offset * exp(X @ theta) by Newton. Returns
	(theta, Hessian)."""
	theta = np.zeros(X.shape[1])
	for _ in range(100):
		lam = offset * np.exp(X @ theta)
		h = X.T @ (X * (w * lam)[:, None])
		step = np.linalg.solve(h, X.T @ (w * (y - lam)))
		theta = theta + step
		if np.abs(step).max() < 1e-10:
			break
	return theta, h


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


def _player_table(rows_side) -> dict:
	"""One row per player-game: role, champion, player, team, weight, K/D/A,
	log length, won, team kill margin, CS (own and team), gold, damage."""
	cols = defaultdict(list)
	for row, side, k, d, a in rows_side:
		st = getattr(row, f"{side}_player_stats")
		nan5 = [np.nan] * 5
		cs, gold, dmg = (np.array(st.get(x) or nan5, float) for x in ("total cs", "totalgold", "damagetochampions"))
		won = int(row.blue_win if side == "blue" else 1 - row.blue_win)
		tk, td = k.sum(), d.sum()
		champs, players = getattr(row, f"{side}_champs"), getattr(row, f"{side}_players")
		for r in range(5):
			for key, v in (("role", r), ("champ", champs[r]), ("player", players[r]), ("team", getattr(row, f"{side}_team")),
			               ("wt", row.wt), ("k", k[r]), ("d", d[r]), ("a", a[r]), ("ll", np.log(row.gamelength / 60.0)), ("won", won),
			               ("kd", (tk - td) / (tk + td + 1)), ("td", td), ("cs", cs[r]), ("team_cs", cs.sum()), ("gold", gold[r]), ("dmg", dmg[r])):
				cols[key].append(v)
	return {key: np.array(v, dtype=object if key in ("champ", "player", "team") else float) for key, v in cols.items()}


def _fit_loglinear(P: dict, ok: np.ndarray, X: np.ndarray, y: np.ndarray, k_champ: float = 30.0, k_player: float = 40.0, use_het: bool = True):
	"""Per role: weighted OLS of y on X over rows ok; then shrunk mean residuals
	by (champion, role) and by player; per-role residual scale. The scale is
	robust (MAD): role swaps and off-meta picks leave a long low tail that would
	otherwise inflate a symmetric log-normal's high tail. Also returns the final
	residuals standardised by a per-role scale linear in the prediction (fit
	to |residual|; see _het_scale), aligned with P (nan outside ok), and that
	scale's (a, b, floor) per role (use_het=False: unit scale, raw residuals). The
	robust sd is of the standardised residuals. CS shares stay raw: their
	residual spread does not track the prediction, and rescaling fattened the
	high tail."""
	roles, champs, players, wts = P["role"][ok].astype(int), P["champ"][ok], P["player"][ok], P["wt"][ok]
	X, y = X[ok], y[ok]
	coef, sd = np.zeros((5, X.shape[1])), np.zeros(5)
	res = np.zeros(len(y))
	for r in range(5):
		m = roles == r
		sw = np.sqrt(wts[m])
		coef[r] = np.linalg.lstsq(X[m] * sw[:, None], y[m] * sw, rcond=None)[0]
		res[m] = y[m] - X[m] @ coef[r]
	champ_sum, champ_n = defaultdict(float), defaultdict(float)
	for r, c, w, e in zip(roles, champs, wts, res):
		champ_sum[(c, r)] += w * e
		champ_n[(c, r)] += w
	champ = {key: champ_sum[key] / (champ_n[key] + k_champ) for key in champ_sum}
	res2 = res - np.array([champ[(c, r)] for r, c in zip(roles, champs)])
	pl_sum, pl_n = defaultdict(float), defaultdict(float)
	for p, w, e in zip(players, wts, res2):
		pl_sum[p] += w * e
		pl_n[p] += w
	player = {key: pl_sum[key] / (pl_n[key] + k_player) for key in pl_sum}
	res3 = res2 - np.array([player[p] for p in players])
	fitted = y - res3
	scale, z = np.zeros((5, 3)), np.zeros(len(y))
	for r in range(5):
		m = roles == r
		e, w = res3[m], wts[m]
		if use_het:
			A = np.column_stack([np.ones(m.sum()), fitted[m]])
			sw = np.sqrt(w)
			scale[r, :2] = np.linalg.lstsq(A * sw[:, None], np.abs(e) * sw, rcond=None)[0]
			scale[r, 2] = 0.25 * _wmedian(np.abs(e), w)
		else:
			scale[r] = (1.0, 0.0, 0.0)
		z[m] = e / _het_scale(scale[r], fitted[m])
		sd[r] = 1.4826 * _wmedian(np.abs(z[m] - _wmedian(z[m], w)), w)
	resid = np.full(len(P["role"]), np.nan)
	resid[ok] = z
	return coef, champ, player, sd, resid, scale


def model_path(worlds_weight: float = 0.0):
	return weighted_path(PATH, worlds_weight)


def save(m: StatsModel, path=PATH) -> None:
	with open(path, "wb") as f:
		pickle.dump(m, f)


def load(path=PATH) -> StatsModel:
	with open(path, "rb") as f:
		return pickle.load(f)

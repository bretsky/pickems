"""Monte Carlo of the full Worlds bracket, one game at a time.

Every game: First Selection -> side / first pick, procedural draft (fearless
pool of the series excluded), draftpredict win probability on that exact
draft, winner drawn, then the game's stats sampled. Rules per stage follow
data/manual/worlds2026.yaml (from Liquipedia).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pickems.history import ROLES
from pickems.sim.draft import Draft, DraftSimulator, MetaTracker, first_selection
from pickems.stats_model import StatsModel
from pickems.teams import Team


@dataclass
class GameRecord:
	stage: str
	round: str
	series_id: int
	game_no: int
	draft: Draft
	p_blue: float
	blue_win: bool
	stats: dict

	@property
	def winner(self) -> Team:
		return self.draft.blue if self.blue_win else self.draft.red


@dataclass
class SeriesResult:
	stage: str
	round: str
	a: Team
	b: Team
	wins_a: int
	wins_b: int
	games: list[GameRecord]

	@property
	def winner(self) -> Team:
		return self.a if self.wins_a > self.wins_b else self.b

	@property
	def loser(self) -> Team:
		return self.b if self.wins_a > self.wins_b else self.a


class WinModel:
	"""draftpredict's live predictor on a simulated draft.

	draftpredict's in-series features (series momentum, series streak, and the
	game-N series-depth terms) are fed by its apply_live_results during real
	use. The simulator plays that role: it tells the predictor which game of
	the series this is, and after each game updates the momentum and streak
	trackers the same way apply_live_results does. Both trackers are restored
	to their end-of-history state at the start of every tournament."""

	_PRISTINE: dict = {}

	def __init__(self, dp_module, predictor, patch: float, league: str):
		self.dp, self.pred = dp_module, predictor
		self.patch, self.league = patch, league
		self.game_number = 1
		key = id(predictor)
		if key not in WinModel._PRISTINE:
			WinModel._PRISTINE[key] = (copy.deepcopy(predictor.series_momentum_tracker), copy.deepcopy(predictor.series_streak_tracker))
		# compute_features reads the game number from the live-results log; point it here.
		dp_module.count_pending_games_in_series = lambda *_a, **_k: self.game_number - 1

	def reset(self) -> None:
		mom, streak = WinModel._PRISTINE[id(self.pred)]
		self.pred.series_momentum_tracker = copy.deepcopy(mom)
		self.pred.series_streak_tracker = copy.deepcopy(streak)

	def start_series(self, a: Team, b: Team) -> None:
		self.game_number = 1
		mom, streak = self.pred.series_momentum_tracker, self.pred.series_streak_tracker
		for t in (a.team_key, b.team_key):
			mom.series_key.pop(t, None)
			mom.cum_residual.pop(t, None)
			if streak.series_key.get(t) is not None:
				streak._finalize(t)

	def record(self, d: Draft, blue_win: bool) -> None:
		from draftpredict.elo import expected_score  # noqa: PLC0415 -- vendored module
		from draftpredict.series_depth import patch_to_step  # noqa: PLC0415

		bk, rk = d.blue.team_key, d.red.team_key
		pair = frozenset((bk, rk))
		base = expected_score(self.pred.blue_side_tracker.get_offset(patch_to_step(self.patch)), 0.0)
		result = 1.0 if blue_win else 0.0
		self.pred.series_momentum_tracker.update(bk, pair, result - base)
		self.pred.series_momentum_tracker.update(rk, pair, (1 - result) - (1.0 - base))
		streak_key = (pd.Timestamp.now().normalize(), pair)
		self.pred.series_streak_tracker.update(bk, streak_key, blue_win)
		self.pred.series_streak_tracker.update(rk, streak_key, not blue_win)
		self.game_number += 1

	def p_blue(self, d: Draft, fearless: bool) -> float:
		def side(t: Team, champs):
			return {"team": t.team_key, "region": t.region, "players": {r: {"player": p, "champion": c} for r, p, c in zip(ROLES, t.players, champs)}}

		game = {
			"patch": self.patch, "is_bridge": d.blue.region != d.red.region, "league": self.league, "fearless": fearless,
			"blue": side(d.blue, d.blue_champs), "red": side(d.red, d.red_champs),
		}
		return self.dp.predict_win_probability(self.pred, game)


class Tournament:
	def __init__(self, cfg: dict, teams: dict[str, Team], drafter: DraftSimulator, win: WinModel, stats: StatsModel,
	             rng: np.random.Generator, projected_presence: np.ndarray, meta_prior_games: float = 40.0):
		self.cfg, self.teams = cfg, teams
		self.drafter, self.win, self.stats, self.rng = drafter, win, stats, rng
		win.reset()
		self.meta = MetaTracker(len(projected_presence), projected_presence, meta_prior_games)
		self.games: list[GameRecord] = []
		self.series: list[SeriesResult] = []
		self._sid = 0

	# --- one series -------------------------------------------------------------
	def play_series(self, stage: str, rnd: str, a: Team, b: Team, best_of: int, fearless: bool, selector: Team | None, dual: bool = False) -> SeriesResult:
		need = best_of // 2 + 1
		self.win.start_series(a, b)
		wins = {a.code: 0, b.code: 0}
		used: set[str] = set()
		games = []
		self._sid += 1
		sel = selector if selector is not None else (a if self.rng.random() < 0.5 else b)
		n = 0
		while max(wins.values()) < need:
			n += 1
			other = b if sel is a else a
			if dual and n == 1:
				sel_blue, sel_fp = True, True
			else:
				sel_blue, sel_fp = first_selection(self.rng)
			blue, red = (sel, other) if sel_blue else (other, sel)
			fp_side = ("Blue" if sel_blue else "Red") if sel_fp else ("Red" if sel_blue else "Blue")
			draft = self.drafter.run(blue, red, fp_side, used if fearless else set(), self.meta)
			p = self.win.p_blue(draft, fearless)
			blue_win = bool(self.rng.random() < p)
			self.win.record(draft, blue_win)
			w, l = (blue, red) if blue_win else (red, blue)
			wc, lc = (draft.blue_champs, draft.red_champs) if blue_win else (draft.red_champs, draft.blue_champs)
			st = self.stats.sample(self.rng, w.team_key, l.team_key, wc, lc, w.players, l.players, p if blue_win else 1 - p)
			rec = GameRecord(stage, rnd, self._sid, n, draft, p, blue_win, st)
			games.append(rec)
			self.games.append(rec)
			wins[w.code] += 1
			used.update(draft.picks)
			sel = l  # loser holds First Selection next game
		res = SeriesResult(stage, rnd, a, b, wins[a.code], wins[b.code], games)
		self.series.append(res)
		return res

	# --- play-in: 4-team double elimination ----------------------------------------
	def play_in(self) -> Team:
		c = self.cfg["play_in"]
		bo, fl = c["best_of"], c["fearless"]
		(a1, b1), (a2, b2) = [[self.teams[x] for x in pair] for pair in c["upper_semis"]]
		u1 = self.play_series("play_in", "UB semi", a1, b1, bo, fl, None)
		u2 = self.play_series("play_in", "UB semi", a2, b2, bo, fl, None)
		ubf = self.play_series("play_in", "UB final", u1.winner, u2.winner, bo, fl, None)
		lbs = self.play_series("play_in", "LB semi", u1.loser, u2.loser, bo, fl, None)
		lbf = self.play_series("play_in", "LB final", ubf.loser, lbs.winner, bo, fl, None)
		q = self.play_series("play_in", "Qualifier", ubf.winner, lbf.winner, bo, fl, ubf.winner, dual=True)
		self.play_in_placement = {q.winner.code: 1, q.loser.code: 2, lbf.loser.code: 3, lbs.loser.code: 4}
		return q.winner

	# --- swiss --------------------------------------------------------------------
	def swiss(self, play_in_winner: Team) -> dict[str, tuple[int, int]]:
		c = self.cfg["swiss"]
		pools = {int(k): [play_in_winner if x == "PLAYIN_WINNER" else self.teams[x] for x in v] for k, v in c["pools"].items()}
		record = {t.code: [0, 0] for p in pools.values() for t in p}
		team_by = {t.code: t for p in pools.values() for t in p}
		met: set[frozenset] = set()
		pairs = self._draw_round1(pools)
		for rnd in range(1, c["rounds"] + 1):
			if rnd > 1:
				pairs = self._draw_swiss(record, team_by, met)
			for a, b in pairs:
				ra, rb = record[a.code], record[b.code]
				decisive = max(ra[0], rb[0]) == c["win_target"] - 1 or max(ra[1], rb[1]) == c["loss_target"] - 1
				bo, fl = (3, c["bo3_fearless"]) if decisive else (1, c["bo1_fearless"])
				sel = a if a.seed < b.seed else b if b.seed < a.seed else None
				s = self.play_series("swiss", f"R{rnd} {ra[0]}-{ra[1]}", a, b, bo, fl, sel)
				record[s.winner.code][0] += 1
				record[s.loser.code][1] += 1
				met.add(frozenset((a.code, b.code)))
		self.swiss_record = {k: tuple(v) for k, v in record.items()}
		return self.swiss_record

	def _draw_round1(self, pools: dict[int, list[Team]]) -> list[tuple[Team, Team]]:
		pairs = []
		for hi, lo in ((1, 4), (2, 3)):
			for _ in range(1000):
				perm = self.rng.permutation(len(pools[lo]))
				cand = list(zip(pools[hi], [pools[lo][i] for i in perm]))
				if all(x.region != y.region for x, y in cand):
					break
			pairs += cand
		return pairs

	def _draw_swiss(self, record, team_by, met) -> list[tuple[Team, Team]]:
		groups: dict[tuple, list[str]] = {}
		for code, (w, l) in record.items():
			if w < self.cfg["swiss"]["win_target"] and l < self.cfg["swiss"]["loss_target"]:
				groups.setdefault((w, l), []).append(code)
		pairs = []
		for key in sorted(groups, key=lambda k: (-k[0], k[1])):
			codes = groups[key]
			for _ in range(2000):
				perm = [codes[i] for i in self.rng.permutation(len(codes))]
				cand = list(zip(perm[0::2], perm[1::2]))
				if all(frozenset(p) not in met for p in cand):
					break
			pairs += [(team_by[a], team_by[b]) for a, b in cand]
		return pairs

	# --- knockouts ------------------------------------------------------------------
	def knockouts(self) -> Team:
		c = self.cfg["knockouts"]
		rec = self.swiss_record
		adv = [code for code, (w, l) in rec.items() if w == 3]
		by_rec = {r: [x for x in adv if rec[x][1] == r] for r in (0, 1, 2)}
		order = lambda xs: [xs[i] for i in self.rng.permutation(len(xs))]
		threes, threetwos, rest = order(by_rec[0]), order(by_rec[2]), order(by_rec[1])
		qfs = []
		for t30 in threes:
			qfs.append((t30, threetwos.pop()))
		pool = order(threetwos + rest)
		qfs += list(zip(pool[0::2], pool[1::2]))
		# 3-0 teams on opposite halves: bracket slots [qf0, qf2 | qf1, qf3] -> SF.
		if len(threes) == 2:
			qfs = [qfs[0], qfs[2], qfs[1], qfs[3]]
		T = self.teams_all
		winners = []
		for a, b in qfs:
			wa, wb = rec[a][1], rec[b][1]
			sel = T[a] if wa < wb else T[b] if wb < wa else None
			winners.append(self.play_series("knockout", "QF", T[a], T[b], c["best_of"], c["fearless"], sel).winner)
		sf = [self.play_series("knockout", "SF", winners[0], winners[1], c["best_of"], c["fearless"], None),
		      self.play_series("knockout", "SF", winners[2], winners[3], c["best_of"], c["fearless"], None)]
		final = self.play_series("knockout", "Final", sf[0].winner, sf[1].winner, c["best_of"], c["fearless"], None)
		self.champion, self.finalist = final.winner, final.loser
		self.semifinalists = [s.loser for s in sf]
		return final.winner

	def run(self) -> "Tournament":
		self.teams_all = dict(self.teams)
		piw = self.play_in()
		self.swiss(piw)
		self.knockouts()
		return self

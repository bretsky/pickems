"""Monte Carlo of the full Worlds bracket, one game at a time.

Every game: First Selection -> side / first pick, procedural draft (fearless
pool of the series excluded), draftpredict win probability on that exact
draft, winner drawn, then the game's stats sampled. Rules per stage follow
data/manual/worlds2026.yaml (from Liquipedia).
"""

from __future__ import annotations

import copy
import os
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pickems.history import ROLES
from pickems.meta import PRIOR_GAMES
from pickems.sim.draft import Draft, DraftSimulator, MetaTracker, first_selection
from pickems.sim.played import PlayedSeries, queue, start_counts
from pickems.stats_model import StatsModel
from pickems.teams import Team
from pickems.variance import Form

# Logit shift toward the previous game's loser, every game after a series'
# first. Real series reverse slightly (tied 1-1 or 2-2, the last game's winner
# takes the next one ~47% of the time across major leagues 2023-26, and
# draftpredict's walk-forward odds agree), but simulated ones ran the other
# way (~52%): the intraday Elo walk moves the winner up as on real data, while
# the procedural drafter doesn't reproduce the loser's draft-side edge (First
# Selection, adapting) that offsets it there. Fitted by
# scripts/calibrate_reversal.py; PICKEMS_SERIES_REVERSAL overrides (0 = off).
SERIES_REVERSAL = float(os.environ.get("PICKEMS_SERIES_REVERSAL", "0.2"))


def blended_parts(pred, raw: float, team_id: str, weight: float) -> tuple[float, float]:
	"""(raw, adjustment): the team's / player's own rating and what blending in its primary league's rating
	(draftpredict's bridge-game blend, elo.blended_power) adds to it; they sum to the blended rating. (raw, 0) when the league is unknown."""
	from draftpredict.elo import blended_power  # noqa: PLC0415 -- vendored module

	region = pred.team_primary_league.get(team_id)
	league = pred.league_tracker.get(region) if region else None
	total = blended_power(raw, league, True, team_weight=weight)
	return raw, total - raw


def blended_rating(pred, raw: float, team_id: str, weight: float) -> float:
	"""raw blended with the rating of the team's primary league the way draftpredict does on a bridge game
	(the cross-region case Worlds always is); raw when the league is unknown."""
	return sum(blended_parts(pred, raw, team_id, weight))


@dataclass
class GameRecord:
	stage: str
	round: str
	series_id: int
	game_no: int
	draft: Draft
	p_blue: float | None  # None for a game already played (pickems.sim.played)
	blue_win: bool
	stats: dict
	real: bool = False

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
	"""draftpredict's live predictor on a simulated draft, walked forward
	through the tournament (pickems.sim.walk): every finished game is applied
	to the predictor's trackers the way the next Oracle's Elixir rebuild would
	apply it, on the simulated calendar (cfg["schedule"]).

	Worlds buff (pickems.worlds_buff): a logit bonus per team added to the
	model's own, for teams that beat their odds at past Worlds.

	Series reversal (SERIES_REVERSAL): a logit shift toward the team that
	lost the previous game of the series.

	Form variance (pickems.variance): each team and player draws a persistent
	logit form for the tournament, so equally strong teams can differ in how
	spread out their outcomes are; a single game's probability is unchanged.

	Series momentum is the one live-only tracker (built empty, fed by
	apply_live_results during real use): the simulator plays that role, and
	also tells the predictor which game of the series this is."""

	def __init__(self, dp_module, predictor, patch: float, league: str, rng: np.random.Generator,
	             worlds_buffs: dict[str, dict] | None = None, form_sds: dict | None = None, teams: dict[str, Team] | None = None):
		from pickems.sim.walk import OracleWalk  # noqa: PLC0415

		self.dp, self.pristine = dp_module, predictor
		self.patch, self.league = patch, league
		self.game_number = 1
		self.last_loser: str | None = None
		self.buff = {k: b["logit"] for k, b in (worlds_buffs or {}).items()}
		self.form = Form(form_sds, teams, rng) if form_sds and teams else None
		self.walk = OracleWalk(dp_module, predictor, rng)
		self._primary: dict[str, str] = {}
		# compute_features reads the game number from the live-results log; point it here.
		dp_module.count_pending_games_in_series = lambda *_a, **_k: self.game_number - 1

	@property
	def pred(self):
		return self.walk.pred

	def reset(self) -> None:
		self.walk.reset()
		self.walk.pred.series_momentum_tracker = copy.deepcopy(self.pristine.series_momentum_tracker)

	def start_day(self, day) -> None:
		self.walk.start_day(day)

	def team_id(self, t: Team) -> str:
		return self.dp._resolve_team(self.pristine, t.team_key)

	def player_id(self, key: str) -> str:
		return self.dp._resolve_player(self.pristine, key)

	def team_parts(self, t: Team) -> tuple[float, float]:
		return blended_parts(self.walk.pred, self.walk.pred.team_tracker.get(self.team_id(t)), self.team_id(t), self.pristine.team_weight)

	def player_parts(self, key: str, t: Team) -> tuple[float, float]:
		return blended_parts(self.walk.pred, self.walk.pred.player_tracker.get(self.player_id(key)), self.team_id(t), self.pristine.player_league_blend_weight)

	def team_rating(self, t: Team) -> float:
		"""Region-blended Elo: draftpredict's bridge-game blend of team and league rating (elo.blended_power)."""
		return sum(self.team_parts(t))

	def player_rating(self, key: str, t: Team) -> float:
		return sum(self.player_parts(key, t))

	def _bridge(self, a: Team, b: Team) -> tuple[str, str, bool, bool]:
		"""Primary leagues and draftpredict's bridge flags (data.load_team_games' rule)."""
		from draftpredict.data import tracked_regions_for  # noqa: PLC0415 -- vendored module

		pa, pb = (self.pristine.team_primary_league.get(self.team_id(t), t.region) for t in (a, b))
		tracked = tracked_regions_for(self.walk.day.year)
		bridge = pa in tracked and pb in tracked and pa != pb
		return pa, pb, bridge, pa != pb and not bridge

	def start_series(self, a: Team, b: Team) -> None:
		self.game_number = 1
		self.last_loser = None
		mom = self.walk.pred.series_momentum_tracker
		for t in (a, b):
			mom.series_key.pop(self.team_id(t), None)
			mom.cum_residual.pop(self.team_id(t), None)

	def _series_state(self, d: Draft, blue_win: bool) -> None:
		"""Series momentum, the live-only tracker (see the class docstring)."""
		from draftpredict.elo import expected_score  # noqa: PLC0415 -- vendored module
		from draftpredict.series_depth import patch_to_step  # noqa: PLC0415

		pred = self.walk.pred
		bk, rk = self.team_id(d.blue), self.team_id(d.red)
		pair = frozenset((bk, rk))
		base = expected_score(pred.blue_side_tracker.get_offset(patch_to_step(self.patch)), 0.0)
		result = 1.0 if blue_win else 0.0
		pred.series_momentum_tracker.update(bk, pair, result - base)
		pred.series_momentum_tracker.update(rk, pair, (1 - result) - (1.0 - base))

	def record_played(self, d: Draft, blue_win: bool) -> None:
		"""A game already played: the predictor's Oracle walk has it, so only
		the series' own state moves on."""
		self._series_state(d, blue_win)
		self.game_number += 1
		self.last_loser = (d.red if blue_win else d.blue).team_key

	def record(self, d: Draft, blue_win: bool, stats: dict) -> None:
		from pickems.sim.walk import WalkGame  # noqa: PLC0415

		self._series_state(d, blue_win)
		result = 1.0 if blue_win else 0.0
		bk, rk = self.team_id(d.blue), self.team_id(d.red)
		bp, rp = (stats["win_players"], stats["lose_players"]) if blue_win else (stats["lose_players"], stats["win_players"])
		pa, pb, bridge, minor = self._bridge(d.blue, d.red)
		kda = lambda ps: [(int(p["kills"]), int(p["deaths"]), int(p["assists"])) for p in ps]
		self.walk.record(WalkGame(
			blue_team=bk, red_team=rk, blue_players=[self.player_id(x) for x in d.blue.players], red_players=[self.player_id(x) for x in d.red.players],
			blue_champs=list(d.blue_champs), red_champs=list(d.red_champs), bans=d.bans, result_blue=result, league=self.league,
			patch=self.patch, game=self.game_number, playoffs=0, primary_blue=pa, primary_red=pb, is_bridge=bridge, is_minor_bridge=minor,
			length_s=float(stats["length"]) * 60, gold_blue=sum(p["gold"] for p in bp), gold_red=sum(p["gold"] for p in rp),
			blue_kda=kda(bp), red_kda=kda(rp),
		))
		self.game_number += 1
		self.last_loser = (d.red if blue_win else d.blue).team_key

	def p_blue(self, d: Draft, fearless: bool) -> float:
		def side(t: Team, champs):
			return {"team": t.team_key, "region": t.region, "players": {r: {"player": p, "champion": c} for r, p, c in zip(ROLES, t.players, champs)}}

		game = {
			"patch": self.patch, "is_bridge": self._bridge(d.blue, d.red)[2], "league": self.league, "fearless": fearless,
			"blue": side(d.blue, d.blue_champs), "red": side(d.red, d.red_champs),
		}
		p = self.walk.predict(game)
		shift = self.buff.get(d.blue.team_key, 0.0) - self.buff.get(d.red.team_key, 0.0)
		if self.last_loser is not None:
			shift += SERIES_REVERSAL if self.last_loser == d.blue.team_key else -SERIES_REVERSAL
		if self.form is None:
			return 1 / (1 + (1 - p) / p * np.exp(-shift)) if shift else p
		logit = self.form.apply(np.log(p / (1 - p)) + shift, d.blue.team_key, d.red.team_key)
		return float(1 / (1 + np.exp(-logit)))


class Schedule:
	"""The simulated calendar (cfg["schedule"]): which day each series is played.
	Entries are in play order, each {date, stage, round, n}: `round` is the
	start of the simulator's round label ("R3 2-0", "R4", "UB semi") and the
	entry takes the next n matching series (all remaining when n is absent)."""

	def __init__(self, entries: list[dict]):
		self.entries = [(pd.Timestamp(str(e["date"])), e["stage"], str(e["round"]), e.get("n")) for e in entries]
		self.used = [0] * len(self.entries)

	def _find(self, stage: str, label: str) -> int | None:
		for i, (_, st, prefix, n) in enumerate(self.entries):
			if st == stage and (label == prefix or label.startswith(prefix + " ")) and (n is None or self.used[i] < n):
				return i
		return None

	def rank(self, stage: str, label: str) -> int:
		i = self._find(stage, label)
		return len(self.entries) if i is None else i

	def take(self, stage: str, label: str) -> pd.Timestamp | None:
		i = self._find(stage, label)
		if i is None:
			return None
		self.used[i] += 1
		return self.entries[i][0]

	def first(self, stage: str) -> pd.Timestamp:
		return min(d for d, st, _, _ in self.entries if st == stage)


class Tournament:
	def __init__(self, cfg: dict, teams: dict[str, Team], drafter: DraftSimulator, win: WinModel, stats: StatsModel,
	             rng: np.random.Generator, projected_presence: np.ndarray, meta_prior_games: float = PRIOR_GAMES,
	             played: list[PlayedSeries] | None = None, score_played: bool = False):
		"""played: series already played (pickems.sim.played), replayed as they happened.
		score_played: price each real game with the win model before feeding it to the walk (win should then be built on a predictor as of
		before those games, see pickems.sim.likelihood); the prices land in `scored`."""
		self.cfg, self.teams = cfg, teams
		self.played: dict[frozenset, deque] = queue(played or [])
		self.starts = start_counts(played or [])
		self.drafter, self.win, self.stats, self.rng = drafter, win, stats, rng
		win.reset()
		self.schedule = Schedule(cfg["schedule"])
		self.day: pd.Timestamp | None = None
		self.elo_start = self._elo()
		self.meta = MetaTracker(len(projected_presence), projected_presence, meta_prior_games)
		self.games: list[GameRecord] = []
		self.series: list[SeriesResult] = []
		self.score_played = score_played
		self.scored: list[dict] = []
		self._sid = 0

	def _elo(self) -> dict:
		"""Current ratings of every Worlds team and player (by role), blended and raw (the difference is the region's adjustment)."""
		pl = {(c, r, n): self.win.player_parts(k, t) for c, t in self.teams.items()
		      for r, n, k in [*zip(ROLES, t.player_names, t.players), *t.subs]}
		tm = {c: self.win.team_parts(t) for c, t in self.teams.items()}
		return {"team": {c: sum(p) for c, p in tm.items()}, "team_raw": {c: p[0] for c, p in tm.items()},
		        "player": {k: sum(p) for k, p in pl.items()}, "player_raw": {k: p[0] for k, p in pl.items()}}

	def lineup(self, team: Team) -> Team:
		"""The team as it plays one simulated game: in each role a player drawn at random in proportion to the real games each started
		so far (the configured starter when none of its games are played yet, or nobody else has played the role)."""
		cnt = self.starts.get(team.code)
		if not cnt:
			return team
		names, keys, changed = list(team.player_names), list(team.players), False
		for r, c in enumerate(cnt):
			if len(c) > 1 or (c and next(iter(c)) != names[r]):
				pick = list(c)[self.rng.choice(len(c), p=np.array(list(c.values()), float) / sum(c.values()))] if len(c) > 1 else next(iter(c))
				names[r], keys[r], changed = pick, team.key_of(ROLES[r], pick), True
		if not changed:
			return team
		return replace(team, players=keys, player_names=names)

	def _score(self, stage: str, rnd: str, n: int, draft: Draft, blue_win: bool, st: dict, fearless: bool, gameid: str) -> None:
		"""The win model's pre-game probability for a real game, on the lineups that played it; then the game goes through the walk."""
		won, lost = (draft.blue, draft.red) if blue_win else (draft.red, draft.blue)
		names = {won.code: st["win_players"], lost.code: st["lose_players"]}
		lineup = lambda t: replace(t, players=[t.key_of(r, p["name"]) for r, p in zip(ROLES, names[t.code])], player_names=[p["name"] for p in names[t.code]])
		d = replace(draft, blue=lineup(draft.blue), red=lineup(draft.red))
		p = self.win.p_blue(d, fearless)
		self.scored.append({"gameid": gameid, "stage": stage, "round": rnd, "game": n, "blue": d.blue.code, "red": d.red.code, "p_blue": float(p), "blue_win": blue_win})
		self.win.record(d, blue_win, st)

	# --- one series -------------------------------------------------------------
	def play_series(self, stage: str, rnd: str, a: Team, b: Team, best_of: int, fearless: bool, selector: Team | None, dual: bool = False,
	                day: pd.Timestamp | None = None) -> SeriesResult:
		"""day: when it's played (default: the schedule's next slot for this round)."""
		day = day if day is not None else self.schedule.take(stage, rnd)
		if day is None:
			raise KeyError(f"cfg['schedule'] has no day for {stage} {rnd!r}")
		if self.day is not None and day < self.day:
			raise ValueError(f"cfg['schedule'] puts {stage} {rnd!r} on {day.date()}, before the series already played on {self.day.date()}")
		self.day = day
		self.win.start_day(day)
		need = best_of // 2 + 1
		self.win.start_series(a, b)
		wins = {a.code: 0, b.code: 0}
		used: set[str] = set()
		games = []
		self._sid += 1
		sel = selector if selector is not None else (a if self.rng.random() < 0.5 else b)
		real = self.played.get(frozenset((a.code, b.code)))
		real = real.popleft().games if real else []
		n = 0
		while max(wins.values()) < need:
			n += 1
			if n <= len(real):  # already played: as it happened
				g = real[n - 1]
				draft, blue_win, st, p = g.draft, g.blue_win, g.stats, None
				if self.score_played:
					self._score(stage, rnd, n, draft, blue_win, st, fearless, g.gameid)
				else:
					self.win.record_played(draft, blue_win)
				self.meta.record([self.drafter.state.cix[c] for _, _, c, _ in draft.actions if c in self.drafter.state.cix])
				w, l = (a, b) if (draft.blue if blue_win else draft.red).code == a.code else (b, a)  # this series' own Team objects
			else:
				other = b if sel is a else a
				if dual and n == 1:
					sel_blue, sel_fp = True, True
				else:
					sel_blue, sel_fp = first_selection(self.rng)
				blue, red = (sel, other) if sel_blue else (other, sel)
				blue, red = self.lineup(blue), self.lineup(red)
				fp_side = ("Blue" if sel_blue else "Red") if sel_fp else ("Red" if sel_blue else "Blue")
				draft = self.drafter.run(blue, red, fp_side, used if fearless else set(), self.meta)
				p = self.win.p_blue(draft, fearless)
				blue_win = bool(self.rng.random() < p)
				w, l = (blue, red) if blue_win else (red, blue)
				wc, lc = (draft.blue_champs, draft.red_champs) if blue_win else (draft.red_champs, draft.blue_champs)
				st = self.stats.sample(self.rng, w.team_key, l.team_key, wc, lc, w.players, l.players, p if blue_win else 1 - p)
				for team, pl in ((w, st["win_players"]), (l, st["lose_players"])):
					for q, nm in zip(pl, team.player_names):
						q["name"] = nm
				self.win.record(draft, blue_win, st)
			rec = GameRecord(stage, rnd, self._sid, n, draft, p, blue_win, st, real=p is None)
			games.append(rec)
			self.games.append(rec)
			wins[w.code] += 1
			used.update(draft.picks)
			sel = a if l.code == a.code else b  # loser holds First Selection next game (l may be a game's lineup copy of a or b)
		if n < len(real):
			raise ValueError(f"{stage} {rnd} {a.code}-{b.code} is Bo{best_of} here, but {len(real)} games of it were played")
		res = SeriesResult(stage, rnd, a, b, wins[a.code], wins[b.code], games)
		self.series.append(res)
		return res

	# --- play-in: 4-team double elimination ----------------------------------------
	def play_in(self) -> Team:
		c = self.cfg["play_in"]
		bo, fl = c["best_of"], c["fearless"]
		if c.get("format") == "single":  # Worlds 2025: one Bo5 for the last Swiss slot
			a, b = (self.teams[x] for x in c["teams"])
			s = self.play_series("play_in", "Qualifier", a, b, bo, fl, None)
			self.play_in_placement = {s.winner.code: 1, s.loser.code: 2}
			return s.winner
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
		team_of = lambda x: play_in_winner if x == "PLAYIN_WINNER" else self.teams[x]
		if c.get("round1"):  # the drawn first round is known (Worlds 2025 config)
			pairs = [(team_of(a), team_of(b)) for a, b in c["round1"]]
			in_swiss = [t for pr in pairs for t in pr]
		else:
			pools = {int(k): [team_of(x) for x in v] for k, v in c["pools"].items()}
			pairs = self._draw_round1(pools)
			in_swiss = [t for p in pools.values() for t in p]
		record = {t.code: [0, 0] for t in in_swiss}
		team_by = {t.code: t for t in in_swiss}
		met: set[frozenset] = set()
		for rnd in range(1, c["rounds"] + 1):
			if rnd > 1:
				pairs = self._fixed_swiss(rnd, team_of, record, team_by, met) or self._draw_swiss(record, team_by, met)
			# Within a round, play series in calendar order.
			label = lambda pr: f"R{rnd} {record[pr[0].code][0]}-{record[pr[0].code][1]}"
			pairs = sorted(pairs, key=lambda pr: self.schedule.rank("swiss", label(pr)))
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

	def _fixed_swiss(self, rnd: int, team_of, record, team_by, met) -> list[tuple[Team, Team]] | None:
		"""The announced pairings for round rnd (swiss.draws), when this run's
		records lead to exactly that round: every pair level on record and not a
		rematch, and the pairs cover the teams still alive. Otherwise None (the
		run drew a different Swiss so far, and the round is drawn as usual)."""
		raw = (self.cfg["swiss"].get("draws") or {}).get(rnd)
		if not raw:
			return None
		pairs = [(team_of(a), team_of(b)) for a, b in raw]
		c = self.cfg["swiss"]
		alive = {k for k, (w, l) in record.items() if w < c["win_target"] and l < c["loss_target"]}
		codes = [t.code for pr in pairs for t in pr]
		if set(codes) != alive or len(codes) != len(alive):
			return None
		for a, b in pairs:
			if a.code not in team_by or b.code not in team_by or record[a.code] != record[b.code] or frozenset((a.code, b.code)) in met:
				return None
		return pairs

	def _played_pairs(self, xs: list, ys: list | None = None, code=lambda t: t) -> tuple[list[tuple], list, list]:
		"""Pairs the real draw already made: (x, y) with a played series still
		queued, x from xs and y from ys (or from xs too). Returns those pairs and
		what's left of xs and ys to draw. Earliest queued series first: two teams
		level now may also meet later (a Swiss rematch of records), and that later
		series must not stand in for this round's real pairing."""
		pairs, xs, ys = [], list(xs), list(xs if ys is None else ys)
		cand = sorted(((q[0].games[0].date, i, j) for i, x in enumerate(xs) for j, y in enumerate(ys)
			if y is not x and (q := self.played.get(frozenset((code(x), code(y)))))), key=lambda c: c[0])
		for _, i, j in cand:
			x, y = xs[i], ys[j]
			if any(x in p or y in p for p in pairs):
				continue
			pairs.append((x, y))
		used = [z for p in pairs for z in p]
		return pairs, [x for x in xs if x not in used], [y for y in ys if y not in used]

	def _draw_round1(self, pools: dict[int, list[Team]]) -> list[tuple[Team, Team]]:
		pairs = []
		for hi, lo in ((1, 4), (2, 3)):
			done, his, los = self._played_pairs(pools[hi], pools[lo], lambda t: t.code)
			pairs += done
			for _ in range(1000):
				perm = self.rng.permutation(len(los))
				cand = list(zip(his, [los[i] for i in perm]))
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
			done, codes, _ = self._played_pairs(groups[key])
			pairs += [(team_by[a], team_by[b]) for a, b in done]
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
		# The real draw, once made (bracket order), when this run's Swiss sent the same teams.
		if c.get("draw") and {x for pr in c["draw"] for x in pr} == set(adv):
			qfs = [tuple(pr) for pr in c["draw"]]
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
		left = [s.score() for q in self.played.values() for s in q]
		if left:
			raise ValueError(f"played series the bracket never reached (draw or schedule out of step with the config?): {left}")
		self.elo_end = self._elo()
		return self

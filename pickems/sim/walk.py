"""Walk-forward updates: every simulated game is fed into draftpredict's
trackers the way its Oracle walk (scripts/predict.build_predictor) would feed
the same game once Oracle's Elixir published it, so later games in the same
simulated tournament are predicted from ratings that include earlier results.

Mirrors build_predictor's final-tracker walks, not build_baseline_dataset's
training walk (the two differ in places, see PlayerWalk notes below): the
predictor starts from build_predictor's state, and the next real rebuild is
what these updates stand in for.

Days. The tuned config has enable_intraday_updates on, so draftpredict's walk
reads, within a calendar day:
  - team / league / player Elo from a scratch copy taken at the day's open and
    chained through that day's earlier games with live-feed-quality inputs
    (win/loss only for players; margin of victory from a degraded length and
    gold reading, see margin_of_victory.live_margin_view). The authoritative
    trackers get the full Oracle update (KDA-blended player score, exact
    margins) and take over at the next day's open.
  - every other tracker live (their update is win/loss only either way),
    except champion overpopularity, which is day-batched.
OracleWalk keeps both views: `reader` (what a prediction sees) and the
authoritative state, and start_day() rolls one into the other.

Not updated (held at their pre-Worlds state):
  - player_champion_lane_tracker: needs each lane's gold difference at 15
    minutes, which the box-score model doesn't simulate.
  - lane_multiplier_state and champion overpopularity's power slopes: the
    predictor only keeps their outputs, not the running sums an update needs.
    Both move by tiny amounts per game (eta ~0.01; slopes fit on ~100k games).

Isolation: each tournament works on copy-on-write layers over the shared
predictor's tracker dicts (ChainMap with a fresh top dict), so the pristine
predictor is never written to and a tournament costs only its own writes.
"""

from __future__ import annotations

import copy
import math
from functools import lru_cache
from collections import ChainMap
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]


@dataclass
class WalkGame:
	"""One finished game in draftpredict's terms (team / player ids, Oracle
	conventions: `game` is the game number within its series, `playoffs`
	Oracle's flag, which is 0 for every Worlds game)."""

	blue_team: str
	red_team: str
	blue_players: list[str]
	red_players: list[str]
	blue_champs: list[str]
	red_champs: list[str]
	bans: list[str]
	result_blue: float
	league: str
	patch: float
	game: int
	playoffs: int
	primary_blue: str
	primary_red: str
	is_bridge: bool
	is_minor_bridge: bool
	length_s: float | None
	gold_blue: float | None
	gold_red: float | None
	blue_kda: list[tuple[int, int, int]]
	red_kda: list[tuple[int, int, int]]
	# Live-feed readings of the margin terms (z, gold_d) for the intraday
	# scratch view; None = draw them like live_margin_view does.
	live: tuple[float, float] | None = None


@lru_cache(maxsize=1)
def walk_config() -> dict:
	"""The tracker knobs build_predictor used (adopted Optuna snapshot, else
	draftpredict's defaults), for the parts of the update the Predictor doesn't
	keep. Mirrors build_predictor's config resolution."""
	from draftpredict.elo import LEAGUE_K_PLAYOFFS_MULTIPLIER  # noqa: PLC0415 -- vendored module
	from draftpredict.player_elo import PLAYER_PERFORMANCE_BLEND  # noqa: PLC0415
	from draftpredict.tuned_config import feature_config_from_params, load_snapshot  # noqa: PLC0415

	snapshot = load_snapshot()
	config = None
	if snapshot is not None and snapshot["best_params"].get("model_type", "logistic") == "logistic":
		config = feature_config_from_params(snapshot["best_params"])
	elo = config["elo_kwargs"] if config else {}
	player = config["player_elo_kwargs"] if config else {}
	for key in ("league_form_scale", "league_standing_weight"):
		if elo.get(key):
			raise NotImplementedError(f"walk-forward updates don't model {key} (domestic records); the adopted snapshot sets it")
	return {
		"intraday": bool(elo.get("enable_intraday_updates", False)),
		"team_smoothing_alpha": elo.get("team_elo_smoothing_alpha", 1.0),
		"league_k_playoffs_multiplier": elo.get("league_k_playoffs_multiplier", LEAGUE_K_PLAYOFFS_MULTIPLIER),
		"performance_blend": player.get("performance_blend", PLAYER_PERFORMANCE_BLEND),
	}


# --- copy-on-write layers -----------------------------------------------------------

def _layer(obj):
	"""Shallow copy whose dict attributes are fresh ChainMap layers over the
	original's: writes land in the copy only, reads fall through."""
	new = copy.copy(obj)
	for k, v in vars(obj).items():
		if isinstance(v, (dict, ChainMap)):
			setattr(new, k, ChainMap({}, v))
	return new


def _snapshot(tracker):
	"""A day-open copy of a layered Elo tracker: its own writes so far frozen
	into a plain dict, over the same pristine base (cheap: only this
	tournament's writes are copied)."""
	from draftpredict.elo import EloTracker  # noqa: PLC0415

	def frozen(m):
		return ChainMap({}, dict(m.maps[0]), *m.maps[1:]) if isinstance(m, ChainMap) else ChainMap({}, m)

	return EloTracker(tracker.default_rating, frozen(tracker.ratings), ChainMap({}), frozen(tracker.smoothed_ratings))


class _Clock:
	"""Stands in for pd.Timestamp inside draftpredict's scripts/predict module,
	so its reads of "today" (series-streak keys, context seasons) see the
	simulated day."""

	def __init__(self):
		self.today: pd.Timestamp | None = None

	def now(self, tz=None):
		return self.today if self.today is not None else pd.Timestamp.now(tz)

	def __call__(self, *a, **k):
		return pd.Timestamp(*a, **k)

	def __getattr__(self, name):
		return getattr(pd.Timestamp, name)


class _SimPandas:
	def __init__(self, clock: _Clock):
		self.Timestamp = clock

	def __getattr__(self, name):
		return getattr(pd, name)


# Every Predictor attribute holding tracker state that reads or updates may write to.
_LAYERED = ["team_tracker", "league_tracker", "player_tracker", "clutch_tracker", "champion_tracker", "player_champion_tracker",
            "series_momentum_tracker", "series_streak_tracker", "series_depth_tracker", "player_series_depth_tracker",
            "game_length_baseline"]


class OracleWalk:
	def __init__(self, dp_module, predictor, rng: np.random.Generator, config: dict | None = None):
		self.dp, self.pristine, self.rng = dp_module, predictor, rng
		self.cfg = config or walk_config()
		if not self.cfg["intraday"]:
			raise NotImplementedError("walk-forward updates assume enable_intraday_updates (the tuned config's setting)")
		self.clock = _Clock()
		dp_module.pd = _SimPandas(self.clock)
		from draftpredict.margin_of_victory import GAME_LENGTH_LOG_SD, live_noise  # noqa: PLC0415

		self.noise, self.log_sd = live_noise(), GAME_LENGTH_LOG_SD
		self.reset()

	# --- tournament / day lifecycle ----------------------------------------------------
	def reset(self) -> None:
		"""Fresh copy-on-write state over the pristine predictor (start of a tournament)."""
		p = copy.copy(self.pristine)
		for name in _LAYERED:
			if getattr(p, name) is not None:
				setattr(p, name, _layer(getattr(p, name)))
		p.blue_side_tracker = copy.copy(p.blue_side_tracker)
		p.ally_trackers = {k: _layer(t) for k, t in p.ally_trackers.items()}
		p.counter_trackers = {k: _layer(t) for k, t in p.counter_trackers.items()}
		if p.champion_overpopularity_tracker is not None:
			op = _layer(p.champion_overpopularity_tracker)
			op.counts, op.games = op.counts.copy(), op.games.copy()
			p.champion_overpopularity_tracker = op
		self.pred = p
		self.day: pd.Timestamp | None = None
		self.pending_presence: list[tuple[float, frozenset]] = []

	def start_day(self, day: pd.Timestamp) -> None:
		"""Open a new calendar day: apply the day-batched updates of the day just
		finished and re-take the day-open scratch copies the reads go through."""
		day = pd.Timestamp(day).normalize()
		if self.day is not None and day == self.day:
			return
		op = self.pred.champion_overpopularity_tracker
		if op is not None:
			for step, champs in self.pending_presence:
				op.update(step, champs)
		self.pending_presence = []
		self.day = day
		self.clock.today = day
		p = self.pred
		self.team_scratch, self.league_scratch, self.player_scratch = _snapshot(p.team_tracker), _snapshot(p.league_tracker), _snapshot(p.player_tracker)
		reader = copy.copy(p)
		reader.team_tracker, reader.league_tracker = self.team_scratch, self.league_scratch
		# Player ratings from the scratch copy; prior-games counts (player
		# experience) from the authoritative tracker, as the walk reads them.
		view = copy.copy(self.player_scratch)
		view.games_played = p.player_tracker.games_played
		reader.player_tracker = view
		self.reader = reader

	def predict(self, game: dict) -> float:
		return self.dp.predict_win_probability(self.reader, game)

	# --- one game ---------------------------------------------------------------------
	def _live_margin(self, g: WalkGame, z: float, gold_d: float) -> tuple[float, float]:
		"""(z, gold_d) as the live feed would have recorded them (live_margin_view, one game)."""
		draw = self.rng.random()
		err = float(self.rng.choice(np.asarray(self.noise["length_errors_s"], dtype=float)))
		live_z = z + (math.log(g.length_s + err) - math.log(g.length_s)) / self.log_sd if g.length_s else math.nan
		return (math.nan if draw < self.noise["length_missing_rate"] else live_z,
		        math.nan if draw < self.noise["gold_missing_rate"] else gold_d)

	def record(self, g: WalkGame) -> None:
		self.record_players(g, self.record_team(g))

	def record_team(self, g: WalkGame) -> dict:
		"""The game-level walks (team / league Elo, team series depth, blue side,
		length baseline, series streak). Returns the pre-game values the
		player-level walks read. Split from record_players only because
		draftpredict orders a day's games differently in the two (by game number
		vs by gameid); the simulation calls both, in play order."""
		from draftpredict.champion_power import patch_to_step  # noqa: PLC0415
		from draftpredict.data import tracked_regions_for  # noqa: PLC0415
		from draftpredict.elo import anchored_league_update, decaying_k, expected_score, stage_k  # noqa: PLC0415
		from draftpredict.margin_of_victory import gold_margin_d, mov_multiplier  # noqa: PLC0415

		p, cfg = self.pred, self.cfg
		result = float(g.result_blue)
		step = patch_to_step(g.patch)
		tb, tr = g.blue_team, g.red_team

		# Pre-game reads every walk shares.
		offset = p.blue_side_tracker.get_offset(step)
		side_prob = expected_score(offset, 0.0) - 0.5
		baseline = p.game_length_baseline
		z = baseline.z(g.league, g.length_s) if baseline is not None else math.nan
		if baseline is not None:
			baseline.update(g.league, g.length_s)
		gold_d = gold_margin_d(g.gold_blue, g.gold_red, result)
		live_z, live_gold_d = g.live if g.live is not None else self._live_margin(g, z, gold_d)

		# --- team / league Elo, team series depth (elo.compute_elo_features) --------------
		team_mov = mov_multiplier(z, p.team_mov_beta, gold_d, p.mov_gold_weight, p.team_mov_clip)
		team_live_mov = mov_multiplier(live_z, p.team_mov_beta, live_gold_d, p.mov_gold_weight, p.team_mov_clip)
		residual = result - expected_score(p.team_tracker.get(tb) + offset, p.team_tracker.get(tr))
		p.series_depth_tracker.update((tb, g.game), residual, 1.0, step)
		p.series_depth_tracker.update((tr, g.game), -residual, 1.0, step)
		k_team = stage_k(is_bridge=g.is_bridge, is_playoffs=bool(g.playoffs), k_table=p.team_k_table)
		alpha = cfg["team_smoothing_alpha"]
		p.team_tracker.update(tb, tr, result, k_team * team_mov, rating_offset_a=offset, smoothing_alpha=alpha)
		self.team_scratch.update(tb, tr, result, k_team * team_live_mov, rating_offset_a=offset, smoothing_alpha=alpha)
		minor = p.minor_league_elo and g.is_minor_bridge
		if g.is_bridge or minor:
			lt = p.league_tracker
			n = (lt.games_played.get(g.primary_blue, 0) + lt.games_played.get(g.primary_red, 0)) / 2
			k_league = decaying_k(n, p.league_k_initial, p.league_k_final, p.league_k_decay_games)
			if g.playoffs:
				k_league *= cfg["league_k_playoffs_multiplier"]
			if g.is_bridge:
				lt.update(g.primary_blue, g.primary_red, result, k_league * team_mov, rating_offset_a=offset)
				self.league_scratch.update(g.primary_blue, g.primary_red, result, k_league * team_live_mov, rating_offset_a=offset)
			else:
				tracked = tracked_regions_for(self.day.year)
				tb_tracked, tr_tracked = g.primary_blue in tracked, g.primary_red in tracked
				anchored_league_update(lt, g.primary_blue, g.primary_red, tb_tracked, tr_tracked, result, k_league * team_mov, offset, p.league_anchor_mult)
				anchored_league_update(self.league_scratch, g.primary_blue, g.primary_red, tb_tracked, tr_tracked, result, k_league * team_live_mov, offset, p.league_anchor_mult)
		p.blue_side_tracker.update(result, step)

		key = (self.day, frozenset((tb, tr)))
		p.series_streak_tracker.update(tb, key, bool(result))
		p.series_streak_tracker.update(tr, key, not bool(result))
		return {"offset": offset, "side_prob": side_prob, "z": z, "gold_d": gold_d, "live_z": live_z, "live_gold_d": live_gold_d}

	def record_players(self, g: WalkGame, pre: dict) -> None:
		"""The per-role walks (player Elo, clutch, player series depth, champion
		power, player-champion, synergy / counter) and the day's overpopularity batch."""
		from draftpredict.champion_matchup import update_counter, update_synergy  # noqa: PLC0415
		from draftpredict.champion_power import BLUE_SIDE_FALLBACK_WIN_PROB_OFFSET, patch_to_step  # noqa: PLC0415
		from draftpredict.elo import expected_score, stage_k  # noqa: PLC0415
		from draftpredict.margin_of_victory import mov_multiplier  # noqa: PLC0415
		from draftpredict.player_elo import clutch_event_weight  # noqa: PLC0415

		p, cfg = self.pred, self.cfg
		result = float(g.result_blue)
		step = patch_to_step(g.patch)
		offset, side_prob, z, gold_d, live_z, live_gold_d = (pre[k] for k in ("offset", "side_prob", "z", "gold_d", "live_z", "live_gold_d"))

		# --- player Elo, clutch, player series depth (player_elo.compute_player_elo_features)
		# As build_predictor runs it: no champion_power_diff column, so the
		# champion adjustment is off (build_baseline_dataset's walk has it on).
		player_k = stage_k(is_bridge=g.is_bridge, is_playoffs=bool(g.playoffs), k_table=p.player_k_table)
		player_mov = mov_multiplier(z, p.player_mov_beta, gold_d, p.mov_gold_weight, p.player_mov_clip)
		player_live_mov = mov_multiplier(live_z, p.player_mov_beta, live_gold_d, p.mov_gold_weight, p.player_mov_clip)
		event_weight = clutch_event_weight(g.league, p.clutch_msi_weight, p.clutch_fst_weight)
		blend = cfg["performance_blend"]
		for kb, kr, (bk, bd, ba), (rk, rd, ra) in zip(g.blue_players, g.red_players, g.blue_kda, g.red_kda):
			kda_b, kda_r = (bk + ba) / max(bd, 1), (rk + ra) / max(rd, 1)
			perf = kda_b / (kda_b + kda_r) if kda_b + kda_r > 0 else 0.5
			score = blend * result + (1 - blend) * perf
			residual = score - expected_score(p.player_tracker.get(kb) + offset, p.player_tracker.get(kr))
			if event_weight > 0:
				p.clutch_tracker.update(kb, residual, event_weight)
				p.clutch_tracker.update(kr, -residual, event_weight)
			p.player_series_depth_tracker.update((kb, g.game), residual, 1.0, step)
			p.player_series_depth_tracker.update((kr, g.game), -residual, 1.0, step)
			p.player_tracker.update(kb, kr, score, player_k * player_mov, rating_offset_a=offset)
			self.player_scratch.update(kb, kr, result, player_k * player_live_mov, rating_offset_a=offset)

		# --- champion power, player-champion (champion_power / champion_matchup) ---------
		for role, cb, cr, kb, kr in zip(ROLES, g.blue_champs, g.red_champs, g.blue_players, g.red_players):
			p.champion_tracker.update(cb, role, result, step, is_blue=True, blue_side_win_prob_offset=side_prob)
			p.champion_tracker.update(cr, role, 1 - result, step, is_blue=False, blue_side_win_prob_offset=side_prob)
			p.player_champion_tracker.update((kb, cb), result)
			p.player_champion_tracker.update((kr, cr), 1 - result)

		# --- synergy / counter: build_predictor's lineups carry no blue_side_offset,
		# so its walk uses the fallback prior.
		champs_b, champs_r = dict(zip(ROLES, g.blue_champs)), dict(zip(ROLES, g.red_champs))
		update_synergy(p.ally_trackers, champs_b, champs_r, result, step, blue_side_win_prob_offset=BLUE_SIDE_FALLBACK_WIN_PROB_OFFSET)
		update_counter(p.counter_trackers, champs_b, champs_r, result, step, blue_side_win_prob_offset=BLUE_SIDE_FALLBACK_WIN_PROB_OFFSET)

		# --- overpopularity: picks and bans, applied at the next day's open --------------
		self.pending_presence.append((step, frozenset(c for c in [*g.bans, *g.blue_champs, *g.red_champs] if isinstance(c, str) and c)))

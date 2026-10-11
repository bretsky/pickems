"""Procedural draft generation: 20 sequential ban/pick actions sampled from the
fitted draft model (pickems.draft.model), conditioned on both rosters, the
fearless pool, the projected Worlds-patch meta, and everything drafted so far.

Meta during the event: the presence feature starts at the projected 26.20
presence (pickems.meta) and is pulled towards what has actually been picked
or banned earlier in the simulated tournament, the way a real event's meta
converges. `meta_prior_games` sets how many games the projection is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pickems.draft.features import DraftContext, Featurizer, NR
from pickems.draft.model import DraftModel
from pickems.draft.trackers import DraftState
from pickems.history import DRAFT_ORDER, ROLES
from pickems.teams import Team

# First Selection (2026 rules), fitted on 2026 major-league series (games 2+,
# where the previous game's loser holds the selection).
P_TAKE_FIRST_PICK = 0.61       # selector takes first pick (else picks a side)
P_OTHER_BLUE_GIVEN_FP = 0.71   # ... then the other team takes blue
P_SELECTOR_RED_GIVEN_SIDE = 0.77


def first_selection(rng: np.random.Generator) -> tuple[bool, bool]:
	"""Returns (selector_is_blue, selector_has_first_pick)."""
	if rng.random() < P_TAKE_FIRST_PICK:
		return rng.random() >= P_OTHER_BLUE_GIVEN_FP, True
	return rng.random() >= P_SELECTOR_RED_GIVEN_SIDE, False


@dataclass
class Draft:
	blue: Team
	red: Team
	first_pick_side: str
	actions: list[tuple[str, str, str, str | None]] = field(default_factory=list)  # (act, side, champ, role)
	blue_champs: list[str] = field(default_factory=lambda: [None] * NR)
	red_champs: list[str] = field(default_factory=lambda: [None] * NR)

	@property
	def bans(self) -> list[str]:
		return [c for a, _, c, _ in self.actions if a == "ban"]

	@property
	def picks(self) -> list[str]:
		return self.blue_champs + self.red_champs


class MetaTracker:
	"""Projected presence blended with presence observed so far in the event."""

	def __init__(self, n: int, projected: np.ndarray, prior_games: float = 40.0):
		self.projected = projected
		self.prior = prior_games
		self.counts = np.zeros(n)
		self.games = 0

	def presence(self) -> np.ndarray:
		return (self.projected * self.prior + self.counts) / (self.prior + self.games)

	def record(self, draft_idx: list[int]) -> None:
		self.counts[list(set(draft_idx))] += 1
		self.games += 1


class DraftSimulator:
	def __init__(self, model: DraftModel, state: DraftState, t: float, rng: np.random.Generator, temperature: float = 1.0):
		self.model = model
		self.state = state
		self.feat = Featurizer(state)
		self.t = t
		self.rng = rng
		self.temperature = temperature

	def run(self, blue: Team, red: Team, first_pick_side: str, fearless_used: set[str], meta: MetaTracker | None) -> Draft:
		s = self.state
		A, B = (blue, red) if first_pick_side == "Blue" else (red, blue)
		side_of = {"A": first_pick_side, "B": "Red" if first_pick_side == "Blue" else "Blue"}
		team_of = {"Blue": blue, "Red": red}
		picks = {"Blue": {}, "Red": {}}
		n_pick = {"Blue": 0, "Red": 0}
		n_ban = {"Blue": 0, "Red": 0}
		unavailable = np.zeros(s.n, dtype=bool)
		for c in fearless_used:
			if c in s.cix:
				unavailable[s.cix[c]] = True
		presence = meta.presence() if meta is not None else None
		draft = Draft(blue, red, first_pick_side)
		for slot, (act, who) in enumerate(DRAFT_ORDER):
			side = side_of[who]
			other = "Red" if side == "Blue" else "Blue"
			me, opp = team_of[side], team_of[other]
			ctx = DraftContext(
				t=self.t, team=me.team_key, opp=opp.team_key, players=me.players, opp_players=opp.players,
				own_picks=picks[side], opp_picks=picks[other], unavailable=unavailable, slot=slot,
				own_pick_ordinal=n_pick[side], own_ban_ordinal=n_ban[side], presence_override=presence,
			)
			if act == "ban":
				p = self.model.ban_probs(self.feat, ctx)
				choice = self._sample(p)
				c, role = choice, None
				n_ban[side] += 1
			else:
				p = self.model.pick_probs(self.feat, ctx)
				choice = self._sample(p)
				c, role = divmod(choice, NR)
				picks[side] = {**picks[side], role: c}
				n_pick[side] += 1
				(draft.blue_champs if side == "Blue" else draft.red_champs)[role] = s.champions[c]
			unavailable = unavailable.copy()
			unavailable[c] = True
			draft.actions.append((act, side, s.champions[c], ROLES[role] if role is not None else None))
		if meta is not None:
			meta.record([s.cix[c] for _, _, c, _ in draft.actions])
		return draft

	def _sample(self, p: np.ndarray) -> int:
		if self.temperature != 1.0:
			lp = np.log(np.clip(p, 1e-300, None)) / self.temperature
			p = np.exp(lp - lp.max())
			p /= p.sum()
		return int(self.rng.choice(len(p), p=p))

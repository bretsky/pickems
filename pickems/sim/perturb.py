"""Parameter uncertainty for the draft sims: one random draw of champion strength per sim.

The plain sims hold the draft model and the projected meta fixed, so pick totals
only vary by draft-to-draft sampling and the tail champions come out (near) never
most picked. Here each sim first draws, per champion,
  - a logit offset ~ N(0, sigma_pick), added to that champion's pick and ban
    scores for the whole tournament (the model being wrong about its priority), and
  - a presence multiplier exp(N(0, sigma_presence) - sigma_presence**2 / 2) on the
    projected meta (the patch projection being wrong about it),
so the spread of pick totals across sims includes not knowing those parameters.
The draws come from their own generator, keyed on the sim seed.

Only scripts/simulate_perturbed.py turns this on; plain runs never import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pickems.draft.features import NR


@dataclass(frozen=True)
class Perturb:
	sigma_pick: float = 0.0
	sigma_presence: float = 0.0

	def draw(self, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
		"""(logit offset, presence multiplier) per champion for the sim with this seed."""
		rng = np.random.default_rng([seed, 0x9E37])
		off = rng.normal(0, self.sigma_pick, n) if self.sigma_pick else np.zeros(n)
		mult = np.exp(rng.normal(0, self.sigma_presence, n) - self.sigma_presence ** 2 / 2) if self.sigma_presence else np.ones(n)
		return off, mult


class PerturbedModel:
	"""A DraftModel with a fixed per-champion logit offset on its pick and ban probabilities."""

	def __init__(self, model, offset: np.ndarray):
		self.model = model
		self.w = np.exp(offset - offset.max())

	def ban_probs(self, feat, ctx) -> np.ndarray:
		p = self.model.ban_probs(feat, ctx) * self.w
		return p / p.sum()

	def pick_probs(self, feat, ctx) -> np.ndarray:
		p = (self.model.pick_probs(feat, ctx).reshape(-1, NR) * self.w[:, None]).ravel()
		return p / p.sum()

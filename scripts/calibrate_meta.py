"""Pick META_ALPHA: simulated presence shift (vs pre-patch) should track the
solo-queue-driven part of the projection with slope ~1."""

import os
import pickle
import subprocess
import sys

import numpy as np

from pickems.sim.summary import STAGES
from pickems.sync_model import ROOT

for alpha in sys.argv[1:] or ["1", "2", "3"]:
	env = dict(os.environ, PICKEMS_META_ALPHA=alpha)
	subprocess.run([sys.executable, str(ROOT / "scripts" / "simulate.py"), "--sims", "32", "--workers", "8", "--name", f"calib_{alpha}"], env=env, check=True, capture_output=True)
	run = pickle.load(open(ROOT / ".cache" / "sims" / f"calib_{alpha}.pkl", "rb"))
	ix, pr = run["ix"], run["projection"]
	arr = np.stack([sum(r["stages"][s]["champ"] for s in STAGES) for r in run["results"]])
	games = np.mean([sum(r["stages"][s]["n_games"] for s in STAGES) for r in run["results"]])
	pres = (arr[:, 0] + arr[:, 1]).mean(0) / games
	pre = np.array([pr.pre.get(c, 0) for c in ix.champions])
	mult = np.array([pr.soloq_mult.get(c, 1) for c in ix.champions])
	# target: pre-presence odds times the solo-queue multiplier
	odds = np.clip(pre, 0.005, 0.99) / (1 - np.clip(pre, 0.005, 0.99)) * mult
	target = odds / (1 + odds)
	sel = pre > 0.05
	slope = np.polyfit(target[sel] - pre[sel], pres[sel] - pre[sel], 1)[0]
	print(f"alpha {alpha}: slope {slope:.2f}  mean|sim-target| {np.abs(pres - target)[sel].mean():.3f}", flush=True)

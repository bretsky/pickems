"""Fit the box-score model and save it.

--worlds-weight w in [0, 1]: Worlds games keep weight 1, every other game gets
1 - w (0 = all games equal, 1 = Worlds games only). A model fit with w > 0 is
saved next to the default one as stats_model_ww<w>.pkl.
"""

import argparse

from pickems.history import load_games
from pickems.stats_model import fit, model_path, save


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--worlds-weight", type=float, default=0.0, help="0 = every game equal, 1 = Worlds games only")
	args = ap.parse_args()
	save(fit(load_games(), worlds_weight=args.worlds_weight), model_path(args.worlds_weight))
	print(f"saved {model_path(args.worlds_weight)}")


if __name__ == "__main__":
	main()

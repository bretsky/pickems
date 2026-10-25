"""Announced Swiss pairings (swiss.draws in the event config), pulled from Leaguepedia.

The simulation takes a later round's real pairings from swiss.draws; without
them it draws the round itself, so a run made after Riot posts a round but
before the config has it wastes the real draw. This fetches the event's
Leaguepedia data page (the MatchSchedule wikitext behind the schedule tabs:
plain MediaWiki API, no cargo, so no credentials), writes every fully drawn
Swiss round into swiss.draws, then checks the config has the round the
history's results lead to next (and the quarterfinal draw once the Swiss is
done, which stays manual: the data page lists quarterfinals by date, not
bracket order).

	python -m pickems.draws            # fetch, update the config, check
	python -m pickems.draws --offline  # only check

Exits 1 when the round the next run needs is still unset.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from pickems.images import LP_API, _get
from pickems.teams import CONFIG, load_config


def fetch_swiss(cfg: dict) -> dict[int, list[list[str]]]:
	"""{round: [[A, B], ...]} for every Swiss round past the first whose pairings are all set on Leaguepedia, as config codes."""
	lp = cfg["leaguepedia"]
	r = _get(LP_API, {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
	                  "titles": lp["page"], "format": "json", "formatversion": 2})
	page = r["query"]["pages"][0]
	if page.get("missing"):
		raise LookupError(f"Leaguepedia has no page {lp['page']!r} (leaguepedia.page in the config)")
	text = page["revisions"][0]["slots"]["main"]["content"]
	codes = lp.get("codes") or {}
	def code(x: str) -> str:
		c = codes.get(x, x)
		if c not in cfg["teams"]:
			raise KeyError(f"Leaguepedia team {x!r} is not a config code; add it to leaguepedia.codes")
		return c
	out = {}
	for tab in text.split("{{MatchSchedule/Start|")[1:]:
		m = re.match(r"\s*tab=Round (\d+)\s*\|", tab)
		if not m or int(m.group(1)) < 2 or int(m.group(1)) > cfg["swiss"]["rounds"]:
			continue
		pairs = [(a.strip(), b.strip()) for a, b in re.findall(r"\|team1=([^|]*)\|team2=([^|\n]*)", tab)]
		if pairs and all(a and b and "TBD" not in (a, b) for a, b in pairs):
			out[int(m.group(1))] = [[code(a), code(b)] for a, b in pairs]
	return out


def _same(x: list, y: list) -> bool:
	return {frozenset(p) for p in x} == {frozenset(p) for p in y}


def write_draws(draws: dict[int, list[list[str]]], path: Path = CONFIG) -> None:
	"""Rewrite the swiss.draws block of the config in place, leaving the rest (comments too) as it is."""
	lines = path.read_text().splitlines(keepends=True)
	swiss = next(i for i, ln in enumerate(lines) if ln.rstrip() == "swiss:")
	start = next(i for i in range(swiss + 1, len(lines)) if re.match(r"  draws:", lines[i]))
	end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= 2), len(lines))
	block = ["  draws: {}\n"] if not draws else ["  draws:\n"] + [
		ln for rnd in sorted(draws) for ln in [f"    {rnd}:\n"] + [f"      - [{a}, {b}]\n" for a, b in draws[rnd]]]
	path.write_text("".join(lines[:start] + block + lines[end:]))


def series_per_round(c: dict) -> list[int]:
	"""Series in each Swiss round: every record group pairs off among itself."""
	n = 2 * len(c["round1"]) if c.get("round1") else sum(len(v) for v in c["pools"].values())
	groups, out = {(0, 0): n}, []
	for _ in range(c["rounds"]):
		out.append(sum(k // 2 for k in groups.values()))
		nxt: dict[tuple[int, int], int] = {}
		for (w, l), k in groups.items():
			for rec in ((w + 1, l), (w, l + 1)):
				if rec[0] < c["win_target"] and rec[1] < c["loss_target"]:
					nxt[rec] = nxt.get(rec, 0) + k // 2
		groups = nxt
	return out


def missing(cfg: dict, swiss_played: int) -> str | None:
	"""What the config lacks for the stage after `swiss_played` real Swiss series, or None."""
	c, done = cfg["swiss"], 0
	for rnd, k in enumerate(series_per_round(c), 1):
		done += k
		if swiss_played < done:
			if rnd == 1 or (c.get("draws") or {}).get(rnd):
				return None
			return f"Swiss Round {rnd} pairings are not in swiss.draws ({swiss_played} Swiss series played)"
	if not cfg["knockouts"].get("draw"):
		return "the Swiss is over but knockouts.draw (the quarterfinal bracket) is empty"
	return None


def swiss_played(cfg: dict) -> int:
	"""Real Swiss series in the training history so far."""
	from pickems.history import load_games  # noqa: PLC0415
	from pickems.sim.played import played_series  # noqa: PLC0415
	from pickems.teams import resolve_teams  # noqa: PLC0415

	games = load_games()
	dates = lambda st: [pd.Timestamp(str(e["date"])) for e in cfg["schedule"] if e["stage"] == st]
	lo, hi = min(dates("swiss")), min(dates("knockout"))
	return sum(lo <= s.games[0].date.normalize() < hi for s in played_series(games, resolve_teams(games, cfg), cfg))


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	ap.add_argument("--offline", action="store_true", help="don't fetch; only check the config")
	args = ap.parse_args()
	cfg = load_config()
	if not args.offline:
		try:
			fetched = fetch_swiss(cfg)
		except Exception as e:  # a failed fetch still leaves the check below
			print(f"warning: couldn't fetch the Swiss draws from Leaguepedia: {e}", file=sys.stderr)
			fetched = {}
		have = dict(cfg["swiss"].get("draws") or {})
		changed = {r: p for r, p in fetched.items() if not _same(have.get(r, []), p)}
		for r, p in sorted(changed.items()):
			print(f"swiss.draws round {r}: {'replaced' if r in have else 'added'} from Leaguepedia: {', '.join(f'{a}-{b}' for a, b in p)}")
		if changed:
			write_draws({**have, **changed})
			cfg = load_config()
		else:
			print(f"swiss.draws up to date (Leaguepedia has rounds {sorted(fetched) or 'none'} past the first)")
	gap = missing(cfg, swiss_played(cfg))
	if gap:
		print(f"MISSING: {gap}. Add it to {CONFIG.relative_to(CONFIG.parents[2])} before simulating, or the run draws it at random (full_pipeline.sh --skip-draws goes ahead anyway).", file=sys.stderr)
		return 1
	print("draws: the config has the pairings the next run needs")
	return 0


if __name__ == "__main__":
	sys.exit(main())

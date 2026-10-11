"""Compact per-tournament summary (numpy arrays keyed by champion / player /
team index, split by stage) and aggregation across simulations into the
probabilities Pick'Em and Crystal Ball questions need."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pickems.sim.tournament import Tournament

STAGES = ["play_in", "swiss", "knockout"]
CHAMP_STATS = ["picks", "bans", "wins", "kills", "deaths", "assists", "pentas", "blue_picks"]
PLAYER_STATS = ["games", "wins", "kills", "deaths", "assists", "firstbloods", "doubles", "triples", "quadras", "pentas"]
TEAM_STATS = ["games", "wins", "kills", "deaths", "dragons", "barons", "towers", "void_grubs", "heralds", "atakhans", "elders", "firstbloods", "game_seconds"]


@dataclass
class Index:
	champions: list[str]
	players: list[tuple[str, str, str]]  # (team code, role, player name)
	teams: list[str]

	def __post_init__(self):
		self.cix = {c: i for i, c in enumerate(self.champions)}
		self.pix = {(t, r): i for i, (t, r, _) in enumerate(self.players)}
		self.tix = {t: i for i, t in enumerate(self.teams)}


def summarize(t: Tournament, ix: Index) -> dict:
	nc, npl, nt = len(ix.champions), len(ix.players), len(ix.teams)
	S = {st: {"champ": np.zeros((len(CHAMP_STATS), nc), np.int32), "player": np.zeros((len(PLAYER_STATS), npl), np.int32),
	          "team": np.zeros((len(TEAM_STATS), nt), np.float64), "n_games": 0, "longest": 0.0, "shortest": 1e9,
	          "max_game_kills": 0, "pentas": 0} for st in STAGES}
	C, P, T = ({k: i for i, k in enumerate(L)} for L in (CHAMP_STATS, PLAYER_STATS, TEAM_STATS))
	roles = ["top", "jng", "mid", "bot", "sup"]
	for g in t.games:
		s = S[g.stage]
		d, st = g.draft, g.stats
		s["n_games"] += 1
		s["longest"] = max(s["longest"], st["length"])
		s["shortest"] = min(s["shortest"], st["length"])
		s["max_game_kills"] = max(s["max_game_kills"], st["win_kills"] + st["lose_kills"])
		for c in d.bans:
			s["champ"][C["bans"], ix.cix[c]] += 1
		w_is_blue = g.blue_win
		for side, team, champs in (("Blue", d.blue, d.blue_champs), ("Red", d.red, d.red_champs)):
			won = (side == "Blue") == w_is_blue
			pl = st["win_players"] if won else st["lose_players"]
			obj = st["win_obj"] if won else st["lose_obj"]
			ti = ix.tix[team.code]
			tk, td = (st["win_kills"], st["lose_kills"]) if won else (st["lose_kills"], st["win_kills"])
			for k, v in (("games", 1), ("wins", int(won)), ("kills", tk), ("deaths", td), ("dragons", obj["dragons"]), ("barons", obj["barons"]),
			             ("towers", obj["towers"]), ("void_grubs", obj["void_grubs"]), ("heralds", obj["heralds"]), ("atakhans", obj["atakhans"]),
			             ("elders", obj["elders"]), ("firstbloods", int(st.get("firstblood_winner") == won and tk + td > 0)), ("game_seconds", st["length"] * 60)):
				s["team"][T[k], ti] += v
			for r, (c, p) in enumerate(zip(champs, pl)):
				ci = ix.cix[c]
				s["champ"][C["picks"], ci] += 1
				s["champ"][C["wins"], ci] += int(won)
				s["champ"][C["kills"], ci] += p["kills"]
				s["champ"][C["deaths"], ci] += p["deaths"]
				s["champ"][C["assists"], ci] += p["assists"]
				s["champ"][C["pentas"], ci] += p["pentakills"]
				s["champ"][C["blue_picks"], ci] += int(side == "Blue")
				pi = ix.pix[(team.code, roles[r])]
				for k, v in (("games", 1), ("wins", int(won)), ("kills", p["kills"]), ("deaths", p["deaths"]), ("assists", p["assists"]),
				             ("firstbloods", p["firstbloodkill"]), ("doubles", p["doublekills"]), ("triples", p["triplekills"]),
				             ("quadras", p["quadrakills"]), ("pentas", p["pentakills"])):
					s["player"][P[k], pi] += v
				s["pentas"] += p["pentakills"]
	result = {
		"stages": S,
		"champion": t.champion.code, "finalist": t.finalist.code, "semifinalists": [x.code for x in t.semifinalists],
		"quarterfinalists": [c for c, (w, _) in t.swiss_record.items() if w == 3],
		"swiss_record": dict(t.swiss_record), "play_in": dict(t.play_in_placement),
		"series": [(s.stage, s.round, s.a.code, s.b.code, s.wins_a, s.wins_b) for s in t.series],
	}
	return result

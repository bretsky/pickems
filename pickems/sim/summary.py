"""Compact per-tournament summary (numpy arrays keyed by champion / player /
team index, split by stage) and aggregation across simulations into the
probabilities Pick'Em and Crystal Ball questions need."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pickems.sim.tournament import Tournament

STAGES = ["play_in", "swiss", "knockout"]
CHAMP_STATS = ["picks", "bans", "wins", "kills", "deaths", "assists", "pentas", "blue_picks"]
PLAYER_STATS = ["games", "wins", "kills", "deaths", "assists", "firstbloods", "doubles", "triples", "quadras", "pentas", "damage", "cs", "team_kills"]
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


# Typical damage to champions per game by role (major leagues 2024-26), used to
# put supports and carries on one scale for the Finals MVP heuristic.
ROLE_DAMAGE = [17000.0, 12000.0, 21000.0, 23000.0, 6500.0]


def finals_mvp(t: Tournament) -> dict:
	"""Finals MVP is a vote; approximated as the champion's player with the best
	series log-KDA plus log damage relative to the role norm."""
	final = [s for s in t.series if s.stage == "knockout" and s.round == "Final"][0]
	win = final.winner
	roles = ["top", "jng", "mid", "bot", "sup"]
	tot = [{"k": 0, "d": 0, "a": 0, "dmg": 0.0, "n": 0} for _ in range(5)]
	for g in final.games:
		won_blue = g.blue_win
		mine = g.stats["win_players"] if (g.draft.blue.code == win.code) == won_blue else g.stats["lose_players"]
		for r, p in enumerate(mine):
			tot[r]["k"] += p["kills"]
			tot[r]["d"] += p["deaths"]
			tot[r]["a"] += p["assists"]
			tot[r]["dmg"] += p["damage"]
			tot[r]["n"] += 1
	score = [np.log((x["k"] + x["a"] + 1) / (x["d"] + 1)) + np.log(x["dmg"] / max(x["n"], 1) / ROLE_DAMAGE[r]) for r, x in enumerate(tot)]
	r = int(np.argmax(score))
	return {"team": win.code, "role": roles[r], "player": win.player_names[r]}


def summarize(t: Tournament, ix: Index) -> dict:
	nc, npl, nt = len(ix.champions), len(ix.players), len(ix.teams)
	S = {st: {"champ": np.zeros((len(CHAMP_STATS), nc), np.int32), "player": np.zeros((len(PLAYER_STATS), npl), np.int32),
	          "team": np.zeros((len(TEAM_STATS), nt), np.float64), "n_games": 0, "longest": 0.0, "shortest": 1e9,
	          "max_game_kills": 0, "pentas": 0, "max_player_kills": 0, "max_player_cs": 0, "games_under_32": 0, "games_over_32": 0,
	          "shortest_winner": None} for st in STAGES}
	C, P, T = ({k: i for i, k in enumerate(L)} for L in (CHAMP_STATS, PLAYER_STATS, TEAM_STATS))
	roles = ["top", "jng", "mid", "bot", "sup"]
	pchamp: dict[tuple[int, int], list[int]] = {}  # (player, champion) -> [games, wins]
	for g in t.games:
		s = S[g.stage]
		d, st = g.draft, g.stats
		s["n_games"] += 1
		s["longest"] = max(s["longest"], st["length"])
		if st["length"] < s["shortest"]:
			s["shortest"] = st["length"]
			s["shortest_winner"] = g.winner.code
		s["games_under_32" if st["length"] < 32 else "games_over_32"] += 1
		everyone = st["win_players"] + st["lose_players"]
		s["max_player_kills"] = max(s["max_player_kills"], max(p["kills"] for p in everyone))
		s["max_player_cs"] = max(s["max_player_cs"], max(p["cs"] for p in everyone))
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
				pc = pchamp.setdefault((pi, ci), [0, 0])
				pc[0] += 1
				pc[1] += int(won)
				for k, v in (("games", 1), ("wins", int(won)), ("kills", p["kills"]), ("deaths", p["deaths"]), ("assists", p["assists"]),
				             ("firstbloods", p["firstbloodkill"]), ("doubles", p["doublekills"]), ("triples", p["triplekills"]),
				             ("quadras", p["quadrakills"]), ("pentas", p["pentakills"]), ("damage", p["damage"]), ("cs", p["cs"]),
				             ("team_kills", p["team_kills"])):
					s["player"][P[k], pi] += v
				s["pentas"] += p["pentakills"]
	result = {
		"stages": S,
		"finals_mvp": finals_mvp(t),
		"play_in_slots": {s.round if s.round != "UB semi" else f"UB semi {s.a.code}-{s.b.code}": s.winner.code for s in t.series if s.stage == "play_in"},
		"champion": t.champion.code, "finalist": t.finalist.code, "semifinalists": [x.code for x in t.semifinalists],
		"quarterfinalists": [c for c, (w, _) in t.swiss_record.items() if w == 3],
		"swiss_record": dict(t.swiss_record), "play_in": dict(t.play_in_placement),
		"series": [(s.stage, s.round, s.a.code, s.b.code, s.wins_a, s.wins_b) for s in t.series],
		"series_games": [[game_log(g) for g in s.games] for s in t.series],
		"player_champs": pchamp,
		"elo": {
			"team_start": np.array([t.elo_start["team"][c] for c in ix.teams], np.float32), "team_end": np.array([t.elo_end["team"][c] for c in ix.teams], np.float32),
			"player_start": np.array([t.elo_start["player"][(c, r)] for c, r, _ in ix.players], np.float32),
			"player_end": np.array([t.elo_end["player"][(c, r)] for c, r, _ in ix.players], np.float32),
		},
	}
	return result


GAME_OBJ = ["towers", "dragons", "barons", "void_grubs", "heralds", "atakhans", "elders", "inhibitors"]  # per-side counts in game logs


def game_log(g) -> dict:
	"""Compact record of one game for the dashboard's single-run view: sides,
	winner, draft order, length, kills, objectives (GAME_OBJ), first blood side
	and each player's K/D/A/CS/damage/gold."""
	d, st = g.draft, g.stats
	bw, rw = (st["win_players"], st["lose_players"]) if g.blue_win else (st["lose_players"], st["win_players"])
	bk, rk = (st["win_kills"], st["lose_kills"]) if g.blue_win else (st["lose_kills"], st["win_kills"])
	line = lambda p: [int(p["kills"]), int(p["deaths"]), int(p["assists"]), int(p["cs"]), int(p["damage"]), int(p["gold"])]
	bo, ro = (st["win_obj"], st["lose_obj"]) if g.blue_win else (st["lose_obj"], st["win_obj"])
	obj = lambda o: [int(o[k]) for k in GAME_OBJ]
	fb = st.get("firstblood_winner")
	return {
		"g": g.game_no, "blue": d.blue.code, "red": d.red.code, "blue_win": bool(g.blue_win), "p_blue": round(float(g.p_blue), 3),
		"fp": d.first_pick_side[0], "len": round(float(st["length"]), 1), "bk": int(bk), "rk": int(rk),
		"draft": [[a[0], side[0], c] for a, side, c, _ in d.actions],
		"bc": list(d.blue_champs), "rc": list(d.red_champs), "bp": [line(p) for p in bw], "rp": [line(p) for p in rw],
		"bo": obj(bo), "ro": obj(ro), "fb": None if fb is None or bk + rk == 0 else ("B" if fb == g.blue_win else "R"),
	}

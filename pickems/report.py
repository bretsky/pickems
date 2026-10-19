"""Aggregate simulation summaries into the numbers the dashboard shows.

"Leader" questions (most picked champion, player with most kills, ...) are
answered as P(this entity finishes first), with ties split evenly, plus the
mean and 10-90% range of the underlying stat. Rate questions (highest win
rate, best KDA) only count entities that meet a minimum-games bar in that
simulation, like Crystal Ball questions usually do.
"""

from __future__ import annotations

import itertools
import pickle
import warnings
from collections import Counter

import numpy as np

from pickems.sim.summary import CHAMP_STATS, PLAYER_STATS, STAGES, TEAM_STATS

SCOPES = ["all", *STAGES]


def _stage_sum(res: dict, scope: str, kind: str) -> np.ndarray:
	stages = STAGES if scope == "all" else [scope]
	return sum(res["stages"][s][kind] for s in stages)


def _leader_probs(values: np.ndarray, eligible: np.ndarray | None = None, highest: bool = True) -> np.ndarray:
	"""values: (sims, n). P(entity is the leader) with ties split."""
	v = values.astype(float).copy()
	if eligible is not None:
		v[~eligible] = -np.inf if highest else np.inf
	best = v.max(axis=1, keepdims=True) if highest else v.min(axis=1, keepdims=True)
	is_best = (v == best) & np.isfinite(v)
	cnt = is_best.sum(axis=1, keepdims=True)
	return (is_best / np.maximum(cnt, 1)).mean(axis=0)


def _table(names, probs, values, eligible=None, highest=True, extra=None) -> list[dict]:
	"""Every entity that ever registers (nonzero value, or eligible for a rate board), by P(top);
	ties (the long P = 0 tail) by average in the board's direction, then name."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		mean = np.nanmean(values, axis=0)
	keep = np.isfinite(mean) & ((values != 0) & ~np.isnan(values)).any(axis=0)
	if eligible is not None:
		keep &= eligible.any(axis=0)
	order = sorted(np.flatnonzero(keep), key=lambda i: (-probs[i], -mean[i] if highest else mean[i], names[i]))
	rows = []
	for i in order:
		r = {"name": names[i], "p": float(probs[i]), "mean": float(mean[i]),
		     "p10": float(np.nanpercentile(values[:, i], 10)), "p90": float(np.nanpercentile(values[:, i], 90))}
		if extra:
			r.update({k: v[i] for k, v in extra.items()})
		rows.append(r)
	return rows


def champion_questions(results, ix, scope: str, min_games: int = 5) -> dict:
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	arr = np.stack([_stage_sum(r, scope, "champ") for r in results])  # (sims, stats, n)
	picks, bans, wins, kills, deaths, assists = (arr[:, C[k]] for k in ("picks", "bans", "wins", "kills", "deaths", "assists"))
	presence = picks + bans
	games = np.array([sum(r["stages"][s]["n_games"] for s in (STAGES if scope == "all" else [scope])) for r in results])
	elig = picks >= min_games
	with np.errstate(divide="ignore", invalid="ignore"):
		wr = np.where(picks > 0, wins / picks, np.nan)
		kda = np.where(picks > 0, (kills + assists) / np.maximum(deaths, 1), np.nan)
	names = ix.champions
	return {
		"most_picked": _table(names, _leader_probs(picks), picks),
		"most_banned": _table(names, _leader_probs(bans), bans),
		"most_presence": _table(names, _leader_probs(presence), presence, extra={"presence_rate": list(presence.mean(0) / max(games.mean(), 1))}),
		"most_kills": _table(names, _leader_probs(kills), kills),
		"most_deaths": _table(names, _leader_probs(deaths), deaths),
		"highest_winrate": _table(names, _leader_probs(np.nan_to_num(wr, nan=-1), elig), wr, elig),
		"lowest_winrate": _table(names, _leader_probs(np.nan_to_num(wr, nan=2), elig, highest=False), wr, elig, highest=False),
		"most_pentas": _table(names, _leader_probs(arr[:, C["pentas"]]), arr[:, C["pentas"]]),
		"best_kda": _table(names, _leader_probs(np.nan_to_num(kda, nan=-1), elig), kda, elig),
		"unique_picked": _dist([(p > 0).sum() for p in picks]),
		"unique_picked_or_banned": _dist([(p > 0).sum() for p in presence]),
		"expected_presence": sorted(
			[{"name": n, "picks": float(picks[:, i].mean()), "bans": float(bans[:, i].mean()), "presence_rate": float(presence[:, i].mean() / max(games.mean(), 1)),
			  "winrate": float(np.nanmean(wr[:, i])) if np.isfinite(wr[:, i]).any() else None}
			 for i, n in enumerate(names)],
			key=lambda r: -r["presence_rate"]),
		"min_games": min_games,
	}


def actual_champions(real: dict | None, ix, scope: str) -> dict:
	"""What the games already played did, per champion (only those picked or
	banned) and per player (only those who played): the part of every stat
	that's the same in all simulations."""
	if not real:
		return {"games": 0, "champ": {}, "player": {}}
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	arr = _stage_sum({"stages": real}, scope, "champ")
	games = sum(real[s]["n_games"] for s in (STAGES if scope == "all" else [scope]))
	keys = ("picks", "bans", "wins", "kills", "deaths", "assists", "pentas")
	P = {k: i for i, k in enumerate(PLAYER_STATS)}
	pl = _stage_sum({"stages": real}, scope, "player")
	pkeys = ("games", "wins", "kills", "deaths", "assists", "firstbloods", "doubles", "triples", "quadras", "pentas", "damage", "cs", "team_kills")
	return {"games": int(games), "champ": {n: {k: int(arr[C[k], i]) for k in keys} for i, n in enumerate(ix.champions) if arr[C["picks"], i] + arr[C["bans"], i] > 0},
	        "player": {f"{nm} ({t})": {k: int(pl[P[k], i]) for k in pkeys} for i, (t, _, nm) in enumerate(ix.players) if pl[P["games"], i] > 0}}


def player_questions(results, ix, scope: str, min_games: int = 5) -> dict:
	P = {k: i for i, k in enumerate(PLAYER_STATS)}
	arr = np.stack([_stage_sum(r, scope, "player") for r in results]).astype(float)
	g, k, d, a = (arr[:, P[x]] for x in ("games", "kills", "deaths", "assists"))
	names = [f"{n} ({t})" for t, _, n in ix.players]
	elig = g >= min_games
	kda = (k + a) / np.maximum(d, 1)
	kda = np.where(g > 0, kda, np.nan)
	per_game = lambda x: np.where(g > 0, x / np.maximum(g, 1), np.nan)
	return {
		"most_kills": _table(names, _leader_probs(k), k),
		"most_deaths": _table(names, _leader_probs(d), d),
		"most_assists": _table(names, _leader_probs(a), a),
		"highest_kda": _table(names, _leader_probs(np.nan_to_num(kda, nan=-1), elig), kda, elig),
		"most_kills_per_game": _table(names, _leader_probs(np.nan_to_num(per_game(k), nan=-1), elig), per_game(k), elig),
		"most_first_bloods": _table(names, _leader_probs(arr[:, P["firstbloods"]]), arr[:, P["firstbloods"]]),
		"most_pentas": _table(names, _leader_probs(arr[:, P["pentas"]]), arr[:, P["pentas"]]),
		"most_games": _table(names, _leader_probs(g), g),
		"min_games": min_games,
	}


def team_questions(results, ix, scope: str) -> dict:
	T = {k: i for i, k in enumerate(TEAM_STATS)}
	arr = np.stack([_stage_sum(r, scope, "team") for r in results])
	g = arr[:, T["games"]]
	out = {}
	for key in ("kills", "dragons", "barons", "towers", "void_grubs", "firstbloods", "deaths"):
		v = arr[:, T[key]]
		out[f"most_{key}"] = _table(ix.teams, _leader_probs(v), v)
	with np.errstate(divide="ignore", invalid="ignore"):
		avg_len = np.where(g > 0, arr[:, T["game_seconds"]] / g / 60, np.nan)
		kpg = np.where(g > 0, arr[:, T["kills"]] / g, np.nan)
	elig = g >= 3
	out["longest_avg_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(avg_len, nan=-1), elig), avg_len, elig)
	out["shortest_avg_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(avg_len, nan=1e9), elig, highest=False), avg_len, elig, highest=False)
	out["most_kills_per_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(kpg, nan=-1), elig), kpg, elig)
	return out


def event_questions(results, scope: str) -> dict:
	stages = STAGES if scope == "all" else [scope]
	pick = lambda key, f: [f(r["stages"][s][key] for s in stages) for r in results]
	return {
		"games": _dist(pick("n_games", sum)),
		"longest_game_min": _dist(pick("longest", max), bins=True),
		"shortest_game_min": _dist(pick("shortest", min), bins=True),
		"max_kills_in_game": _dist(pick("max_game_kills", max)),
		"pentakills": _dist(pick("pentas", sum)),
	}


def _dist(values, bins: bool = False) -> dict:
	v = np.asarray(values, float)
	out = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90))}
	if bins:
		h, e = np.histogram(v, bins=12)
		out["hist"] = {"counts": h.tolist(), "edges": e.round(1).tolist()}
	else:
		c = Counter(np.round(v).astype(int).tolist())
		out["pmf"] = {str(k): c[k] / len(v) for k in sorted(c)}
	return out


def team_outcomes(results, teams: list[str]) -> list[dict]:
	n = len(results)
	rows = []
	for t in teams:
		rec = Counter(r["swiss_record"].get(t) for r in results if t in r["swiss_record"])
		in_swiss = sum(rec.values())
		rows.append({
			"team": t,
			"champion": sum(r["champion"] == t for r in results) / n,
			"final": sum(t in (r["champion"], r["finalist"]) for r in results) / n,
			"semis": sum(t in (r["champion"], r["finalist"], *r["semifinalists"]) for r in results) / n,
			"quarters": sum(t in r["quarterfinalists"] for r in results) / n,
			"swiss": in_swiss / n,
			"play_in_win": sum(r["play_in"].get(t) == 1 for r in results) / n,
			"swiss_3_0": rec.get((3, 0), 0) / n,
			"swiss_3_1": rec.get((3, 1), 0) / n,
			"swiss_3_2": rec.get((3, 2), 0) / n,
			"swiss_2_3": rec.get((2, 3), 0) / n,
			"swiss_1_3": rec.get((1, 3), 0) / n,
			"swiss_0_3": rec.get((0, 3), 0) / n,
		})
	return sorted(rows, key=lambda r: -r["champion"])


# Riot's 2026 Swiss Pick'Em: 8 teams to advance; the first three are also
# called as the 3-0, 3-1 and 3-2 team. Points per correct advancing team, plus a
# bonus when a slot pick finishes with that exact record.
SWISS_SLOTS = ("3-0", "3-1", "3-2")
SWISS_POINTS = {"advance": 10, "record": 15}


def best_swiss_pickem(outcomes: list[dict], n_adv: int = 8, points: dict | None = None) -> dict:
	"""Maximize expected points: every pick earns points["advance"] * P(advance),
	and each slot pick adds points["record"] * P(that exact record). Exhaustive
	over the plausible slot candidates; the rest are the best remaining advancers."""
	pts = points or SWISS_POINTS
	o = {r["team"]: r for r in outcomes if r["swiss"] > 0}
	prec = {k: {t: r["swiss_" + k.replace("-", "_")] for t, r in o.items()} for k in SWISS_SLOTS}
	padv = {t: sum(prec[k][t] for k in SWISS_SLOTS) for t in o}
	value = {k: {t: pts["advance"] * padv[t] + pts["record"] * prec[k][t] for t in o} for k in SWISS_SLOTS}
	cands = {k: sorted(o, key=lambda t: -value[k][t])[:8] for k in SWISS_SLOTS}
	by_adv = sorted(o, key=lambda t: -padv[t])
	best = (-1.0, None)
	for slot in itertools.product(*(cands[k] for k in SWISS_SLOTS)):
		if len(set(slot)) < len(slot):
			continue
		rest = [t for t in by_adv if t not in slot][: n_adv - len(slot)]
		ev = sum(value[k][t] for k, t in zip(SWISS_SLOTS, slot)) + pts["advance"] * sum(padv[t] for t in rest)
		if ev > best[0]:
			best = (ev, {**{k: t for k, t in zip(SWISS_SLOTS, slot)}, "advance": rest})
	ev, picks = best
	return {"expected_points": ev, "max_points": n_adv * pts["advance"] + len(SWISS_SLOTS) * pts["record"], "points": pts, "picks": picks,
	        "expected_correct": sum(padv[t] for t in [*(picks[k] for k in SWISS_SLOTS), *picks["advance"]]),
	        "detail": {t: {"advance": padv[t], **{k: prec[k][t] for k in SWISS_SLOTS}} for t in o}}


def swiss_matchups(results: list[dict]) -> list[dict]:
	"""Swiss series whose pairing is already set (the same in every simulation:
	the drawn Round 1, or a later round once its draw is real), with each side's
	chance of winning. round is the full label ("R1 0-0"), matching played_slots."""
	n = len(results)
	pairs: dict[tuple, list] = {}
	for r in results:
		for st, rnd, a, b, wa, wb in r["series"]:
			if st == "swiss":
				pairs.setdefault((rnd, a, b), []).append(wa > wb)
	out = []
	for (rnd, a, b), w in pairs.items():
		if len(w) < n:
			continue
		out.append({"round": rnd, "a": a, "b": b, "p_a": sum(w) / n})
	return sorted(out, key=lambda m: m["round"])


def _sections(run: dict, results: list[dict]) -> dict:
	"""Every aggregate the dashboard shows, over `results` (all sims, or only those
	won by one team for the "given champion" view)."""
	ix = run["ix"]
	outcomes = team_outcomes(results, ix.teams)
	from pickems.knockout import most_likely_records  # noqa: PLC0415

	out = {"n": len(results), "outcomes": outcomes, "swiss_pickem": best_swiss_pickem(outcomes), "swiss_matchups": swiss_matchups(results), "records": most_likely_records(outcomes), "scopes": {}}
	if all("play_in_slots" in r for r in results):
		out["pickems"] = pickem_answers(run, results)
	if all("elo" in r for r in results):
		out["elo"] = elo_section(results, ix, run.get("elo_pre"), run.get("elo_scale"))
	out["player_stats"] = player_stats(results, ix)
	for scope in SCOPES:
		out["scopes"][scope] = {
			"champions": champion_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"players": player_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"teams": team_questions(results, ix, scope),
			"event": event_questions(results, scope),
		}
	return out


# Leader tables list every entity in the main view; each "given champion" view keeps this many.
GIVEN_TOP = 15


def _cap_leaders(sec: dict, top: int = GIVEN_TOP) -> dict:
	"""Trim every leader table in a section to its first `top` rows (keeps the page small).
	Expected presence is the meta table's per-champion totals, not a leaderboard: it keeps
	every champion that was picked or banned (the page subtracts real games from these, so
	a missing row would show as negative simulated picks)."""
	for sc in sec["scopes"].values():
		for kind in ("champions", "players", "teams"):
			for k, v in sc[kind].items():
				if k == "expected_presence":
					sc[kind][k] = [r for r in v if r["picks"] or r["bans"]]
				elif isinstance(v, list):
					sc[kind][k] = v[:top]
	return sec


# A team gets a "given champion" view when it wins at least this many simulations
# (every team that wins any; the page flags small samples).
GIVEN_MIN_SIMS = 1
def build(run: dict) -> dict:
	results, ix = run["results"], run["ix"]
	from pickems.images import load as load_images  # noqa: PLC0415

	from pickems.history import load_games, worlds_share  # noqa: PLC0415

	ww = run.get("worlds_weight", 0.0)
	report = {
		"meta": {"sims": run["sims"], "run_at": run["run_at"], "model": run["manifest"], "teams": run["teams"],
		         "worlds_weight": ww, "worlds_share": worlds_share(load_games(), ww) if ww else None, "tuned_params": run.get("tuned_params"), "latest_game": run.get("latest_game"),
		         "worlds_buff_scale": run.get("worlds_buff_scale"),
		         "data_through": run.get("data_through")},
		**_sections(run, results),
		"projection": run["projection"].reset_index().round(4).to_dict(orient="records"),
		"images": load_images(),
		"player_stat_cols": PLAYER_STATS,
	}
	report["champs"] = list(ix.champions)
	# Games already played (pickems.sim.played): the page splits champion stats into played / predicted / total.
	report["played"] = run.get("played") or []
	report["played_series"] = played_slots(results[0]) if results else []
	report["actual"] = {scope: actual_champions(run.get("real"), ix, scope) for scope in SCOPES}
	if all("play_in_slots" in r for r in results):
		report["knockout"] = knockout_section(run, report["outcomes"])
		# Single-run mode: the page lists every simulation and fetches each one's
		# single_run() file on demand (written next to the page by dashboard.render).
		report["run_index"] = [[i, r["champion"]] for i, r in enumerate(results)]
	wins = Counter(r["champion"] for r in results)
	report["given"] = {t: _cap_leaders(_sections(run, [r for r in results if r["champion"] == t]))
	                   for t, c in wins.most_common() if c >= GIVEN_MIN_SIMS}
	return report


# --- Riot Pick'Em questions (Worlds 2026) -------------------------------------------

def _placement(r: dict, team: str) -> int:
	"""Final placement rank (1 = champion; tied ranks share the stage's best)."""
	if r["champion"] == team:
		return 1
	if r["finalist"] == team:
		return 2
	if team in r["semifinalists"]:
		return 3
	if team in r["quarterfinalists"]:
		return 5
	if team in r["swiss_record"]:
		w, _ = r["swiss_record"][team]
		return {2: 9, 1: 12, 0: 15}[w]
	return 16 + r["play_in"].get(team, 4)


def _player_index(ix, name: str) -> int:
	return next(i for i, (_, _, n) in enumerate(ix.players) if n == name)


def _choice(question: str, pts: int, options: list[tuple[str, float]], note: str = "", detail: list[dict] | None = None) -> dict:
	options = sorted(options, key=lambda o: -o[1])
	return {"question": question, "pts": pts, "answer": options[0][0], "p": options[0][1],
	        "options": [{"label": l, "p": p} for l, p in options], "note": note, "detail": detail or []}


def _leader_choice(question, names, values, eligible=None, highest=True, fmt="{:.1f}", note="", cols=None):
	"""cols: {label: ((sims, n) array, fmt)} extra per-entity averages for the detail table.
	The detail lists every entity with a chance or an average (the page shows the top 8 until expanded)."""
	probs = _leader_probs(values, eligible, highest)
	means = np.nanmean(np.where(eligible, values, np.nan) if eligible is not None else values, axis=0) if values.size else []
	order = np.argsort(-probs)
	opts = [(names[i], float(probs[i])) for i in order if probs[i] > 0]
	cols = cols or {}
	cm = {k: np.nanmean(v, axis=0) for k, (v, _) in cols.items()}
	detail = [{"label": names[i], "p": float(probs[i]), "value": fmt.format(means[i]) if np.isfinite(means[i]) else "-",
	           "extra": [f.format(cm[k][i]) if np.isfinite(cm[k][i]) else "-" for k, (_, f) in cols.items()]}
	          for i in order if probs[i] > 0 or np.isfinite(means[i])]
	out = _choice(question, 50, opts, note, detail)
	out["detail_cols"] = list(cols)
	return out


def _viz_dist(title: str, values, mark: float | None = None, unit: str = "", bins: bool = False) -> dict:
	"""A distribution panel for a Crystal Ball card; mark = the question's threshold."""
	return {"type": "dist", "title": title, "dist": _dist(values, bins=bins), "mark": mark, "unit": unit}


def pickem_answers(run: dict, results: list[dict] | None = None) -> dict:
	results = run["results"] if results is None else results
	ix, teams = run["ix"], run["teams"]
	n = len(results)
	C = {k: i for i, k in enumerate(CHAMP_STATS)}
	P = {k: i for i, k in enumerate(PLAYER_STATS)}
	champ = np.stack([_stage_sum(r, "all", "champ") for r in results]).astype(float)
	player = np.stack([_stage_sum(r, "all", "player") for r in results]).astype(float)
	games = np.array([sum(r["stages"][s]["n_games"] for s in STAGES) for r in results], float)
	picks, bans = champ[:, C["picks"]], champ[:, C["bans"]]
	pnames = [f"{nm} ({t})" for t, _, nm in ix.players]
	pg = player[:, P["games"]]
	elig_c = picks >= 5
	elig_p = pg >= 5
	with np.errstate(divide="ignore", invalid="ignore"):
		presence = (picks + bans) / games[:, None]
		wr = champ[:, C["wins"]] / picks
		dpg = champ[:, C["deaths"]] / picks
		kda = (player[:, P["kills"]] + player[:, P["assists"]]) / np.maximum(player[:, P["deaths"]], 1)
		dmg = player[:, P["damage"]] / pg
		kp = (player[:, P["kills"]] + player[:, P["assists"]]) / player[:, P["team_kills"]]
	# Entity-specific questions come from the config (`pickem:`), so a backtest
	# of another year asks the same questions about that year's field.
	pk = {"duel": ["Peyz", "Gumayusi"], "milestone": {"player": "Faker", "kills": 84, "text": "Will Faker reach 700 career Worlds Kills this year? (616 entering: needs 84)"},
	      "team_duel": ["HLE", "T1"], "region": "LCS", **run["cfg"].get("pickem", {})}
	q = []
	cg = {"Games": (picks, "{:.1f}")}
	real = run.get("real")
	if real:  # split the champion totals into what's been played and what's predicted for the games still ahead
		played = _stage_sum({"stages": real}, "all", "champ").astype(float)[None]
		split = lambda label, col, f: {f"{label} played": (np.repeat(played[:, C[col]], n, 0), "{:.0f}"),
		                               f"{label} predicted": (champ[:, C[col]] - played[:, C[col]], f), label: (champ[:, C[col]], f)}
		cg = split("Games", "picks", "{:.1f}")
	pgc = {"Games": (pg, "{:.1f}")}
	q.append(_leader_choice("Which champion will have the highest Presence (pct of games picked or banned) across all drafts?", ix.champions, presence, fmt="{:.0%}",
	                        cols={**(split("Picks", "picks", "{:.1f}") if real else {"Picks": (picks, "{:.1f}")}),
	                              **(split("Bans", "bans", "{:.1f}") if real else {"Bans": (bans, "{:.1f}")})},
	                        note=f"Average {games.mean():.0f} games per simulation."))
	q.append(_leader_choice("Which champion will have the highest Win Rate (min 5 games played)?", ix.champions, np.nan_to_num(wr, nan=-1), elig_c, fmt="{:.0%}",
	                        note="Small-sample winners dominate this one; the favourite is rarely above 10%.", cols={**cg, **(split("Wins", "wins", "{:.1f}") if real else {"Wins": (champ[:, C["wins"]], "{:.1f}")})}))
	q.append(_leader_choice("Which champion will have the highest average Deaths per Game (min 5 games played)?", ix.champions, np.nan_to_num(dpg, nan=-1), elig_c, fmt="{:.2f}",
	                        cols={**cg, **(split("Deaths", "deaths", "{:.1f}") if real else {"Deaths": (champ[:, C["deaths"]], "{:.1f}")})}))
	q.append(_leader_choice("Which pro will finish with the highest average KDA?", pnames, np.nan_to_num(kda, nan=-1), elig_p, fmt="{:.2f}",
	                        note="Assumes a 5-game minimum, like the champion questions.",
	                        cols={**pgc, "K": (player[:, P["kills"]], "{:.0f}"), "D": (player[:, P["deaths"]], "{:.0f}"), "A": (player[:, P["assists"]], "{:.0f}")}))
	q.append(_leader_choice("Which pro will finish with the highest average Damage per Game?", pnames, np.nan_to_num(dmg, nan=-1), elig_p, fmt="{:,.0f}", cols=pgc))
	q.append(_leader_choice("Which pro will finish with the highest average Kill Participation ((Kills + Assists) / Team Kills)?", pnames, np.nan_to_num(kp, nan=-1), elig_p, fmt="{:.0%}",
	                        cols={**pgc, "K+A": (player[:, P["kills"]] + player[:, P["assists"]], "{:.0f}"), "Team K": (player[:, P["team_kills"]], "{:.0f}")}))
	k = player[:, P["kills"]]
	d1, d2 = pk["duel"]
	peyz, guma = _player_index(ix, d1), _player_index(ix, d2)
	tie = float((k[:, peyz] == k[:, guma]).mean())
	q.append(_choice(f"{d1} vs {d2}: Who will finish with more total Kills?", 100,
	                 [(d1, float((k[:, peyz] > k[:, guma]).mean()) + tie / 2), (d2, float((k[:, guma] > k[:, peyz]).mean()) + tie / 2)],
	                 f"Mean kills: {d1} {k[:, peyz].mean():.0f}, {d2} {k[:, guma].mean():.0f}. Ties ({tie:.1%}) split."))
	ms = pk["milestone"]
	faker = k[:, _player_index(ix, ms["player"])]
	q.append(_choice(ms["text"], 100,
	                 [("Yes", float((faker >= ms["kills"]).mean())), ("No", float((faker < ms["kills"]).mean()))],
	                 f"Simulated {ms['player']} kills this Worlds: median {np.median(faker):.0f}, 10-90% {np.percentile(faker, 10):.0f}-{np.percentile(faker, 90):.0f}."))
	teemo = picks[:, ix.champions.index("Teemo")]
	q.append(_choice("Will Teemo be picked at least once?", 100, [("Yes", float((teemo > 0).mean())), ("No", float((teemo == 0).mean()))],
	                 f"Expected Teemo picks: {teemo.mean():.2f}" + (f" ({int(_stage_sum({"stages": real}, 'all', 'champ')[C['picks'], ix.champions.index('Teemo')])} played so far)." if real else ".")))
	uniq = (picks > 0).sum(axis=1)
	edges = list(range(int(np.percentile(uniq, 1)) // 5 * 5, int(np.percentile(uniq, 99)) + 5, 5))
	bins = [(f"{a}-{a + 4}", float(((uniq >= a) & (uniq <= a + 4)).mean())) for a in edges]
	q.append(_choice("How many unique champions will be picked?", 50, bins,
	                 f"Median {np.median(uniq):.0f}, 10-90% {np.percentile(uniq, 10):.0f}-{np.percentile(uniq, 90):.0f}. Pick the bucket Riot offers that covers the median."))
	under = np.array([sum(r["stages"][s]["games_under_32"] for s in STAGES) for r in results])
	over = np.array([sum(r["stages"][s]["games_over_32"] for s in STAGES) for r in results])
	q.append(_choice("Will more games finish before or after the 32 minute mark?", 50,
	                 [("Before 32:00", float((under > over).mean() + (under == over).mean() / 2)), ("After 32:00", float((over > under).mean() + (under == over).mean() / 2))],
	                 f"Average {under.mean():.0f} games under vs {over.mean():.0f} over."))
	mk = np.array([max(r["stages"][s]["max_player_kills"] for s in STAGES) for r in results])
	q.append(_choice("Will we see any pro get at least 15 kills in a single game?", 50, [("Yes", float((mk >= 15).mean())), ("No", float((mk < 15).mean()))],
	                 "Held-out check: the sim produces a 15-kill game slightly more often than real 2026 games (2.9% vs 2.1% of games), so this leans a bit high."))
	mc = np.array([max(r["stages"][s]["max_player_cs"] for s in STAGES) for r in results])
	q.append(_choice("Will we see any pro reach a Creep Score (CS) of at least 450 in a single game?", 50, [("Yes", float((mc >= 450).mean())), ("No", float((mc < 450).mean()))],
	                 "Real 2026 major-league games: 5.9% have a 450+ CS player; ~100 games at Worlds."))
	roles = Counter(r["finals_mvp"]["role"] for r in results)
	mvp_players = Counter(f'{r["finals_mvp"]["player"]} ({r["finals_mvp"]["team"]})' for r in results)
	q.append(_choice("What will be the role of the Finals MVP winner?", 50, [(rl.capitalize(), roles.get(rl, 0) / n) for rl in ("top", "jng", "mid", "bot", "sup")],
	                 "MVP is a vote; modelled as the champion's player with the best series KDA and damage relative to role. Recent Finals MVPs went mostly to mid, jungle and bot.",
	                 [{"label": p, "p": c / n, "value": ""} for p, c in mvp_players.most_common()]))
	regions = Counter(len({teams[t]["region"] for t in r["quarterfinalists"]}) for r in results)
	q.append(_choice("How many different regions will be represented in Knockouts?", 50, [(str(k_), v / n) for k_, v in sorted(regions.items())]))
	lcs = sum(any(teams[t]["region"] == "LCS" for t in r["quarterfinalists"]) for r in results) / n
	q.append(_choice("Will a team from LCS reach Knockouts?", 50, [("Yes", lcs), ("No", 1 - lcs)]))
	hle = np.array([_placement(r, "HLE") for r in results])
	t1 = np.array([_placement(r, "T1") for r in results])
	same = float((hle == t1).mean())
	q.append(_choice("Which team will place higher: HLE or T1?", 50, [("HLE", float((hle < t1).mean())), ("T1", float((t1 < hle).mean()))],
	                 f"Both go out in the same round in {same:.0%} of sims (e.g. both lose quarterfinals); those are left out of the split."))
	short = []
	for r in results:
		st = min(STAGES, key=lambda s: r["stages"][s]["shortest"])
		short.append(r["stages"][st]["shortest_winner"])
	sc = Counter(short)
	q.append(_choice("Which team will win the shortest game (duration)?", 50, [(teams[t]["name"], c / n) for t, c in sc.most_common(10)]))
	# Extra context per card (same order as above).
	q[6]["viz"] = [_viz_dist(f"{d1} kills minus {d2} kills (gold: {d1} ahead)", k[:, peyz] - k[:, guma], mark=1),
	               _viz_dist(f"{d1} kills", k[:, peyz]), _viz_dist(f"{d2} kills", k[:, guma])]
	# Probabilities of single-run answers that aren't options (split evenly above).
	q[6]["other"] = {"Tie": tie}
	q[10]["other"] = {"Tie": float((under == over).mean())}
	q[16]["other"] = {"Same round": same}
	q[7]["viz"] = [_viz_dist(f"{ms['player']} kills this Worlds", faker, mark=ms["kills"])]
	q[8]["viz"] = [_viz_dist("Teemo picks", teemo, mark=1), {"type": "stats", "rows": [["Sims with at least one Teemo pick", f"{(teemo > 0).mean():.1%}"], ["Avg picks", f"{teemo.mean():.2f}"]]}]
	q[9]["viz"] = [_viz_dist("Unique champions picked", uniq),
	               {"type": "champ_grid", "title": "Chance each champion is picked at least once",
	                "items": sorted([[c, float((picks[:, j] > 0).mean())] for j, c in enumerate(ix.champions)], key=lambda x: -x[1])}]
	q[9]["exact"] = {str(v): c / n for v, c in Counter(uniq.tolist()).items()}
	q[10]["viz"] = [_viz_dist("Games before 32:00 minus games after (gold: more before)", under - over, mark=1), _viz_dist("Games before 32:00", under), _viz_dist("Games after 32:00", over)]
	q[11]["viz"] = [_viz_dist("Most kills by one pro in a game", mk, mark=15)]
	q[12]["viz"] = [_viz_dist("Highest CS by one pro in a game", mc, mark=450, bins=True)]
	q[13]["viz"] = [{"type": "stats", "rows": [[f"{o['label']} (all players)", f"{o['p']:.1%}"] for o in q[13]["options"]]}]
	reg_names = sorted({t["region"] for t in teams.values()})
	q[14]["viz"] = [{"type": "region_grid", "title": "Chance each region has a team in Knockouts",
	                 "items": [[rg, float(np.mean([any(teams[t]["region"] == rg for t in r["quarterfinalists"]) for r in results]))] for rg in reg_names]}]
	o = {x["team"]: x for x in team_outcomes(results, ix.teams)}
	q[15]["viz"] = [{"type": "team_table", "title": "LCS teams", "cols": ["Reach Swiss", "Knockouts"],
	                 "rows": [[c, f'{o[c]["swiss"]:.0%}', f'{o[c]["quarters"]:.0%}'] for c, t in teams.items() if t["region"] == "LCS"]}]
	place = lambda arr: [f"{(arr == v).mean():.0%}" for v in (1, 2, 3, 5, 9, 12, 15)] + [f"{(arr > 15).mean():.0%}"]
	q[16]["viz"] = [{"type": "team_table", "title": "Where each finishes", "cols": ["Win", "Final", "Semis", "Quarters", "2-3", "1-3", "0-3", "Play-In"],
	                 "rows": [["HLE", *place(hle)], ["T1", *place(t1)]]}]
	slen = [min(r["stages"][s_]["shortest"] for s_ in STAGES) for r in results]
	q[17]["viz"] = [_viz_dist("Shortest game of the event", slen, unit=" min", bins=True)]
	return {"crystal_ball": q, "play_in_bracket": best_play_in_bracket(results, run["cfg"])}


def best_play_in_bracket(results: list[dict], cfg: dict) -> dict:
	"""Enumerate every consistent 6-pick bracket; maximize expected points
	(5 per match, 10 for the qualifier/finals)."""
	if cfg["play_in"].get("format") == "single":
		return {}
	(a1, b1), (a2, b2) = cfg["play_in"]["upper_semis"]
	s1, s2 = f"UB semi {a1}-{b1}", f"UB semi {a2}-{b2}"
	pts = {s1: 5, s2: 5, "UB final": 5, "LB semi": 5, "LB final": 5, "Qualifier": 10}
	slots = [r["play_in_slots"] for r in results]
	best = []
	for w1 in (a1, b1):
		l1 = b1 if w1 == a1 else a1
		for w2 in (a2, b2):
			l2 = b2 if w2 == a2 else a2
			for ubf in (w1, w2):
				ubf_l = w2 if ubf == w1 else w1
				for lbs in (l1, l2):
					for lbf in (ubf_l, lbs):
						for q in (ubf, lbf):
							picks = {s1: w1, s2: w2, "UB final": ubf, "LB semi": lbs, "LB final": lbf, "Qualifier": q}
							ev = np.mean([sum(pts[k] for k, v in picks.items() if s.get(k) == v) for s in slots])
							best.append((ev, picks))
	best.sort(key=lambda x: -x[0])
	ev, picks = best[0]
	match_p = {k: float(np.mean([s.get(k) == v for s in slots])) for k, v in picks.items()}
	return {"picks": picks, "expected_points": float(ev), "max_points": 35, "pick_prob": match_p,
	        "runner_up": {"picks": best[1][1], "expected_points": float(best[1][0])}}


# --- Elo and per-player stats --------------------------------------------------------

def _spread(start: np.ndarray, end: np.ndarray) -> list[dict]:
	"""start / end: (sims, n) ratings -> per entity start, mean end and 10-90% of end."""
	return [{"start": float(start[0, i]), "mean": float(end[:, i].mean()), "p10": float(np.percentile(end[:, i], 10)),
	         "p90": float(np.percentile(end[:, i], 90))} for i in range(start.shape[1])]


ELO_BASE = 1500.0  # draftpredict's default rating, which scaling leaves fixed


def standard_elo(x, ix, scale: dict | None, team=None):
	"""Ratings on the standard Elo scale, so a gap of d points means a 1 / (1 + 10^(-d/400)) win chance (see runner.elo_scale).
	x: team ratings (entity axis last), or with `team` (the same-shaped team ratings) player ratings. The win model adds the team's rating
	gap and each player's, so a player's figure is team plus player: the win chance of that team with that player in the role, the rest
	average, against an average side. Runs that didn't record a scale are left as they were."""
	x = np.asarray(x, dtype=float)
	if not scale:
		return x
	if team is None:
		return ELO_BASE + scale["team"] * (x - ELO_BASE)
	tix = np.array([ix.tix[t] for t, _, _ in ix.players])
	sp = np.array([scale["player"][ro] for _, ro, _ in ix.players])
	return ELO_BASE + scale["team"] * (np.asarray(team, dtype=float)[..., tix] - ELO_BASE) + sp * (x - ELO_BASE)


def elo_section(results, ix, pre: dict | None = None, scale: dict | None = None) -> dict:
	"""Team and player Elo going into Worlds and after it (walked forward through each simulation, see pickems.sim.walk).
	start is the rating the sims start from, i.e. after the games already played; pre (runner.elo_before_worlds,
	None when nothing is played or for runs made before it was recorded) is the rating before Worlds."""
	tk = lambda k: k.replace("player", "team")
	conv = lambda src, k: standard_elo(src(k), ix, scale, src(tk(k)) if k.startswith("player") else None)
	E = lambda k: conv(lambda j: np.stack([r["elo"][j] for r in results]), k)
	P = lambda k, n: [float(x) for x in conv(lambda j: pre[j], k)] if pre is not None and k in pre else [None] * n
	# Raw (unblended) rating of each (pre / start / mean end); blended less raw is the region's adjustment. Absent in runs made before it was recorded.
	raw = "team_raw_start" in results[0]["elo"]
	R = lambda k, n: [{"pre": p, "start": float(s[0, i]), "mean": float(e[:, i].mean())} for i, p, s, e in
	                  zip(range(n), P(k + "_raw", n), [E(k + "_raw_start")] * n, [E(k + "_raw_end")] * n)] if raw else [None] * n
	teams = [{"team": t, "pre": p, "raw": rg, **x} for t, p, rg, x in zip(ix.teams, P("team", len(ix.teams)), R("team", len(ix.teams)), _spread(E("team_start"), E("team_end")))]
	players = [{"i": i, "team": t, "role": ro, "name": n, "pre": p, "raw": rg, **x}
	           for i, ((t, ro, n), p, rg, x) in enumerate(zip(ix.players, P("player", len(ix.players)), R("player", len(ix.players)), _spread(E("player_start"), E("player_end"))))]
	return {"teams": teams, "players": players, "played": pre is not None}


def player_stats(results, ix) -> list[dict]:
	"""Each player's tournament stats: mean totals per simulation by stage
	("all" plus each stage, zero games counted as zero) and the champions they
	played (mean games per simulation, pooled win rate)."""
	n = len(results)
	out = [{"i": i, "team": t, "role": ro, "name": nm, "stages": {}} for i, (t, ro, nm) in enumerate(ix.players)]
	for scope in SCOPES:
		arr = np.stack([_stage_sum(r, scope, "player") for r in results]).astype(float)  # (sims, stats, players)
		played = arr[:, PLAYER_STATS.index("games")] > 0
		for i, o in enumerate(out):
			o["stages"][scope] = {"mean": arr[:, :, i].mean(0).tolist(), "p_played": float(played[:, i].mean())}
	# Real games each player has played on each champion so far (the same in every simulation: read off the first one's game logs).
	real_pc: dict[tuple[int, int], list[int]] = {}  # (player, champion) -> [games, wins]
	roles = ["top", "jng", "mid", "bot", "sup"]
	for series in results[0].get("series_games") or []:
		for g in series:
			if g.get("p_blue") is not None:
				continue
			for team, champs, names, won in ((g["blue"], g["bc"], g.get("bn"), g["blue_win"]), (g["red"], g["rc"], g.get("rn"), not g["blue_win"])):
				for r, c in enumerate(champs):
					pi = ix.pnix.get((team, roles[r], names[r]), ix.pix[(team, roles[r])]) if names and names[r] else ix.pix[(team, roles[r])]
					acc = real_pc.setdefault((pi, ix.cix[c]), [0, 0])
					acc[0] += 1
					acc[1] += int(won)
	if all("player_champs" in r for r in results):
		pc: dict[tuple[int, int], list[int]] = {}
		for r in results:
			for key, (g, w) in r["player_champs"].items():
				acc = pc.setdefault(key, [0, 0])
				acc[0] += g
				acc[1] += w
		for (pi, ci), (g, w) in pc.items():
			out[pi].setdefault("champs", []).append([ix.champions[ci], g / n, w / g, *real_pc.get((pi, ci), [0, 0])])
		for o in out:
			o["champs"] = sorted(o.get("champs", []), key=lambda x: -x[1])[:15]
	return out


# --- single-run mode ------------------------------------------------------------------

STAGE_NAME = {1: "Champion", 2: "Finalist", 3: "Semifinals", 5: "Quarterfinals", 9: "Swiss 2-3", 12: "Swiss 1-3", 15: "Swiss 0-3"}


def _pack_game(g: dict, cix: dict) -> dict:
	"""Shrink a game log for the page: champions become indices into
	report["champs"], each draft action one int (champ * 4 + 2 * ban + red)."""
	return {**g, "draft": [cix[c] * 4 + 2 * (a == "b") + (s == "R") for a, s, c in g["draft"]],
	        "bc": [cix[c] for c in g["bc"]], "rc": [cix[c] for c in g["rc"]]}


def played_slots(r: dict) -> list[dict]:
	"""Series already played (every game replayed as it happened, so no p_blue),
	with the bracket slot each landed in: the same in every simulation, so the
	page can show them as results instead of picks."""
	games = r.get("series_games") or []
	return [{"stage": st, "round": rnd, "a": a, "b": b, "wa": wa, "wb": wb, "winner": a if wa > wb else b}
	        for (st, rnd, a, b, wa, wb), g in zip(r["series"], games) if g and all(x.get("p_blue") is None for x in g)]


def _series_rows(r: dict, stage: str, cix: dict | None = None) -> list[dict]:
	games = r.get("series_games") or [None] * len(r["series"])  # runs from before per-game logging have none
	return [{"round": rnd, "a": a, "b": b, "wa": wa, "wb": wb, "winner": a if wa > wb else b,
	         "games": [_pack_game(x, cix) for x in g] if g and cix else None}
	        for (st, rnd, a, b, wa, wb), g in zip(r["series"], games) if st == stage]


def run_files(run: dict):
	"""(index, single_run payload) for every simulation, for dashboard.render to write out."""
	if all("play_in_slots" in r for r in run["results"]):
		for i, r in enumerate(run["results"]):
			yield i, single_run(r, i, run["ix"], run["teams"], run.get("elo_scale"))


def single_run(r: dict, i: int, ix, teams: dict, scale: dict | None = None) -> dict:
	"""One simulated tournament as it happened: brackets plus that run's own
	answer to each Crystal Ball question (same order as pickem_answers)."""
	C = {k: j for j, k in enumerate(CHAMP_STATS)}
	P = {k: j for j, k in enumerate(PLAYER_STATS)}
	champ = _stage_sum(r, "all", "champ").astype(float)
	pl = _stage_sum(r, "all", "player").astype(float)
	games = sum(r["stages"][s]["n_games"] for s in STAGES)
	picks, bans = champ[C["picks"]], champ[C["bans"]]
	names = ix.champions
	pnames = [f"{n} ({t})" for t, _, n in ix.players]
	pg = pl[P["games"]]

	def lead(values, labels, eligible, fmt):
		v = np.where(eligible, values, -np.inf)
		best = v.max()
		who = [labels[j] for j in np.flatnonzero(v == best)]
		return {"answer": " / ".join(who), "value": fmt.format(best)}

	with np.errstate(divide="ignore", invalid="ignore"):
		ce, pe = picks >= 5, pg >= 5
		k, a, d = pl[P["kills"]], pl[P["assists"]], pl[P["deaths"]]
		ans = [
			lead((picks + bans) / games, names, np.ones_like(picks, bool), "{:.0%} presence"),
			lead(champ[C["wins"]] / picks, names, ce, "{:.0%} win rate"),
			lead(champ[C["deaths"]] / picks, names, ce, "{:.2f} deaths/game"),
			lead((k + a) / np.maximum(d, 1), pnames, pe, "{:.2f} KDA"),
			lead(pl[P["damage"]] / pg, pnames, pe, "{:,.0f} dmg/game"),
			lead((k + a) / pl[P["team_kills"]], pnames, pe, "{:.0%} KP"),
		]
	peyz, guma = _player_index(ix, "Peyz"), _player_index(ix, "Gumayusi")
	ans.append({"answer": "Peyz" if k[peyz] > k[guma] else "Gumayusi" if k[guma] > k[peyz] else "Tie", "value": f"{k[peyz]:.0f} vs {k[guma]:.0f} kills"})
	fk = k[_player_index(ix, "Faker")]
	ans.append({"answer": "Yes" if fk >= 84 else "No", "value": f"{fk:.0f} kills ({616 + fk:.0f} career)"})
	teemo = picks[names.index("Teemo")]
	ans.append({"answer": "Yes" if teemo > 0 else "No", "value": f"{teemo:.0f} picks"})
	ans.append({"answer": str(int((picks > 0).sum())), "value": "unique champions"})
	under = sum(r["stages"][s]["games_under_32"] for s in STAGES)
	over = sum(r["stages"][s]["games_over_32"] for s in STAGES)
	ans.append({"answer": "Before 32:00" if under > over else "After 32:00" if over > under else "Tie", "value": f"{under} before, {over} after"})
	mk = max(r["stages"][s]["max_player_kills"] for s in STAGES)
	ans.append({"answer": "Yes" if mk >= 15 else "No", "value": f"max {mk} kills"})
	mc = max(r["stages"][s]["max_player_cs"] for s in STAGES)
	ans.append({"answer": "Yes" if mc >= 450 else "No", "value": f"max {mc} CS"})
	mvp = r["finals_mvp"]
	ans.append({"answer": mvp["role"].capitalize(), "value": f'{mvp["player"]} ({mvp["team"]})'})
	regs = sorted({teams[t]["region"] for t in r["quarterfinalists"]})
	ans.append({"answer": str(len(regs)), "value": ", ".join(regs)})
	lcs = [t for t in r["quarterfinalists"] if teams[t]["region"] == "LCS"]
	ans.append({"answer": "Yes" if lcs else "No", "value": ", ".join(lcs) or "none"})
	ph, pt = _placement(r, "HLE"), _placement(r, "T1")
	ans.append({"answer": "HLE" if ph < pt else "T1" if pt < ph else "Same round", "value": f"HLE {STAGE_NAME.get(ph, ph)}, T1 {STAGE_NAME.get(pt, pt)}"})
	st = min(STAGES, key=lambda s: r["stages"][s]["shortest"])
	sw = r["stages"][st]["shortest_winner"]
	ans.append({"answer": teams[sw]["name"], "value": f'{r["stages"][st]["shortest"]:.1f} min'})
	extra = {}
	if "elo" in r:
		el = r["elo"]
		extra["elo"] = {"team": np.round(standard_elo(el["team_end"], ix, scale), 2).tolist(), "player": np.round(standard_elo(el["player_end"], ix, scale, el["team_end"]), 1).tolist()}
		if "team_raw_end" in el:
			extra["elo"].update(team_raw=np.round(standard_elo(el["team_raw_end"], ix, scale), 2).tolist(), player_raw=np.round(standard_elo(el["player_raw_end"], ix, scale, el["team_raw_end"]), 1).tolist())
	# The game logs carry K/D/A/CS/damage/gold; first bloods and multikills come from here (sparse: players with any).
	cols = [P[k] for k in ("firstbloods", "doubles", "triples", "quadras", "pentas")]
	extra["pmk"] = {str(j): pl[cols, j].astype(int).tolist() for j in range(pl.shape[1]) if pl[cols, j].any()}
	return {
		**extra, "id": i, "champion": r["champion"],
		"play_in": _series_rows(r, "play_in", ix.cix), "swiss": _series_rows(r, "swiss", ix.cix), "knockout": _series_rows(r, "knockout", ix.cix),
		"swiss_record": {t: f"{w}-{l}" for t, (w, l) in r["swiss_record"].items()},
		"answers": ans,
	}


def knockout_section(run: dict, outcomes: list[dict]) -> dict:
	from pickems.knockout import best_bracket, most_likely_records, placeholder_draw  # noqa: PLC0415
	from pickems.sim.runner import OUT  # noqa: PLC0415

	teams, cfg = run["teams"], run["cfg"]
	records = most_likely_records(outcomes)
	ko = cfg.get("knockouts") or {}
	real = bool(ko.get("draw"))
	if real:
		draw = [list(x) for x in ko["draw"]]
	else:
		adv = {t: r for t, r in records.items() if r.startswith("3-")}
		strength = {o["team"]: o["champion"] * 3 + o["semis"] for o in outcomes}
		draw = placeholder_draw(adv, {k: v["region"] for k, v in teams.items()}, strength)
	out = {"records": records, "draw": draw, "draw_is_real": real}
	# This run's own head-to-head file (scripts/h2h.py --run <name>), else the shared one older runs used.
	path = next((p for p in (OUT / f"h2h_{run.get('name')}.pkl", OUT / "h2h_knockout.pkl") if p.exists()), None)
	if path is not None:
		h = pickle.loads(path.read_bytes())
		if set(h["teams"]) >= {t for pair in draw for t in pair}:
			out["bracket"] = best_bracket(draw, h["p"], ko.get("pickem_points"))
			out["h2h"] = {"teams": h["teams"], "p": h["p"], "series": h["series"], "run_at": h["run_at"]}
	return out

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


def _table(names, probs, values, top=15, extra=None) -> list[dict]:
	order = np.argsort(-probs)[:top]
	rows = []
	for i in order:
		if probs[i] <= 0 and len(rows) >= 5:
			break
		r = {"name": names[i], "p": float(probs[i]), "mean": float(np.nanmean(values[:, i])),
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
		"highest_winrate": _table(names, _leader_probs(np.nan_to_num(wr, nan=-1), elig), wr),
		"lowest_winrate": _table(names, _leader_probs(np.nan_to_num(wr, nan=2), elig, highest=False), wr),
		"most_pentas": _table(names, _leader_probs(arr[:, C["pentas"]]), arr[:, C["pentas"]]),
		"best_kda": _table(names, _leader_probs(np.nan_to_num(kda, nan=-1), elig), kda),
		"unique_picked": _dist([(p > 0).sum() for p in picks]),
		"unique_picked_or_banned": _dist([(p > 0).sum() for p in presence]),
		"expected_presence": sorted(
			[{"name": n, "picks": float(picks[:, i].mean()), "bans": float(bans[:, i].mean()), "presence_rate": float(presence[:, i].mean() / max(games.mean(), 1)),
			  "winrate": float(np.nanmean(wr[:, i])) if np.isfinite(wr[:, i]).any() else None}
			 for i, n in enumerate(names) if presence[:, i].mean() > 0.05],
			key=lambda r: -r["presence_rate"]),
		"min_games": min_games,
	}


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
		"highest_kda": _table(names, _leader_probs(np.nan_to_num(kda, nan=-1), elig), kda),
		"most_kills_per_game": _table(names, _leader_probs(np.nan_to_num(per_game(k), nan=-1), elig), per_game(k)),
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
	out["longest_avg_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(avg_len, nan=-1), elig), avg_len)
	out["shortest_avg_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(avg_len, nan=1e9), elig, highest=False), avg_len)
	out["most_kills_per_game"] = _table(ix.teams, _leader_probs(np.nan_to_num(kpg, nan=-1), elig), kpg)
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


def best_swiss_pickem(outcomes: list[dict], n30: int = 2, n03: int = 2, n_adv: int = 5) -> dict:
	"""Riot's Swiss Pick'Em: n30 teams to go 3-0, n03 to go 0-3, n_adv more to
	advance with 3-1/3-2; a team can only be in one bucket. Maximizes expected
	correct picks (exhaustive over the plausible candidates)."""
	o = {r["team"]: r for r in outcomes if r["swiss"] > 0}
	p30 = {t: r["swiss_3_0"] for t, r in o.items()}
	p03 = {t: r["swiss_0_3"] for t, r in o.items()}
	padv = {t: r["swiss_3_1"] + r["swiss_3_2"] for t, r in o.items()}
	c30 = sorted(o, key=lambda t: -p30[t])[: n30 + 4]
	c03 = sorted(o, key=lambda t: -p03[t])[: n03 + 4]
	best = (-1, None)
	for a in itertools.combinations(c30, n30):
		for b in itertools.combinations([t for t in c03 if t not in a], n03):
			rest = sorted([t for t in o if t not in a and t not in b], key=lambda t: -padv[t])[:n_adv]
			ev = sum(p30[t] for t in a) + sum(p03[t] for t in b) + sum(padv[t] for t in rest)
			if ev > best[0]:
				best = (ev, {"3-0": list(a), "0-3": list(b), "advance": rest})
	ev, picks = best
	return {"expected_correct": ev, "picks": picks,
	        "detail": {t: {"3-0": p30[t], "0-3": p03[t], "advance_3_1_or_3_2": padv[t]} for t in o}}


def _sections(run: dict, results: list[dict]) -> dict:
	"""Every aggregate the dashboard shows, over `results` (all sims, or only those
	won by one team for the "given champion" view)."""
	ix = run["ix"]
	outcomes = team_outcomes(results, ix.teams)
	from pickems.knockout import most_likely_records  # noqa: PLC0415

	out = {"n": len(results), "outcomes": outcomes, "swiss_pickem": best_swiss_pickem(outcomes), "records": most_likely_records(outcomes), "scopes": {}}
	if all("play_in_slots" in r for r in results):
		out["pickems"] = pickem_answers(run, results)
	if all("elo" in r for r in results):
		out["elo"] = elo_section(results, ix)
	out["player_stats"] = player_stats(results, ix)
	for scope in SCOPES:
		out["scopes"][scope] = {
			"champions": champion_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"players": player_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"teams": team_questions(results, ix, scope),
			"event": event_questions(results, scope),
		}
	return out


# A team gets a "given champion" view when it wins at least this many simulations.
GIVEN_MIN_SIMS = 10
# Single-run mode samples this many simulations, plus up to GIVEN_MIN_RUNS of each
# champion's wins so every "given champion" view has runs to browse.
N_RUNS, GIVEN_MIN_RUNS = 400, 25


def _sample_runs(results: list[dict]) -> list[int]:
	rng = np.random.default_rng(0)
	pick = set(rng.choice(len(results), size=min(N_RUNS, len(results)), replace=False).tolist())
	by: dict[str, list[int]] = {}
	for i, r in enumerate(results):
		by.setdefault(r["champion"], []).append(i)
	for idx in by.values():
		have = [i for i in idx if i in pick]
		rest = [i for i in idx if i not in pick]
		need = min(GIVEN_MIN_RUNS, len(idx)) - len(have)
		if need > 0:
			pick.update(rng.choice(rest, size=need, replace=False).tolist())
	return sorted(pick)


def build(run: dict) -> dict:
	results, ix = run["results"], run["ix"]
	from pickems.images import load as load_images  # noqa: PLC0415

	from pickems.history import load_games, worlds_share  # noqa: PLC0415

	ww = run.get("worlds_weight", 0.0)
	report = {
		"meta": {"sims": run["sims"], "run_at": run["run_at"], "model": run["manifest"], "teams": run["teams"],
		         "worlds_weight": ww, "worlds_share": worlds_share(load_games(), ww) if ww else None, "tuned_params": run.get("tuned_params"), "latest_game": run.get("latest_game"),
		         "data_through": run.get("data_through")},
		**_sections(run, results),
		"projection": run["projection"].reset_index().head(60).round(4).to_dict(orient="records"),
		"images": load_images(),
		"player_stat_cols": PLAYER_STATS,
	}
	report["champs"] = list(ix.champions)
	if all("play_in_slots" in r for r in results):
		report["knockout"] = knockout_section(run, report["outcomes"])
		report["runs"] = [single_run(results[i], int(i), ix, run["teams"]) for i in _sample_runs(results)]
	wins = Counter(r["champion"] for r in results)
	report["given"] = {t: _sections(run, [r for r in results if r["champion"] == t])
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
	"""cols: {label: ((sims, n) array, fmt)} extra per-entity averages for the detail table."""
	probs = _leader_probs(values, eligible, highest)
	means = np.nanmean(np.where(eligible, values, np.nan) if eligible is not None else values, axis=0) if values.size else []
	order = np.argsort(-probs)
	opts = [(names[i], float(probs[i])) for i in order if probs[i] > 0]
	cols = cols or {}
	cm = {k: np.nanmean(v, axis=0) for k, (v, _) in cols.items()}
	detail = [{"label": names[i], "p": float(probs[i]), "value": fmt.format(means[i]) if np.isfinite(means[i]) else "-",
	           "extra": [f.format(cm[k][i]) if np.isfinite(cm[k][i]) else "-" for k, (_, f) in cols.items()]}
	          for i in order[:8] if probs[i] > 0]
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
	q = []
	cg = {"Games": (picks, "{:.1f}")}
	pgc = {"Games": (pg, "{:.1f}")}
	q.append(_leader_choice("Which champion will have the highest Presence (pct of games picked or banned) across all drafts?", ix.champions, presence, fmt="{:.0%}",
	                        cols={"Picks": (picks, "{:.1f}"), "Bans": (bans, "{:.1f}"), "of games": (np.repeat(games[:, None], picks.shape[1], 1), "{:.0f}")}))
	q.append(_leader_choice("Which champion will have the highest Win Rate (min 5 games played)?", ix.champions, np.nan_to_num(wr, nan=-1), elig_c, fmt="{:.0%}",
	                        note="Small-sample winners dominate this one; the favourite is rarely above 10%.", cols={**cg, "Wins": (champ[:, C["wins"]], "{:.1f}")}))
	q.append(_leader_choice("Which champion will have the highest average Deaths per Game (min 5 games played)?", ix.champions, np.nan_to_num(dpg, nan=-1), elig_c, fmt="{:.2f}",
	                        cols={**cg, "Deaths": (champ[:, C["deaths"]], "{:.1f}")}))
	q.append(_leader_choice("Which pro will finish with the highest average KDA?", pnames, np.nan_to_num(kda, nan=-1), elig_p, fmt="{:.2f}",
	                        note="Assumes a 5-game minimum, like the champion questions.",
	                        cols={**pgc, "K": (player[:, P["kills"]], "{:.0f}"), "D": (player[:, P["deaths"]], "{:.0f}"), "A": (player[:, P["assists"]], "{:.0f}")}))
	q.append(_leader_choice("Which pro will finish with the highest average Damage per Game?", pnames, np.nan_to_num(dmg, nan=-1), elig_p, fmt="{:,.0f}", cols=pgc))
	q.append(_leader_choice("Which pro will finish with the highest average Kill Participation ((Kills + Assists) / Team Kills)?", pnames, np.nan_to_num(kp, nan=-1), elig_p, fmt="{:.0%}",
	                        cols={**pgc, "K+A": (player[:, P["kills"]] + player[:, P["assists"]], "{:.0f}"), "Team K": (player[:, P["team_kills"]], "{:.0f}")}))
	k = player[:, P["kills"]]
	peyz, guma = _player_index(ix, "Peyz"), _player_index(ix, "Gumayusi")
	tie = float((k[:, peyz] == k[:, guma]).mean())
	q.append(_choice("Peyz vs Gumayusi: Who will finish with more total Kills?", 100,
	                 [("Peyz", float((k[:, peyz] > k[:, guma]).mean()) + tie / 2), ("Gumayusi", float((k[:, guma] > k[:, peyz]).mean()) + tie / 2)],
	                 f"Mean kills: Peyz {k[:, peyz].mean():.0f}, Gumayusi {k[:, guma].mean():.0f}. Ties ({tie:.1%}) split."))
	faker = k[:, _player_index(ix, "Faker")]
	q.append(_choice("Will Faker reach 700 career Worlds Kills this year? (616 entering: needs 84)", 100,
	                 [("Yes", float((faker >= 84).mean())), ("No", float((faker < 84).mean()))],
	                 f"Simulated Faker kills this Worlds: median {np.median(faker):.0f}, 10-90% {np.percentile(faker, 10):.0f}-{np.percentile(faker, 90):.0f}."))
	teemo = picks[:, ix.champions.index("Teemo")]
	q.append(_choice("Will Teemo be picked at least once?", 100, [("Yes", float((teemo > 0).mean())), ("No", float((teemo == 0).mean()))],
	                 f"Expected Teemo picks: {teemo.mean():.2f}."))
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
	                 [{"label": p, "p": c / n, "value": ""} for p, c in mvp_players.most_common(6)]))
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
	q[6]["viz"] = [_viz_dist("Peyz kills minus Gumayusi kills (gold: Peyz ahead)", k[:, peyz] - k[:, guma], mark=1),
	               _viz_dist("Peyz kills", k[:, peyz]), _viz_dist("Gumayusi kills", k[:, guma])]
	# Probabilities of single-run answers that aren't options (split evenly above).
	q[6]["other"] = {"Tie": tie}
	q[10]["other"] = {"Tie": float((under == over).mean())}
	q[16]["other"] = {"Same round": same}
	q[7]["viz"] = [_viz_dist("Faker kills this Worlds", faker, mark=84)]
	tb = bans[:, ix.champions.index("Teemo")]
	q[8]["viz"] = [_viz_dist("Teemo picks", teemo, mark=1), {"type": "stats", "rows": [["P(banned at least once)", f"{(tb > 0).mean():.1%}"], ["Avg bans", f"{tb.mean():.2f}"]]}]
	q[9]["viz"] = [_viz_dist("Unique champions picked", uniq),
	               {"type": "champ_grid", "title": "Chance each champion is picked at least once",
	                "items": sorted([[c, float((picks[:, j] > 0).mean())] for j, c in enumerate(ix.champions)], key=lambda x: -x[1])}]
	q[9]["exact"] = {str(v): c / n for v, c in Counter(uniq.tolist()).items()}
	q[10]["viz"] = [_viz_dist("Games before 32:00 minus games after (gold: more before)", under - over, mark=1), _viz_dist("Games before 32:00", under), _viz_dist("Games after 32:00", over)]
	q[11]["viz"] = [_viz_dist("Most kills by one pro in a game", mk, mark=15)]
	q[12]["viz"] = [_viz_dist("Highest CS by one pro in a game", mc, mark=450, bins=True)]
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


def elo_section(results, ix) -> dict:
	"""Team and player Elo going into Worlds and after it (walked forward through each simulation, see pickems.sim.walk)."""
	E = lambda k: np.stack([r["elo"][k] for r in results]).astype(float)
	teams = [{"team": t, **x} for t, x in zip(ix.teams, _spread(E("team_start"), E("team_end")))]
	players = [{"i": i, "team": t, "role": ro, "name": n, **x} for i, ((t, ro, n), x) in enumerate(zip(ix.players, _spread(E("player_start"), E("player_end"))))]
	return {"teams": teams, "players": players}


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
	if all("player_champs" in r for r in results):
		pc: dict[tuple[int, int], list[int]] = {}
		for r in results:
			for key, (g, w) in r["player_champs"].items():
				acc = pc.setdefault(key, [0, 0])
				acc[0] += g
				acc[1] += w
		for (pi, ci), (g, w) in pc.items():
			out[pi].setdefault("champs", []).append([ix.champions[ci], g / n, w / g])
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


def _series_rows(r: dict, stage: str, cix: dict | None = None) -> list[dict]:
	games = r.get("series_games") or [None] * len(r["series"])  # runs from before per-game logging have none
	return [{"round": rnd, "a": a, "b": b, "wa": wa, "wb": wb, "winner": a if wa > wb else b,
	         "games": [_pack_game(x, cix) for x in g] if g and cix else None}
	        for (st, rnd, a, b, wa, wb), g in zip(r["series"], games) if st == stage]


def single_run(r: dict, i: int, ix, teams: dict) -> dict:
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
		extra["elo"] = {"team": np.round(r["elo"]["team_end"], 1).tolist(), "player": np.round(r["elo"]["player_end"], 1).tolist()}
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
	path = OUT / "h2h_knockout.pkl"
	if path.exists():
		h = pickle.loads(path.read_bytes())
		if set(h["teams"]) >= {t for pair in draw for t in pair}:
			out["bracket"] = best_bracket(draw, h["p"], ko.get("pickem_points"))
			out["h2h"] = {"teams": h["teams"], "p": h["p"], "series": h["series"], "run_at": h["run_at"]}
	return out

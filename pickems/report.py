"""Aggregate simulation summaries into the numbers the dashboard shows.

"Leader" questions (most picked champion, player with most kills, ...) are
answered as P(this entity finishes first), with ties split evenly, plus the
mean and 10-90% range of the underlying stat. Rate questions (highest win
rate, best KDA) only count entities that meet a minimum-games bar in that
simulation, like Crystal Ball questions usually do.
"""

from __future__ import annotations

import itertools
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


def build(run: dict) -> dict:
	results, ix = run["results"], run["ix"]
	outcomes = team_outcomes(results, ix.teams)
	report = {
		"meta": {"sims": run["sims"], "run_at": run["run_at"], "model": run["manifest"], "teams": run["teams"]},
		"outcomes": outcomes,
		"swiss_pickem": best_swiss_pickem(outcomes),
		"scopes": {},
		"projection": run["projection"].reset_index().head(60).round(4).to_dict(orient="records"),
	}
	for scope in SCOPES:
		report["scopes"][scope] = {
			"champions": champion_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"players": player_questions(results, ix, scope, min_games=5 if scope != "play_in" else 2),
			"teams": team_questions(results, ix, scope),
			"event": event_questions(results, scope),
		}
	return report

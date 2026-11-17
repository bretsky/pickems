"""How the published forecasts did once the event is over: every snapshot in
dashboard/versions.json (each forecast as it stood before a day's games) scored
against the real results.

Read from the snapshots' own pages (dashboard/out/runs/<run>.html carries its
report as JSON), so what's scored is exactly what was published, not a re-run
with today's model:

  pickem      the Pick'Em picks each snapshot recommended, scored with the real
              results: Play-In bracket, Swiss picks, knockout bracket (only on a
              real quarterfinal draw) and Crystal Ball
  series      every real series, with the pre-series chance the latest snapshot
              that hadn't seen it gave its winner (Swiss: swiss_matchups, else
              swiss_h2h; knockouts: the knockout head-to-head odds)
  forecasts   per snapshot, the chance it gave the real champion and finalists,
              and Brier scores of its team-stage outcomes (quarters, semis, final,
              champion) against a no-skill forecast

The real answers come from the final run's report (every game replayed, so each
Crystal Ball question has one answer), except Finals MVP, a vote: its role is
`results.finals_mvp.role` in the event's config. Questions Riot scored on its top 3
(`results.crystal_ball_top`) pay each pick by its rank on Riot's published list.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dashboard" / "out"
VERSIONS = ROOT / "dashboard" / "versions.json"
STAGES = ["quarters", "semis", "final", "champion"]
# Which snapshot each Pick'Em section was locked at (Riot locks each before its stage starts).
LOCKS = {"play_in": "pre-playins", "swiss": "before-swiss-d1", "knockout": "before-knockouts", "crystal": "pre-playins"}


def snapshot_report(run_name: str) -> dict | None:
	page = OUT / "runs" / f"{run_name}.html"
	if not page.exists():
		return None
	m = re.search(r'<script id="report" type="application/json">(.*?)</script>', page.read_text(), re.S)
	return json.loads(m.group(1)) if m else None


def _snapshots() -> list[dict]:
	"""versions.json's snapshots, oldest first, each with its report (those whose page is missing are skipped)."""
	if not VERSIONS.exists():
		return []
	out = []
	for s in sorted(json.loads(VERSIONS.read_text())["snapshots"], key=lambda s: s["run_at"]):
		r = snapshot_report(s["run_name"])
		if r is not None:
			out.append({**s, "report": r})
	return out


def _key(x: dict) -> tuple:
	return (x["stage"], frozenset((x["a"], x["b"])), x["round"])


# --- real results --------------------------------------------------------------------

def _play_in_winners(series: list[dict], picks: dict) -> dict:
	"""Pick key -> real winner. "UB semi A-B" keys name their pair; the rest are one series each."""
	out = {}
	for k in picks:
		if k.startswith("UB semi "):
			pair = set(k.removeprefix("UB semi ").split("-"))
			s = next((x for x in series if x["stage"] == "play_in" and x["round"] == "UB semi" and {x["a"], x["b"]} == pair), None)
		else:
			s = next((x for x in series if x["stage"] == "play_in" and x["round"] == k), None)
		out[k] = s["winner"] if s else None
	return out


def _knockout_winners(series: list[dict], draw: list[list[str]]) -> dict:
	"""QF1-4, SF1-2, Final -> real winner, following the bracket from the real draw."""
	ko = [x for x in series if x["stage"] == "knockout"]

	def won(a, b):
		s = next((x for x in ko if {x["a"], x["b"]} == {a, b}), None)
		return s["winner"] if s else None

	w = {f"QF{i + 1}": won(*pair) for i, pair in enumerate(draw)}
	w["SF1"] = won(w["QF1"], w["QF2"]) if w["QF1"] and w["QF2"] else None
	w["SF2"] = won(w["QF3"], w["QF4"]) if w["QF3"] and w["QF4"] else None
	w["Final"] = won(w["SF1"], w["SF2"]) if w["SF1"] and w["SF2"] else None
	return w


def _cb_hit(pick: str, answer: str) -> bool:
	return pick == answer or pick in answer.split(" / ")


def _cb_p(q: dict, label: str) -> float | None:
	"""The chance a snapshot's Crystal Ball card gave an answer (None when it isn't among the card's options)."""
	ps = {o["label"]: o["p"] for o in q.get("options", [])}
	parts = label.split(" / ")
	if len(parts) > 1:
		return sum(ps.get(p, 0.0) for p in parts) or None
	return ps.get(label)


def _results(cfg: dict) -> dict:
	"""The event's `results:` config. A run simulated before a result was recorded reads it from
	today's config, when that config is the same event."""
	res = cfg.get("results") or {}
	from pickems.teams import load_config  # noqa: PLC0415
	today = load_config()
	if today.get("name") == cfg.get("name"):
		res = {**(today.get("results") or {}), **res}
	return res


def _cb_answers(final: dict, res: dict) -> list[str]:
	ans = [q["answer"] for q in final["pickems"]["crystal_ball"]]
	mvp = (res.get("finals_mvp") or {}).get("role")
	for i, q in enumerate(final["pickems"]["crystal_ball"]):
		if "Finals MVP" in q["question"] and mvp:
			ans[i] = mvp
	return ans


def _cb_top(final: dict, res: dict) -> list[dict | None]:
	"""Per Crystal Ball question, Riot's ranked answers and points ({"top": [...], "points": [...]})
	when it scored more than the winner, else None."""
	top = res.get("crystal_ball_top") or {}
	pts, qs = top.get("points") or [], top.get("questions") or {}
	return [next(({"top": v, "points": pts} for k, v in qs.items() if k in q["question"]), None) for q in final["pickems"]["crystal_ball"]]


# --- scoring -------------------------------------------------------------------------

def _score_play_in(rep: dict, winners: dict) -> dict | None:
	B = (rep.get("pickems") or {}).get("play_in_bracket")
	if not B or not B.get("picks"):
		return None
	rows = [{"slot": k, "pick": v, "winner": winners.get(k), "pts": 10 if k == "Qualifier" else 5, "p": B["pick_prob"].get(k)} for k, v in B["picks"].items()]
	return _total(rows, B["expected_points"], B["max_points"])


def _score_swiss(rep: dict, records: dict) -> dict | None:
	S = rep.get("swiss_pickem")
	if not S or not S.get("picks"):
		return None
	pts, picks, D = S["points"], S["picks"], S["detail"]
	rows = []
	for slot in ("3-0", "3-1", "3-2"):
		t = picks.get(slot)
		if t is None:
			continue
		adv = records.get(t, "").startswith("3-")
		exact = records.get(t) == slot
		rows.append({"slot": slot, "pick": t, "winner": records.get(t, "out"), "pts": pts["advance"] + pts["record"],
		             "got": pts["advance"] * adv + pts["record"] * exact, "hit": exact, "p": D.get(t, {}).get(slot)})
	for t in picks.get("advance", []):
		adv = records.get(t, "").startswith("3-")
		rows.append({"slot": "advance", "pick": t, "winner": records.get(t, "out"), "pts": pts["advance"], "got": pts["advance"] * adv, "hit": adv,
		             "p": D.get(t, {}).get("advance")})
	return _total(rows, S["expected_points"], S["max_points"])


def _score_knockout(rep: dict, series: list[dict]) -> dict | None:
	K = rep.get("knockout") or {}
	B = K.get("bracket")
	if not K.get("draw_is_real") or not B:
		return None
	winners = _knockout_winners(series, B["draw"])
	pts = B["points"]
	rows = [{"slot": k, "pick": v, "winner": winners.get(k), "pts": pts["Final" if k == "Final" else k[:2]], "p": B["pick_prob"].get(k)} for k, v in B["picks"].items()]
	return _total(rows, B["expected_points"], B["max_points"])


def _score_crystal(rep: dict, answers: list[str], tops: list[dict | None]) -> dict | None:
	"""Riot's top-3 questions pay by the pick's rank (row "rank", 1-based, None off the list).
	Expected points use the card's own `exp` when it has one (reports made since the picks counted
	places 2-3); older snapshots carry only P(leads), so theirs count the winner alone."""
	Q = (rep.get("pickems") or {}).get("crystal_ball")
	if not Q or len(Q) != len(answers):
		return None
	rows = []
	for q, a, t in zip(Q, answers, tops):
		r = {"slot": q["question"], "pick": q["answer"], "winner": a, "pts": q["pts"], "hit": _cb_hit(q["answer"], a),
		     "p": q["p"], "p_actual": _cb_p(q, a)}
		if t:
			rank = next((i for i, x in enumerate(t["top"]) if _cb_hit(q["answer"], x)), None)
			r.update(top=t["top"], rank=None if rank is None else rank + 1, hit=rank == 0,
			         got=t["points"][rank] if rank is not None and rank < len(t["points"]) else 0)
		rows.append(r)
	exp = sum(q.get("exp", r["pts"] * r["p"]) for q, r in zip(Q, rows))
	return _total(rows, exp, sum(r["pts"] for r in rows))


def _total(rows: list[dict], expected: float, max_points: float) -> dict:
	for r in rows:
		r.setdefault("hit", r["winner"] is not None and r["pick"] == r["winner"])
		r.setdefault("got", r["pts"] if r["hit"] else 0)
	return {"rows": rows, "points": sum(r["got"] for r in rows), "expected": round(expected, 2), "max": max_points,
	        "correct": sum(r["hit"] for r in rows), "n": len(rows)}


def _series_odds(rep: dict, x: dict) -> float | None:
	"""The chance a snapshot gave x's real winner before the series."""
	a, b, w = x["a"], x["b"], x["winner"]
	if x["stage"] == "swiss":
		m = next((m for m in rep.get("swiss_matchups") or [] if m["round"] == x["round"] and {m["a"], m["b"]} == {a, b}), None)
		if m is not None:
			return m["p_a"] if m["a"] == w else 1 - m["p_a"]
		bo = 3 if max(x["wa"], x["wb"]) >= 2 else 1
		lo, hi = sorted((a, b))
		p = (rep.get("swiss_h2h") or {}).get(f"{lo}|{hi}|{bo}")
		return None if p is None else (p if lo == w else 1 - p)
	if x["stage"] == "knockout":
		h = ((rep.get("knockout") or {}).get("h2h") or {}).get("p") or {}
		loser = b if w == a else a
		p = (h.get(w) or {}).get(loser)
		return p
	return None


def _series_section(snaps: list[dict], series: list[dict]) -> dict:
	rows = []
	for x in series:
		prior = [s for s in snaps if _key(x) not in {_key(y) for y in s["report"].get("played_series") or []}]
		src = next((s for s in reversed(prior) if _series_odds(s["report"], x) is not None), None)
		p = _series_odds(src["report"], x) if src else None
		rows.append({**{k: x[k] for k in ("stage", "round", "a", "b", "wa", "wb", "winner")}, "p": p,
		             "snapshot": src["label"] if src else None, "snapshot_title": src["title"] if src else None})
	priced = [r for r in rows if r["p"] is not None]
	n = len(priced)
	fav = sum(r["p"] > 0.5 for r in priced)
	exp_fav = sum(max(r["p"], 1 - r["p"]) for r in priced)
	ll = -sum(math.log(max(r["p"], 1e-6)) for r in priced) / n if n else None
	brier = sum((1 - r["p"]) ** 2 for r in priced) / n if n else None
	# Reliability: favourites binned by the chance they were given.
	bins = []
	for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
		xs = [max(r["p"], 1 - r["p"]) for r in priced if lo <= max(r["p"], 1 - r["p"]) < hi]
		won = [r["p"] > 0.5 for r in priced if lo <= max(r["p"], 1 - r["p"]) < hi]
		if xs:
			bins.append({"lo": lo, "hi": min(hi, 1.0), "n": len(xs), "predicted": sum(xs) / len(xs), "actual": sum(won) / len(won)})
	by_stage = {}
	for st in ("swiss", "knockout"):
		ps = [r["p"] for r in priced if r["stage"] == st]
		if ps:
			by_stage[st] = {"n": len(ps), "favorites": sum(p > 0.5 for p in ps), "expected_favorites": sum(max(p, 1 - p) for p in ps),
			                "logloss": -sum(math.log(max(p, 1e-6)) for p in ps) / len(ps)}
	return {"rows": rows, "n": n, "favorites": fav, "expected_favorites": exp_fav, "logloss": ll, "brier": brier, "bins": bins, "by_stage": by_stage}


def _forecast_row(s: dict, actual: dict[str, dict], champion: str, finalists: set[str]) -> dict:
	o = {x["team"]: x for x in s["report"]["outcomes"]}
	teams = [t for t in actual if actual[t].get("swiss", 0) > 0.5]  # the main-stage field (play-in losers never reach a stage scored here)
	n = len(teams)
	# No-skill reference: every main-stage team equally likely to reach each stage.
	base = {"quarters": 8 / n, "semis": 4 / n, "final": 2 / n, "champion": 1 / n}
	brier, ref = {}, {}
	for st in STAGES:
		brier[st] = sum((o.get(t, {}).get(st, 0.0) - actual[t][st]) ** 2 for t in teams) / n
		ref[st] = sum((base[st] - actual[t][st]) ** 2 for t in teams) / n
	tot, tot_ref = sum(brier.values()), sum(ref.values())
	top = sorted(o.values(), key=lambda x: -x["champion"])[:6]
	return {"label": s["label"], "title": s["title"], "url": s["url"], "data_through": s["data_through"],
	        "p_champion": o.get(champion, {}).get("champion", 0.0), "rank_champion": 1 + sum(x["champion"] > o.get(champion, {}).get("champion", 0.0) for x in o.values()),
	        "p_final": {t: o.get(t, {}).get("final", 0.0) for t in sorted(finalists)},
	        "brier": brier, "brier_ref": ref, "skill": 1 - tot / tot_ref if tot_ref else None,
	        "top": [{"team": x["team"], "p": x["champion"]} for x in top],
	        "champion_odds": {x["team"]: x["champion"] for x in o.values() if x.get("swiss", 0) > 0}}


def build(final: dict, cfg: dict) -> dict | None:
	"""The evaluation section for a finished event's report, or None while the final is unplayed."""
	series = final.get("played_series") or []
	outcomes = {o["team"]: o for o in final["outcomes"]}
	champs = [t for t, o in outcomes.items() if o["champion"] > 0.999]
	if len(champs) != 1:
		return None
	champion = champs[0]
	finalists = {t for t, o in outcomes.items() if o["final"] > 0.999}
	snaps = _snapshots()
	if not snaps:
		return None
	records = (final.get("knockout") or {}).get("records") or {}
	res = _results(cfg)
	answers, tops = _cb_answers(final, res), _cb_top(final, res)
	pickem = []
	for s in snaps:
		rep = s["report"]
		pickem.append({"label": s["label"], "title": s["title"], "url": s["url"],
		               "play_in": _score_play_in(rep, _play_in_winners(series, (rep.get("pickems") or {}).get("play_in_bracket", {}).get("picks", {}))),
		               "swiss": _score_swiss(rep, records),
		               "knockout": _score_knockout(rep, series),
		               "crystal": _score_crystal(rep, answers, tops)})
	return {"champion": champion, "finalists": sorted(finalists), "records": records, "locks": LOCKS,
	        "crystal_answers": answers, "pickem": pickem,
	        "series": _series_section(snaps, series),
	        "forecasts": [_forecast_row(s, outcomes, champion, finalists) for s in snaps]}

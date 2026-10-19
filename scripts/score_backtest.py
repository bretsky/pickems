"""Score a backtest run against what actually happened (PICKEMS_ASOF=<year>).

  PICKEMS_ASOF=2025 .venv/bin/python scripts/score_backtest.py bt2025_1000 [--mvp-role mid]

Answers the same Crystal Ball questions as the simulator (report.pickem_answers)
from the real Worlds games in the full Oracle's Elixir CSV, then reports for each
question the model's favourite, the probability it gave the real outcome, and a
log score; plus the Swiss pick'em (3-0 / 0-3 / advance) and title odds.
"""

import argparse
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pickems.report import SWISS_POINTS, SWISS_SLOTS, best_swiss_pickem, pickem_answers, team_outcomes
from pickems.sync_model import ASOF, CACHE_DIR, ROOT

ROLES = ["top", "jng", "mid", "bot", "sup"]
SRC = ROOT / "vendor" / "dp"  # the full, uncut data


def real_games(year: int, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	df = pd.read_csv(SRC / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
	df = df[(df.league == "WLDs") & (df.date >= start)]
	return df[df.position == "team"].copy(), df[df.position.isin(ROLES)].copy()


def series_table(team: pd.DataFrame) -> list[dict]:
	"""Series from consecutive games of the same pair (game counter restarts at 1)."""
	rows, open_ = [], {}
	for gid, g in team.sort_values(["date", "gameid"]).groupby("gameid", sort=False):
		b, r = g[g.side == "Blue"].iloc[0], g[g.side == "Red"].iloc[0]
		pair = frozenset((b.teamname, r.teamname))
		if pair not in open_ or b.game == 1:
			open_[pair] = {"a": b.teamname, "b": r.teamname, "wa": 0, "wb": 0, "date": b.date[:10]}
			rows.append(open_[pair])
		s = open_[pair]
		winner = b.teamname if b.result == 1 else r.teamname
		s["wa" if winner == s["a"] else "wb"] += 1
	return rows


def actual_answers(team: pd.DataFrame, ply: pd.DataFrame, cfg: dict, champs: list[str]) -> list:
	code = {t["name"]: c for c, t in cfg["teams"].items()}
	region = {t["name"]: t["region"] for t in cfg["teams"].values()}
	pk = {"duel": ["Peyz", "Gumayusi"], "milestone": {"player": "Faker", "kills": 84}, **cfg.get("pickem", {})}
	n = team.gameid.nunique()
	picks = Counter(ply.champion.dropna())
	bans = Counter(c for col in [f"ban{i}" for i in range(1, 6)] for c in team[col].dropna())
	c = ply.groupby("champion").agg(p=("gameid", "size"), w=("result", "sum"), d=("deaths", "sum"))
	c = c[c.p >= 5]
	ps = ply.groupby(["playername", "teamname"]).agg(g=("gameid", "size"), k=("kills", "sum"), d=("deaths", "sum"), a=("assists", "sum"), dmg=("damagetochampions", "sum"))
	tk = ply.groupby(["teamname"]).kills.sum()
	# Team kills belong to the games the player played in; players all play every game of their team here.
	ps["tk"] = [ply[(ply.teamname == t)].kills.sum() for (_, t) in ps.index]
	ps = ps[ps.g >= 5]
	lab = lambda i: f"{i[0]} ({code[i[1]]})"
	out = []
	pres = {ch: (picks.get(ch, 0) + bans.get(ch, 0)) / n for ch in set(picks) | set(bans)}
	out.append(max(pres, key=pres.get))
	out.append((c.w / c.p).idxmax())
	out.append((c.d / c.p).idxmax())
	out.append(lab(((ps.k + ps.a) / ps.d.clip(lower=1)).idxmax()))
	out.append(lab((ps.dmg / ps.g).idxmax()))
	out.append(lab(((ps.k + ps.a) / ps.tk).idxmax()))
	kills = ply.groupby("playername").kills.sum()
	d1, d2 = pk["duel"]
	out.append(d1 if kills[d1] > kills[d2] else d2 if kills[d2] > kills[d1] else "Tie")
	out.append("Yes" if kills[pk["milestone"]["player"]] >= pk["milestone"]["kills"] else "No")
	out.append("Yes" if picks.get("Teemo", 0) > 0 else "No")
	out.append(len(picks))
	t = team.drop_duplicates("gameid")
	out.append("Before 32:00" if (t.gamelength < 1920).sum() > (t.gamelength > 1920).sum() else "After 32:00")
	out.append("Yes" if ply.kills.max() >= 15 else "No")
	out.append("Yes" if ply["total cs"].max() >= 450 else "No")
	out.append(None)  # Finals MVP role: a vote, supplied by hand
	ser = series_table(team)
	return out, ser, code, region, kills, ply


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run")
	ap.add_argument("--mvp-role", help="role of the real Finals MVP (top/jng/mid/bot/sup)")
	args = ap.parse_args()
	run = pickle.load(open(CACHE_DIR / "sims" / f"{args.run}.pkl", "rb"))
	cfg, year = run["cfg"], int(ASOF)
	start = min(str(e["date"]) for e in cfg["schedule"])
	team, ply = real_games(year, start)
	ans, ser, code, region, kills, ply = actual_answers(team, ply, cfg, run["ix"].champions)
	pred = pickem_answers(run)["crystal_ball"]
	n_q = len(pred)

	# Knockout facts from the real Swiss records / series.
	swiss = [s for s in ser if s["date"] <= str(cfg["schedule"][[e["round"] for e in cfg["schedule"]].index("R5")]["date"])]
	rec = defaultdict(lambda: [0, 0])
	for s in swiss:
		if s["wa"] == s["wb"]:
			continue
		w, l = (s["a"], s["b"]) if s["wa"] > s["wb"] else (s["b"], s["a"])
		if {s["a"], s["b"]} == set(t for t in (cfg["play_in"].get("teams") and [cfg["teams"][x]["name"] for x in cfg["play_in"]["teams"]] or [])):
			continue  # the play-in series
		rec[w][0] += 1
		rec[l][1] += 1
	qf = [t for t, (w, _) in rec.items() if w == 3]
	ko = [s for s in ser if s["date"] > "2025-10-25"]
	champion = [s for s in ko][-1]
	champ = champion["a"] if champion["wa"] > champion["wb"] else champion["b"]
	place = lambda name: 1 if name == champ else 2 if name in (champion["a"], champion["b"]) else 3 if any(name in (s["a"], s["b"]) for s in ko[-3:-1]) else 5 if name in qf else 9
	ans.append(len({region[t] for t in qf}))
	ans.append("Yes" if any(region[t] == "LCS" for t in qf) else "No")
	hle, t1 = place("Hanwha Life Esports"), place("T1")
	ans.append("HLE" if hle < t1 else "T1" if t1 < hle else "Same round")
	g = team.drop_duplicates("gameid").sort_values("gamelength").iloc[0]
	short = team[team.gameid == g.gameid]
	ans.append(short[short.result == 1].teamname.iloc[0])
	# Order: the 18 questions are ... 14 MVP role, 15 regions, 16 LCS, 17 HLE/T1, 18 shortest. Re-slot.
	q_ans = ans[:13] + [args.mvp_role.capitalize() if args.mvp_role else None] + [str(ans[14]), ans[15], ans[16], ans[17]]
	print(f"Backtest {args.run}: {run['sims']} sims, data through {run['data_through'][:10]}, Worlds {year}\n")
	logs, hits, scored = [], 0, 0
	for i, (q, a) in enumerate(zip(pred, q_ans), 1):
		opts = {o["label"]: o["p"] for o in q["options"]}
		if i == 10:
			exact = q.get("exact", {})
			p_act = exact.get(str(a), 0.0)
			bucket = next((o for o in opts if o.split("-")[0].isdigit() and int(o.split("-")[0]) <= a <= int(o.split("-")[1])), None)
			p_act_b = opts.get(bucket, 0.0)
			print(f"{i:2}. {q['question']}\n    model bucket {q['answer']} ({q['p']:.0%}); actual {a} (bucket {bucket}: {p_act_b:.0%}, exact {p_act:.1%})")
			scored += 1
			hits += q["answer"] == bucket
			logs.append(np.log(max(p_act_b, 1e-3)))
			continue
		if a is None:
			print(f"{i:2}. {q['question']}\n    model {q['answer']} ({q['p']:.0%}); actual: unknown (pass --mvp-role)")
			continue
		p_act = opts.get(a, 0.0)
		rank = 1 + sum(p > p_act for p in opts.values())
		print(f"{i:2}. {q['question']}\n    model {q['answer']} ({q['p']:.0%}); actual {a} (model gave {p_act:.1%}, rank {rank}) {'HIT' if q['answer'] == a else 'miss'}")
		scored += 1
		hits += q["answer"] == a
		logs.append(np.log(max(p_act, 1e-3)))
	print(f"\nfavourite right on {hits}/{scored} scored questions; mean log prob of the real outcome {np.mean(logs):.2f}")

	# Swiss pick'em and title odds.
	outs = team_outcomes(run["results"], run["ix"].teams)
	pick = best_swiss_pickem(outs)["picks"]
	nm = {c: t["name"] for c, t in cfg["teams"].items()}
	pts = SWISS_POINTS
	real = {t: f"{r[0]}-{r[1]}" for t, r in rec.items()}
	picked = [pick[k] for k in SWISS_SLOTS] + pick["advance"]
	score = sum(pts["advance"] for t in picked if real.get(nm[t], "").startswith("3-")) + sum(pts["record"] for k in SWISS_SLOTS if real.get(nm[pick[k]]) == k)
	print(f"\nSwiss pick'em (model's best picks): " + ", ".join(f"{k} {pick[k]}" for k in SWISS_SLOTS) + f", advance {pick['advance']}")
	print(f"real: {sorted((r, t) for t, r in real.items() if r.startswith('3-'))}; points {score}/{8 * pts['advance'] + 3 * pts['record']}")
	by = {r["team"]: r for r in outs}
	qf_code = [c for c, t in nm.items() if t in qf]
	print("Predicted knockout odds of the real 8:", {c: f"{by[c]['quarters']:.0%}" for c in qf_code})
	ll = np.mean([np.log(by[c]["quarters"]) if nm[c] in qf else np.log(1 - by[c]["quarters"]) for c in by if by[c]["swiss"] > 0.5])
	print(f"mean log prob of each team's reach-knockouts outcome: {ll:.3f}")
	cc = next(c for c, t in nm.items() if t == champ)
	print(f"champion {champ}: model gave {by[cc]['champion']:.1%}; model's top 5:", [(r['team'], f"{r['champion']:.0%}") for r in outs[:5]])


if __name__ == "__main__":
	sys.exit(main())

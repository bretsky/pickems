"""One-way sync of draftpredict into vendor/dp/: the win-probability model the
simulator loads, and the place Worlds-weighted Optuna retunes run.

draftpredict (~/draftpredict) is the source of truth and is never written to
from here. vendor/dp is a git clone of it with no remote and a pre-push hook
that refuses every push, so nothing done in it can reach ~/draftpredict or
GitHub. Its branches:

  upstream        the draftpredict commit it's based on (only committed work:
                  uncommitted edits in ~/draftpredict are not taken)
  worlds-weight   upstream + tune/dp_worlds_weight.patch, checked out: with
                  DP_WORLDS_WEIGHT=w every fit and walk-forward score weights a
                  Worlds game 1 and any other 1 - w; adopted retunes and
                  predictor caches are kept per weight; worker logs stay out of
                  the /tmp dir the original repo's runs share. At w=0 it
                  behaves exactly like upstream.

Each sync fetches draftpredict's HEAD (or --ref), rebases the patch onto it and
re-exports tune/dp_worlds_weight.patch, archives Worlds-retuned snapshots
tuned on the old code (.cache/snapshots/), clones changed season CSVs with APFS
copy-on-write (`cp -c`), and keeps vendor/dp/.venv (used by the tuning
scripts) on draftpredict's package versions. The first sync clones; an older
plain-file mirror at vendor/dp is moved aside.

vendor/dp/MANIFEST.json records what was synced so the dashboard can show
which model produced a run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path.home() / "draftpredict"
ROOT = Path(__file__).resolve().parent.parent
# PICKEMS_ASOF=<year> switches the whole pipeline to a backtest of that Worlds:
# data copy vendor/dp_asof_<year> (scripts/make_asof.py), its own .cache subdir,
# and data/manual/worlds<year>.yaml. PICKEMS_ASOF=<year>-<mmdd> is the same Worlds
# cut mid-event (data before that day, so the games played by then are replayed):
# its own data copy and cache, that year's config and patch fit.
ASOF = os.environ.get("PICKEMS_ASOF", "")
ASOF_YEAR = int(ASOF[:4]) if ASOF else None
DEST = ROOT / "vendor" / (f"dp_asof_{ASOF}" if ASOF else "dp")
CACHE_DIR = ROOT / ".cache" / (f"asof_{ASOF}" if ASOF else "")
PATCH = ROOT / "tune" / "dp_worlds_weight.patch"
BRANCH = "worlds-weight"
# Packages the simulator / tuner never import (and psycopg2 needs pg_config to build).
SKIP_REQS = re.compile(r"^(-e |.*@ |psycopg2|py-clob|windows-curses)", re.I)


def _rel(path: Path) -> str:
	return os.path.relpath(path, ROOT)


def _run(*cmd: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
	return subprocess.run(list(cmd), cwd=cwd, check=check, capture_output=True, text=True)


def _git(*args: str, check: bool = True) -> str:
	return _run("git", "-C", str(DEST), *args, check=check).stdout.strip()


def _source_git(*args: str) -> str:
	"""Read-only git commands against ~/draftpredict."""
	return _run("git", "-C", str(SOURCE), *args).stdout.strip()


def _tuning_pids() -> list[int]:
	"""Tuning / ablation processes whose working directory is vendor/dp."""
	found = _run("pgrep", "-f", r"scripts\.(tune_optuna|ablate_flags)", check=False).stdout.split()
	return [int(p) for p in found if f"n{DEST}" in _run("lsof", "-a", "-p", p, "-d", "cwd", "-Fn", check=False).stdout.splitlines()]


def _clone() -> None:
	if DEST.exists():
		aside = DEST.with_name(f"dp.mirror-{datetime.now():%Y%m%d_%H%M%S}")
		DEST.rename(aside)
		print(f"moved the old vendor/dp mirror to {_rel(aside)} (safe to delete)")
	_run("git", "clone", "-q", "--no-hardlinks", str(SOURCE), str(DEST))
	_git("remote", "remove", "origin")
	hook = DEST / ".git" / "hooks" / "pre-push"
	hook.write_text('#!/bin/sh\necho "vendor/dp is a one-way copy of ~/draftpredict: pushing is disabled" >&2\nexit 1\n')
	hook.chmod(0o755)
	with open(DEST / ".git" / "info" / "exclude", "a") as f:
		f.write("MANIFEST.json\noptuna_best_params_ww*.json\n")
	_git("config", "user.name", "pickems")
	_git("config", "user.email", "pickems@localhost")
	_git("branch", "-f", "upstream", "HEAD")
	_git("checkout", "-q", "-b", BRANCH)
	_git("am", "-q", str(PATCH))
	print(f"cloned draftpredict {_git('rev-parse', '--short', 'upstream')} into vendor/dp and applied {_rel(PATCH)}")


def _save_and_discard_edits() -> None:
	"""vendor/dp holds only committed code (upstream + the patch). Edits left in it by
	runs (e.g. a regenerated live_margin_noise.json) or stray experiments would block
	the rebase: keep them as a patch under .cache/discarded/ and clean the tree."""
	status = _git("status", "--short", "--untracked-files=no")
	if not status:
		return
	saved = DEST / ".cache" / "discarded" / f"{datetime.now():%Y%m%d_%H%M%S}.patch"
	saved.parent.mkdir(parents=True, exist_ok=True)
	saved.write_text(_git("diff", "HEAD") + "\n")
	_git("checkout", "-q", "HEAD", "--", ".")
	print(f"discarded uncommitted edits in vendor/dp (saved to {_rel(saved)}; 'git apply' it to restore):\n{status}")


def _update(ref: str) -> bool:
	"""Move upstream to draftpredict's `ref` and rebase the patch onto it."""
	_git("fetch", "-q", str(SOURCE), ref)
	new, old = _git("rev-parse", "FETCH_HEAD"), _git("rev-parse", "upstream")
	if new == old:
		return False
	_save_and_discard_edits()
	for snap in sorted(DEST.glob("optuna_best_params_ww*.json")):
		archive = DEST / ".cache" / "snapshots" / f"{datetime.now():%Y%m%d_%H%M%S}_base{old[:7]}_{snap.name}"
		archive.parent.mkdir(parents=True, exist_ok=True)
		snap.rename(archive)
		print(f"archived {snap.name} (tuned on {old[:7]}) to {_rel(archive)}: retune for the new code")
	_git("branch", "-f", "upstream", new)
	if _run("git", "-C", str(DEST), "rebase", "-q", "upstream", BRANCH, check=False).returncode:
		raise SystemExit(f"the Worlds-weighting patch conflicts with draftpredict {new[:7]}: resolve it in vendor/dp, "
		                 "run 'git rebase --continue', then sync again")
	PATCH.write_text(_git("format-patch", "upstream.." + BRANCH, "--stdout") + "\n")
	print(f"rebased the patch from draftpredict {old[:7]} onto {new[:7]} ({_rel(PATCH)} updated: commit it)")
	return True


def _season_files() -> list[str]:
	return sorted(p.name for p in SOURCE.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))


def _clone_csvs() -> list[str]:
	"""Copy-on-write clone of every season CSV whose size/mtime changed."""
	updated = []
	for name in _season_files():
		src, dst = SOURCE / name, DEST / name
		s = src.stat()
		if dst.exists():
			d = dst.stat()
			if d.st_size == s.st_size and int(d.st_mtime) == int(s.st_mtime):
				continue
			dst.unlink()
		# -c: clonefile (APFS CoW); -p: keep mtime so the next sync can skip it.
		subprocess.run(["cp", "-c", "-p", str(src), str(dst)], check=True)
		updated.append(name)
	return updated


def _sync_venv() -> bool:
	"""vendor/dp/.venv with draftpredict's package versions (the tuning scripts
	call .venv/bin/python). The simulator itself runs in pickems' own venv."""
	src_py = SOURCE / ".venv" / "bin" / "python"
	if not src_py.exists():
		print("note: ~/draftpredict/.venv not found; vendor/dp/.venv left as is (needed only for retunes)")
		return False
	reqs = "\n".join(line for line in _run(str(src_py), "-m", "pip", "freeze").stdout.splitlines() if not SKIP_REQS.match(line)) + "\n"
	cached = DEST / ".cache" / "requirements.txt"
	venv_py = DEST / ".venv" / "bin" / "python"
	if venv_py.exists() and cached.exists() and cached.read_text() == reqs:
		return False
	if not venv_py.exists():
		base_py = _run(str(src_py), "-c", "import sys; print(sys._base_executable)").stdout.strip()
		_run(base_py, "-m", "venv", str(DEST / ".venv"))
		_run(str(venv_py), "-m", "pip", "install", "-q", "--upgrade", "pip")
	cached.parent.mkdir(parents=True, exist_ok=True)
	cached.write_text(reqs)
	print("installing draftpredict's package versions into vendor/dp/.venv ...", flush=True)
	_run(str(venv_py), "-m", "pip", "install", "-q", "-r", str(cached))
	return True


def sync(ref: str | None = None) -> dict:
	busy = _tuning_pids()
	if busy:
		raise SystemExit(f"a retune is running in vendor/dp (pids {busy}); wait for it or stop it first")
	if not (DEST / ".git").exists():
		_clone()
	_update(ref or "HEAD")
	updated = _clone_csvs()
	_sync_venv()
	params = json.loads((DEST / "optuna_best_params.json").read_text())
	manifest = {
		"synced_at": datetime.now(timezone.utc).isoformat(),
		"source": str(SOURCE),
		"ref": ref or "HEAD",
		"commit": _git("rev-parse", "upstream"),
		# vendor/dp only ever holds committed draftpredict code.
		"dirty": False,
		"source_uncommitted": bool(_source_git("status", "--porcelain", "--", "draftpredict", "scripts", "optuna_best_params.json")),
		"patch_commit": _git("rev-parse", BRANCH),
		"params_version": params.get("version"),
		"params_best_value": params.get("best_value"),
		"params_saved_at": params.get("saved_at"),
		"csvs_updated": updated,
	}
	(DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
	if manifest["source_uncommitted"]:
		print("note: ~/draftpredict has uncommitted changes; vendor/dp has its committed HEAD only", file=sys.stderr)
	return manifest


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", help="draftpredict commit/branch to sync (default: its HEAD)")
	args = ap.parse_args()
	print(json.dumps(sync(args.ref), indent=2))


if __name__ == "__main__":
	main()

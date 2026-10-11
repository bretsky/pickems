"""One-way sync of the draftpredict model into vendor/dp/.

draftpredict (~/draftpredict) is the source of truth for the win-probability
model and is never written to from here. Every run of this module copies the
current code + adopted params into vendor/dp/ and clones the Oracle's Elixir
season CSVs with APFS copy-on-write (`cp -c`), so the copies are independent
files: nothing done under pickems can change a byte in draftpredict.

Default source is draftpredict's working tree (the same files its own
predict.py would use right now); pass --ref <git ref> to take code and params
from a commit instead, e.g. when the working tree is mid-edit. The CSVs are
gitignored there, so they always come from the working tree.

vendor/dp/MANIFEST.json records what was synced (commit, dirty flag, params
version) so the dashboard can show which model produced a run.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path.home() / "draftpredict"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "vendor" / "dp"

# Code and data files build_predictor() needs, relative to the draftpredict root.
CODE_PATHS = ["draftpredict", "scripts/predict.py"]
DATA_FILES = ["optuna_best_params.json", "live_margin_noise.json", "auto_draft_config.json"]


def _git(*args: str) -> str:
	return subprocess.run(["git", "-C", str(SOURCE), *args], check=True, capture_output=True, text=True).stdout.strip()


def _copy_tree_from_worktree(dest: Path) -> None:
	for rel in CODE_PATHS + DATA_FILES:
		src = SOURCE / rel
		if not src.exists():
			continue
		target = dest / rel
		target.parent.mkdir(parents=True, exist_ok=True)
		if src.is_dir():
			shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
		else:
			shutil.copy2(src, target)


def _copy_tree_from_ref(dest: Path, ref: str) -> None:
	tracked = set(_git("ls-tree", "-r", "--name-only", ref).splitlines())
	paths = [p for p in CODE_PATHS + DATA_FILES if p in tracked or any(t.startswith(p + "/") for t in tracked)]
	blob = subprocess.run(["git", "-C", str(SOURCE), "archive", ref, *paths], check=True, capture_output=True).stdout
	with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
		tar.extractall(dest, filter="data")


def _season_files() -> list[str]:
	return sorted(p.name for p in SOURCE.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))


def _clone_csvs(dest: Path) -> list[str]:
	"""Copy-on-write clone of every season CSV whose size/mtime changed."""
	updated = []
	for name in _season_files():
		src, dst = SOURCE / name, dest / name
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


def sync(ref: str | None = None) -> dict:
	DEST.mkdir(parents=True, exist_ok=True)
	for rel in CODE_PATHS + DATA_FILES:
		p = DEST / rel
		if p.is_dir():
			shutil.rmtree(p)
		elif p.exists():
			p.unlink()
	if ref:
		_copy_tree_from_ref(DEST, ref)
		commit = _git("rev-parse", ref)
		dirty = False
	else:
		_copy_tree_from_worktree(DEST)
		commit = _git("rev-parse", "HEAD")
		dirty = bool(_git("status", "--porcelain", "--", *CODE_PATHS, *DATA_FILES))
	(DEST / "scripts" / "__init__.py").touch()
	updated = _clone_csvs(DEST)
	params = json.loads((DEST / "optuna_best_params.json").read_text())
	manifest = {
		"synced_at": datetime.now(timezone.utc).isoformat(),
		"source": str(SOURCE),
		"ref": ref or "worktree",
		"commit": commit,
		"dirty": dirty,
		"params_version": params.get("version"),
		"params_best_value": params.get("best_value"),
		"params_saved_at": params.get("saved_at"),
		"csvs_updated": updated,
	}
	(DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
	return manifest


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", help="git ref to take code/params from (default: working tree)")
	args = ap.parse_args()
	print(json.dumps(sync(args.ref), indent=2))


if __name__ == "__main__":
	main()

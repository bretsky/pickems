"""Worlds Pick'Em / Crystal Ball simulator."""

import os
from pathlib import Path

# KEY=value lines from the repo's .env (gitignored), e.g. POLYMARKET_WALLET; the real
# environment wins over the file.
_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
	for _line in _ENV.read_text().splitlines():
		_k, _, _v = _line.strip().partition("=")
		if _k and not _k.startswith("#") and _v:
			os.environ.setdefault(_k.strip().removeprefix("export ").strip(), _v.strip().strip("\"'"))

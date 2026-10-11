"""Leaguepedia cargo client (same mwrogue + LP_USERNAME/LP_PASSWORD setup as
python-sharp's psharp/leaguepedia.py). Anonymous cargo queries are rate-limited
to near-uselessness, so credentials are required: put LP_USERNAME/LP_PASSWORD
in pickems/.env."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from mwclient.errors import APIError, MaximumRetriesExceeded
from mwrogue.auth_credentials import AuthCredentials
from mwrogue.esports_client import EsportsClient

from pickems.sync_model import ROOT

load_dotenv(ROOT / ".env")


class LPClient:
	def __init__(self, max_backoff: int = 64):
		user, password = os.getenv("LP_USERNAME"), os.getenv("LP_PASSWORD")
		if not user or not password:
			raise RuntimeError("LP_USERNAME/LP_PASSWORD missing from pickems/.env")
		self.client = EsportsClient("lol", credentials=AuthCredentials(username=user, password=password))
		self.max_backoff = max_backoff

	def query(self, **kwargs) -> list[dict]:
		"""Paged cargoquery (tables, fields, where, join_on, order_by, ...)."""
		rows: list[dict] = []
		backoff = 2
		while True:
			try:
				page = self.client.cargo_client.query(limit=500, offset=len(rows), **kwargs)
			except (APIError, MaximumRetriesExceeded):
				if backoff > self.max_backoff:
					raise
				time.sleep(backoff)
				backoff *= 2
				continue
			backoff = 2
			rows.extend(page)
			if len(page) < 500:
				return rows

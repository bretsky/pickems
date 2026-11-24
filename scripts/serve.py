"""Serve dashboard/out the way the deployed site behaves (instead of a plain http.server).

  .venv/bin/python scripts/serve.py [port]     # default 8765

Pages get the same <meta> stamps scripts/stage_site.py adds at deploy, /versions.json comes from
dashboard/versions.json, and /v/<label>/ serves each snapshot straight from its run in
dashboard/out/runs (what functions/v/[[path]].js proxies to on Cloudflare), so the Forecast picker,
snapshot banner and single-run mode all work locally.
"""

import functools
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_site import FAVICON, OUT, inject, load_versions  # noqa: E402

PAGE = re.compile(r"^(?:index\.html|runs/[^/]+\.html)?$")


class Handler(SimpleHTTPRequestHandler):
	def send_bytes(self, body, ctype, status=200):
		self.send_response(status)
		self.send_header("Content-Type", ctype)
		self.send_header("Content-Length", str(len(body)))
		self.send_header("Cache-Control", "no-cache")
		self.end_headers()
		if self.command != "HEAD":
			self.wfile.write(body)

	def page(self, path, snapshot):
		if not path.is_file():
			return self.send_error(404)
		self.send_bytes(inject(path.read_text(), snapshot).encode(), "text/html; charset=utf-8")

	def route(self):
		url = urlsplit(self.path)
		path = unquote(url.path)
		if path == "/versions.json":
			return self.send_bytes(json.dumps(load_versions()).encode(), "application/json")
		if path == "/favicon.svg":
			return self.send_bytes(FAVICON.read_bytes(), "image/svg+xml")
		if m := re.fullmatch(r"/v/([a-z0-9-]{1,28})(/.*)?", path):
			label, rest = m.group(1), m.group(2)
			if rest is None:
				self.send_response(308)
				self.send_header("Location", f"/v/{label}/{'?' + url.query if url.query else ''}")
				return self.end_headers()
			snap = next((s for s in load_versions()["snapshots"] if s["label"] == label), None)
			if snap is None:
				return self.send_error(404, "No such snapshot")
			rest = rest.lstrip("/")
			if rest in ("", "index.html"):
				return self.page(OUT / "runs" / f"{snap['run_name']}.html", label)
			if PAGE.match(rest):
				return self.page(OUT / rest, label)
			if not rest.startswith("runs/"):
				return self.send_error(404)  # a snapshot ships only its page and runs
			self.path = "/" + rest
			return super().do_GET() if self.command == "GET" else super().do_HEAD()
		if PAGE.match(path.lstrip("/")):
			return self.page(OUT / (path.lstrip("/") or "index.html"), None)
		return super().do_GET() if self.command == "GET" else super().do_HEAD()

	do_GET = do_HEAD = route


def main():
	port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
	server = ThreadingHTTPServer(("", port), functools.partial(Handler, directory=str(OUT)))
	print(f"serving {OUT} like the deployed site at http://localhost:{port}/", flush=True)
	server.serve_forever()


if __name__ == "__main__":
	main()

"""Local read-only dashboard: history, skill audit, and config — nothing else.

Stdlib http.server only (no flask/fastapi): this is a viewer, not a service.
Binds 127.0.0.1 unconditionally and answers GET only; never imports keyring.
Cost tracking doesn't exist in Nero yet, so the page notes it arrives later
rather than inventing a tracker.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import NeroConfig
from nero.core.audit_log import AuditLog, default_audit_path
from nero.memory.history_store import HistoryStore, default_history_path

logger = logging.getLogger("nero.dashboard")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nero Agent — Dashboard</title>
<style>
  body { background: #10121a; color: #d8dee9; font-family: ui-monospace, Menlo, Consolas, monospace;
         margin: 0; padding: 2rem; }
  h1 { color: #8be9fd; font-size: 1.3rem; margin-bottom: 0; }
  .sub { color: #6b7280; margin-top: 0.25rem; margin-bottom: 2rem; }
  section { margin-bottom: 2rem; }
  h2 { color: #f1fa8c; font-size: 1rem; border-bottom: 1px solid #2a2d3a; padding-bottom: 0.3rem; }
  pre { background: #1a1d29; border: 1px solid #2a2d3a; border-radius: 6px; padding: 1rem;
        overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  .note { color: #6b7280; font-style: italic; }
</style>
</head>
<body>
<h1>Nero Agent</h1>
<p class="sub">Local dashboard — read-only, localhost only.</p>

<section>
  <h2>History</h2>
  <pre id="history">loading…</pre>
</section>

<section>
  <h2>Skill audit</h2>
  <pre id="audit">loading…</pre>
</section>

<section>
  <h2>Config</h2>
  <pre id="config">loading…</pre>
  <p class="note">Cost tracking arrives with rate/cost limits — not shown here yet.</p>
</section>

<script>
  function load(id, url) {
    fetch(url)
      .then((r) => r.json())
      .then((data) => { document.getElementById(id).textContent = JSON.stringify(data, null, 2); })
      .catch((err) => { document.getElementById(id).textContent = "error: " + err; });
  }
  load("history", "/api/history");
  load("audit", "/api/audit");
  load("config", "/api/config");
</script>
</body>
</html>
"""


def _load_config() -> NeroConfig:
    """A missing or invalid config file must never crash the dashboard."""
    try:
        return ConfigManager().load()
    except ConfigError as exc:
        logger.warning("Could not load config for dashboard: %s", exc)
        return NeroConfig()


def _config_payload(config: NeroConfig) -> dict:
    """Only the whitelisted sections — never anything from the keyring."""
    return {
        "mode": config.mode,
        "llm": config.llm.model_dump(),
        "skills": {"enabled": config.skills.enabled.model_dump()},
        "voice": {"enabled": config.voice.enabled},
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """GET-only: history, audit, config, and the page that renders them."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # quiet: this runs on every request otherwise

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/history":
            store = HistoryStore(default_history_path(), session_id="dashboard")
            self._send_json(200, store.recent())
        elif self.path == "/api/audit":
            entries = AuditLog(default_audit_path()).recent(limit=100)
            self._send_json(200, [entry.model_dump(mode="json") for entry in entries])
        elif self.path == "/api/config":
            self._send_json(200, _config_payload(_load_config()))
        else:
            self._send_json(404, {"error": "not found"})

    def _method_not_allowed(self) -> None:
        self._send_json(405, {"error": "method not allowed"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _method_not_allowed


def run_dashboard(port: int) -> None:
    """Serve the dashboard on 127.0.0.1:port until Ctrl+C."""
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    try:
        host, bound_port = server.server_address
        print(f"Nero dashboard: http://{host}:{bound_port} (Ctrl+C to stop)")
        server.serve_forever()
    finally:
        server.server_close()

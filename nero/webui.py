"""The dashboard: Nero in a browser, on localhost.

Stdlib `http.server` only — a handful of GET endpoints and one POST, not a
reason to take on flask or fastapi. The page itself is the React/shadcn app in
web/, built into webui_dist/ and served from here; nero/dashboard.py is the
read side it asks for.

A turn here is the same turn as one in the terminal: `ChatLoop.ask` runs it, so
the fallback chain, key rotation, memory recall and skills all behave
identically.

## Why a local port still needs auth

This accepts POST and can open apps and read files, which makes a localhost
port a real trust boundary:

- **Any website you have open can POST to 127.0.0.1.** A form or `fetch` with a
  simple content type is sent cross-origin without a preflight, and the reply
  being unreadable to the attacker does not help — running the skill was the
  damage. So every request carries a token, generated per run and printed once.
- **DNS rebinding** points an attacker's hostname at 127.0.0.1 to get
  same-origin access, so the `Host` header is checked against loopback rather
  than trusted.
- **Other accounts on the machine** can reach a loopback port; the token is
  what keeps this to whoever can read the terminal that started it.

The pages behind the sidebar also *write*: settings go through the same
validate-then-save path as `nero config set`, and a stored conversation or
remembered fact can be deleted. So they sit behind the same token as chat —
they used to be open on loopback, back when the dashboard was a separate
GET-only server, but a page that can run a skill and rewrite the config has no
business answering anything that asks.

Two things stay out of reach on purpose: **API keys**, which live in the OS
keyring and are neither read nor written here, and the **audit log**, which is
the record of what Nero actually did. A record you can quietly edit from a
browser is not a record.

The token is in the URL you open, then held in the page and sent as a header.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import secrets
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger("nero.web")

DEFAULT_PORT = 8643
TOKEN_HEADER = "X-Nero-Token"
# A turn can legitimately be a pasted paragraph, but not a megabyte.
MAX_BODY_BYTES = 64 * 1024
_LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}


def new_token() -> str:
    return secrets.token_urlsafe(24)


def host_is_loopback(host_header: str | None) -> bool:
    """Whether `Host` names this machine.

    Unchecked, a hostname an attacker controls can resolve to 127.0.0.1 and the
    browser will treat their page as same-origin with this server.
    """
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):  # [::1]:8643
        host = host[: host.find("]") + 1]
    elif ":" in host:
        host = host.rsplit(":", 1)[0]
    return host.lower() in _LOOPBACK


DIST = Path(__file__).parent / "webui_dist"
INDEX = DIST / "index.html"
BUILD_HINT = (
    "The web UI has not been built. From the repo: cd web && npm install && npm run build"
)


def _asset(path: str) -> Path | None:
    """The file `path` names inside the built bundle, or None.

    Resolved and re-checked against DIST rather than joined and trusted: a
    request for `/assets/../../../etc/passwd` must not escape the bundle, and
    `..` can also arrive percent-encoded or through a symlink.
    """
    candidate = (DIST / path.lstrip("/")).resolve()
    try:
        candidate.relative_to(DIST.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def make_handler(ask: Callable[[str], str | None], history, assistant_name: str, token: str,
                 registry=None):
    """A request handler bound to one session. `ask` is `ChatLoop.ask`.

    `registry` is the live SkillRegistry, so the Skills page can say what the
    model may actually call right now rather than re-deriving it from config.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "nero"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet: this fires on every request otherwise

        # --- plumbing ---------------------------------------------------

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # Nothing here should ever be framed by another page or sniffed
            # into a different content type.
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, supplied: str | None) -> bool:
            if not host_is_loopback(self.headers.get("Host")):
                logger.warning("rejected a request with Host=%r", self.headers.get("Host"))
                return False
            return bool(supplied) and secrets.compare_digest(supplied, token)

        # --- routes -----------------------------------------------------

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            kind, _ = mimetypes.guess_type(path.name)
            self.send_response(200)
            self.send_header("Content-Type", kind or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path, _, _query = self.path.partition("?")
            # The bundle holds no secret and cannot be read cross-origin, so it
            # is served on loopback without a token. That is also what lets a
            # reload work once the page has taken the token out of the address
            # bar. Everything under /api/ is what the token actually guards.
            if not host_is_loopback(self.headers.get("Host")):
                self._json(403, {"error": "forbidden"})
                return
            if path == "/":
                if not INDEX.is_file():
                    self._json(503, {"error": BUILD_HINT})
                    return
                self._send_file(INDEX)
                return
            if path.startswith("/assets/") or path in ("/favicon.ico", "/vite.svg"):
                asset = _asset(path)
                if asset is None:
                    self._json(404, {"error": "not found"})
                    return
                self._send_file(asset)
                return
            if path == "/api/meta":
                if not self._authorized(self.headers.get(TOKEN_HEADER)):
                    self._json(403, {"error": "forbidden"})
                    return
                # The bundle is static, so a renamed assistant cannot be baked
                # in at build time — the page asks for it instead.
                self._json(200, {"name": assistant_name})
                return
            if path in ("/api/audit", "/api/config", "/api/state"):
                if not self._authorized(self.headers.get(TOKEN_HEADER)):
                    self._json(403, {"error": "forbidden"})
                    return
                from nero import dashboard

                readers = {
                    "/api/audit": dashboard.audit_payload,
                    "/api/config": dashboard.config_payload,
                    "/api/state": lambda: dashboard.state_payload(registry),
                }
                try:
                    self._json(200, readers[path]())
                except Exception as exc:  # noqa: BLE001 — a bad read is not a dead page
                    logger.debug("%s failed", path, exc_info=True)
                    self._json(500, {"error": str(exc)})
                return
            if path == "/api/history":
                if not self._authorized(self.headers.get(TOKEN_HEADER)):
                    self._json(403, {"error": "forbidden"})
                    return
                self._json(200, history.recent() if history is not None else [])
                return
            self._json(404, {"error": "not found"})

        def _body(self) -> dict | None:
            """The JSON body of a POST, or None once an error has been sent.

            A browser only sends application/json cross-origin after a preflight
            this server never answers, so requiring it is a second lock on the
            same door as the token.
            """
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._json(415, {"error": "expected application/json"})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json(400, {"error": "bad length"})
                return None
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"error": "message too large"})
                return None
            try:
                payload = json.loads(self.rfile.read(length))
            except ValueError:
                self._json(400, {"error": "expected a JSON object"})
                return None
            if not isinstance(payload, dict):
                self._json(400, {"error": "expected a JSON object"})
                return None
            return payload

        def do_POST(self) -> None:
            if self.path not in ("/api/chat", "/api/edit"):
                self._json(404, {"error": "not found"})
                return
            if not self._authorized(self.headers.get(TOKEN_HEADER)):
                self._json(403, {"error": "forbidden"})
                return
            payload = self._body()
            if payload is None:
                return
            if self.path == "/api/edit":
                self._edit(payload)
                return
            try:
                text = str(payload["text"]).strip()
            except (KeyError, TypeError):
                self._json(400, {"error": "expected a JSON object with a text field"})
                return
            if not text:
                self._json(400, {"error": "empty message"})
                return
            try:
                reply = ask(text)
            except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the server
                logger.debug("turn failed", exc_info=True)
                self._json(200, {"reply": f"Something went wrong with that turn: {exc}"})
                return
            self._json(200, {"reply": reply or "(no reply)"})

        def _edit(self, payload: dict) -> None:
            """Apply one edit, then answer with the state it produced.

            The reply is the whole new state rather than an acknowledgement, so
            the page renders what Nero actually saved. An optimistic update
            would show a value the validator may have coerced or refused.
            """
            from nero import dashboard

            try:
                dashboard.apply_edit(
                    str(payload.get("action") or ""),
                    str(payload.get("key") or ""),
                    payload.get("value"),
                )
            except dashboard.EditError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 — a failed edit is not a dead page
                logger.debug("edit failed", exc_info=True)
                self._json(500, {"error": str(exc)})
                return
            self._json(200, dashboard.state_payload(registry))

        def _not_allowed(self) -> None:
            self._json(405, {"error": "method not allowed"})

        do_PUT = do_DELETE = do_PATCH = _not_allowed

    return Handler


def serve(ask, history, assistant_name: str, port: int = DEFAULT_PORT,
          token: str | None = None, on_ready: Callable[[str], None] = print,
          registry=None):
    """Serve the dashboard on 127.0.0.1:port until interrupted."""
    token = token or new_token()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(ask, history, assistant_name, token, registry)
    )
    try:
        host, bound = server.server_address
        on_ready(f"http://{host}:{bound}/?token={token}")
        server.serve_forever()
    finally:
        server.server_close()

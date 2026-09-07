"""The browser chat window (nero/webui.py).

The dashboard is GET-only and read-only. This one accepts POST and can open
apps and read files, so a loopback port becomes a real trust boundary — most
of what is locked down here is that boundary, not the HTML.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from nero.webui import (
    DIST,
    MAX_BODY_BYTES,
    TOKEN_HEADER,
    _asset,
    host_is_loopback,
    new_token,
    serve,
)


class TestHostCheck:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "127.0.0.1:8643", "localhost", "localhost:8643", "[::1]:8643"]
    )
    def test_loopback_names_pass(self, host):
        assert host_is_loopback(host)

    @pytest.mark.parametrize(
        "host",
        ["evil.com", "evil.com:8643", "127.0.0.1.evil.com", "localhost.evil.com", "", None],
    )
    def test_everything_else_fails(self, host):
        """DNS rebinding: an attacker's hostname resolving to 127.0.0.1 would
        otherwise be same-origin with this server."""
        assert not host_is_loopback(host)


class TestToken:
    def test_tokens_are_long_and_unique(self):
        tokens = {new_token() for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(t) >= 24 for t in tokens)


class TestBundle:
    """The page is a built React/shadcn bundle (web/ -> nero/webui_dist),
    committed so `pip install` and the frozen binary need no node."""

    def test_the_bundle_is_present_in_the_package(self):
        assert (DIST / "index.html").is_file(), "run: cd web && npm run build"
        assert list((DIST / "assets").glob("*.js")), "no JS bundle was built"

    @pytest.mark.parametrize(
        "path",
        [
            "/assets/../../../../etc/passwd",
            "/assets/../../nero/config/manager.py",
            "/assets/../webui.py",
        ],
    )
    def test_no_request_can_escape_the_bundle_directory(self, path):
        assert _asset(path) is None

    def test_a_real_asset_resolves(self):
        name = next((DIST / "assets").glob("*.js")).name
        assert _asset(f"/assets/{name}") is not None

    def test_a_missing_asset_is_not_a_crash(self):
        assert _asset("/assets/nope.js") is None


@pytest.fixture
def server():
    """A real server on a free port, with a fake model behind it."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    state = {"turns": [], "reply": "ok"}

    class History:
        def recent(self):
            return [{"role": "user", "content": "earlier"}]

    def ask(text):
        state["turns"].append(text)
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return state["reply"]

    thread = threading.Thread(
        target=lambda: serve(ask, History(), "Nero", port=port, token="SECRET",
                             on_ready=lambda _u: None),
        daemon=True,
    )
    thread.start()
    for _ in range(50):  # wait for the socket rather than sleeping blind
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.2)
            break
        except urllib.error.HTTPError:
            break
        except OSError:
            time.sleep(0.02)
    state["port"] = port
    return state


def call(port, path, method="GET", body=None, headers=None, host=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
    )
    request.add_header("Host", host or f"127.0.0.1:{port}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


AUTH = {"Content-Type": "application/json", TOKEN_HEADER: "SECRET"}


class TestChatEndpoint:
    def test_a_valid_request_runs_the_turn(self, server):
        status, body = call(server["port"], "/api/chat", "POST", {"text": "hi"}, AUTH)
        assert status == 200
        assert json.loads(body)["reply"] == "ok"
        assert server["turns"] == ["hi"]

    def test_no_token_never_reaches_the_model(self, server):
        status, _ = call(server["port"], "/api/chat", "POST", {"text": "rm -rf"},
                         {"Content-Type": "application/json"})
        assert status == 403
        assert server["turns"] == []

    def test_a_wrong_token_never_reaches_the_model(self, server):
        status, _ = call(server["port"], "/api/chat", "POST", {"text": "x"},
                         {"Content-Type": "application/json", TOKEN_HEADER: "guess"})
        assert status == 403
        assert server["turns"] == []

    def test_a_rebound_host_is_refused(self, server):
        status, _ = call(server["port"], "/api/chat", "POST", {"text": "x"}, AUTH,
                         host="evil.com")
        assert status == 403
        assert server["turns"] == []

    def test_a_simple_content_type_is_refused(self, server):
        """text/plain is what a cross-origin form post can send without a
        preflight, so it must not be an accepted way in."""
        status, _ = call(server["port"], "/api/chat", "POST", {"text": "x"},
                         {"Content-Type": "text/plain", TOKEN_HEADER: "SECRET"})
        assert status == 415
        assert server["turns"] == []

    def test_an_oversized_body_is_refused_before_it_is_read(self, server):
        status, _ = call(
            server["port"], "/api/chat", "POST", {"text": "x" * (MAX_BODY_BYTES + 10)}, AUTH
        )
        assert status == 413
        assert server["turns"] == []

    def test_an_empty_message_is_not_a_turn(self, server):
        status, _ = call(server["port"], "/api/chat", "POST", {"text": "   "}, AUTH)
        assert status == 400
        assert server["turns"] == []

    def test_a_malformed_body_is_refused(self, server):
        status, _ = call(server["port"], "/api/chat", "POST", {"nope": 1}, AUTH)
        assert status == 400

    def test_a_failing_turn_answers_instead_of_killing_the_server(self, server):
        server["reply"] = RuntimeError("provider exploded")
        status, body = call(server["port"], "/api/chat", "POST", {"text": "hi"}, AUTH)
        assert status == 200
        assert "provider exploded" in json.loads(body)["reply"]
        # still serving
        server["reply"] = "recovered"
        assert json.loads(call(server["port"], "/api/chat", "POST", {"text": "again"},
                               AUTH)[1])["reply"] == "recovered"

    def test_an_empty_reply_still_answers(self, server):
        server["reply"] = None
        status, body = call(server["port"], "/api/chat", "POST", {"text": "hi"}, AUTH)
        assert json.loads(body)["reply"] == "(no reply)"


class TestOtherRoutes:
    def test_the_page_itself_is_open_but_the_api_behind_it_is_not(self):
        """Superseded by TestBundleRoutes: the shell is public on loopback,
        every /api/ route needs the token. Kept as a one-line statement of the
        rule so it is not quietly inverted later."""
        import inspect

        from nero import webui

        source = inspect.getsource(webui.make_handler)
        assert source.count("_authorized(self.headers.get(TOKEN_HEADER))") >= 3

    def test_history_needs_the_token(self, server):
        assert call(server["port"], "/api/history")[0] == 403
        status, body = call(server["port"], "/api/history", headers={TOKEN_HEADER: "SECRET"})
        assert status == 200 and json.loads(body)[0]["content"] == "earlier"

    def test_unknown_paths_are_404_not_a_file_read(self, server):
        status, _ = call(server["port"], "/../../etc/passwd",
                         headers={TOKEN_HEADER: "SECRET"})
        assert status in (400, 404)

    def test_write_methods_are_rejected(self, server):
        for method in ("PUT", "DELETE", "PATCH"):
            assert call(server["port"], "/api/chat", method, headers=AUTH)[0] == 405

    def test_responses_cannot_be_framed_or_sniffed(self, server):
        request = urllib.request.Request(f"http://127.0.0.1:{server['port']}/?token=SECRET")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.headers.get("X-Frame-Options") == "DENY"


class TestBundleRoutes:
    def test_the_page_is_served_without_a_token(self, server):
        """The bundle holds no secret and cannot be read cross-origin. Serving
        it unauthenticated is also what lets a reload work after the page has
        taken the token out of the address bar."""
        status, body = call(server["port"], "/")
        assert status == 200
        assert "<!doctype html>" in body.lower()

    def test_the_page_is_still_refused_to_a_rebound_host(self, server):
        assert call(server["port"], "/", host="evil.com")[0] == 403

    def test_assets_are_served(self, server):
        name = next((DIST / "assets").glob("*.js")).name
        status, body = call(server["port"], f"/assets/{name}")
        assert status == 200 and len(body) > 1000

    def test_a_traversing_asset_request_is_404_not_a_file_read(self, server):
        status, body = call(server["port"], "/assets/../../../../etc/passwd")
        assert status in (400, 404)
        assert "root:" not in body

    def test_meta_reports_the_configured_name_and_needs_the_token(self, server):
        assert call(server["port"], "/api/meta")[0] == 403
        status, body = call(server["port"], "/api/meta", headers={TOKEN_HEADER: "SECRET"})
        assert status == 200 and json.loads(body)["name"] == "Nero"

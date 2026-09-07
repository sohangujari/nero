"""The dashboard's read side (nero/dashboard.py) and its routes in the web UI.

The page is now the React/shadcn app; this module is only the data behind the
Activity and Config tabs. The whitelist is the part that matters — it decides
what a browser is allowed to see, and it must never reach the keyring.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pytest

from nero import dashboard
from nero.config.manager import ConfigManager
from nero.core.audit_log import AuditEntry, AuditLog
from nero.memory.history_store import HistoryStore
from nero.webui import TOKEN_HEADER, serve


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every store at tmp_path — never the developer's real files."""
    audit_path = tmp_path / "audit.db"
    history_path = tmp_path / "history.db"
    monkeypatch.setattr("nero.dashboard.default_audit_path", lambda: audit_path)
    monkeypatch.setattr("nero.dashboard.default_history_path", lambda: history_path)
    monkeypatch.setattr(
        "nero.dashboard.ConfigManager", lambda: ConfigManager(config_dir=tmp_path / "config")
    )
    return audit_path, history_path


class TestPayloads:
    def test_config_excludes_anything_from_the_keyring(self, isolated):
        body = json.dumps(dashboard.config_payload())
        assert "api_key" not in body.lower()
        assert "token" not in body.lower()

    def test_config_includes_the_provider_and_model(self, isolated):
        payload = dashboard.config_payload()
        assert payload["llm"]["provider"] == "claude"
        assert payload["llm"]["model"]
        assert set(payload) == {"mode", "llm", "skills", "voice"}

    def test_a_broken_config_file_does_not_crash_the_view(self, tmp_path, monkeypatch):
        from nero.config.manager import ConfigError

        def boom():
            raise ConfigError("bad yaml")

        monkeypatch.setattr("nero.dashboard.ConfigManager", lambda: type("M", (), {"load": staticmethod(boom)})())
        assert dashboard.config_payload()["llm"]["provider"] == "claude"  # defaults

    def test_audit_reports_recorded_entries(self, isolated):
        audit_path, _ = isolated
        AuditLog(audit_path).record(
            AuditEntry(
                timestamp=datetime.now(UTC),
                skill_name="open_app",
                arguments={"app_name": "Safari"},
                result_summary="Opened Safari.",
                provider="claude",
            )
        )
        entries = dashboard.audit_payload()
        assert len(entries) == 1
        assert entries[0]["skill_name"] == "open_app"
        assert entries[0]["provider"] == "claude"

    def test_history_reports_appended_turns(self, isolated):
        _, history_path = isolated
        HistoryStore(history_path, session_id="s1").append_turn("hi", "hello")
        assert dashboard.history_payload() == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_empty_stores_are_empty_lists_not_errors(self, isolated):
        assert dashboard.audit_payload() == []
        assert dashboard.history_payload() == []


@pytest.fixture
def server(isolated):
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    thread = threading.Thread(
        target=lambda: serve(lambda t: "ok", None, "Nero", port=port, token="SECRET",
                             on_ready=lambda _u: None),
        daemon=True,
    )
    thread.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.2)
            break
        except urllib.error.HTTPError:
            break
        except OSError:
            time.sleep(0.02)
    return port


def call(port, path, headers=None, host=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    request.add_header("Host", host or f"127.0.0.1:{port}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


class TestRoutes:
    """These used to be open on a loopback port. They now sit behind the token,
    because the same server also accepts chat turns that can run skills."""

    def test_audit_needs_the_token(self, server):
        assert call(server, "/api/audit")[0] == 403
        status, body = call(server, "/api/audit", {TOKEN_HEADER: "SECRET"})
        assert status == 200 and json.loads(body) == []

    def test_config_needs_the_token(self, server):
        assert call(server, "/api/config")[0] == 403
        status, body = call(server, "/api/config", {TOKEN_HEADER: "SECRET"})
        assert status == 200 and json.loads(body)["llm"]["provider"] == "claude"

    def test_config_over_the_wire_still_carries_no_key(self, server):
        _, body = call(server, "/api/config", {TOKEN_HEADER: "SECRET"})
        assert "api_key" not in body.lower()

    def test_a_rebound_host_is_refused(self, server):
        assert call(server, "/api/config", {TOKEN_HEADER: "SECRET"}, host="evil.com")[0] == 403

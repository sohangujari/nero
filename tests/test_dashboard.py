import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer

import pytest

from nero.config.manager import ConfigManager
from nero.core.audit_log import AuditEntry, AuditLog
from nero.dashboard import DashboardHandler
from nero.memory.history_store import HistoryStore


@pytest.fixture
def dashboard_server(tmp_path, monkeypatch):
    """A real dashboard server on an ephemeral port, pointed at tmp_path DBs."""
    audit_path = tmp_path / "audit.db"
    history_path = tmp_path / "history.db"
    monkeypatch.setattr("nero.dashboard.default_audit_path", lambda: audit_path)
    monkeypatch.setattr("nero.dashboard.default_history_path", lambda: history_path)
    # Never read the developer's real config file.
    monkeypatch.setattr(
        "nero.dashboard.ConfigManager", lambda: ConfigManager(config_dir=tmp_path / "config")
    )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, audit_path, history_path
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _url(httpd, path: str) -> str:
    port = httpd.server_address[1]
    return f"http://127.0.0.1:{port}{path}"


def _get(httpd, path: str):
    return urllib.request.urlopen(_url(httpd, path))


class TestDashboard:
    def test_index_page_is_html_with_nero_agent(self, dashboard_server):
        httpd, _, _ = dashboard_server
        resp = _get(httpd, "/")
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        assert "Nero Agent" in resp.read().decode()

    def test_audit_endpoint_returns_recorded_entries(self, dashboard_server):
        httpd, audit_path, _ = dashboard_server
        AuditLog(audit_path).record(
            AuditEntry(
                timestamp=datetime.now(UTC),
                skill_name="open_app",
                arguments={"app_name": "Safari"},
                result_summary="Opened Safari.",
                provider="claude",
            )
        )
        data = json.loads(_get(httpd, "/api/audit").read())
        assert len(data) == 1
        assert data[0]["skill_name"] == "open_app"
        assert data[0]["provider"] == "claude"

    def test_history_endpoint_returns_appended_turns(self, dashboard_server):
        httpd, _, history_path = dashboard_server
        HistoryStore(history_path, session_id="s1").append_turn("hi", "hello")
        data = json.loads(_get(httpd, "/api/history").read())
        assert data == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_config_endpoint_includes_provider_excludes_keys(self, dashboard_server):
        httpd, _, _ = dashboard_server
        body = _get(httpd, "/api/config").read().decode()
        assert "api_key" not in body
        data = json.loads(body)
        assert data["llm"]["provider"] == "claude"
        assert data["llm"]["model"]

    def test_empty_dbs_render_as_empty_lists(self, dashboard_server):
        httpd, _, _ = dashboard_server
        assert json.loads(_get(httpd, "/api/audit").read()) == []
        assert json.loads(_get(httpd, "/api/history").read()) == []

    def test_post_is_rejected(self, dashboard_server):
        httpd, _, _ = dashboard_server
        request = urllib.request.Request(_url(httpd, "/api/audit"), method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 405

    def test_unknown_path_is_404(self, dashboard_server):
        httpd, _, _ = dashboard_server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(httpd, "/nope")
        assert exc_info.value.code == 404

    def test_binds_localhost_only(self, dashboard_server):
        httpd, _, _ = dashboard_server
        assert httpd.server_address[0] == "127.0.0.1"

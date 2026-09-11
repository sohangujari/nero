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
        assert set(payload) == set(dashboard.CONFIG_SECTIONS)

    def test_config_never_carries_the_one_section_that_stores_secrets(self, isolated):
        """MCPServerConfig.env holds literal environment values, commonly API
        keys. It has its own page, which reports env by key name only."""
        assert "mcp" not in dashboard.config_payload()
        assert "mcp" not in dashboard.CONFIG_SECTIONS

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


def post(port, path, body, headers=None, content_type="application/json", host=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Host", host or f"127.0.0.1:{port}")
    request.add_header("Content-Type", content_type)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


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


class FakeRegistry:
    """Two skills: one live, one switched off. Enough to prove the Skills page
    reports `available` from the registry rather than re-deriving it."""

    NAMES = {"open_app", "run_shell"}

    def known_names(self):
        return set(self.NAMES)

    def get(self, name):
        tier = "destructive" if name == "run_shell" else "state_changing"
        meta = type(
            "Meta", (), {"description": f"{name} does a thing", "permission_tier": tier,
                         "category": "Apps" if name == "open_app" else "Code",
                         "requires_network": name == "open_app"}
        )()
        return type("Skill", (), {"meta": meta})()

    def is_enabled(self, name):
        return name == "open_app"

    def is_available(self, name):
        return name == "open_app"


class TestState:
    """`/api/state` is what every read-only page in the sidebar reads."""

    def test_it_carries_a_section_per_sidebar_page(self, isolated):
        state = dashboard.state_payload()
        assert {"channels", "models", "skills", "routines", "sessions", "mcp",
                "memory", "counts"} <= set(state)

    def test_skills_distinguish_disabled_from_available(self, isolated):
        rows = {s["name"]: s for s in dashboard.skills_payload(FakeRegistry())}
        assert rows["open_app"]["available"] is True
        assert rows["run_shell"]["enabled"] is False
        assert rows["run_shell"]["tier"] == "destructive"
        assert rows["run_shell"]["category"] == "Code"

    def test_no_registry_means_no_skills_rather_than_a_crash(self, isolated):
        assert dashboard.skills_payload(None) == []

    def test_channels_name_every_way_in_including_this_one(self, isolated):
        channels = dashboard.channels_payload()
        assert set(channels) == {
            "terminal", "dashboard", "voice",
            "telegram", "discord", "slack", "googlechat",
        }
        assert channels["dashboard"]["enabled"] is True

    def test_mcp_reports_env_keys_but_never_env_values(self, isolated):
        """MCP env vars are commonly API keys. The names are useful; the values
        are exactly what must not reach a browser."""
        from nero.config.schema import MCPServerConfig, NeroConfig

        config = NeroConfig()
        config.mcp.servers["x"] = MCPServerConfig(
            command="run", env={"SECRET_TOKEN": "hunter2"}
        )
        payload = dashboard.mcp_payload(config)
        assert payload[0]["env_keys"] == ["SECRET_TOKEN"]
        assert "hunter2" not in json.dumps(payload)

    def test_sessions_group_the_transcript_by_session(self, isolated):
        _, history_path = isolated
        HistoryStore(history_path, session_id="terminal-1").append_turn("hi", "hello")
        HistoryStore(history_path, session_id="telegram-9").append_turn("a", "b")
        rows = {s["session_id"]: s for s in dashboard.sessions_payload()}
        assert rows["terminal-1"]["turns"] == 2
        assert set(rows) == {"terminal-1", "telegram-9"}

    def test_counts_agree_with_the_lists_they_summarise(self, isolated):
        _, history_path = isolated
        HistoryStore(history_path, session_id="s1").append_turn("hi", "hello")
        state = dashboard.state_payload(FakeRegistry())
        assert state["counts"]["skills_total"] == 2
        assert state["counts"]["skills_available"] == 1
        assert state["counts"]["sessions"] == len(state["sessions"]) == 1
        assert state["counts"]["turns"] == 2

    def test_a_broken_launchd_dir_does_not_break_routines(self, isolated, monkeypatch):
        monkeypatch.setattr(
            "nero.routines.default_agents_dir", lambda: (_ for _ in ()).throw(OSError("no home"))
        )
        assert dashboard.routines_payload() == []


class TestEdits:
    """The dashboard writes as well as reads. Everything routes through
    ConfigManager, so a value the CLI would reject is rejected here too."""

    def test_a_setting_is_saved_through_the_same_path_as_the_cli(self, isolated, tmp_path):
        dashboard.apply_edit("set", "llm.model", "claude-opus-5")
        assert ConfigManager(config_dir=tmp_path / "config").load().llm.model == "claude-opus-5"

    def test_an_invalid_value_is_refused_and_nothing_is_written(self, isolated, tmp_path):
        before = ConfigManager(config_dir=tmp_path / "config").load().llm.provider
        with pytest.raises(dashboard.EditError):
            dashboard.apply_edit("set", "llm.provider", "not-a-provider")
        assert ConfigManager(config_dir=tmp_path / "config").load().llm.provider == before

    def test_an_unknown_key_is_refused(self, isolated):
        with pytest.raises(dashboard.EditError, match="Unknown config key"):
            dashboard.apply_edit("set", "llm.nope", "x")

    def test_an_unknown_action_is_refused(self, isolated):
        with pytest.raises(dashboard.EditError, match="Unknown action"):
            dashboard.apply_edit("drop_everything", "llm.model")

    def test_a_routine_can_be_removed(self, isolated, tmp_path):
        from nero.config.schema import RoutineConfig

        manager = ConfigManager(config_dir=tmp_path / "config")
        config = manager.load()
        config.routines.routines["morning"] = RoutineConfig(schedule="0 8 * * *", prompt="hi")
        manager.save(config)
        dashboard.apply_edit("remove", "routines.routines.morning")
        assert manager.load().routines.routines == {}

    def test_removing_something_that_is_not_there_is_refused(self, isolated):
        with pytest.raises(dashboard.EditError, match="Unknown config key"):
            dashboard.apply_edit("remove", "routines.routines.ghost")

    def test_one_session_can_be_forgotten_without_touching_the_others(self, isolated):
        _, history_path = isolated
        HistoryStore(history_path, session_id="keep").append_turn("hi", "hello")
        HistoryStore(history_path, session_id="drop").append_turn("secret", "sure")
        dashboard.apply_edit("forget_session", "drop")
        assert [s["session_id"] for s in dashboard.sessions_payload()] == ["keep"]

    def test_a_forgotten_session_is_gone_from_recall_too(self, isolated):
        """Deleting a conversation that semantic search could still surface is
        not deleting it."""
        _, history_path = isolated
        store = HistoryStore(history_path, session_id="drop")
        store.append_turn("the passphrase is hunter2", "noted")
        dashboard.apply_edit("forget_session", "drop")
        assert store.search_keys("passphrase") == []

    def test_a_fact_can_be_forgotten(self, isolated, tmp_path, monkeypatch):
        from nero.memory.facts import FactStore

        path = tmp_path / "facts.db"
        monkeypatch.setattr("nero.memory.facts.default_facts_path", lambda: path)
        FactStore(path).remember("colour", "blue")
        dashboard.apply_edit("forget_fact", "colour")
        assert FactStore(path).all() == []

    def test_an_empty_key_is_refused_before_anything_is_touched(self, isolated):
        with pytest.raises(dashboard.EditError):
            dashboard.apply_edit("set", "", "x")


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

    def test_state_needs_the_token(self, server):
        assert call(server, "/api/state")[0] == 403
        status, body = call(server, "/api/state", {TOKEN_HEADER: "SECRET"})
        assert status == 200 and json.loads(body)["mode"] == "online"

    def test_state_over_the_wire_still_carries_no_key(self, server):
        _, body = call(server, "/api/state", {TOKEN_HEADER: "SECRET"})
        assert "api_key" not in body.lower()

    def test_a_failed_read_is_a_500_not_a_dead_page(self, server, monkeypatch):
        """One unreadable store must not take the whole dashboard down."""
        monkeypatch.setattr(
            "nero.dashboard.load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert call(server, "/api/state", {TOKEN_HEADER: "SECRET"})[0] == 500

    def test_an_edit_needs_the_token(self, server):
        """The write door is the one that matters most: this can rewrite the
        config of an agent that opens apps and reads files."""
        assert post(server, "/api/edit", {"action": "set", "key": "llm.model", "value": "x"})[0] == 403

    def test_an_edit_needs_a_json_content_type(self, server):
        """A cross-origin form post carries a simple content type and needs no
        preflight; requiring JSON is a second lock on the same door."""
        status, _ = post(
            server,
            "/api/edit",
            {"action": "set", "key": "llm.model", "value": "x"},
            {TOKEN_HEADER: "SECRET"},
            content_type="text/plain",
        )
        assert status == 415

    def test_an_edit_answers_with_the_state_it_produced(self, server):
        status, body = post(
            server,
            "/api/edit",
            {"action": "set", "key": "llm.model", "value": "claude-opus-5"},
            {TOKEN_HEADER: "SECRET"},
        )
        assert status == 200
        assert json.loads(body)["models"]["model"] == "claude-opus-5"

    def test_a_refused_edit_says_why(self, server):
        status, body = post(
            server,
            "/api/edit",
            {"action": "set", "key": "llm.provider", "value": "nonsense"},
            {TOKEN_HEADER: "SECRET"},
        )
        assert status == 400
        assert "llm.provider" in json.loads(body)["error"]

    def test_a_rebound_host_is_refused(self, server):
        assert call(server, "/api/config", {TOKEN_HEADER: "SECRET"}, host="evil.com")[0] == 403

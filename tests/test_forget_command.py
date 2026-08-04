from typer.testing import CliRunner

from nero import cli
from nero.config.schema import NeroConfig
from nero.memory.history_store import HistoryStore

runner = CliRunner()


class TestBuildHistory:
    def test_returns_none_when_memory_disabled(self):
        config = NeroConfig()
        config.memory.enabled = False
        assert cli._build_history(config) is None

    def test_returns_store_when_enabled(self, isolate_audit_log):
        store = cli._build_history(NeroConfig())
        assert isinstance(store, HistoryStore)
        assert store.max_turns == 20


class TestForget:
    def test_empty_says_so(self, isolate_audit_log):
        result = runner.invoke(cli.app, ["forget"], input="y\n")
        assert result.exit_code == 0
        assert "already empty" in result.stdout.lower()

    def test_clears_after_confirmation(self, isolate_audit_log):
        HistoryStore(isolate_audit_log.parent / "history.db", session_id="s").append_turn("q", "a")
        result = runner.invoke(cli.app, ["forget"], input="y\n")
        assert "cleared" in result.stdout.lower()

    def test_declined_leaves_history_intact(self, isolate_audit_log):
        store = HistoryStore(isolate_audit_log.parent / "history.db", session_id="s")
        store.append_turn("q", "a")
        result = runner.invoke(cli.app, ["forget"], input="n\n")
        assert store.recent() != []
        assert "cancel" in result.stdout.lower() or "kept" in result.stdout.lower()

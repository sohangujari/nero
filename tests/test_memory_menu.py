"""Phase 5b: memory settings must appear in `nero config` like every other section."""
from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


def _manager(tmp_path):
    m = ConfigManager(config_dir=tmp_path)
    m.save(NeroConfig())
    return m


class TestMemoryInShowTable:
    def test_show_lists_memory(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "show"])
        assert "Memory" in result.stdout


class TestMemoryInInteractiveMenu:
    def test_menu_has_a_memory_row(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Enter nothing -> menu renders once, then finish.
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "Memory" in result.stdout

    def test_toggle_memory_enabled(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        assert manager.load().memory.enabled is True
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 12 toggles memory.enabled; then blank line to finish.
        runner.invoke(cli.app, ["config"], input="12\n\n")
        assert manager.load().memory.enabled is False

    def test_set_max_history_turns(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 13 prompts for a new value; then blank line to finish.
        runner.invoke(cli.app, ["config"], input="13\n5\n\n")
        assert manager.load().memory.max_history_turns == 5


class TestBargeInRowHint:
    """Row 14 displays barge_in_active but toggles the raw flag; when VAD is
    off the row can never show "yes" no matter how often it's toggled, so it
    must explain itself instead of looking stuck."""

    def test_hint_appears_only_when_vad_disabled(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)

        assert manager.load().voice.vad.enabled is True
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "needs VAD auto-stop" not in result.stdout

        config = manager.load()
        config.voice.vad.enabled = False
        manager.save(config)
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "needs VAD auto-stop" in result.stdout


class TestInvalidMaxTurnsDoesNotCrash:
    def test_negative_is_rejected_gracefully(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 13, feed -1 (violates ge=0), then finish. Menu must not crash.
        result = runner.invoke(cli.app, ["config"], input="13\n-1\n\n")
        assert result.exit_code == 0
        assert manager.load().memory.max_history_turns == 8  # unchanged

"""`nero facts` / `nero facts forget` and `nero notes index` / `nero notes
search` — CLI surfaces for the memory-expansion skills.

isolate_audit_log (autouse, tests/conftest.py) redirects default_facts_path
and default_notes_index_path into tmp_path, so nothing here touches real
user data.
"""

from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig
from nero.memory.facts import FactStore

runner = CliRunner()


def _manager(tmp_path):
    m = ConfigManager(config_dir=tmp_path)
    m.save(NeroConfig())
    return m


class TestFactsCommand:
    def test_empty_says_so(self, isolate_audit_log):
        result = runner.invoke(cli.app, ["facts"])
        assert result.exit_code == 0
        assert "No facts" in result.stdout

    def test_lists_remembered_facts(self, isolate_audit_log):
        FactStore(isolate_audit_log.parent / "facts.db").remember("favorite_editor", "vim")
        result = runner.invoke(cli.app, ["facts"])
        assert result.exit_code == 0
        assert "favorite_editor" in result.stdout
        assert "vim" in result.stdout

    def test_forget_removes_a_fact(self, isolate_audit_log):
        store = FactStore(isolate_audit_log.parent / "facts.db")
        store.remember("k", "v")
        result = runner.invoke(cli.app, ["facts", "forget", "k"])
        assert result.exit_code == 0
        assert "Forgot" in result.stdout
        assert store.recall("k") is None

    def test_forget_missing_key_says_so(self, isolate_audit_log):
        result = runner.invoke(cli.app, ["facts", "forget", "nope"])
        assert result.exit_code == 0
        assert "No fact stored" in result.stdout


class TestNotesCommands:
    def test_index_without_notes_dir_is_actionable(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["notes", "index"])
        assert result.exit_code == 1
        assert "notes_dir" in result.stdout

    def test_index_reports_counts(self, monkeypatch, tmp_path, isolate_audit_log):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        manager = _manager(tmp_path)
        manager.set_value("memory.notes_dir", str(notes_dir))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["notes", "index"])
        assert result.exit_code == 0
        assert "added=1" in result.stdout

    def test_search_without_notes_dir_is_actionable(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["notes", "search", "hello"])
        assert result.exit_code == 1
        assert "notes_dir" in result.stdout

    def test_search_auto_indexes_and_finds(self, monkeypatch, tmp_path, isolate_audit_log):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("the deadline is friday")
        manager = _manager(tmp_path)
        manager.set_value("memory.notes_dir", str(notes_dir))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["notes", "search", "deadline"])
        assert result.exit_code == 0
        assert "deadline" in result.stdout

    def test_search_no_match(self, monkeypatch, tmp_path, isolate_audit_log):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        manager = _manager(tmp_path)
        manager.set_value("memory.notes_dir", str(notes_dir))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["notes", "search", "zzz_nomatch"])
        assert result.exit_code == 0
        assert "No notes match" in result.stdout

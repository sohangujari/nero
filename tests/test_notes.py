"""Notes search: NoteIndex (nero/memory/notes.py) and search_notes
(nero/skills/notes/server.py).

Every index and every note file lives only under pytest's tmp_path.
"""

import asyncio

from nero.memory.notes import NoteIndex
from nero.skills.notes.server import SearchNotesSkill


def run(coro):
    return asyncio.run(coro)


class TestReindex:
    def test_adds_new_files(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        (notes_dir / "b.txt").write_text("goodbye world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        assert index.reindex() == (2, 0, 0)

    def test_reindex_again_with_no_changes_is_a_noop(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()
        assert index.reindex() == (0, 0, 0)

    def test_unchanged_files_are_not_re_read(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()

        from pathlib import Path

        original_read_bytes = Path.read_bytes

        def tracking_read_bytes(self):
            tracking_read_bytes.calls += 1
            return original_read_bytes(self)

        tracking_read_bytes.calls = 0
        monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
        index.reindex()
        assert tracking_read_bytes.calls == 0

    def test_modified_file_is_updated(self, tmp_path):
        import os
        import time

        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        target = notes_dir / "a.md"
        target.write_text("version one")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()

        time.sleep(0.01)
        target.write_text("version two, much longer than before")
        os.utime(target, None)  # ensure a fresh mtime
        assert index.reindex() == (0, 1, 0)
        results = index.search("longer")
        assert len(results) == 1
        assert "longer" in results[0][1]

    def test_removed_file_is_dropped(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        target = notes_dir / "a.md"
        target.write_text("hello world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()

        target.unlink()
        assert index.reindex() == (0, 0, 1)
        assert index.search("hello") == []

    def test_oversize_file_skipped(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "big.txt").write_text("x" * 200)
        index = NoteIndex(tmp_path / "index.db", str(notes_dir), max_bytes=100)
        assert index.reindex() == (0, 0, 0)
        assert index.search("x") == []

    def test_binary_file_skipped(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "bin.txt").write_bytes(b"abc\x00def")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        assert index.reindex() == (0, 0, 0)

    def test_non_note_extension_ignored(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "script.py").write_text("print('hi')")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        assert index.reindex() == (0, 0, 0)

    def test_recurses_into_subdirectories(self, tmp_path):
        notes_dir = tmp_path / "notes"
        (notes_dir / "sub").mkdir(parents=True)
        (notes_dir / "sub" / "nested.md").write_text("nested content")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        assert index.reindex() == (1, 0, 0)

    def test_missing_notes_dir_is_a_noop(self, tmp_path):
        index = NoteIndex(tmp_path / "index.db", str(tmp_path / "nope"))
        assert index.reindex() == (0, 0, 0)


class TestSearch:
    def test_finds_term_with_snippet(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("the quick brown fox jumps")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()
        results = index.search("fox")
        assert len(results) == 1
        path, snippet = results[0]
        assert path.endswith("a.md")
        assert "fox" in snippet

    def test_operator_characters_do_not_raise(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("some content here")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()
        for tricky in ['weird"quote', "a - b", "NOT AND OR", "*trailing", '"unterminated']:
            index.search(tricky)  # must not raise

    def test_empty_query_returns_nothing(self, tmp_path):
        index = NoteIndex(tmp_path / "index.db", None)
        assert index.search("") == []

    def test_no_match_returns_empty(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        index.reindex()
        assert index.search("nonexistentterm") == []


class TestSearchNotesSkill:
    def test_tier_and_ingests(self):
        assert SearchNotesSkill.meta.permission_tier == "read_only"
        assert SearchNotesSkill.meta.ingests_external_content is True

    def test_unset_notes_dir_is_actionable_not_an_error(self):
        result = run(SearchNotesSkill(None).execute(query="anything"))
        assert "Error" not in result
        assert "notes_dir" in result

    def test_finds_and_envelopes_results(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("project deadline is friday")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        result = run(SearchNotesSkill(index).execute(query="deadline"))
        assert "<untrusted_content" in result
        assert "deadline" in result

    def test_auto_reindexes_when_index_is_empty(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("auto reindex works")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        # No manual reindex() call — the skill should do it on first use.
        result = run(SearchNotesSkill(index).execute(query="reindex"))
        assert "auto reindex works" in result

    def test_no_match_message(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "a.md").write_text("hello world")
        index = NoteIndex(tmp_path / "index.db", str(notes_dir))
        result = run(SearchNotesSkill(index).execute(query="zzz_nomatch"))
        assert "No notes match" in result

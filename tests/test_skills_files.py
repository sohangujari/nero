"""Filesystem skills (nero/skills/files/server.py): read_file, write_file,
edit_file, delete_path, move_path.

Every test operates only inside pytest's tmp_path — never the repo or the
user's home directory.
"""

import asyncio

from nero.skills.files.server import (
    DeletePathSkill,
    EditFileSkill,
    MovePathSkill,
    ReadFileSkill,
    WriteFileSkill,
)


def run(coro):
    return asyncio.run(coro)


class TestReadFile:
    def test_tier_and_ingests(self):
        meta = ReadFileSkill.meta
        assert meta.permission_tier == "read_only"
        assert meta.ingests_external_content is True

    def test_happy_path_wraps_in_envelope(self, tmp_path):
        target = tmp_path / "hello.txt"
        target.write_text("hello world")
        result = run(ReadFileSkill().execute(path=str(target)))
        assert "<untrusted_content" in result
        assert "hello world" in result
        assert str(target.absolute()) in result

    def test_missing_file_refused(self, tmp_path):
        result = run(ReadFileSkill().execute(path=str(tmp_path / "nope.txt")))
        assert "Error" in result

    def test_binary_file_refused(self, tmp_path):
        target = tmp_path / "bin.dat"
        target.write_bytes(b"abc\x00def")
        result = run(ReadFileSkill().execute(path=str(target)))
        assert "Error" in result
        assert "binary" in result

    def test_oversize_file_refused(self, tmp_path):
        target = tmp_path / "big.txt"
        target.write_text("x" * 100)
        result = run(ReadFileSkill().execute(path=str(target), max_bytes=10))
        assert "Error" in result
        assert "limit" in result

    def test_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "note.txt"
        target.write_text("hi")
        result = run(ReadFileSkill().execute(path="~/note.txt"))
        assert "hi" in result


class TestWriteFile:
    def test_tier(self):
        assert WriteFileSkill.meta.permission_tier == "destructive"
        assert WriteFileSkill.meta.ingests_external_content is False

    def test_creates_new_file_and_reports_created(self, tmp_path):
        target = tmp_path / "new.txt"
        result = run(WriteFileSkill().execute(path=str(target), content="hi"))
        assert target.read_text() == "hi"
        assert "Created" in result

    def test_overwrites_existing_file_and_reports_overwrote(self, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("old")
        result = run(WriteFileSkill().execute(path=str(target), content="new"))
        assert target.read_text() == "new"
        assert "Overwrote" in result

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        run(WriteFileSkill().execute(path=str(target), content="deep"))
        assert target.read_text() == "deep"


class TestEditFile:
    def test_tier(self):
        assert EditFileSkill.meta.permission_tier == "destructive"

    def test_happy_path_replaces_unique_match(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("foo bar baz")
        result = run(EditFileSkill().execute(path=str(target), old_string="bar", new_string="qux"))
        assert target.read_text() == "foo qux baz"
        assert "Edited" in result

    def test_refuses_missing_old_string(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("foo bar")
        result = run(EditFileSkill().execute(path=str(target), old_string="nope", new_string="x"))
        assert "Error" in result
        assert target.read_text() == "foo bar"

    def test_refuses_ambiguous_match_and_names_the_count(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("dup dup dup")
        result = run(EditFileSkill().execute(path=str(target), old_string="dup", new_string="x"))
        assert "Error" in result
        assert "3" in result
        assert target.read_text() == "dup dup dup"

    def test_refuses_non_file(self, tmp_path):
        result = run(
            EditFileSkill().execute(path=str(tmp_path), old_string="a", new_string="b")
        )
        assert "Error" in result


class TestDeletePath:
    def test_tier(self):
        assert DeletePathSkill.meta.permission_tier == "destructive"

    def test_deletes_file(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("x")
        result = run(DeletePathSkill().execute(path=str(target)))
        assert not target.exists()
        assert "Deleted" in result

    def test_refuses_missing_path(self, tmp_path):
        result = run(DeletePathSkill().execute(path=str(tmp_path / "nope")))
        assert "Error" in result

    def test_refuses_nonempty_dir_without_recursive(self, tmp_path):
        directory = tmp_path / "d"
        directory.mkdir()
        (directory / "child.txt").write_text("x")
        result = run(DeletePathSkill().execute(path=str(directory)))
        assert "Error" in result
        assert directory.exists()

    def test_deletes_nonempty_dir_with_recursive(self, tmp_path):
        directory = tmp_path / "d"
        directory.mkdir()
        (directory / "child.txt").write_text("x")
        result = run(DeletePathSkill().execute(path=str(directory), recursive=True))
        assert not directory.exists()
        assert "Deleted" in result

    def test_deletes_empty_dir_without_recursive(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        run(DeletePathSkill().execute(path=str(directory)))
        assert not directory.exists()

    def test_deletes_symlink_without_following_it(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "keepme.txt").write_text("x")
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        result = run(DeletePathSkill().execute(path=str(link), recursive=True))
        assert not link.exists()
        assert real_dir.exists()  # the target must survive
        assert (real_dir / "keepme.txt").exists()
        assert "symlink" in result.lower()


class TestMovePath:
    def test_tier(self):
        assert MovePathSkill.meta.permission_tier == "destructive"

    def test_moves_file(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("content")
        dest = tmp_path / "dst.txt"
        result = run(MovePathSkill().execute(source=str(source), destination=str(dest)))
        assert not source.exists()
        assert dest.read_text() == "content"
        assert "Moved" in result

    def test_refuses_missing_source(self, tmp_path):
        result = run(
            MovePathSkill().execute(
                source=str(tmp_path / "nope"), destination=str(tmp_path / "dst")
            )
        )
        assert "Error" in result

    def test_refuses_existing_destination_no_clobber(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("a")
        dest = tmp_path / "dst.txt"
        dest.write_text("b")
        result = run(MovePathSkill().execute(source=str(source), destination=str(dest)))
        assert "Error" in result
        assert source.exists()
        assert dest.read_text() == "b"

    def test_creates_destination_parent_dirs(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("a")
        dest = tmp_path / "nested" / "dst.txt"
        run(MovePathSkill().execute(source=str(source), destination=str(dest)))
        assert dest.read_text() == "a"

import pytest

from nero.memory.history_store import HistoryStore


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history.db", session_id="s1", max_turns=20)


class TestRoundtrip:
    def test_append_and_recent(self, store):
        store.append_turn("hello", "hi there")
        assert store.recent() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_empty_is_empty(self, store):
        assert store.recent() == []

    def test_oldest_first_across_turns(self, store):
        store.append_turn("one", "1")
        store.append_turn("two", "2")
        assert [m["content"] for m in store.recent()] == ["one", "1", "two", "2"]

    def test_limit_counts_exchanges_not_rows(self, store):
        for i in range(5):
            store.append_turn(f"q{i}", f"a{i}")
        recent = store.recent(limit=2)  # 2 exchanges = 4 rows
        assert [m["content"] for m in recent] == ["q3", "a3", "q4", "a4"]

    def test_default_limit_is_max_turns(self, tmp_path):
        store = HistoryStore(tmp_path / "h.db", session_id="s", max_turns=1)
        store.append_turn("old", "O")
        store.append_turn("new", "N")
        assert [m["content"] for m in store.recent()] == ["new", "N"]

    def test_session_id_is_stamped(self, tmp_path):
        import sqlite3

        path = tmp_path / "h.db"
        HistoryStore(path, session_id="abc").append_turn("q", "a")
        rows = sqlite3.connect(path).execute("SELECT DISTINCT session_id FROM turns").fetchall()
        assert rows == [("abc",)]


class TestClear:
    def test_clear_returns_count_and_empties(self, store):
        store.append_turn("a", "b")
        store.append_turn("c", "d")
        assert store.clear() == 4  # rows, not exchanges
        assert store.recent() == []

    def test_clear_empty_returns_zero(self, store):
        assert store.clear() == 0


class TestBestEffort:
    def test_append_on_unwritable_path_does_not_raise(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        store = HistoryStore(blocker / "history.db", session_id="s")
        store.append_turn("q", "a")  # must not raise
        assert store.recent() == []  # unreadable too -> empty, no raise

    def test_corrupt_db_reads_empty(self, tmp_path):
        corrupt = tmp_path / "history.db"
        corrupt.write_text("not sqlite at all")
        assert HistoryStore(corrupt, session_id="s").recent() == []

import sqlite3
from pathlib import Path

from platformdirs import user_state_dir

NOTE_EXTENSIONS = {".md", ".txt", ".markdown"}
DEFAULT_MAX_BYTES = 2_000_000

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(path UNINDEXED, body);
CREATE TABLE IF NOT EXISTS note_files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL
);
"""


def default_notes_index_path() -> Path:
    return Path(user_state_dir("nero")) / "notes_index.db"


def _escape_fts_query(query: str) -> str:
    """Quote the whole query as one FTS5 phrase, so a user's `-`, `"`, or any
    other operator character can't be parsed as query syntax or raise."""
    return '"' + query.replace('"', '""') + '"'


class NoteIndex:
    """FTS5 keyword search over a directory of the user's own notes.

    Reindex-on-demand only — no watcher, no background thread. A `note_files`
    side table (path, mtime, size) lets reindex() skip files that haven't
    changed since the last pass.
    """

    def __init__(self, index_path: Path | str, notes_dir: str | None, max_bytes: int = DEFAULT_MAX_BYTES):
        self.index_path = Path(index_path)
        self.notes_dir = notes_dir
        self.max_bytes = max_bytes

    def _connect(self) -> sqlite3.Connection:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        connection.executescript(_SCHEMA)
        return connection

    def reindex(self) -> tuple[int, int, int]:
        """Walk notes_dir, index new/changed files, drop rows for files that
        no longer exist. Returns (added, updated, removed)."""
        added = updated = removed = 0
        connection = self._connect()
        try:
            notes_dir = Path(self.notes_dir) if self.notes_dir else None
            if notes_dir and notes_dir.is_dir():
                for file_path in sorted(notes_dir.rglob("*")):
                    if not file_path.is_file() or file_path.suffix.lower() not in NOTE_EXTENSIONS:
                        continue
                    outcome = self._index_one(connection, file_path)
                    if outcome == "added":
                        added += 1
                    elif outcome == "updated":
                        updated += 1

            existing_paths = [
                row[0] for row in connection.execute("SELECT path FROM note_files").fetchall()
            ]
            for path_str in existing_paths:
                if not Path(path_str).is_file():
                    with connection:
                        connection.execute("DELETE FROM notes WHERE path = ?", (path_str,))
                        connection.execute("DELETE FROM note_files WHERE path = ?", (path_str,))
                    removed += 1
        finally:
            connection.close()
        return added, updated, removed

    def _index_one(self, connection: sqlite3.Connection, file_path: Path) -> str:
        """Index one candidate file. Returns "added", "updated", "unchanged",
        or "skipped" (unreadable, oversize, or binary)."""
        path_str = str(file_path)
        try:
            stat = file_path.stat()
        except OSError:
            return "skipped"
        if stat.st_size > self.max_bytes:
            return "skipped"
        existing = connection.execute(
            "SELECT mtime, size FROM note_files WHERE path = ?", (path_str,)
        ).fetchone()
        if existing and existing[0] == stat.st_mtime and existing[1] == stat.st_size:
            return "unchanged"  # mtime-skip: unchanged files aren't re-read
        try:
            raw = file_path.read_bytes()
        except OSError:
            return "skipped"
        if b"\x00" in raw[:8192]:
            return "skipped"
        body = raw.decode("utf-8", errors="replace")
        with connection:
            connection.execute("DELETE FROM notes WHERE path = ?", (path_str,))
            connection.execute("INSERT INTO notes (path, body) VALUES (?, ?)", (path_str, body))
            connection.execute(
                "INSERT INTO note_files (path, mtime, size) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size",
                (path_str, stat.st_mtime, stat.st_size),
            )
        return "updated" if existing else "added"

    def is_empty(self) -> bool:
        connection = self._connect()
        try:
            row = connection.execute("SELECT COUNT(*) FROM note_files").fetchone()
        finally:
            connection.close()
        return not row or row[0] == 0

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """FTS5 keyword search, best matches first: (path, snippet) pairs."""
        if not query.strip():
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT path, snippet(notes, 1, '', '', '...', 8) FROM notes "
                "WHERE notes MATCH ? ORDER BY rank LIMIT ?",
                (_escape_fts_query(query), limit),
            ).fetchall()
        finally:
            connection.close()
        return [(row[0], row[1]) for row in rows]

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_state_dir

logger = logging.getLogger("nero.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def default_history_path() -> Path:
    return Path(user_state_dir("nero")) / "history.db"


class HistoryStore:
    """Recent conversation, persisted across restarts (Phase 5b).

    Only user + final-assistant text turns are stored — never tool-call or
    tool-result messages — so any window restored from it is independently valid
    and can't start mid-tool-sequence. Connection-per-call, like AuditLog, since
    voice mode drives playback on a background thread.
    """

    def __init__(self, path: Path | str, session_id: str, max_turns: int = 20):
        self.path = Path(path)
        self.session_id = session_id
        self.max_turns = max_turns

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_SCHEMA)
        return connection

    def append_turn(self, user: str, assistant: str) -> None:
        """Persist one exchange as two rows. Best-effort: a broken history file
        must never take down the conversation it only observes."""
        now = datetime.now(UTC).isoformat()
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.executemany(
                        "INSERT INTO turns (session_id, role, content, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        [
                            (self.session_id, "user", user, now),
                            (self.session_id, "assistant", assistant, now),
                        ],
                    )
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not persist conversation turn: %s", exc)

    def recent(self, limit: int | None = None) -> list[dict]:
        """The last `limit` exchanges (default max_turns) as oldest-first
        {"role", "content"} dicts. Empty on any read failure."""
        exchanges = self.max_turns if limit is None else limit
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?",
                    (exchanges * 2,),
                ).fetchall()
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not read conversation history: %s", exc)
            return []
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def clear(self) -> int:
        """Delete all stored turns; returns the number of rows removed."""
        try:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute("DELETE FROM turns")
                    return cursor.rowcount
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not clear conversation history: %s", exc)
            return 0

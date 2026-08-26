import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_state_dir

logger = logging.getLogger("nero.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source TEXT,
    updated_at TEXT NOT NULL
)
"""

# Prompt-injection caps (spec: "the prompt cannot grow without bound").
MAX_PROMPT_FACTS = 40
MAX_PROMPT_CHARS = 2000


def default_facts_path() -> Path:
    return Path(user_state_dir("nero")) / "facts.db"


@dataclass
class Fact:
    key: str
    value: str
    source: str | None
    updated_at: str


class FactStore:
    """Structured user facts/preferences: explicit keys, never a free-text
    pile the model has to re-read. Connection-per-call, like AuditLog/
    HistoryStore — cheap for the call volume this sees and sidesteps sqlite's
    thread affinity."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_SCHEMA)
        return connection

    def remember(self, key: str, value: str, source: str | None = None) -> Fact:
        """Upsert. Returns the fact as stored."""
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO facts (key, value, source, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                    "source=excluded.source, updated_at=excluded.updated_at",
                    (key, value, source, now),
                )
        finally:
            connection.close()
        return Fact(key=key, value=value, source=source, updated_at=now)

    def recall(self, key: str) -> str | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row else None

    def all(self) -> list[Fact]:
        """Every fact, oldest-updated first — the order the prompt-cap logic
        relies on to drop the oldest facts first."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT key, value, source, updated_at FROM facts ORDER BY updated_at ASC"
            ).fetchall()
        finally:
            connection.close()
        return [Fact(key=r[0], value=r[1], source=r[2], updated_at=r[3]) for r in rows]

    def forget(self, key: str) -> bool:
        """Delete one fact. Returns whether a row was actually removed."""
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute("DELETE FROM facts WHERE key = ?", (key,))
                return cursor.rowcount > 0
        finally:
            connection.close()

    def search(self, text: str) -> list[Fact]:
        """LIKE over key and value — the fact set is small; FTS5 is reserved
        for notes, not this."""
        pattern = f"%{text}%"
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT key, value, source, updated_at FROM facts "
                "WHERE key LIKE ? OR value LIKE ? ORDER BY updated_at ASC",
                (pattern, pattern),
            ).fetchall()
        finally:
            connection.close()
        return [Fact(key=r[0], value=r[1], source=r[2], updated_at=r[3]) for r in rows]


def facts_prompt_block(facts: list[tuple[str, str]]) -> str:
    """The system-prompt appendix for known facts, or "" for none.

    Caps at MAX_PROMPT_FACTS entries and MAX_PROMPT_CHARS characters, dropping
    the oldest first — callers pass facts oldest-first (FactStore.all()'s
    order), so dropping from the front of the list IS dropping the oldest.
    """
    remaining = list(facts)[-MAX_PROMPT_FACTS:]
    while remaining:
        lines = "\n".join(f"- {key}: {value}" for key, value in remaining)
        block = f"\n\nWhat you know about this user (from earlier sessions):\n{lines}"
        if len(block) <= MAX_PROMPT_CHARS:
            return block
        remaining = remaining[1:]  # drop the oldest, retry
    return ""

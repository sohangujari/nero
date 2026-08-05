import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from platformdirs import user_state_dir
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("nero.audit")

SUMMARY_LIMIT = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result_summary TEXT NOT NULL,
    provider TEXT NOT NULL
)
"""


class AuditEntry(BaseModel):
    """One skill invocation, successful or refused."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    skill_name: str
    arguments: dict
    result_summary: str
    provider: str


def default_audit_path() -> Path:
    return Path(user_state_dir("nero")) / "audit.db"


def summarize(result: str, limit: int = SUMMARY_LIMIT) -> str:
    """Collapse a skill result to one tidy line for the history table."""
    text = " ".join((result or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class AuditLog:
    """Append-only SQLite record of what Nero has actually done.

    A connection is opened per call rather than held: voice mode drives playback
    on a background thread while asyncio owns the main one, and per-call connect
    sidesteps sqlite's thread affinity entirely for a write that happens a few
    times a minute.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_SCHEMA)
        return connection

    def record(self, entry: AuditEntry) -> bool:
        """Persist one entry. Returns whether it was written.

        Never raises: an unwritable or corrupt log must not take down the skill
        call it was only supposed to observe.
        """
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "INSERT INTO invocations "
                        "(timestamp, skill_name, arguments, result_summary, provider) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            entry.timestamp.isoformat(),
                            entry.skill_name,
                            json.dumps(entry.arguments, ensure_ascii=False),
                            entry.result_summary,
                            entry.provider,
                        ),
                    )
            finally:
                connection.close()
            return True
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            # TypeError matters: json.dumps raises it (not ValueError) on a value
            # it can't encode, and "never raises" has to mean never.
            logger.warning("Could not write audit entry: %s", exc)
            return False

    def recent(self, limit: int = 20) -> list[AuditEntry]:
        """The most recent entries, newest first. Empty on any read failure."""
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT timestamp, skill_name, arguments, result_summary, provider "
                    "FROM invocations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not read audit log: %s", exc)
            return []
        # Skip only the unreadable row rather than discarding the whole log:
        # one bad timestamp (e.g. from manual DB tampering or a future schema
        # change) shouldn't hide every other entry `nero history` would show.
        entries = []
        for row in rows:
            try:
                timestamp = datetime.fromisoformat(row[0])
            except ValueError:
                logger.warning("Skipping audit row with unreadable timestamp: %r", row[0])
                continue
            entries.append(
                AuditEntry(
                    timestamp=timestamp,
                    skill_name=row[1],
                    arguments=_loads(row[2]),
                    result_summary=row[3],
                    provider=row[4],
                )
            )
        return entries


def _loads(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

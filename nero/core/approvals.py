"""The human-in-the-loop approval queue.

A routine runs headlessly (`nero routine run <name>`, launched by launchd with
no one watching). Nothing destructive may run unattended, so a destructive
skill call made during a routine is refused immediately — the same fail-closed
rule every headless/no-confirm caller gets — and recorded here for a human to
approve later, at `nero approvals run <id>`.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_state_dir
from pydantic import BaseModel, ConfigDict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    routine TEXT NOT NULL,
    skill TEXT NOT NULL,
    arguments TEXT NOT NULL,
    requested_at TEXT NOT NULL
)
"""


class PendingApproval(BaseModel):
    """One destructive skill call a routine wasn't allowed to run unattended."""

    model_config = ConfigDict(extra="forbid")

    id: int
    routine: str
    skill: str
    arguments: dict
    requested_at: datetime


def default_approvals_path() -> Path:
    return Path(user_state_dir("nero")) / "approvals.db"


class ApprovalQueue:
    """SQLite-backed queue of destructive calls awaiting human approval.

    Connection-per-call, like AuditLog: this is written from a headless
    `routine run` invocation and read from an interactive session, never both
    at once from the same process, so there's no reason to hold one open.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_SCHEMA)
        return connection

    def record(self, routine: str, skill: str, arguments: dict) -> int:
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO pending (routine, skill, arguments, requested_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        routine,
                        skill,
                        json.dumps(arguments, ensure_ascii=False),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                return cursor.lastrowid
        finally:
            connection.close()

    def pending(self) -> list[PendingApproval]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT id, routine, skill, arguments, requested_at "
                "FROM pending ORDER BY id"
            ).fetchall()
        finally:
            connection.close()
        return [_row_to_model(row) for row in rows]

    def get(self, entry_id: int) -> PendingApproval | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, routine, skill, arguments, requested_at "
                "FROM pending WHERE id = ?",
                (entry_id,),
            ).fetchone()
        finally:
            connection.close()
        return _row_to_model(row) if row else None

    def discard(self, entry_id: int) -> bool:
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute("DELETE FROM pending WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
        finally:
            connection.close()

    def clear(self) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM pending")
        finally:
            connection.close()


def _row_to_model(row) -> PendingApproval:
    return PendingApproval(
        id=row[0],
        routine=row[1],
        skill=row[2],
        arguments=json.loads(row[3]) if row[3] else {},
        requested_at=datetime.fromisoformat(row[4]),
    )


def queue_confirm(queue: ApprovalQueue, routine: str):
    """Build the headless confirm callback used by `nero routine run`.

    Records the request for later human review and returns False — the call
    is refused right now, exactly as if no confirm callback existed at all
    (fail-closed is preserved, not bypassed). The model sees the normal
    refusal string and the routine continues sensibly.
    """

    def confirm(name: str, tier: str, arguments: dict) -> bool:
        queue.record(routine, name, arguments)
        return False

    return confirm

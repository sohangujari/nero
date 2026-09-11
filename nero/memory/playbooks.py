"""What Nero has learned about doing a recurring job, and how it gets back.

A playbook is a named, versioned procedure: when it applies, the steps that
worked, and the things that did not. Nero writes them by reviewing its own
audit log (`nero/learn.py`), and reads the relevant one back into the prompt
when a matching task comes round again.

## A playbook is read, never run

This is the whole safety argument, and it is worth stating plainly. A playbook
is text the model is shown, exactly like a recalled conversation. It is not
code and nothing here executes anything. When a step says
`uv run pytest -q`, running it still means the model calling `run_shell`,
which is off by default, gated by the confirm callback, checked against the
denylist, and written to the audit log. Learning changes what Nero *knows*,
never what it is *allowed to do*.

That also means a wrong playbook is a wrong suggestion rather than a wrong
command, which is the failure mode worth having: `nero playbooks edit` and
`nero playbooks forget` fix it, and every earlier version is still there.

## Where it goes in the prompt

Onto the user's message, not the system prompt — the same place recall goes
and for the same measured reason. Anything at the front of the prompt
invalidates a provider's cache of everything after it, and a store that
changes as Nero learns is the last thing that should live there. See
`nero/llm/client.py:current_time_line` for what that cost when a clock did it.

## Versioning

`playbooks` holds the current version of each; `playbook_revisions` holds every
version that came before, including the one being replaced. Nothing is ever
overwritten in place, so a review that makes a playbook worse is one
`nero playbooks restore` away from being undone.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_state_dir

from nero.memory.recall import query_terms

logger = logging.getLogger("nero.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS playbooks (
    name TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    steps TEXT NOT NULL,
    avoid TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    uses INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE TABLE IF NOT EXISTS playbook_revisions (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    task TEXT NOT NULL,
    steps TEXT NOT NULL,
    avoid TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    saved_at TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
"""

# One playbook per turn. Two competing procedures in a prompt is how a small
# model ends up following half of each, and the retrieval score is a ranking,
# not a certainty — so the runner-up is far more often noise than a second
# useful answer.
MAX_IN_PROMPT = 1
# Hard ceiling on what a playbook can add to a turn. A procedure that has grown
# past this has stopped being a procedure.
MAX_BLOCK_CHARS = 1200
# Term overlap a playbook needs before it is considered a match at all. One
# shared word between "deploy the docs" and "deploy" is a coincidence often
# enough that acting on it is worse than not matching.
MIN_TERM_OVERLAP = 2


def default_playbook_path() -> Path:
    return Path(user_state_dir("nero")) / "playbooks.db"


@dataclass(frozen=True)
class Playbook:
    name: str
    task: str
    steps: str
    avoid: str = ""
    version: int = 1
    uses: int = 0
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None

    def render(self) -> str:
        """The playbook as the model sees it."""
        lines = [f"How to {self.task.strip()}:", self.steps.strip()]
        if self.avoid.strip():
            lines.append("Avoid:\n" + self.avoid.strip())
        return "\n".join(lines)

    @property
    def terms(self) -> list[str]:
        """The words this playbook is matched on. The name and task describe
        *when* it applies; the steps describe how, and matching on those too
        would retrieve a git playbook for any mention of a commit."""
        return query_terms(f"{self.name.replace('-', ' ')} {self.task}")


@dataclass(frozen=True)
class Revision:
    name: str
    version: int
    task: str
    steps: str
    avoid: str
    note: str
    saved_at: str


class PlaybookStore:
    """Procedures Nero has learned, with their history.

    Connection-per-call, like FactStore and AuditLog: cheap at this volume, and
    it sidesteps sqlite's thread affinity, which matters because the chat
    bridges read this from their own threads.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or default_playbook_path())

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.executescript(_SCHEMA)
        return connection

    def save(self, name: str, task: str, steps: str, avoid: str = "", note: str = "") -> Playbook:
        """Write a playbook, keeping whatever was there as a revision.

        Returns the stored version. A name that already exists is *revised*,
        not replaced: the previous text is copied into `playbook_revisions`
        first, so `restore` can always get it back.
        """
        name = _slug(name)
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    "SELECT version, task, steps, avoid, created_at, uses "
                    "FROM playbooks WHERE name = ?",
                    (name,),
                ).fetchone()
                if row is None:
                    version, created_at, uses = 1, now, 0
                else:
                    version, created_at, uses = row[0] + 1, row[4], row[5]
                    connection.execute(
                        "INSERT OR REPLACE INTO playbook_revisions "
                        "(name, version, task, steps, avoid, note, saved_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (name, row[0], row[1], row[2], row[3], note, now),
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO playbooks "
                    "(name, task, steps, avoid, version, uses, created_at, updated_at, "
                    " last_used_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    "        (SELECT last_used_at FROM playbooks WHERE name = ?))",
                    (name, task, steps, avoid, version, uses, created_at, now, name),
                )
        finally:
            connection.close()
        return Playbook(
            name=name, task=task, steps=steps, avoid=avoid, version=version,
            uses=uses, created_at=created_at, updated_at=now,
        )

    def get(self, name: str) -> Playbook | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT name, task, steps, avoid, version, uses, created_at, updated_at, "
                "last_used_at FROM playbooks WHERE name = ?",
                (_slug(name),),
            ).fetchone()
        finally:
            connection.close()
        return _playbook(row) if row else None

    def all(self) -> list[Playbook]:
        """Every playbook, most recently updated first."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT name, task, steps, avoid, version, uses, created_at, updated_at, "
                "last_used_at FROM playbooks ORDER BY updated_at DESC"
            ).fetchall()
        finally:
            connection.close()
        return [_playbook(row) for row in rows]

    def history(self, name: str) -> list[Revision]:
        """Every earlier version, newest first. The current one is not here —
        it is in `playbooks` until something replaces it."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT name, version, task, steps, avoid, note, saved_at "
                "FROM playbook_revisions WHERE name = ? ORDER BY version DESC",
                (_slug(name),),
            ).fetchall()
        finally:
            connection.close()
        return [Revision(*row) for row in rows]

    def restore(self, name: str, version: int) -> Playbook | None:
        """Bring an earlier version back as a new one.

        Deliberately additive: restoring v2 over v5 writes v6 with v2's text
        and leaves v5 in the history, so an accidental restore is itself
        undoable. Nothing in this module ever loses a version.
        """
        revisions = {revision.version: revision for revision in self.history(name)}
        wanted = revisions.get(version)
        if wanted is None:
            return None
        return self.save(
            name, wanted.task, wanted.steps, wanted.avoid,
            note=f"restored from v{version}",
        )

    def forget(self, name: str, keep_history: bool = False) -> bool:
        """Delete a playbook. Returns whether one was actually removed.

        History goes too unless `keep_history` — a playbook the user explicitly
        deleted should not keep answering `nero playbooks history`.
        """
        name = _slug(name)
        connection = self._connect()
        try:
            with connection:
                removed = connection.execute(
                    "DELETE FROM playbooks WHERE name = ?", (name,)
                ).rowcount
                if not keep_history:
                    connection.execute(
                        "DELETE FROM playbook_revisions WHERE name = ?", (name,)
                    )
        finally:
            connection.close()
        return removed > 0

    def mark_used(self, name: str) -> None:
        """Count a retrieval. Never raises: this is bookkeeping on a turn that
        has already been decided, and must not be able to fail it."""
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "UPDATE playbooks SET uses = uses + 1, last_used_at = ? WHERE name = ?",
                        (datetime.now(UTC).isoformat(), _slug(name)),
                    )
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.debug("could not record playbook use: %s", exc)

    def match(self, text: str, limit: int = MAX_IN_PROMPT) -> list[Playbook]:
        """The playbooks worth showing for `text`, best first.

        Scored on shared words rather than embeddings: there are tens of these,
        not thousands, and a second model call on the critical path is exactly
        the cost `nero/memory/recall.py` exists to avoid.
        """
        wanted = set(query_terms(text))
        if not wanted:
            return []
        scored = []
        for playbook in self.all():
            terms = set(playbook.terms)
            if not terms:
                continue
            shared = len(wanted & terms)
            # A short task description can be fully matched by two words, so
            # the floor is "most of it, or at least MIN_TERM_OVERLAP words".
            if shared < min(MIN_TERM_OVERLAP, len(terms)):
                continue
            scored.append((shared / len(terms), shared, playbook))
        scored.sort(key=lambda row: (-row[0], -row[1], row[2].name))
        return [playbook for _ratio, _shared, playbook in scored[:limit]]


def _playbook(row) -> Playbook:
    return Playbook(
        name=row[0], task=row[1], steps=row[2], avoid=row[3], version=row[4],
        uses=row[5], created_at=row[6], updated_at=row[7], last_used_at=row[8],
    )


def _slug(name: str) -> str:
    """Playbook names are matched, listed and typed at a shell, so they are
    normalised once here rather than at every call site."""
    return "-".join(str(name).strip().lower().split())


def playbook_block(store, text: str) -> str:
    """The matching playbook, formatted to prefix the user's message — or "".

    Tagged rather than headed, and for the reason recorded in
    `nero/memory/recall.py`: a bare procedure above a question reads to a small
    model like something the user pasted, and gets answered instead of
    followed. The tag pairs with the system prompt, which says what it is.

    Never raises. A store that cannot be read costs the turn a hint, not the
    turn.
    """
    if store is None:
        return ""
    try:
        matches = store.match(text)
    except Exception:  # noqa: BLE001 — see docstring
        logger.debug("playbook lookup failed", exc_info=True)
        return ""
    if not matches:
        return ""
    rendered = []
    budget = MAX_BLOCK_CHARS
    for playbook in matches:
        body = playbook.render()
        if len(body) > budget:
            break
        budget -= len(body)
        rendered.append(body)
        store.mark_used(playbook.name)
    if not rendered:
        return ""
    return "<playbook>\n" + "\n\n".join(rendered) + "\n</playbook>\n\n"

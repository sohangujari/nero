import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_state_dir

logger = logging.getLogger("nero.memory")

# `porter` stems on both sides of the index, so "running" finds "run" and
# "colours" finds "colour". Free, and it is what the 86%-recall keyword
# baseline in the literature is actually measured with.
_TOKENIZER = "porter unicode61"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Keyword index. An external-content FTS5 table stores no copy of the text --
-- it points back at `turns` -- so the index costs a fraction of the transcript
-- and can never drift out of sync with it. Rows are only ever inserted or
-- deleted wholesale, so two triggers cover every write.
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
    USING fts5(content, content='turns', content_rowid='id', tokenize='{_TOKENIZER}');
CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts (rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS turns_fts_delete AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts (turns_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

-- Semantic index. Separate table rather than a column on `turns`, so a store
-- running without an embedder never carries the weight, and adding one later
-- needs no migration of the transcript itself.
CREATE TABLE IF NOT EXISTS turn_vectors (
    turn_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL
);
"""

# An exchange is keyed by its timestamp: append_turn writes both of its rows
# with one `created_at`, and that is what pairs a question with its answer.
_ROWS_FOR_KEYS = """
SELECT created_at, role, content FROM turns
WHERE created_at IN ({placeholders}) ORDER BY id
"""


def default_history_path() -> Path:
    return Path(user_state_dir("nero")) / "history.db"


class HistoryStore:
    """Recent conversation, persisted across restarts, and searchable.

    Only user + final-assistant text turns are stored — never tool-call or
    tool-result messages — so any window restored from it is independently
    valid and can't start mid-tool-sequence. Connection-per-call, like
    AuditLog, since voice mode drives playback on a background thread.

    Search comes in two flavours that `nero/memory/recall.py` fuses: keyword
    (always) and semantic (whenever `embedder` is available).
    """

    def __init__(self, path: Path | str, session_id: str, max_turns: int = 20, embedder=None):
        self.path = Path(path)
        self.session_id = session_id
        self.max_turns = max_turns
        self.embedder = embedder
        # (highest turn id seen, ids, matrix). Vectors are immutable once
        # written, so a rising id is the only thing that can invalidate this.
        self._vectors = None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        existing = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'turns_fts'"
        ).fetchone()
        if existing and _TOKENIZER not in (existing[0] or ""):
            # An index built by an older version, with a different tokenizer.
            # Rebuilding is cheap and beats silently searching with the wrong one.
            connection.executescript(
                "DROP TRIGGER IF EXISTS turns_fts_insert;"
                "DROP TRIGGER IF EXISTS turns_fts_delete;"
                "DROP TABLE IF EXISTS turns_fts;"
            )
            existing = None
        connection.executescript(_SCHEMA)
        if not existing:
            with connection:
                connection.execute("INSERT INTO turns_fts (turns_fts) VALUES ('rebuild')")
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
                # After the write, and after the reply is already on screen —
                # the only place in the turn where a 100 ms cost is invisible.
                self._embed_pending(connection)
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
        self._vectors = None
        try:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute("DELETE FROM turns")
                    connection.execute("DELETE FROM turn_vectors")
                    # The delete trigger fires per row; this only matters if a
                    # crash ever left the index ahead of the table.
                    connection.execute("INSERT INTO turns_fts (turns_fts) VALUES ('rebuild')")
                    return cursor.rowcount
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not clear conversation history: %s", exc)
            return 0

    # --- search -----------------------------------------------------------

    def search_keys(self, query: str, limit: int = 5) -> list[str]:
        """Exchange keys matching the FTS5 `query`, best match first. Empty on
        any failure or on a query FTS5 rejects — recall is an optimisation,
        never a turn-breaker."""
        if not query.strip():
            return []
        try:
            connection = self._connect()
            try:
                ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT rowid FROM turns_fts WHERE turns_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (query, limit * 2),
                    )
                ]
                return self._keys_for(connection, ids)[:limit]
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not search conversation history: %s", exc)
            return []

    def search_semantic_keys(self, text: str, limit: int = 5) -> list[str]:
        """Exchange keys whose meaning is close to `text`, best first.

        Empty when there is no embedder, no vectors yet, anything goes wrong —
        or, importantly, when nothing clears SIMILARITY_FLOOR. "Nothing is
        relevant" is a real answer, and the common one."""
        if self.embedder is None or not self.embedder.available:
            return []
        try:
            import numpy as np

            from nero.memory.embeddings import SIMILARITY_FLOOR, unpack_all

            connection = self._connect()
            try:
                loaded = self._vector_matrix(connection)
                if loaded is None:
                    return []
                ids, matrix = loaded
                packed = self.embedder.embed([text])
                if not packed:
                    return []
                query = unpack_all(packed)[0]
                # Both sides unit-normalised, so the dot product is cosine.
                scores = matrix @ (query / (np.linalg.norm(query) or 1.0))
                best = np.argsort(-scores)[: limit * 2]
                # The floor is what lets this stream return nothing. Without
                # it, argsort always has a "best" match and an unrelated
                # question still drags old exchanges into the prompt.
                ranked = [int(ids[i]) for i in best if scores[i] >= SIMILARITY_FLOOR]
                return self._keys_for(connection, ranked)[:limit]
            finally:
                connection.close()
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.debug("semantic search failed: %s", exc, exc_info=True)
            return []

    def exchanges(self, keys: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
        """(key, rows) for each key, in the order given. A stored answer
        without its question is not usable context, so both rows of each come
        back together. The key rides along so a caller can still tell which
        search produced a hit after fusion has mixed them."""
        if not keys:
            return []
        try:
            connection = self._connect()
            try:
                sql = _ROWS_FOR_KEYS.format(placeholders=",".join("?" * len(keys)))
                grouped: dict[str, list[tuple[str, str]]] = {}
                for key, role, content in connection.execute(sql, keys):
                    grouped.setdefault(key, []).append((role, content))
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not read conversation history: %s", exc)
            return []
        return [(key, grouped[key]) for key in keys if key in grouped]

    # --- internals --------------------------------------------------------

    @staticmethod
    def _keys_for(connection, ids: list[int]) -> list[str]:
        """Row ids to exchange keys, keeping the ids' ranking and dropping the
        duplicate that appears when both halves of one exchange matched."""
        if not ids:
            return []
        by_id = dict(
            connection.execute(
                f"SELECT id, created_at FROM turns WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
        )
        # dict preserves insertion order, so this dedupes without reordering.
        return list(dict.fromkeys(by_id[i] for i in ids if i in by_id))

    def _vector_matrix(self, connection):
        """(ids, unit-normalised matrix) for every embedded turn, cached until
        a new turn is written."""
        import numpy as np

        from nero.memory.embeddings import unpack_all

        highest = connection.execute("SELECT MAX(turn_id) FROM turn_vectors").fetchone()[0]
        if highest is None:
            return None
        if self._vectors is not None and self._vectors[0] == highest:
            return self._vectors[1], self._vectors[2]
        rows = connection.execute(
            "SELECT turn_id, vector FROM turn_vectors ORDER BY turn_id"
        ).fetchall()
        ids = np.array([row[0] for row in rows])
        matrix = unpack_all([row[1] for row in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0, 1.0, norms)
        self._vectors = (highest, ids, matrix)
        return ids, matrix

    def _embed_pending(self, connection, limit: int = 32) -> int:
        """Embed turns that have no vector yet, newest first, up to `limit`.

        Bounded on purpose: this also backfills a transcript that predates the
        embedder, a batch per turn, instead of stalling one startup on it.

        Never raises. It runs inside append_turn, which guards only sqlite and
        OS errors — an embedder that throws must not cost the user their turn.
        """
        if self.embedder is None or not self.embedder.available:
            return 0
        try:
            return self._embed_batch(connection, limit)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.debug("embedding backfill failed: %s", exc, exc_info=True)
            return 0

    def _embed_batch(self, connection, limit: int) -> int:
        rows = connection.execute(
            "SELECT t.id, t.content FROM turns t "
            "LEFT JOIN turn_vectors v ON v.turn_id = t.id "
            "WHERE v.turn_id IS NULL ORDER BY t.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return 0
        packed = self.embedder.embed([content for _id, content in rows])
        if packed is None:
            return 0
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO turn_vectors (turn_id, vector) VALUES (?, ?)",
                [(row[0], blob) for row, blob in zip(rows, packed, strict=True)],
            )
        self._vectors = None
        return len(rows)

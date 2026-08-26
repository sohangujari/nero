"""search_notes: FTS5 keyword search over the user's own notes directory
(nero/memory/notes.py).

Reindexes on demand: if the index is empty, reindex() runs once so first use
works without ceremony (spec §2 — no watcher, no background thread).
"""

from nero.memory.notes import NoteIndex
from nero.security import envelope
from nero.skills.base import Skill, SkillMeta

DEFAULT_LIMIT = 5


class SearchNotesSkill(Skill):
    meta = SkillMeta(
        name="search_notes",
        description=(
            "Search the user's own notes (their configured notes directory) for "
            "a keyword or phrase. Use this when the user asks you to find, look "
            "up, or recall something from their notes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for."},
                "limit": {"type": "integer", "description": f"Max results (default {DEFAULT_LIMIT})."},
            },
            "required": ["query"],
        },
        requires_network=False,
        permission_tier="read_only",
        # Note files are the user's own content, but still external text the
        # model shouldn't treat as instructions.
        ingests_external_content=True,
    )

    def __init__(self, index: NoteIndex | None):
        self._index = index

    async def execute(self, **kwargs) -> str:
        query = str(kwargs.get("query") or "")
        limit = int(kwargs.get("limit") or DEFAULT_LIMIT)
        if self._index is None:
            return (
                "No notes directory is configured. Tell the user they can set one "
                "with `nero config set memory.notes_dir <path>`."
            )
        if self._index.is_empty():
            self._index.reindex()
        results = self._index.search(query, limit=limit)
        if not results:
            return f"No notes match {query!r}."
        text = "\n\n".join(f"{path}:\n{snippet}" for path, snippet in results)
        return envelope(f"notes:{query}", text)

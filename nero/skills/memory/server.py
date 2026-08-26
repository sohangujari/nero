"""Structured-fact skills: remember_fact, recall_facts, forget_fact.

Backed by one shared FactStore instance (nero/memory/facts.py) — explicit
keys, never a free-text pile the model has to re-read.
"""

from nero.memory.facts import FactStore
from nero.skills.base import Skill, SkillMeta


class RememberFactSkill(Skill):
    meta = SkillMeta(
        name="remember_fact",
        description=(
            "Store one fact or preference about the user under an explicit key, "
            "for recall in future sessions. Use this when the user tells you "
            "something worth remembering long-term (a preference, a detail about "
            "them). Overwrites any existing value for the same key."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier, e.g. 'favorite_editor'."},
                "value": {"type": "string", "description": "The fact to remember."},
            },
            "required": ["key", "value"],
        },
        requires_network=False,
        permission_tier="state_changing",
    )

    def __init__(self, store: FactStore):
        self._store = store

    async def execute(self, **kwargs) -> str:
        key = str(kwargs.get("key") or "")
        value = str(kwargs.get("value") or "")
        fact = self._store.remember(key, value, source="skill")
        return f"Remembered {fact.key}: {fact.value}"


class RecallFactsSkill(Skill):
    meta = SkillMeta(
        name="recall_facts",
        description=(
            "Look up facts previously remembered about the user. Pass `query` "
            "to search, or omit it to list everything remembered."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional search text."},
            },
            "required": [],
        },
        requires_network=False,
        permission_tier="read_only",
        # This is the user's own store, not external content — nothing here
        # should be treated as untrusted.
        ingests_external_content=False,
    )

    def __init__(self, store: FactStore):
        self._store = store

    async def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        facts = self._store.search(str(query)) if query else self._store.all()
        if not facts:
            return "No facts remembered yet." if not query else f"No facts match {query!r}."
        return "\n".join(f"- {fact.key}: {fact.value}" for fact in facts)


class ForgetFactSkill(Skill):
    meta = SkillMeta(
        name="forget_fact",
        description="Delete a previously remembered fact by its key.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to forget."},
            },
            "required": ["key"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    def __init__(self, store: FactStore):
        self._store = store

    async def execute(self, **kwargs) -> str:
        key = str(kwargs.get("key") or "")
        removed = self._store.forget(key)
        return f"Forgot {key}." if removed else f"No fact stored under {key!r}."

"""Structured facts: FactStore (nero/memory/facts.py), the prompt-block cap,
and the three skills (nero/skills/memory/server.py).

Every store operates only inside pytest's tmp_path.
"""

import asyncio

from nero.memory.facts import FactStore, facts_prompt_block
from nero.skills.memory.server import ForgetFactSkill, RecallFactsSkill, RememberFactSkill


def run(coro):
    return asyncio.run(coro)


class TestFactStore:
    def test_creates_db_lazily(self, tmp_path):
        path = tmp_path / "sub" / "facts.db"
        assert not path.exists()
        FactStore(path).remember("k", "v")
        assert path.exists()

    def test_remember_then_recall(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("favorite_color", "blue")
        assert store.recall("favorite_color") == "blue"

    def test_recall_missing_key_is_none(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        assert store.recall("nope") is None

    def test_upsert_overwrites(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("k", "first")
        store.remember("k", "second")
        assert store.recall("k") == "second"
        assert len(store.all()) == 1

    def test_forget_removes_and_reports(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("k", "v")
        assert store.forget("k") is True
        assert store.recall("k") is None

    def test_forget_missing_key_returns_false(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        assert store.forget("nope") is False

    def test_search_matches_key_or_value(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("favorite_editor", "vim")
        store.remember("favorite_color", "blue")
        store.remember("os", "macos")
        assert {f.key for f in store.search("favorite")} == {"favorite_editor", "favorite_color"}
        assert {f.key for f in store.search("vim")} == {"favorite_editor"}
        assert store.search("nothingmatches") == []

    def test_all_ordered_oldest_updated_first(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("first", "1")
        store.remember("second", "2")
        store.remember("first", "1-updated")  # bump first's updated_at to newest
        assert [f.key for f in store.all()] == ["second", "first"]


class TestFactsPromptBlock:
    def test_empty_is_empty_string(self):
        assert facts_prompt_block([]) == ""

    def test_basic_block_shape(self):
        block = facts_prompt_block([("k1", "v1"), ("k2", "v2")])
        assert block == (
            "\n\nWhat you know about this user (from earlier sessions):\n"
            "- k1: v1\n- k2: v2"
        )

    def test_caps_at_40_facts_dropping_oldest(self):
        facts = [(f"k{i}", "v") for i in range(50)]
        block = facts_prompt_block(facts)
        assert "k0:" not in block
        assert "k9:" not in block
        assert "k49: v" in block
        assert block.count("\n- ") == 40

    def test_caps_at_2000_chars_dropping_oldest(self):
        facts = [(f"k{i}", "x" * 100) for i in range(30)]
        block = facts_prompt_block(facts)
        assert len(block) <= 2000
        # newest survives, oldest was dropped to make room
        assert "k29:" in block
        assert "k0:" not in block


class TestRememberFactSkill:
    def test_tier(self):
        assert RememberFactSkill.meta.permission_tier == "state_changing"

    def test_stores_and_reports_what_it_stored(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        result = run(RememberFactSkill(store).execute(key="k", value="v"))
        assert "k" in result and "v" in result
        assert store.recall("k") == "v"

    def test_source_is_skill(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        run(RememberFactSkill(store).execute(key="k", value="v"))
        assert store.all()[0].source == "skill"


class TestRecallFactsSkill:
    def test_tier_and_not_external(self):
        assert RecallFactsSkill.meta.permission_tier == "read_only"
        assert RecallFactsSkill.meta.ingests_external_content is False

    def test_lists_all_when_no_query(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("a", "1")
        store.remember("b", "2")
        result = run(RecallFactsSkill(store).execute())
        assert "a: 1" in result and "b: 2" in result

    def test_filters_by_query(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("favorite_editor", "vim")
        store.remember("os", "macos")
        result = run(RecallFactsSkill(store).execute(query="favorite"))
        assert "vim" in result and "macos" not in result

    def test_empty_store_says_so(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        assert "No facts" in run(RecallFactsSkill(store).execute())


class TestForgetFactSkill:
    def test_tier(self):
        assert ForgetFactSkill.meta.permission_tier == "destructive"

    def test_removes_existing_key(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        store.remember("k", "v")
        result = run(ForgetFactSkill(store).execute(key="k"))
        assert "Forgot" in result
        assert store.recall("k") is None

    def test_missing_key_reports_no_fact(self, tmp_path):
        store = FactStore(tmp_path / "facts.db")
        result = run(ForgetFactSkill(store).execute(key="nope"))
        assert "No fact" in result

    def test_refused_via_registry_with_no_confirm_callback(self, tmp_path):
        from nero.skills.registry import SkillRegistry

        store = FactStore(tmp_path / "facts.db")
        store.remember("k", "v")
        registry = SkillRegistry([ForgetFactSkill(store)])
        result = run(registry.execute("forget_fact", {"key": "k"}))
        assert "declined" in result
        assert store.recall("k") == "v"  # untouched

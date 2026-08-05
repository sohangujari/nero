import asyncio

from nero.skills.base import Skill, SkillMeta
from nero.skills.registry import SkillRegistry


def make_skill(name, requires_network=False, result="ok", error=None):
    class Stub(Skill):
        meta = SkillMeta(
            name=name,
            description=f"{name} description",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            requires_network=requires_network,
            permission_tier="read_only",
            offline_message=(
                f"{name} needs an internet connection, and you're in offline mode right now."
                if requires_network
                else None
            ),
        )

        async def execute(self, **kwargs):
            if error:
                raise error
            return result

    return Stub()


def run(registry, name, arguments, provider="claude"):
    return asyncio.run(registry.execute(name, arguments, provider))


class TestAvailability:
    def test_all_available_by_default(self):
        registry = SkillRegistry([make_skill("a"), make_skill("b")])
        assert {s.meta.name for s in registry.available()} == {"a", "b"}

    def test_disabled_skill_is_not_available(self):
        registry = SkillRegistry([make_skill("a"), make_skill("b")], enabled={"b": False})
        assert [s.meta.name for s in registry.available()] == ["a"]

    def test_offline_hides_network_skills_only(self):
        registry = SkillRegistry(
            [make_skill("local"), make_skill("remote", requires_network=True)],
            mode="offline",
        )
        assert [s.meta.name for s in registry.available()] == ["local"]

    def test_online_keeps_network_skills(self):
        registry = SkillRegistry(
            [make_skill("remote", requires_network=True)], mode="online"
        )
        assert [s.meta.name for s in registry.available()] == ["remote"]

    def test_known_names_includes_unavailable_skills(self):
        registry = SkillRegistry(
            [make_skill("a"), make_skill("b"), make_skill("c", requires_network=True)],
            enabled={"b": False},
            mode="offline",
        )
        assert registry.known_names() == {"a", "b", "c"}
        assert [s.meta.name for s in registry.available()] == ["a"]


class TestToolDefinitions:
    def test_openai_function_shape_for_available_only(self):
        registry = SkillRegistry([make_skill("a"), make_skill("b")], enabled={"b": False})
        definitions = registry.tool_definitions()
        assert definitions == [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "description": "a description",
                    "parameters": registry.get("a").meta.input_schema,
                },
            }
        ]


class TestExecuteGates:
    def test_unknown_skill(self):
        registry = SkillRegistry([make_skill("a")])
        result = run(registry, "format_disk", {})
        assert "Error" in result and "format_disk" in result

    def test_disabled_skill_is_refused_not_executed(self):
        registry = SkillRegistry([make_skill("a", result="RAN")], enabled={"a": False})
        result = run(registry, "a", {"q": "x"})
        assert "RAN" not in result
        assert "turned off" in result
        assert "nero config" in result

    def test_offline_refusal_uses_the_skill_message(self):
        registry = SkillRegistry(
            [make_skill("weather", requires_network=True, result="RAN")], mode="offline"
        )
        result = run(registry, "weather", {"q": "x"})
        assert "RAN" not in result
        assert "weather needs an internet connection" in result.lower()
        assert "offline mode" in result.lower()

    def test_offline_does_not_block_local_skills(self):
        registry = SkillRegistry([make_skill("a", result="RAN")], mode="offline")
        assert run(registry, "a", {"q": "x"}) == "RAN"

    def test_unparseable_arguments(self):
        registry = SkillRegistry([make_skill("a")])
        result = run(registry, "a", None)
        assert "Error" in result and "JSON" in result

    def test_skill_exception_becomes_error_string(self):
        registry = SkillRegistry([make_skill("a", error=RuntimeError("boom"))])
        result = run(registry, "a", {"q": "x"})
        assert "Error" in result and "boom" in result

    def test_disabled_and_offline_reports_disabled_not_offline(self):
        # Pins the gate ordering (unknown -> disabled -> offline): a skill that
        # is BOTH disabled and blocked by offline mode must fail on the first
        # gate it hits, not the second. A refactor that swapped the checks
        # would still refuse the call, but with the wrong message — this
        # catches that silently.
        registry = SkillRegistry(
            [make_skill("weather", requires_network=True, result="RAN")],
            enabled={"weather": False},
            mode="offline",
        )
        result = run(registry, "weather", {"q": "x"})
        assert "turned off" in result
        assert "internet connection" not in result.lower()

    def test_unknown_and_disabled_reports_unknown_not_disabled(self):
        # Pins that the unknown-tool gate runs before the disabled gate: the
        # `enabled` map keys off a name the registry has never heard of.
        registry = SkillRegistry([make_skill("a")], enabled={"format_disk": False})
        result = run(registry, "format_disk", {})
        assert "Error" in result and "unknown tool" in result.lower()
        assert "turned off" not in result


class TestValidate:
    def test_valid_arguments(self):
        registry = SkillRegistry([make_skill("a")])
        assert registry.validate("a", {"q": "x"}) is True

    def test_missing_required_argument(self):
        registry = SkillRegistry([make_skill("a")])
        assert registry.validate("a", {}) is False

    def test_unknown_skill_is_invalid(self):
        registry = SkillRegistry([make_skill("a")])
        assert registry.validate("nope", {"q": "x"}) is False

    def test_disabled_skill_still_validates(self):
        # Validation is about argument shape, not permission. The disabled gate
        # lives in execute() so the refusal gets audited rather than discarded.
        registry = SkillRegistry([make_skill("a")], enabled={"a": False})
        assert registry.validate("a", {"q": "x"}) is True

import asyncio

from nero.config.schema import LLMConfig
from nero.core.audit_log import AuditLog
from nero.llm.client import LLMClient
from nero.skills.base import Skill, SkillMeta
from nero.skills.registry import SkillRegistry


class RecordingSkill(Skill):
    meta = SkillMeta(
        name="get_weather",
        description="Get the weather.",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        requires_network=True,
        permission_tier="read_only",
        offline_message="Weather needs an internet connection, and you're in offline mode right now.",
    )

    def __init__(self):
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return "It is 14°C and raining."


class OtherSkill(Skill):
    """A second, always-available local skill. Present in the `== []`
    schema-visibility tests below so they can't pass against an
    implementation that returns no tools at all — only against one that
    correctly drops the disabled/offline skill while keeping this one."""

    meta = SkillMeta(
        name="open_app",
        description="Open an application.",
        input_schema={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
        requires_network=False,
        permission_tier="state_changing",
    )

    async def execute(self, **kwargs):
        return "Opened."


def make_client(registry):
    return LLMClient(
        config=LLMConfig(provider="claude", model="claude-sonnet-5"),
        assistant_name="Nero",
        registry=registry,
        api_key="sk-test",
    )


def execute(client, name, arguments):
    return asyncio.run(client._execute_tool(name, arguments))


class TestSchemaVisibility:
    def test_enabled_skill_is_offered_to_the_model(self):
        client = make_client(SkillRegistry([RecordingSkill()]))
        names = [d["function"]["name"] for d in client._tool_definitions()]
        assert names == ["get_weather"]

    def test_disabled_skill_is_absent_from_the_schema(self):
        registry = SkillRegistry(
            [RecordingSkill(), OtherSkill()], enabled={"get_weather": False}
        )
        names = {d["function"]["name"] for d in make_client(registry)._tool_definitions()}
        assert names == {"open_app"}

    def test_offline_removes_network_skill_from_the_schema(self):
        registry = SkillRegistry([RecordingSkill(), OtherSkill()], mode="offline")
        names = {d["function"]["name"] for d in make_client(registry)._tool_definitions()}
        assert names == {"open_app"}

    def test_disabled_skill_stays_in_known_names(self):
        # So a stale-schema call classifies as a tool call and reaches the
        # registry's refusal, instead of being discarded as MALFORMED.
        registry = SkillRegistry([RecordingSkill()], enabled={"get_weather": False})
        assert registry.known_names() == {"get_weather"}


class TestBackstopRefusals:
    def test_stale_schema_call_to_disabled_skill_is_refused(self):
        skill = RecordingSkill()
        registry = SkillRegistry([skill], enabled={"get_weather": False})
        result = execute(make_client(registry), "get_weather", {"location": "Oslo"})
        assert skill.calls == []
        assert "turned off" in result

    def test_offline_refusal_message_is_honest(self):
        skill = RecordingSkill()
        registry = SkillRegistry([skill], mode="offline")
        result = execute(make_client(registry), "get_weather", {"location": "Oslo"})
        assert skill.calls == []
        assert result == (
            "Weather needs an internet connection, and you're in offline mode right now."
        )


class TestAuditing:
    def test_successful_call_is_audited(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        registry = SkillRegistry([RecordingSkill()], audit=log)
        execute(make_client(registry), "get_weather", {"location": "Oslo"})
        entries = log.recent()
        assert len(entries) == 1
        assert entries[0].skill_name == "get_weather"
        assert entries[0].arguments == {"location": "Oslo"}
        assert entries[0].provider == "claude"

    def test_disabled_refusal_is_still_audited(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        registry = SkillRegistry(
            [RecordingSkill()], enabled={"get_weather": False}, audit=log
        )
        execute(make_client(registry), "get_weather", {"location": "Oslo"})
        entries = log.recent()
        assert len(entries) == 1
        assert "turned off" in entries[0].result_summary

    def test_offline_refusal_is_still_audited(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        registry = SkillRegistry([RecordingSkill()], mode="offline", audit=log)
        execute(make_client(registry), "get_weather", {"location": "Oslo"})
        assert "internet connection" in log.recent()[0].result_summary

    def test_unknown_skill_is_still_audited(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        registry = SkillRegistry([RecordingSkill()], audit=log)
        execute(make_client(registry), "format_disk", {})
        assert log.recent()[0].skill_name == "format_disk"

    def test_broken_audit_log_does_not_break_the_skill(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        registry = SkillRegistry([RecordingSkill()], audit=AuditLog(blocker / "audit.db"))
        result = execute(make_client(registry), "get_weather", {"location": "Oslo"})
        assert result == "It is 14°C and raining."

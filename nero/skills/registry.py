import logging

from nero.skills.base import Skill, validate_arguments

logger = logging.getLogger("nero.skills")


class SkillRegistry:
    """The single gate between the model and any skill.

    Two name views on purpose (spec §Registry):

    - `tool_definitions()` exposes *available* skills — this is what the model
      is told it can call.
    - `known_names()` exposes *every* skill. The tool-call classifier uses this
      so a call naming a disabled skill is still recognised as a tool call
      rather than falling through to MALFORMED and being silently discarded.
      It then reaches `execute()`, is refused, and is still audited.
    """

    def __init__(
        self,
        skills: list[Skill],
        enabled: dict[str, bool] | None = None,
        mode: str = "online",
        audit=None,
    ):
        self._skills = {skill.meta.name: skill for skill in skills}
        self._enabled = enabled or {}
        self._mode = mode
        self._audit = audit

    def known_names(self) -> set[str]:
        return set(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def is_enabled(self, name: str) -> bool:
        # Absent from the map means enabled: a skill the config doesn't mention
        # yet (e.g. added by a newer version) shouldn't silently vanish.
        return self._enabled.get(name, True)

    def is_available(self, name: str) -> bool:
        skill = self._skills.get(name)
        if skill is None or not self.is_enabled(name):
            return False
        return self._mode != "offline" or not skill.meta.requires_network

    def available(self) -> list[Skill]:
        return [s for s in self._skills.values() if self.is_available(s.meta.name)]

    def tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": skill.meta.name,
                    "description": skill.meta.description,
                    "parameters": skill.meta.input_schema,
                },
            }
            for skill in self.available()
        ]

    def validate(self, name: str, arguments) -> bool:
        """Argument-shape validation only — permission gates live in execute()."""
        skill = self._skills.get(name)
        return skill is not None and validate_arguments(skill.meta.input_schema, arguments)

    async def execute(self, name: str, arguments: dict | None, provider: str = "unknown") -> str:
        result = await self._dispatch(name, arguments)
        self._record(name, arguments, result, provider)
        return result

    async def _dispatch(self, name: str, arguments: dict | None) -> str:
        skill = self._skills.get(name)
        if skill is None:
            return f"Error: unknown tool {name!r}."
        if not self.is_enabled(name):
            return (
                f"The {name} skill is turned off right now. "
                "Tell the user they can enable it with `nero config`."
            )
        if self._mode == "offline" and skill.meta.requires_network:
            return skill.meta.offline_message or (
                "That needs an internet connection, and you're in offline mode right now."
            )
        if arguments is None:
            return "Error: tool arguments were not valid JSON."
        try:
            return await skill.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error: {exc}"

    def _record(self, name: str, arguments: dict | None, result: str, provider: str) -> None:
        """Audit every call, including all three refusal cases. Never raises:
        a logging subsystem must not take down the feature it observes."""
        if self._audit is None:
            return
        from datetime import UTC, datetime

        from nero.core.audit_log import AuditEntry, summarize

        try:
            self._audit.record(
                AuditEntry(
                    timestamp=datetime.now(UTC),
                    skill_name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    result_summary=summarize(result),
                    provider=provider,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not record audit entry for %r: %s", name, exc)


def build_registry(config, audit=None, on_location_resolved=None) -> SkillRegistry:
    """Construct the registry from a NeroConfig.

    `on_location_resolved` lets the weather skill persist a newly learned
    default location without importing ConfigManager (added in Task 8).
    """
    from nero.skills.open_app.server import OpenAppSkill
    from nero.skills.open_website.server import OpenWebsiteSkill
    from nero.skills.play_music.server import PlayMusicSkill
    from nero.skills.weather.server import WeatherSkill

    skills: list[Skill] = [
        OpenAppSkill(),
        OpenWebsiteSkill(),
        WeatherSkill(
            default_location=config.skills.weather.default_location,
            on_location_resolved=on_location_resolved,
        ),
        PlayMusicSkill(),
    ]
    return SkillRegistry(
        skills=skills,
        enabled=config.skills.enabled.model_dump(),
        mode=config.mode,
        audit=audit,
    )

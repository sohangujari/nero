import logging
import re

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
        confirm=None,
    ):
        self._skills = {skill.meta.name: skill for skill in skills}
        self._enabled = enabled or {}
        self._mode = mode
        self._audit = audit
        # confirm: Callable[[str, str, dict], bool] | None, receiving
        # (skill_name, tier, arguments). None => fail closed: a destructive
        # skill is refused, never auto-approved (voice loop, tests, headless
        # routine runs all pass no confirm today).
        self._confirm = confirm
        # Per-turn taint: set once a completed skill result came from a skill
        # with ingests_external_content=True. ChatLoop clears it every turn
        # via reset_turn(). Exposed read-only via `tainted` so a confirm
        # callback built before the registry exists can still consult it.
        self._tainted = False

    @property
    def tainted(self) -> bool:
        return self._tainted

    def mark_tainted(self) -> None:
        self._tainted = True

    def reset_turn(self) -> None:
        """Clear per-turn taint. Called by ChatLoop at the start of each
        user turn."""
        self._tainted = False

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
        """Argument-shape validation only — permission gates live in execute().

        Cleaned first, and with exactly the same pipeline `execute` uses. The
        two must agree: if this said no to `{"mute": "true"}` while execute
        would have accepted it, the call would be discarded as malformed before
        ever reaching the skill that could have run it.
        """
        skill = self._skills.get(name)
        if skill is None or not isinstance(arguments, dict):
            return False
        return validate_arguments(skill.meta.input_schema, clean(skill.meta.input_schema, arguments))

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
        arguments = clean(skill.meta.input_schema, arguments)
        if skill.meta.permission_tier == "destructive":
            # Fail closed: no confirm callback (tests, voice loop, headless
            # routine runs) means the call is refused, never auto-approved.
            if self._confirm is None or not self._confirm(name, skill.meta.permission_tier, arguments):
                return f"The user declined the {name} call."
        try:
            result = await skill.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error: {exc}"
        if skill.meta.ingests_external_content:
            self.mark_tainted()
        return result

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


# What a model sends when it means "I have nothing for this". Seen in the wild
# as `get_weather {"location": "null"}` — which was then geocoded as a place
# called "null", failed, and cost another provider round trip to recover from.
_PLACEHOLDERS = {"null", "none", "undefined", "nil", "n/a", "not specified", ""}


def _drop_placeholders(input_schema: dict, arguments: dict) -> dict:
    """Remove optional string arguments that only say "nothing here".

    Optional only, and only on an exact match: a *required* field is the
    model's problem to get right, and a skill that legitimately receives the
    four characters "null" as content must still receive them.
    """
    optional = set(input_schema.get("properties") or {}) - set(input_schema.get("required") or [])
    cleaned = {
        key: value
        for key, value in arguments.items()
        if not (
            key in optional
            and isinstance(value, str)
            and value.strip().lower() in _PLACEHOLDERS
        )
    }
    return cleaned


# What a model writes when it means True or False but has emitted JSON as text.
_TRUE = {"true", "yes", "1", "on"}
_FALSE = {"false", "no", "0", "off"}


def _coerce_types(input_schema: dict, arguments: dict) -> dict:
    """Arguments retyped to what the schema asks for, where that is unambiguous.

    Small local models routinely send every argument as a string —
    `{"mute": "true", "level": "30"}` for what the schema declares as a boolean
    and an integer. The call is perfectly clear; only its encoding is wrong, and
    refusing it means the user's "mute it" comes back as "I didn't quite catch
    that". Measured on llama3.2, this alone was every one of three failed
    `set_volume` calls.

    Conservative on purpose. Only string values are touched, only where the
    schema names a type, and only when the string maps to exactly one value —
    anything else is left as it is, so a genuinely wrong argument still fails
    validation rather than being coerced into something plausible.
    """
    properties = input_schema.get("properties") or {}
    converted = {}
    for key, value in arguments.items():
        wanted = (properties.get(key) or {}).get("type")
        converted[key] = _as_type(wanted, value) if isinstance(value, str) else value
    return converted


def _as_type(wanted: str | None, text: str):
    """`text` as `wanted`, or `text` unchanged when that is not clearly possible."""
    stripped = text.strip()
    if wanted == "boolean":
        if stripped.lower() in _TRUE:
            return True
        if stripped.lower() in _FALSE:
            return False
    elif wanted == "integer":
        try:
            return int(stripped)
        except ValueError:
            # "30%" and "30 percent" are the model quoting the user rather than
            # answering the schema, and the number in them is unambiguous.
            digits = re.match(r"[-+]?\d+", stripped)
            if digits:
                return int(digits.group())
    elif wanted == "number":
        try:
            return float(stripped)
        except ValueError:
            pass
    return text


def clean(input_schema: dict, arguments: dict) -> dict:
    """Arguments as the skill should receive them: placeholders dropped, then
    strings retyped to what the schema asks for.

    One function so `validate` and `execute` can never disagree about what a
    call means.
    """
    return _coerce_types(input_schema, _drop_placeholders(input_schema, arguments))


def _remember(remember_setting, key: str):
    """Bind a `remember_setting(key, value)` seam to one config key, or None.

    Skills take a one-argument callback because they know their own value, not
    where it lives. This is the only place that pairs the two.
    """
    if remember_setting is None:
        return None
    return lambda value: remember_setting(key, value)


def build_registry(
    config, audit=None, remember_setting=None, confirm=None, extra_skills=None,
    spotify_auth=None,
) -> SkillRegistry:
    """Construct the registry from a NeroConfig.

    `remember_setting(key, value)` lets a skill persist something it learned —
    a default weather location, the music player the user picked — without
    importing ConfigManager. One seam rather than one callback per skill.
    `confirm` gates destructive skills — see SkillRegistry.__init__.
    `extra_skills` appends dynamically discovered skills (MCP tools), whose
    process lifetime the caller owns.
    """
    from nero.skills.execution.server import (
        GitCommandSkill,
        RunJavascriptSkill,
        RunPythonSkill,
        RunShellSkill,
    )
    from nero.memory.facts import FactStore, default_facts_path
    from nero.memory.notes import NoteIndex, default_notes_index_path
    from nero.skills.files.server import (
        DeletePathSkill,
        EditFileSkill,
        MovePathSkill,
        ReadFileSkill,
        WriteFileSkill,
    )
    from nero.skills.memory.server import ForgetFactSkill, RecallFactsSkill, RememberFactSkill
    from nero.skills.notes.server import SearchNotesSkill
    from nero.skills.open_app.server import CloseAppSkill, OpenAppSkill
    from nero.skills.open_website.server import OpenWebsiteSkill
    from nero.skills.play_music.server import PlayMusicSkill
    from nero.skills.volume.server import SetVolumeSkill
    from nero.skills.weather.server import WeatherSkill
    from nero.skills.search.server import WebSearchSkill
    from nero.skills.web.server import FetchWebPageSkill

    fact_store = FactStore(default_facts_path())
    notes_index = (
        NoteIndex(default_notes_index_path(), config.memory.notes_dir, config.memory.notes_max_bytes)
        if config.memory.notes_dir
        else None
    )

    skills: list[Skill] = [
        OpenAppSkill(),
        CloseAppSkill(),
        OpenWebsiteSkill(
            preferred_browser=config.skills.browser.preferred,
            on_browser_chosen=_remember(remember_setting, "skills.browser.preferred"),
        ),
        WeatherSkill(
            default_location=config.skills.weather.default_location,
            on_location_resolved=_remember(remember_setting, "skills.weather.default_location"),
        ),
        SetVolumeSkill(),
        PlayMusicSkill(
            preferred_app=config.skills.music.preferred_app,
            on_app_chosen=_remember(remember_setting, "skills.music.preferred_app"),
            # Read on demand, not at build time: credentials added mid-session
            # should work without a restart.
            spotify_auth=spotify_auth,
        ),
        ReadFileSkill(),
        WriteFileSkill(),
        EditFileSkill(),
        DeletePathSkill(),
        MovePathSkill(),
        FetchWebPageSkill(),
        WebSearchSkill(),
        RunShellSkill(security=config.security),
        GitCommandSkill(security=config.security),
        RunPythonSkill(),
        RunJavascriptSkill(),
        RememberFactSkill(fact_store),
        RecallFactsSkill(fact_store),
        ForgetFactSkill(fact_store),
        SearchNotesSkill(notes_index),
    ]
    skills.extend(extra_skills or [])
    return SkillRegistry(
        skills=skills,
        enabled=config.skills.enabled.model_dump(),
        mode=config.mode,
        audit=audit,
        confirm=confirm,
    )

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Kept hand-written because pydantic needs the Literal at the type level; it
# cannot be generated from nero.llm.providers.PROVIDERS without losing static
# typing. tests/test_providers.py asserts the two stay identical.
Provider = Literal[
    "claude", "openai", "gemini", "ollama", "mistral", "deepseek", "minimax",
    "kimi", "qwen", "xai", "glm", "openrouter", "groq", "custom",
    "custom_anthropic", "bedrock", "huggingface", "cohere", "perplexity",
    "replicate",
]
Mode = Literal["online", "offline"]


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Nero"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider = "claude"
    model: str = "claude-sonnet-5"
    base_url: str | None = None
    # Only meaningful under provider "bedrock"; persists inertly otherwise,
    # exactly like base_url — the guard is LLMClient.aws_region.
    aws_region: str | None = None
    # ONE fallback pair, used mid-turn when the primary model fails with a
    # transient error. Both None means the feature is off; a half-set pair is
    # legal and inert (no cross-field validator) — `config set` nudges toward
    # completing or clearing it, but never mutates the other field, the same
    # warn-only rule aws_region and base_url follow.
    fallback_provider: Provider | None = None
    fallback_model: str | None = None
    # Priority-ordered fallback chain: "provider/model" entries, tried in order
    # on a transient primary failure. Non-empty wins over the scalar pair above
    # (no migration — the pair is never rewritten into this list). Constrains
    # AUTOMATIC selection only; explicit config set always wins.
    fallback_chain: list[str] = []
    # Exact model ids (no globs). blacklist: never auto-selected. whitelist: if
    # non-empty, automatic selection uses ONLY these. Both are warn-only against
    # explicit choices (llm.model, the fallback pair/chain) — never enforced.
    model_blacklist: list[str] = []
    model_whitelist: list[str] = []
    # v1.6.0 routing: exactly one dimension orders the fallback chain (never
    # the primary). "off" is byte-identical to today's chain order. See
    # nero/llm/routing.py for what each dimension actually measures.
    route_by: Literal["off", "cost", "latency", "quality"] = "off"
    quality_rank: list[str] = []  # model ids, best first; unranked sorts last
    health_check: bool = True  # skip a chain entry after 2 consecutive failures this session
    coding_model: str | None = None  # "provider/model" resolved for /code

    @model_validator(mode="after")
    def _validate_fallback_chain(self):
        for entry in self.fallback_chain:
            provider, sep, _model = entry.partition("/")
            if not sep:
                raise ValueError(
                    f'llm.fallback_chain entry {entry!r} must be "provider/model"'
                )
            if provider not in Provider.__args__:
                raise ValueError(
                    f"llm.fallback_chain entry {entry!r} has unknown provider {provider!r}"
                )
        return self

    @model_validator(mode="after")
    def _normalize_base_url(self):
        """Shape only — deliberately not coupled to `provider`.

        A base_url left over from a previous custom endpoint stays in the file
        inertly when the user switches to a named provider, and comes back if
        they switch to custom again. Rejecting it here would turn every provider
        switch into a two-step chore; auto-clearing it would mutate one field as
        a side effect of setting another, which `_warn_if_model_mismatched`
        already establishes this codebase does not do. The guard that matters is
        `LLMClient.api_base`.
        """
        if self.base_url is None:
            return self
        self.base_url = self.base_url.rstrip("/")
        if not re.match(r"^https?://.+", self.base_url):
            raise ValueError("llm.base_url must start with http:// or https://")
        return self


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected_ram_gb: float | None = None
    detected_cpu_cores: int | None = None
    recommended_local_model: str | None = None


class STTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["faster-whisper"] = "faster-whisper"
    model: str = "base"
    # Pinned rather than auto-detected: detection measured ~500 ms per turn,
    # and Nero only speaks English back (Kokoro's voices are all en). Set to
    # null to restore per-utterance auto-detection.
    language: str | None = "en"


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["kokoro", "chatterbox", "cloud"] = "kokoro"
    voice_id: str = "af_bella"  # Kokoro default female; male equivalent: am_michael


class VADConfig(BaseModel):
    """Voice-activity detection tuning.

    All four numbers are physical-world calibration: a slow talker needs longer
    silence, a noisy room a higher threshold. No default is right for every mic.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    silence_ms: int = Field(default=800, ge=200)  # silence that ends a turn
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Generous on purpose: dictating a long message is a real use case, not an
    # error. These caps exist to stop a stuck stream, not to police the user.
    max_utterance_seconds: int = Field(default=180, ge=1)
    # Doubles as the hands-free idle timeout: this much silence with nobody
    # speaking releases the microphone and puts the session to sleep until a
    # keypress. 30 s was tuned for a press-to-talk turn that a human had just
    # started deliberately; a session that re-arms itself after every reply
    # needs longer before it decides the room is empty.
    wait_for_speech_seconds: int = Field(default=60, ge=1)


class VoiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    input_mode: Literal["press_to_talk", "text_only"] = "press_to_talk"
    barge_in: bool = True
    # Nero's own TTS output leaking through built-in speakers into the mic
    # reads as continuous human speech, so barge-in is auto-suppressed on
    # built-in speakers (see audio_io.output_is_builtin_speakers). This
    # bypasses that suppression for users whose speakers sit far enough from
    # the mic that self-hearing isn't a problem. Does not affect
    # `barge_in_active` -- the speaker check is a device-level concern
    # layered on top at the call site, not part of the config truth table.
    force_barge_in: bool = False
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()

    @property
    def barge_in_active(self) -> bool:
        """Barge-in needs a detector. With VAD off it is inert regardless of
        `barge_in`, so every consumer must ask this rather than read the flag."""
        return self.barge_in and self.vad.enabled


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Exchanges (user+assistant pairs) restored from disk when a session
    # starts. Matched to compact_after_messages on purpose: restoring more
    # than the live window holds only to trim it away again is wasted work,
    # and everything older is retrieved on demand anyway.
    max_history_turns: int = Field(default=8, ge=0)
    # A directory of the user's own .md/.txt/.markdown files, indexed for
    # search_notes. None means notes search is unconfigured (actionable
    # message, not an error).
    notes_dir: str | None = None
    notes_max_bytes: int = Field(default=2_000_000, gt=0)  # per-file guard
    # The live context window: messages beyond this are trimmed off the front
    # of what Nero sends (nero/memory/recall.py). Small on purpose — its only
    # job is to carry the live thread so "it" and "that one" still resolve.
    # Anything older is *retrieved* when it's relevant rather than re-sent
    # every turn, which is what makes a long session cost the same as a short
    # one. 0 disables trimming and sends the whole transcript again.
    compact_after_messages: int = Field(default=16, ge=0)
    # Fuse a local embedding model into recall alongside keyword search, when
    # one is available (see nero/memory/embeddings.py — it needs a running
    # ollama and numpy, and silently stays off otherwise). Worth about +9
    # points of recall; costs ~25 ms on the turn and nothing on the network.
    semantic_recall: bool = True


class SkillToggles(BaseModel):
    """One field per skill, rather than dict[str, bool], so a typo'd skill name
    is rejected instead of silently ignored. Adding a skill means adding a field
    — the honest cost of strict validation."""

    model_config = ConfigDict(extra="forbid")

    open_app: bool = True
    open_website: bool = True
    get_weather: bool = True
    play_music: bool = True
    read_file: bool = True
    write_file: bool = False
    edit_file: bool = False
    delete_path: bool = False
    move_path: bool = False
    fetch_web_page: bool = True
    run_shell: bool = False
    git_command: bool = False
    run_python: bool = False
    run_javascript: bool = False
    remember_fact: bool = True
    recall_facts: bool = True
    forget_fact: bool = False
    search_notes: bool = True


class WeatherSkillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_location: str | None = None


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: SkillToggles = SkillToggles()
    weather: WeatherSkillConfig = WeatherSkillConfig()


class SecurityConfig(BaseModel):
    """Command allow/denylist and session limits. Consumed by the confirm
    gate (nero/security.py) and, in the next build, the shell/git skills.
    All defaults are inert: an empty allowlist permits everything, and 0
    limits mean unlimited — installing this feature changes nothing until a
    user opts in."""

    model_config = ConfigDict(extra="forbid")

    command_denylist: list[str] = [
        "rm -rf", "sudo", "shutdown", "mkfs", "dd if=", "curl | sh", "curl|sh",
        "wget | sh", "git push --force", "git reset --hard", ":(){", "chmod 777",
    ]
    # Non-empty => a command must match one of these, else refused outright.
    command_allowlist: list[str] = []
    max_turns_per_session: int = Field(default=0, ge=0)  # 0 = unlimited
    max_cost_usd_per_session: float = Field(default=0.0, ge=0.0)  # 0 = unlimited


class MCPServerConfig(BaseModel):
    """One stdio MCP server.

    `env` values expand ${VAR} from the shell environment so credentials stay
    out of this file. `trusted` relaxes the per-call confirmation to the same
    tier the built-in skills use; leaving it False means every call from that
    server is confirmed, because it runs third-party code.
    """

    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    enabled: bool = True
    trusted: bool = False
    # Conservative: offline mode hides the server's tools. Flip it for servers
    # that are purely local (filesystem, sqlite).
    requires_network: bool = True
    timeout_seconds: int = 60


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: dict[str, MCPServerConfig] = {}


class RoutineConfig(BaseModel):
    """One scheduled routine: a prompt run headlessly on a cron schedule via
    launchd. Destructive skills it triggers never run unattended — they queue
    for approval instead (nero/core/approvals.py)."""

    model_config = ConfigDict(extra="forbid")

    schedule: str  # 5-field cron: "minute hour day month weekday"
    prompt: str
    enabled: bool = True


class RoutinesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routines: dict[str, RoutineConfig] = {}


class NeroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant: AssistantConfig = AssistantConfig()
    llm: LLMConfig = LLMConfig()
    # Whether Nero may use the network at all. This is a user *intent*, not a
    # connectivity probe: offline hides network skills from the model entirely
    # (see spec D2). Runtime network failures are handled per skill.
    mode: Mode = "online"
    hardware: HardwareConfig = HardwareConfig()
    voice: VoiceConfig = VoiceConfig()
    skills: SkillsConfig = SkillsConfig()
    memory: MemoryConfig = MemoryConfig()
    security: SecurityConfig = SecurityConfig()
    mcp: MCPConfig = MCPConfig()
    routines: RoutinesConfig = RoutinesConfig()

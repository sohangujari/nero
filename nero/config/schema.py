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
    wait_for_speech_seconds: int = Field(default=30, ge=1)


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
    max_history_turns: int = Field(default=20, ge=0)  # counts exchanges (user+assistant pairs)


class SkillToggles(BaseModel):
    """One field per skill, rather than dict[str, bool], so a typo'd skill name
    is rejected instead of silently ignored. Adding a skill means adding a field
    — the honest cost of strict validation."""

    model_config = ConfigDict(extra="forbid")

    open_app: bool = True
    open_website: bool = True
    get_weather: bool = True
    play_music: bool = True


class WeatherSkillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_location: str | None = None


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: SkillToggles = SkillToggles()
    weather: WeatherSkillConfig = WeatherSkillConfig()


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

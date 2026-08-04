from typing import Literal

from pydantic import BaseModel, ConfigDict

Provider = Literal["claude", "openai", "gemini", "ollama"]
Mode = Literal["online", "offline"]


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Nero"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider = "claude"
    model: str = "claude-sonnet-5"


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


class VoiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    input_mode: Literal["press_to_talk", "text_only"] = "press_to_talk"
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_history_turns: int = 20  # counts exchanges (user+assistant pairs)


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

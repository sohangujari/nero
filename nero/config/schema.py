from typing import Literal

from pydantic import BaseModel, ConfigDict

Provider = Literal["claude", "openai", "gemini", "ollama"]


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


class NeroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant: AssistantConfig = AssistantConfig()
    llm: LLMConfig = LLMConfig()
    hardware: HardwareConfig = HardwareConfig()
    voice: VoiceConfig = VoiceConfig()

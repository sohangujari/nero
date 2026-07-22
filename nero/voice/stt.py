from __future__ import annotations

from abc import ABC, abstractmethod

from nero.voice.errors import VoiceDependencyError


class STTEngine(ABC):
    @abstractmethod
    async def transcribe(self, audio, sample_rate: int) -> str:
        """Transcribe mono float32 audio to text."""


class FasterWhisperSTT(STTEngine):
    """faster-whisper STT. Expects 16 kHz mono float32 audio (what audio_io records)."""

    def __init__(self, model: str = "base", _model=None):
        self._model = _model if _model is not None else self._load(model)

    @staticmethod
    def _load(model: str):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise VoiceDependencyError(
                'Voice mode needs extra packages. Install them with: pip install "nero[voice]"'
            ) from exc
        return WhisperModel(model, device="cpu", compute_type="int8")

    async def transcribe(self, audio, sample_rate: int) -> str:
        segments, _info = self._model.transcribe(audio, language=None)
        return " ".join(segment.text.strip() for segment in segments).strip()

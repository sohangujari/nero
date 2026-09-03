from __future__ import annotations

from abc import ABC, abstractmethod

from nero.voice.errors import VoiceDependencyError

# Curated faster-whisper sizes for the `nero config` picker. Deliberately not
# exhaustive — the picker keeps a free-text escape row for distil-* builds and
# Hugging Face repo ids, so this list only has to cover the common choices.
STT_MODELS: list[tuple[str, str]] = [
    ("tiny", "fastest, least accurate"),
    ("base", "fast, fine for short commands"),
    ("small", "balanced"),
    ("medium", "slower, more accurate"),
    ("large-v3", "slowest, most accurate"),
    ("large-v3-turbo", "large-v3 accuracy at roughly small's speed"),
]


class STTEngine(ABC):
    @abstractmethod
    async def transcribe(self, audio, sample_rate: int) -> str:
        """Transcribe mono float32 audio to text."""

    def warmup(self) -> None:
        """Optional: pay a cold-start cost now rather than on the first turn."""


class FasterWhisperSTT(STTEngine):
    """faster-whisper STT. Expects 16 kHz mono float32 audio (what audio_io records)."""

    def __init__(self, model: str = "base", language: str | None = "en", _model=None):
        # Language auto-detection measured 1137-1286 ms on a 2.3 s utterance
        # against 671 ms with the language pinned -- roughly 500 ms per turn,
        # every turn, to re-answer a question whose answer never changes.
        # `None` restores auto-detect for anyone who needs it.
        self._language = language
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
        segments, _info = self._model.transcribe(audio, language=self._language)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def warmup(self) -> None:
        """Decode half a second of silence to force CTranslate2's first-run setup."""
        import numpy as np

        self._model.transcribe(np.zeros(8000, dtype=np.float32), language=self._language)

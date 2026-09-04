from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nero.voice.errors import TTSLoadError, VoiceDependencyError

# Curated Kokoro voices with human-readable labels so config can offer a
# gender-labeled picker without exposing Kokoro's naming scheme.
VOICE_CATALOG: list[tuple[str, str, str]] = [
    ("af_bella", "Bella", "female"),
    ("af_sarah", "Sarah", "female"),
    ("af_nicole", "Nicole", "female"),
    ("am_michael", "Michael", "male"),
    ("am_adam", "Adam", "male"),
    ("bf_emma", "Emma", "female"),
    ("bm_george", "George", "male"),
]


# Kokoro pads every chunk with silence -- measured 22-41 ms of lead and
# 57-152 ms of tail. Between two sentences that is a natural breath, but a
# reply is cut at a clause boundary too (see sentence_buffer.FIRST_MAX_LEN), and
# there the padding lands mid-sentence: ~190 ms of dead air right after a comma,
# on every reply, which is what "it pauses too long at commas" describes.
_SILENCE = 0.005  # amplitude below which a sample counts as silence


def trim_silence(samples, keep_tail: bool):
    """Drop a chunk's padding. `keep_tail` leaves the trailing silence intact --
    that is the pause between two sentences, and speech without it sounds
    rushed. Mid-sentence chunks trim both ends so the clause flows on."""
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    loud = np.flatnonzero(np.abs(data) > _SILENCE)
    if loud.size == 0:
        return data[:0]
    return data[loud[0] : (data.size if keep_tail else loud[-1] + 1)]


def _ends_a_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", '."', ".'", '!"', '?"'))


class TTSEngine(ABC):
    SAMPLE_RATE: int = 24000

    @abstractmethod
    async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator:
        """Yield an audio chunk per sentence as each becomes ready."""

    def warmup(self) -> None:
        """Optional: pay a cold-start cost now rather than on the first turn."""


class KokoroTTS(TTSEngine):
    """Kokoro TTS via kokoro-onnx (onnxruntime, no PyTorch — bundle-friendly).

    The ONNX model + voices file are fetched to a local cache on first use, so
    the shipped binary stays small.
    """

    SAMPLE_RATE = 24000

    def __init__(self, voice_id: str = "af_bella", _model=None):
        self._voice = voice_id
        self._model = _model if _model is not None else self._load()

    @staticmethod
    def _load():
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise VoiceDependencyError(
                'Voice mode needs extra packages. Install them with: pip install "nero[voice]"'
            ) from exc
        from nero.voice.models import ensure_kokoro_model

        try:
            model_path, voices_path = ensure_kokoro_model()
            return Kokoro(str(model_path), str(voices_path))
        except VoiceDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001 — missing/corrupt model files, etc.
            raise TTSLoadError(
                "Kokoro's model could not be loaded. Re-run setup to (re)download the "
                "voice model, then try again."
            ) from exc

    async def synthesize_stream(self, text_stream):
        async for sentence in text_stream:
            samples, _sample_rate = self._model.create(
                sentence, voice=self._voice, speed=1.0, lang="en-us"
            )
            yield trim_silence(samples, keep_tail=_ends_a_sentence(sentence))

    def warmup(self) -> None:
        """Synthesize one throwaway character.

        Kokoro's first call measured 1383 ms against ~830 ms warm for the same
        five-word sentence — a cold start the user would otherwise pay while
        waiting for Nero's very first word.
        """
        self._model.create("a", voice=self._voice, speed=1.0, lang="en-us")


class ChatterboxTTS(TTSEngine):
    SAMPLE_RATE = 24000

    def __init__(self, _model=None):
        self._model = _model if _model is not None else self._load()

    @staticmethod
    def _load():
        try:
            from chatterbox.tts import ChatterboxTTS as _Model
        except ImportError as exc:
            raise VoiceDependencyError(
                'Premium voice needs extra packages. Install with: pip install "nero[premium-voice]"'
            ) from exc
        try:
            return _Model.from_pretrained(device="cpu")
        except Exception as exc:  # noqa: BLE001
            raise TTSLoadError(
                "Chatterbox's model could not be loaded. Re-run setup to download it."
            ) from exc

    async def synthesize_stream(self, text_stream):
        async for sentence in text_stream:
            yield self._model.generate(sentence)


def build_tts(engine: str, voice_id: str) -> TTSEngine:
    if engine == "chatterbox":
        return ChatterboxTTS()
    if engine == "cloud":
        # Cloud TTS is out of scope for this phase; fall back to Kokoro.
        return KokoroTTS(voice_id=voice_id)
    return KokoroTTS(voice_id=voice_id)

import asyncio
import types

from nero.voice.errors import (
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
    TTSLoadError,
    VoiceDependencyError,
    VoiceError,
)


def test_error_hierarchy():
    for cls in (
        MicUnavailableError,
        MicPermissionError,
        PlaybackError,
        TTSLoadError,
        VoiceDependencyError,
    ):
        assert issubclass(cls, VoiceError)


# --- Task 7: STT ---
from nero.voice.stt import FasterWhisperSTT, STTEngine


class FakeWhisper:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        info = types.SimpleNamespace(language="en")
        return iter(self._segments), info


def seg(text):
    return types.SimpleNamespace(text=text)


def test_faster_whisper_joins_segments():
    stt = FasterWhisperSTT(_model=FakeWhisper([seg(" Open"), seg(" calculator.")]))
    result = asyncio.run(stt.transcribe(object(), 16000))
    assert result == "Open calculator."


def test_faster_whisper_empty_transcription():
    stt = FasterWhisperSTT(_model=FakeWhisper([]))
    assert asyncio.run(stt.transcribe(object(), 16000)) == ""


def test_is_sttengine():
    assert issubclass(FasterWhisperSTT, STTEngine)


# --- Task 8: TTS (kokoro-onnx) ---
from nero.voice.tts import VOICE_CATALOG, KokoroTTS, TTSEngine, build_tts


class FakeKokoro:
    """Mimics kokoro_onnx.Kokoro: .create(text, voice=...) -> (samples, sample_rate)."""

    def __init__(self):
        self.calls = []

    def create(self, text, voice=None, speed=1.0, lang="en-us"):
        self.calls.append((text, voice, speed, lang))
        return f"AUDIO[{voice}:{text}]", 24000


async def _drain(engine, sentences):
    async def stream():
        for s in sentences:
            yield s

    return [audio async for audio in engine.synthesize_stream(stream())]


def test_kokoro_synthesizes_each_sentence_with_voice():
    model = FakeKokoro()
    tts = KokoroTTS(voice_id="am_michael", _model=model)
    out = asyncio.run(_drain(tts, ["Hello.", "Bye."]))
    assert out == ["AUDIO[am_michael:Hello.]", "AUDIO[am_michael:Bye.]"]
    assert [(t, v) for t, v, _s, _l in model.calls] == [
        ("Hello.", "am_michael"),
        ("Bye.", "am_michael"),
    ]


def test_kokoro_is_ttsengine_with_sample_rate():
    assert issubclass(KokoroTTS, TTSEngine)
    assert KokoroTTS(_model=FakeKokoro()).SAMPLE_RATE == 24000


def test_voice_catalog_has_labeled_genders():
    ids = {vid for vid, _name, _gender in VOICE_CATALOG}
    assert "af_bella" in ids and "am_michael" in ids
    assert all(gender in ("female", "male") for _vid, _name, gender in VOICE_CATALOG)


def test_build_tts_returns_kokoro(monkeypatch):
    monkeypatch.setattr(KokoroTTS, "_load", staticmethod(lambda: FakeKokoro()))
    engine = build_tts("kokoro", "af_bella")
    assert isinstance(engine, KokoroTTS)


# --- Model cache (no network) ---
from nero.voice import models


def test_ensure_kokoro_model_uses_cache_without_download(monkeypatch, tmp_path):
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"model")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"voices")
    monkeypatch.setattr(models, "voice_cache_dir", lambda: tmp_path)

    def no_network(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not download when cached")

    monkeypatch.setattr(models.httpx, "stream", no_network)
    model_path, voices_path = models.ensure_kokoro_model()
    assert model_path.name == "kokoro-v1.0.onnx"
    assert voices_path.name == "voices-v1.0.bin"

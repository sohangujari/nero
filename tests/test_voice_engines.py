import asyncio
import types

import numpy as np

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
from nero.voice.tts import (
    VOICE_CATALOG,
    KokoroTTS,
    TTSEngine,
    build_tts,
    trim_silence,
)


class FakeKokoro:
    """Mimics kokoro_onnx.Kokoro: .create(text, voice=...) -> (samples, sample_rate).

    Real samples, not a sentinel string: synthesize_stream trims each chunk's
    padding now, so a double that hands back something unarray-like would be
    asserting against a shape Kokoro never produces. Every chunk is padded with
    silence at both ends, exactly as Kokoro's are.
    """

    PAD = 240  # 10 ms at 24 kHz

    def __init__(self):
        self.calls = []

    def create(self, text, voice=None, speed=1.0, lang="en-us"):
        self.calls.append((text, voice, speed, lang))
        body = np.full(len(text) * 10, 0.5, dtype=np.float32)
        pad = np.zeros(self.PAD, dtype=np.float32)
        return np.concatenate([pad, body, pad]), 24000


async def _drain(engine, sentences):
    async def stream():
        for s in sentences:
            yield s

    return [audio async for audio in engine.synthesize_stream(stream())]


def test_kokoro_synthesizes_each_sentence_with_voice():
    model = FakeKokoro()
    tts = KokoroTTS(voice_id="am_michael", _model=model)
    out = asyncio.run(_drain(tts, ["Hello.", "Bye."]))
    assert [len(chunk) for chunk in out] == [len("Hello.") * 10 + FakeKokoro.PAD,
                                             len("Bye.") * 10 + FakeKokoro.PAD]
    assert [call[0] for call in model.calls] == ["Hello.", "Bye."]
    assert {call[1] for call in model.calls} == {"am_michael"}


class TestTrimSilence:
    """Kokoro pads every chunk. Between sentences that padding is a breath;
    inside one, it is the ~190 ms stall people hear after a comma."""

    @staticmethod
    def _padded():
        return np.concatenate([
            np.zeros(240, dtype=np.float32),
            np.full(1000, 0.5, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
        ])

    def test_a_sentence_keeps_its_trailing_pause(self):
        assert len(trim_silence(self._padded(), keep_tail=True)) == 1000 + 480

    def test_a_mid_sentence_chunk_trims_both_ends(self):
        assert len(trim_silence(self._padded(), keep_tail=False)) == 1000

    def test_the_lead_in_always_goes(self):
        """The previous chunk's own tail already supplies the gap."""
        for keep_tail in (True, False):
            assert trim_silence(self._padded(), keep_tail=keep_tail)[0] == 0.5

    def test_a_silent_chunk_survives_as_empty(self):
        assert len(trim_silence(np.zeros(500, dtype=np.float32), keep_tail=True)) == 0

    def test_a_clause_cut_is_treated_as_mid_sentence(self):
        """This is the case that matters: the first segment of a reply is cut
        at a comma, so its padding lands inside a sentence."""
        model = FakeKokoro()
        tts = KokoroTTS(voice_id="af_bella", _model=model)
        clause, sentence = asyncio.run(_drain(tts, ["Sure thing,", "here it is."]))
        assert len(clause) == len("Sure thing,") * 10          # both ends trimmed
        assert len(sentence) == len("here it is.") * 10 + FakeKokoro.PAD


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


# --- Latency: language pinning and cold-start warmup ---
class _RecordingWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append((getattr(audio, "size", None), language))
        return ([], None)


def test_stt_pins_the_configured_language():
    """Auto-detect measured ~500ms per turn to re-answer a fixed question."""
    import asyncio

    from nero.voice.stt import FasterWhisperSTT

    model = _RecordingWhisper()
    stt = FasterWhisperSTT(_model=model)
    asyncio.run(stt.transcribe(np.zeros(16000, dtype=np.float32), 16000))
    assert model.calls[0][1] == "en"


def test_stt_language_none_restores_auto_detect():
    import asyncio

    from nero.voice.stt import FasterWhisperSTT

    model = _RecordingWhisper()
    stt = FasterWhisperSTT(language=None, _model=model)
    asyncio.run(stt.transcribe(np.zeros(16000, dtype=np.float32), 16000))
    assert model.calls[0][1] is None


def test_stt_warmup_decodes_silence_with_the_same_language():
    from nero.voice.stt import FasterWhisperSTT

    model = _RecordingWhisper()
    FasterWhisperSTT(_model=model).warmup()
    assert model.calls == [(8000, "en")]


def test_tts_warmup_synthesizes_a_throwaway_token():
    from nero.voice.tts import KokoroTTS

    class _RecordingKokoro:
        def __init__(self):
            self.created = []

        def create(self, text, voice, speed, lang):
            self.created.append(text)
            return (np.zeros(4, dtype=np.float32), 24000)

    model = _RecordingKokoro()
    KokoroTTS(_model=model).warmup()
    assert model.created == ["a"]


def test_prewarm_never_lets_a_failure_stop_startup():
    """An optimization is never a prerequisite for `nero talk` starting."""
    from nero import cli

    class Exploding:
        def warmup(self):
            raise RuntimeError("no model")

    cli._prewarm(Exploding(), object())  # object() has no warmup at all


class TestNoSpeechFilter:
    """Hands-free listening feeds whisper every noise the VAD mistook for
    speech, and whisper answers those with a fluent hallucination rather than
    an empty string. Measured against the real model: room tone and a thump
    decode to 'You' at p=0.65-0.73, real speech at p=0.018."""

    class FakeSegment:
        def __init__(self, text, no_speech_prob):
            self.text = text
            self.no_speech_prob = no_speech_prob

    class FakeModel:
        def __init__(self, segments):
            self._segments = segments

        def transcribe(self, audio, language=None):
            return iter(self._segments), None

    def _transcribe(self, segments):
        stt = FasterWhisperSTT(_model=self.FakeModel(segments))
        return asyncio.run(stt.transcribe(np.zeros(16000, dtype=np.float32), 16000))

    def test_a_hallucination_on_noise_is_dropped(self):
        assert self._transcribe([self.FakeSegment(" You", 0.727)]) == ""

    def test_real_speech_is_kept(self):
        assert self._transcribe([self.FakeSegment(" Hello there.", 0.018)]) == "Hello there."

    def test_only_the_noise_segment_is_dropped(self):
        assert self._transcribe([
            self.FakeSegment(" Turn on the lights.", 0.02),
            self.FakeSegment(" Thanks for watching!", 0.91),
        ]) == "Turn on the lights."

    def test_a_segment_without_the_field_is_kept(self):
        """Not every engine reports one; absence must not silence a transcript."""
        segment = types.SimpleNamespace(text=" Hello.")
        assert self._transcribe([segment]) == "Hello."

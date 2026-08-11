import asyncio
import io
import threading
import sys
import types

import numpy as np
import pytest
from rich.console import Console

from nero.voice import audio_io
from nero.voice.audio_io import Player
from nero.voice.errors import (
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
)


class FakeInputStream:
    """Mimics sd.InputStream: hands out fixed frames on each read()."""

    def __init__(self, frames, raise_on_start=None, **kwargs):
        self._frames = frames
        self._raise = raise_on_start
        self._i = 0

    def __enter__(self):
        if self._raise:
            raise self._raise
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        block = self._frames[self._i]
        self._i = (self._i + 1) % len(self._frames)
        return block, False


def make_fake_sd(frames=None, input_error_msg=None, play_error=None):
    fake = types.ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    fake.PortAudioError = PortAudioError
    frames = frames if frames is not None else [np.zeros((160, 1), dtype="float32")]
    input_error = PortAudioError(input_error_msg) if input_error_msg else None

    def InputStream(**kwargs):
        return FakeInputStream(frames, raise_on_start=input_error, **kwargs)

    fake.InputStream = InputStream
    fake.played = []

    def play(audio, sr):
        if play_error:
            raise play_error
        fake.played.append((audio, sr))

    def wait():
        pass

    fake.play = play
    fake.wait = wait
    return fake


def install(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_record_stops_on_enter_and_returns_float32(monkeypatch):
    fake = make_fake_sd(frames=[np.ones((160, 1), dtype="float32")])
    install(monkeypatch, fake)
    audio = audio_io.record_until_enter(Console(), input_fn=lambda: "")
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # flattened to mono


def test_record_maps_permission_error(monkeypatch):
    fake = make_fake_sd(input_error_msg="Error opening InputStream: Permission denied")
    install(monkeypatch, fake)
    with pytest.raises(MicPermissionError):
        audio_io.record_until_enter(Console(), input_fn=lambda: "")


def test_record_maps_generic_device_error(monkeypatch):
    fake = make_fake_sd(input_error_msg="No default input device")
    install(monkeypatch, fake)
    with pytest.raises(MicUnavailableError):
        audio_io.record_until_enter(Console(), input_fn=lambda: "")


def test_play_forwards_to_sounddevice(monkeypatch):
    fake = make_fake_sd()
    install(monkeypatch, fake)
    audio = np.zeros(240, dtype="float32")
    audio_io.play(audio, 24000)
    assert fake.played == [(audio, 24000)]


def test_play_maps_error(monkeypatch):
    fake = make_fake_sd(play_error=RuntimeError("boom"))
    install(monkeypatch, fake)
    with pytest.raises(PlaybackError):
        audio_io.play(np.zeros(10, dtype="float32"), 24000)


# --- Task 6: Player ---
class FakeTTS:
    """synthesize_stream maps each sentence to a marker 'audio' object."""

    async def synthesize_stream(self, text_stream):
        async for sentence in text_stream:
            yield f"AUDIO({sentence})"


class ExplodingTTS:
    async def synthesize_stream(self, text_stream):
        async for _sentence in text_stream:
            raise RuntimeError("synthesis blew up")
            yield  # pragma: no cover


def test_player_synthesizes_and_plays_in_order():
    played = []
    player = Player(FakeTTS(), sample_rate=24000, play_fn=lambda a, sr: played.append((a, sr)))
    player.start()
    player.enqueue("First sentence.")
    player.enqueue("Second one.")
    player.close()
    player.join()
    assert played == [
        ("AUDIO(First sentence.)", 24000),
        ("AUDIO(Second one.)", 24000),
    ]


def test_player_join_reraises_thread_error():
    player = Player(ExplodingTTS(), sample_rate=24000, play_fn=lambda a, sr: None)
    player.start()
    player.enqueue("boom")
    player.close()
    with pytest.raises(RuntimeError, match="synthesis blew up"):
        player.join()


# --- Clean shutdown: an aborted turn must not leave threads parked ---
class BlockingTTS:
    """Consumes sentences forever; never completes on its own."""

    async def synthesize_stream(self, text_stream):
        async for _s in text_stream:
            yield "AUDIO"


def test_shutdown_stops_thread_when_turn_aborted():
    """No sentinel was sent (simulating Ctrl+C mid-turn): shutdown must still stop it."""
    player = Player(BlockingTTS(), sample_rate=24000, play_fn=lambda a, sr: None)
    player.start()
    player.enqueue("something.")
    player.shutdown(timeout=5)
    assert player._thread is not None
    assert not player._thread.is_alive(), "playback thread still parked -> would hang exit"


def test_shutdown_is_idempotent_after_normal_close():
    played = []
    player = Player(FakeTTS(), sample_rate=24000, play_fn=lambda a, sr: played.append(a))
    player.start()
    player.enqueue("Hi.")
    player.close()
    player.join()
    player.shutdown()  # must not raise or hang
    player.shutdown()
    assert played == ["AUDIO(Hi.)"]


def test_recorder_stops_capture_thread_on_keyboard_interrupt(monkeypatch):
    """Ctrl+C at the Enter prompt must still release the mic."""
    fake = make_fake_sd()
    install(monkeypatch, fake)

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        audio_io.record_until_enter(Console(), input_fn=interrupt)
    # If the finally didn't run, capture threads would still be alive.
    leftover = [t for t in threading.enumerate() if t.name.startswith("Thread-") and t.is_alive()]
    assert all("capture" not in t.name for t in leftover)


class ScriptedVAD:
    """Returns a scripted speech/silence verdict per frame."""

    FRAME_SAMPLES = 512

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def is_speech(self, frame):
        return self._verdicts.pop(0) if self._verdicts else False


def vad_frames(count):
    """`count` frames of the exact size record_until_silence reads."""
    return [np.zeros((512, 1), dtype="float32") for _ in range(count)]


class TestRecordUntilSilence:
    def test_stops_after_configured_silence(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        # speech, speech, then silence long enough to end the turn.
        # 800ms / 32ms per frame = 25 silent frames.
        vad = ScriptedVAD([True, True] + [False] * 25)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, silence_ms=800
        )
        assert audio.ndim == 1
        assert audio.size > 0

    def test_prefix_audio_lands_at_the_head(self, monkeypatch):
        """Barge-in hands its buffered audio forward; without this the user's
        first word — the one that triggered the interrupt — is clipped."""
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([True] + [False] * 25)
        prefix = np.full(512, 0.7, dtype=np.float32)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, silence_ms=800, prefix=prefix
        )
        assert np.allclose(audio[:512], 0.7)

    def test_gives_up_when_speech_never_starts(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([False] * 5000)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, wait_for_speech_seconds=1
        )
        assert audio.size == 0

    def test_hard_cap_stops_an_endless_utterance(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([True] * 100000)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, max_utterance_seconds=1
        )
        # 1 second at 16 kHz, allowing one frame of overshoot.
        assert 0 < audio.size <= 16000 + 512

    def test_vad_state_is_reset_per_recording(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([True] + [False] * 25)
        audio_io.record_until_silence(Console(file=io.StringIO()), vad, silence_ms=800)
        assert vad.reset_calls == 1

    def test_prefix_is_discarded_when_speech_never_starts(self, monkeypatch):
        """A spurious barge-in (Nero hearing itself) must not reach the STT.

        The prefix can contain Nero's own voice. If no real speech follows, we
        must return empty — otherwise Nero's words get transcribed as the
        user's and fed back to the model.
        """
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([False] * 5000)
        prefix = np.full(512, 0.7, dtype=np.float32)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, wait_for_speech_seconds=1, prefix=prefix
        )
        assert audio.size == 0

    def test_mic_permission_error_is_translated(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1), input_error_msg="Permission denied")
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        with pytest.raises(MicPermissionError):
            audio_io.record_until_silence(
                Console(file=io.StringIO()), ScriptedVAD([]), silence_ms=800
            )

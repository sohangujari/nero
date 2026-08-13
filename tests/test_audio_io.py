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
    fake.stop_calls = []

    def stop():
        fake.stop_calls.append(True)

    fake.stop = stop
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
        # 2 speech frames + 25 silence frames are each appended to `blocks`
        # before the 25th silent frame trips `silent_run >= silent_needed`
        # (silent_needed = round(800/1000 * 16000/512) = 25) and breaks the
        # loop. 27 frames * 512 samples/frame = 13824. Asserting the exact
        # count (not just `> 0`) is what actually proves the silence-break
        # fired, rather than the 180s frame cap.
        assert audio.size == 27 * 512 == 13824

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

    def test_indicator_is_silent_on_a_non_terminal_console(self, monkeypatch):
        """Tests and piped output are not a terminal; a Live region there would
        spam output, so the indicator must stay off and fall back to the plain
        'Listening' line already printed."""
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([True, True] + [False] * 25)
        out = io.StringIO()
        console = Console(file=out, force_terminal=False)
        audio_io.record_until_silence(console, vad, silence_ms=800)
        output = out.getvalue()
        assert "Listening" in output
        # No cursor-control / ANSI escapes -- proof no Live region ever opened.
        assert "\x1b[" not in output

    def test_indicator_on_a_terminal_does_not_change_returned_audio(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)

        vad_plain = ScriptedVAD([True, True] + [False] * 25)
        audio_plain = audio_io.record_until_silence(
            Console(file=io.StringIO(), force_terminal=False), vad_plain, silence_ms=800
        )

        vad_terminal = ScriptedVAD([True, True] + [False] * 25)
        terminal_out = io.StringIO()
        console = Console(file=terminal_out, force_terminal=True, width=80)
        audio_terminal = audio_io.record_until_silence(
            console, vad_terminal, silence_ms=800
        )

        assert np.array_equal(audio_plain, audio_terminal)
        # On a real terminal the Live region does render something.
        assert terminal_out.getvalue() != ""


class SentenceTTS:
    """One audio chunk per sentence, mirroring the real TTSEngine contract."""

    async def synthesize_stream(self, text_stream):
        async for sentence in text_stream:
            yield np.full(4, len(sentence), dtype=np.float32)


class PrefetchingTTS:
    """Pulls 3 sentences from the stream before yielding any audio.

    Still one audio chunk per sentence, in order — just with the source
    lookahead a real batching TTS engine could exhibit.
    """

    async def synthesize_stream(self, text_stream):
        buffered = []
        async for sentence in text_stream:
            buffered.append(sentence)
            if len(buffered) == 3:
                break
        for sentence in buffered:
            yield np.full(4, len(sentence), dtype=np.float32)


class TestSpokenTracking:
    def test_spoken_text_is_empty_before_anything_plays(self):
        player = Player(SentenceTTS(), 24000, play_fn=lambda a, sr: None)
        assert player.spoken_text() == ""

    def test_spoken_text_reports_played_sentences_in_order(self):
        player = Player(SentenceTTS(), 24000, play_fn=lambda a, sr: None)
        player.start()
        player.enqueue("One.")
        player.enqueue("Two.")
        player.close()
        player.join(timeout=5)
        assert player.spoken_text() == "One. Two."

    def test_queued_but_unplayed_sentences_are_not_reported(self, monkeypatch):
        """The whole point of D3: only what reached the speaker is recorded."""
        monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd())
        started = threading.Event()
        release = threading.Event()

        def blocking_play(audio, sr):
            started.set()
            release.wait(timeout=5)

        player = Player(SentenceTTS(), 24000, play_fn=blocking_play)
        player.start()
        player.enqueue("First.")
        player.enqueue("Second.")
        player.enqueue("Third.")
        assert started.wait(timeout=5)
        # Only sentence 1 has been dispatched; 2 and 3 are still queued.
        assert player.spoken_text() == "First."
        release.set()
        player.stop_now()

    def test_prefetched_sentences_are_not_reported_until_played(self, monkeypatch):
        """Discriminates the play-counter from mere yielded-to-TTS tracking.

        A real TTS engine can pull several sentences ahead of what it has
        actually turned into played audio (batching/lookahead). This fake
        mimics that: it drains three sentences from the stream (so all three
        land in `_yielded`) before yielding any audio. If `spoken_text()`
        were `" ".join(self._yielded)` with no `[:self._spoken]` slice, this
        would report all three while only the first has reached the speaker.
        """
        monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd())
        started = threading.Event()
        release = threading.Event()

        def blocking_play(audio, sr):
            started.set()
            release.wait(timeout=5)

        player = Player(PrefetchingTTS(), 24000, play_fn=blocking_play)
        player.start()
        player.enqueue("First.")
        player.enqueue("Second.")
        player.enqueue("Third.")
        player.close()
        assert started.wait(timeout=5)
        # All three sentences were pulled by the TTS engine already...
        assert player._yielded == ["First.", "Second.", "Third."]
        # ...but only the first has been dispatched to play.
        assert player.spoken_text() == "First."
        release.set()
        player.join(timeout=5)
        assert player.spoken_text() == "First. Second. Third."

    def test_stop_now_prevents_already_buffered_chunks_from_playing(self, monkeypatch):
        """Fix 2: PrefetchingTTS pulls all 3 sentences and has all 3 audio
        chunks ready to hand out before the outer loop has dispatched more
        than one to `_play`. If `_consume` only checked `_stop` inside
        `_next_item` (queue reads), it would still play chunks 2 and 3 once
        the first `_play` call unblocks, because they don't require another
        queue read. stop_now() must cut that off too."""
        monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd())
        played = []
        started = threading.Event()
        release = threading.Event()

        def blocking_play(audio, sr):
            played.append(audio)
            started.set()
            release.wait(timeout=5)

        player = Player(PrefetchingTTS(), 24000, play_fn=blocking_play)
        player.start()
        player.enqueue("First.")
        player.enqueue("Second.")
        player.enqueue("Third.")
        player.close()
        assert started.wait(timeout=5)
        # First chunk is mid-play; the engine already has chunks 2 and 3 ready
        # to yield with no further queue read required.
        player.stop_now()
        release.set()
        player.join(timeout=5)
        assert len(played) == 1
        assert np.allclose(played[0], np.full(4, len("First."), dtype=np.float32))

    def test_stop_now_halts_playback(self, monkeypatch):
        fake_sd = make_fake_sd()
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
        player = Player(SentenceTTS(), 24000, play_fn=lambda a, sr: None)
        player.start()
        player.enqueue("One.")
        player.stop_now()
        player.join(timeout=5)
        assert player._stop.is_set()
        assert fake_sd.stop_calls == [True]


class _ExhaustibleVAD(ScriptedVAD):
    """ScriptedVAD that sets `stop` once its script runs out.

    Lets a test end the watch thread deterministically, from the scripted
    frame count, instead of from a wall-clock timer.
    """

    def __init__(self, verdicts, stop):
        super().__init__(verdicts)
        self._stop = stop

    def is_speech(self, frame):
        if not self._verdicts:
            self._stop.set()
            return False
        return super().is_speech(frame)


class TestBargeInMonitor:
    def test_sustained_speech_fires_with_buffered_prefix(self, monkeypatch):
        fake = make_fake_sd(frames=[np.full((512, 1), 0.4, dtype="float32")])
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        # 400ms / 32ms = 12.5 -> round() (banker's rounding on the .5 case)
        # gives 12, not 13. 12 frames is ~384ms, still within the spec's
        # "~400ms" sustained gate.
        vad = ScriptedVAD([True] * 20)
        fired = []
        stop = threading.Event()
        thread = audio_io.listen_for_barge_in(vad, fired.append, stop)
        thread.join(timeout=5)
        assert len(fired) == 1
        assert fired[0].size > 0
        assert np.allclose(fired[0], 0.4)

    def test_short_burst_does_not_fire(self, monkeypatch):
        """Nero's own voice leaking into the mic produces brief blips; the
        sustained gate is what stops Nero interrupting itself."""
        fake = make_fake_sd(frames=[np.full((512, 1), 0.4, dtype="float32")])
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        stop = threading.Event()
        # The VAD itself ends the thread once its script is exhausted, so the
        # test is bounded by a known frame count rather than the wall clock.
        vad = _ExhaustibleVAD(([True] * 5 + [False] * 5) * 20, stop)
        fired = []
        thread = audio_io.listen_for_barge_in(vad, fired.append, stop)
        thread.join(timeout=5)
        assert fired == []

    def test_stop_event_ends_the_thread(self, monkeypatch):
        fake = make_fake_sd(frames=[np.zeros((512, 1), dtype="float32")])
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([False] * 100000)
        stop = threading.Event()
        thread = audio_io.listen_for_barge_in(vad, lambda prefix: None, stop)
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_mic_failure_does_not_raise_into_the_caller(self, monkeypatch):
        """A dead mic must not kill Nero's reply — the turn matters more."""
        fake = make_fake_sd(frames=[np.zeros((512, 1), dtype="float32")],
                            input_error_msg="Device unavailable")
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        stop = threading.Event()
        errors = []
        thread = audio_io.listen_for_barge_in(
            ScriptedVAD([]), lambda prefix: None, stop, on_error=errors.append
        )
        thread.join(timeout=5)
        assert len(errors) == 1

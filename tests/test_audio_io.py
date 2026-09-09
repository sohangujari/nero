import asyncio
import io
import threading
import sys
import time
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
    """Mimics sd.InputStream: hands out fixed frames on each read().

    Supports both shapes the module uses: the context manager
    `record_until_enter` opens per recording, and the explicit
    start/stop/close AudioSource keeps open for the session.
    """

    def __init__(self, frames, raise_on_start=None, overflow_after=None, **kwargs):
        self._frames = frames
        self._raise = raise_on_start
        self._overflow_after = overflow_after
        self._i = 0
        self.read_count = 0
        self.started = False
        self.closed = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        return False

    def start(self):
        if self._raise:
            raise self._raise
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def read(self, n):
        self.read_count += 1
        block = self._frames[self._i]
        self._i = (self._i + 1) % len(self._frames)
        overflowed = (
            self._overflow_after is not None and self.read_count > self._overflow_after
        )
        return block, overflowed


def make_fake_sd(frames=None, input_error_msg=None, play_error=None, overflow_after=None):
    fake = types.ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    fake.PortAudioError = PortAudioError
    frames = frames if frames is not None else [np.zeros((160, 1), dtype="float32")]
    input_error = PortAudioError(input_error_msg) if input_error_msg else None

    fake.streams = []

    def InputStream(**kwargs):
        stream = FakeInputStream(
            frames, raise_on_start=input_error, overflow_after=overflow_after, **kwargs
        )
        fake.streams.append(stream)
        return stream

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

    fake.out_streams = []
    fake.on_write = None
    fake.output_error = None

    def OutputStream(**kwargs):
        stream = FakeOutputStream(fake, **kwargs)
        fake.out_streams.append(stream)
        return stream

    fake.OutputStream = OutputStream
    return fake


class FakeOutputStream:
    """Mimics sd.OutputStream: records every write so block-level cuts show up."""

    def __init__(self, module, **kwargs):
        self._module = module
        self.kwargs = kwargs
        self.writes = []
        self.started = self.closed = self.aborted = False

    def start(self):
        if self._module.output_error:
            raise self._module.output_error
        self.started = True

    def write(self, data):
        self.writes.append(np.asarray(data).copy())
        if self._module.on_write is not None:
            self._module.on_write(self)

    def abort(self):
        self.aborted = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


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

    def test_prefix_followup_wait_uses_the_short_window(self, monkeypatch):
        """A barge-in handoff (prefix given) with no real speech following
        must give up after the short BARGE_IN_FOLLOWUP_SECONDS window, not
        the long wait_for_speech_seconds one -- otherwise a spurious trigger
        (Nero hearing itself) strands the session for the full 30s."""
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([False] * 5000)
        prefix = np.full(512, 0.7, dtype=np.float32)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()),
            vad,
            wait_for_speech_seconds=30,
            prefix=prefix,
        )
        assert audio.size == 0
        expected_frames = (
            round(
                audio_io.BARGE_IN_FOLLOWUP_SECONDS
                * audio_io.RECORD_SAMPLE_RATE
                / audio_io.VAD_FRAME
            )
            + 1
        )
        assert fake.streams[-1].read_count == expected_frames

    def test_no_prefix_wait_still_uses_wait_for_speech_seconds(self, monkeypatch):
        """Without a prefix (a deliberate Enter press), the long
        wait_for_speech_seconds window must apply unchanged."""
        fake = make_fake_sd(frames=vad_frames(1))
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        vad = ScriptedVAD([False] * 5000)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()),
            vad,
            wait_for_speech_seconds=5,
        )
        assert audio.size == 0
        expected_frames = round(5 * audio_io.RECORD_SAMPLE_RATE / audio_io.VAD_FRAME) + 1
        assert fake.streams[-1].read_count == expected_frames

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

    def test_the_meter_shows_how_loud_the_frame_was(self, monkeypatch):
        """A recorder's strip is driven by the microphone, not by a canned
        animation — the point is seeing your own voice in it."""
        quiet = np.zeros(audio_io.VAD_FRAME, dtype=np.float32)
        loud = np.full(audio_io.VAD_FRAME, 0.25, dtype=np.float32)
        assert audio_io._frame_level(quiet) == 0
        assert audio_io._frame_level(loud) == len(audio_io._BARS) - 1

    def test_ordinary_speech_uses_the_middle_of_the_range(self, monkeypatch):
        """A linear map pins everything you actually say to the bottom two bars
        and only moves the top half when you shout."""
        levels = [audio_io._level(rms) for rms in (0.01, 0.03, 0.06, 0.10, 0.16)]
        assert levels == sorted(levels)
        assert 1 <= levels[0] and levels[-1] >= 5

    def test_a_frame_that_cannot_be_measured_is_a_flat_bar_not_a_crash(self):
        """Cosmetic code may never cost a turn."""
        assert audio_io._frame_level("not audio") == 0
        assert audio_io._frame_level(None) == 0

    def test_the_strip_is_one_bar_per_frame_newest_last(self):
        from collections import deque

        window = deque([(0, False), (7, True), (3, True)], maxlen=8)
        assert audio_io._indicator_text(window).plain == "  " + "▁█▄"

    def test_loudness_and_speech_are_shown_separately(self):
        """Height is how loud; colour is whether the VAD counted it. A loud but
        dim strip means it is hearing the room, not you."""
        from collections import deque

        text = audio_io._indicator_text(deque([(7, False), (7, True)]))
        styles = [str(span.style) for span in text.spans]
        assert styles == ["dim", "bold red"]

    def test_a_level_never_indexes_past_the_bars(self):
        for rms in (0.0, 0.25, 1.0, 99.0, -1.0):
            assert 0 <= audio_io._level(rms) < len(audio_io._BARS)

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


class TestOutputIsBuiltinSpeakers:
    """Name heuristic used to auto-suppress barge-in on built-in speakers."""

    @staticmethod
    def _fake_sd_with_output(name):
        fake = types.ModuleType("sounddevice")

        def query_devices(kind=None):
            assert kind == "output"
            return {"name": name}

        fake.query_devices = query_devices
        return fake

    @pytest.mark.parametrize(
        "name",
        [
            "MacBook Air Speakers",
            "MacBook Pro Speakers",
            "iMac Speakers",
            "Built-in Output",
            "Built-in Audio",
            "Internal Speakers",
            "Speakers (Realtek High Definition Audio)",
            # case-insensitive match
            "macbook air speakers",
        ],
    )
    def test_matches_known_builtin_shapes(self, monkeypatch, name):
        install(monkeypatch, self._fake_sd_with_output(name))
        assert audio_io.output_is_builtin_speakers() is True

    @pytest.mark.parametrize(
        "name",
        ["AirPods Pro", "External Headphones", "USB Audio Device"],
    )
    def test_does_not_match_headphone_like_names(self, monkeypatch, name):
        install(monkeypatch, self._fake_sd_with_output(name))
        assert audio_io.output_is_builtin_speakers() is False

    def test_fails_open_when_device_enumeration_raises(self, monkeypatch):
        fake = types.ModuleType("sounddevice")

        def query_devices(kind=None):
            raise RuntimeError("no default output device")

        fake.query_devices = query_devices
        install(monkeypatch, fake)
        assert audio_io.output_is_builtin_speakers() is False

    def test_fails_open_when_sounddevice_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        assert audio_io.output_is_builtin_speakers() is False


# --- Pipelined playback: synthesis must overlap the speaker ---
class RecordingTTS:
    """Notes when each sentence is synthesized, so overlap is observable."""

    def __init__(self):
        self.synthesized = []

    async def synthesize_stream(self, text_stream):
        async for sentence in text_stream:
            self.synthesized.append(sentence)
            yield np.full(4, len(sentence), dtype=np.float32)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestPipelinedPlayback:
    def test_synthesis_runs_ahead_of_playback(self):
        """The defect this whole rewrite exists for.

        The old single-threaded `async for audio in synth(): await play(audio)`
        could not pull sentence N+1 out of the engine until sentence N had
        finished playing, so every sentence boundary cost a full synthesis
        (0.8-5.9s measured) of dead air. With playback parked on chunk 1, the
        remaining sentences must already be synthesized.
        """
        tts = RecordingTTS()
        started = threading.Event()
        release = threading.Event()

        def blocking_play(audio, sr):
            started.set()
            release.wait(timeout=5)

        player = Player(tts, 24000, play_fn=blocking_play)
        player.start()
        for sentence in ("One.", "Two.", "Three."):
            player.enqueue(sentence)
        player.close()
        assert started.wait(timeout=5)
        assert _wait_until(lambda: tts.synthesized == ["One.", "Two.", "Three."])
        # ...while only the first has actually been spoken.
        assert player.spoken_text() == "One."
        release.set()
        player.join(timeout=5)

    def test_prefetch_is_bounded(self):
        """Backpressure: a long reply must not synthesize itself into memory.

        Playback holds chunk 1, the queue holds PREFETCH_CHUNKS more, and the
        synthesis thread parks handing over one more — so the engine runs at
        most PREFETCH_CHUNKS + 2 sentences ahead, never all ten.
        """
        tts = RecordingTTS()
        started = threading.Event()
        release = threading.Event()

        def blocking_play(audio, sr):
            started.set()
            release.wait(timeout=5)

        player = Player(tts, 24000, play_fn=blocking_play)
        player.start()
        for i in range(10):
            player.enqueue(f"Sentence {i}.")
        player.close()
        assert started.wait(timeout=5)
        ceiling = Player.PREFETCH_CHUNKS + 2
        assert _wait_until(lambda: len(tts.synthesized) >= ceiling)
        time.sleep(0.3)  # give an unbounded implementation room to run away
        assert len(tts.synthesized) == ceiling
        release.set()
        player.stop_now()
        player.shutdown()

    def test_one_output_stream_serves_every_chunk(self, monkeypatch):
        """Per-sentence `sd.play()` measured ~130ms of device open/teardown.
        One reused stream costs 10ms once."""
        fake = make_fake_sd()
        install(monkeypatch, fake)
        player = Player(SentenceTTS(), 24000)
        player.start()
        for sentence in ("One.", "Two.", "Three."):
            player.enqueue(sentence)
        player.close()
        player.join(timeout=5)
        player.shutdown()
        assert len(fake.out_streams) == 1
        assert fake.out_streams[0].kwargs["samplerate"] == 24000
        assert sum(w.size for w in fake.out_streams[0].writes) == 3 * 4

    def test_stop_now_cuts_inside_a_chunk(self, monkeypatch):
        """Barge-in must not wait out the sentence already handed to the device."""
        fake = make_fake_sd()
        install(monkeypatch, fake)
        blocks = 10
        big = np.ones(audio_io._StreamWriter._WRITE_BLOCK * blocks, dtype=np.float32)

        class OneBigChunkTTS:
            async def synthesize_stream(self, text_stream):
                async for _sentence in text_stream:
                    yield big

        player = Player(OneBigChunkTTS(), 24000)

        def cut_after_two(stream):
            if len(stream.writes) == 2:
                player.stop_now()

        fake.on_write = cut_after_two
        player.start()
        player.enqueue("A very long sentence.")
        player.close()
        player.join(timeout=5)
        stream = fake.out_streams[0]
        assert len(stream.writes) == 2 < blocks
        assert stream.aborted

    def test_playback_device_failure_surfaces_on_join(self, monkeypatch):
        fake = make_fake_sd()
        fake.output_error = RuntimeError("no output device")
        install(monkeypatch, fake)
        player = Player(SentenceTTS(), 24000)
        player.start()
        player.enqueue("One.")
        player.close()
        with pytest.raises(PlaybackError, match="no output device"):
            player.join(timeout=5)

    def test_shutdown_releases_the_output_device(self, monkeypatch):
        fake = make_fake_sd()
        install(monkeypatch, fake)
        player = Player(SentenceTTS(), 24000)
        player.start()
        player.enqueue("One.")
        player.close()
        player.join(timeout=5)
        player.shutdown()
        assert fake.out_streams[0].closed


# --- Shared microphone + pre-roll padding ---
class TestAudioSource:
    def test_one_stream_serves_many_recordings(self, monkeypatch):
        """Opening an InputStream measured 112ms; a turn used to pay it twice."""
        fake = make_fake_sd(frames=vad_frames(1))
        install(monkeypatch, fake)
        source = audio_io.AudioSource()
        console = Console(file=io.StringIO())
        for _ in range(2):
            audio_io.record_until_silence(
                console, ScriptedVAD([True] + [False] * 25), silence_ms=800, source=source
            )
        assert len(fake.streams) == 1
        source.close()
        assert fake.streams[0].closed

    def test_without_a_shared_source_each_call_opens_its_own(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1))
        install(monkeypatch, fake)
        console = Console(file=io.StringIO())
        for _ in range(2):
            audio_io.record_until_silence(
                console, ScriptedVAD([True] + [False] * 25), silence_ms=800
            )
        assert len(fake.streams) == 2
        assert all(stream.closed for stream in fake.streams)

    def test_permission_error_is_translated(self, monkeypatch):
        fake = make_fake_sd(frames=vad_frames(1), input_error_msg="Permission denied")
        install(monkeypatch, fake)
        with pytest.raises(MicPermissionError):
            audio_io.AudioSource().read_frame()

    def test_overflow_is_logged_not_raised(self, monkeypatch, caplog):
        """PortAudio's overflow flag was discarded; dropped input then showed
        up only as a mysteriously garbled transcript."""
        fake = make_fake_sd(frames=vad_frames(1), overflow_after=0)
        install(monkeypatch, fake)
        source = audio_io.AudioSource()
        with caplog.at_level("DEBUG", logger="nero.voice.audio"):
            frame = source.read_frame()
        assert frame.shape == (512,)
        assert "overflow" in caplog.text.lower()

    def test_barge_in_monitor_reuses_the_shared_stream(self, monkeypatch):
        fake = make_fake_sd(frames=[np.full((512, 1), 0.4, dtype="float32")])
        install(monkeypatch, fake)
        source = audio_io.AudioSource()
        source.read_frame()  # opens it
        stop = threading.Event()
        fired = []
        thread = audio_io.listen_for_barge_in(
            ScriptedVAD([True] * 20), fired.append, stop, source=source
        )
        thread.join(timeout=5)
        assert len(fired) == 1
        assert len(fake.streams) == 1
        assert not fake.streams[0].closed, "a shared mic outlives one monitor"


class TestPreroll:
    def test_frames_before_speech_onset_are_kept(self, monkeypatch):
        """Silero fires on the frame where speech is already audible, so the
        frame that trips it usually holds the first phoneme. Dropping what came
        before clipped word onsets."""
        fake = make_fake_sd(frames=vad_frames(1))
        install(monkeypatch, fake)
        vad = ScriptedVAD([False] * 15 + [True] + [False] * 25)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, silence_ms=800
        )
        kept = audio.size // 512
        # 1 speech + 25 trailing silence + PREROLL_FRAMES of lead-in.
        assert kept == 26 + audio_io.PREROLL_FRAMES

    def test_preroll_is_bounded(self, monkeypatch):
        """A long silent wait must not accumulate minutes of lead-in."""
        fake = make_fake_sd(frames=vad_frames(1))
        install(monkeypatch, fake)
        vad = ScriptedVAD([False] * 300 + [True] + [False] * 25)
        audio = audio_io.record_until_silence(
            Console(file=io.StringIO()), vad, silence_ms=800, wait_for_speech_seconds=30
        )
        assert audio.size // 512 == 26 + audio_io.PREROLL_FRAMES


class TestEnterInterrupt:
    """The keyboard is the interrupt that works on built-in speakers, where the
    acoustic monitor has to stay off or Nero interrupts itself."""

    def watcher(self):
        """A fake tty backed by an os.pipe, so select() sees it for real."""
        import os

        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "r")
        stream.isatty = lambda: True
        return stream, write_fd

    def test_enter_fires_the_callback(self):
        import os

        stream, write_fd = self.watcher()
        fired = threading.Event()
        stop = threading.Event()
        thread = audio_io.watch_for_enter(fired.set, stop, stream=stream)
        os.write(write_fd, b"\n")
        assert fired.wait(2), "Enter did not reach the callback"
        stop.set()
        thread.join(timeout=2)
        os.close(write_fd)

    def test_nothing_fires_without_a_keypress(self):
        import os

        stream, write_fd = self.watcher()
        fired = threading.Event()
        stop = threading.Event()
        thread = audio_io.watch_for_enter(fired.set, stop, stream=stream)
        time.sleep(0.2)
        assert not fired.is_set()
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        os.close(write_fd)

    def test_the_watcher_stops_when_the_reply_ends(self):
        """A thread parked in a blocking read cannot be cancelled — it would
        still be sitting there and would swallow the Enter that starts the
        *next* turn. This polls, so the stop flag always lands."""
        import os

        stream, write_fd = self.watcher()
        stop = threading.Event()
        thread = audio_io.watch_for_enter(lambda: None, stop, stream=stream)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        os.close(write_fd)

    def test_a_keypress_after_the_stop_flag_is_ignored(self):
        """The reply finished on its own; that Enter belongs to the next turn."""
        import os

        stream, write_fd = self.watcher()
        fired = []
        stop = threading.Event()
        thread = audio_io.watch_for_enter(lambda: fired.append(1), stop, stream=stream)
        stop.set()
        os.write(write_fd, b"\n")
        thread.join(timeout=2)
        assert fired == []
        os.close(write_fd)

    def test_input_typed_before_the_watch_is_discarded(self):
        """The bug this exists to stop: Enter pressed while Nero was *listening*
        sits in the terminal buffer, gets read the instant the next reply
        starts, and kills that reply before a word of it is spoken. The user
        presses Enter again to retry, which stocks the buffer for the turn after
        that — it never recovers on its own."""
        import os

        stream, write_fd = self.watcher()
        os.write(write_fd, b"\n")
        time.sleep(0.1)

        fired = threading.Event()
        stop = threading.Event()
        thread = audio_io.watch_for_enter(fired.set, stop, stream=stream)
        assert not fired.wait(0.3), "a stale keypress interrupted the reply"

        # A real one, pressed while Nero is speaking, still works.
        os.write(write_fd, b"\n")
        assert fired.wait(2)
        stop.set()
        thread.join(timeout=2)
        os.close(write_fd)

    def test_several_stale_keypresses_are_all_discarded(self):
        """Someone who pressed Enter three times waiting for it to work must not
        have three interrupts queued up."""
        import os

        stream, write_fd = self.watcher()
        os.write(write_fd, b"\n\n\n")
        time.sleep(0.1)
        fired = threading.Event()
        stop = threading.Event()
        thread = audio_io.watch_for_enter(fired.set, stop, stream=stream)
        assert not fired.wait(0.3)
        stop.set()
        thread.join(timeout=2)
        os.close(write_fd)

    def test_draining_a_stream_with_no_descriptor_is_harmless(self):
        assert audio_io.watch_for_enter(lambda: None, threading.Event(), stream=io.StringIO()) is None

    def test_a_non_terminal_stdin_starts_nothing(self):
        """Piped input and test runs must be untouched."""
        assert audio_io.watch_for_enter(lambda: None, threading.Event(), stream=io.StringIO()) is None

    def test_a_handler_that_raises_does_not_kill_the_thread_uncaught(self):
        """An interrupt must never crash the turn it is interrupting."""
        import os

        stream, write_fd = self.watcher()
        stop = threading.Event()
        def boom():
            raise RuntimeError("nope")
        thread = audio_io.watch_for_enter(boom, stop, stream=stream)
        os.write(write_fd, b"\n")
        thread.join(timeout=2)
        assert not thread.is_alive()
        stop.set()
        os.close(write_fd)

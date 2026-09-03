from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections import deque
from collections.abc import Callable

from nero.voice.errors import (
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
    VoiceDependencyError,
)

logger = logging.getLogger("nero.voice.audio")

RECORD_SAMPLE_RATE = 16000
_BLOCK = 1600  # 0.1s blocks at 16 kHz
VAD_FRAME = 512  # 32 ms at 16 kHz — the only size silero accepts at this rate

_INDICATOR_FRAMES = 20  # ~20 VAD frames * 32ms = ~0.6s of rolling history shown


def _indicator_text(window: "deque[bool]"):
    from rich.text import Text

    text = Text("  ")
    for speech in window:
        text.append("●" if speech else "·", style="bold red" if speech else "dim")
    return text


def _start_indicator(console, window: "deque[bool]"):
    """Best-effort live speech-activity strip; None if not a real terminal.

    Purely cosmetic — every call site guards against this raising, because a
    rendering hiccup must never cost a turn. In non-terminal contexts (tests,
    piped output) this returns None and callers fall back to the plain
    "Listening" line already printed, so nothing new is emitted there.
    """
    if not console.is_terminal:
        return None
    try:
        from rich.live import Live

        live = Live(_indicator_text(window), console=console, refresh_per_second=15, transient=True)
        live.start()
        return live
    except Exception:  # noqa: BLE001 — cosmetic only, must never break recording
        return None


def _update_indicator(live, window: "deque[bool]", speech: bool) -> None:
    if live is None:
        return
    window.append(speech)
    try:
        live.update(_indicator_text(window))
    except Exception:  # noqa: BLE001 — cosmetic only, must never break recording
        pass


def _stop_indicator(live) -> None:
    if live is None:
        return
    try:
        live.stop()
    except Exception:  # noqa: BLE001 — cosmetic only, must never break recording
        pass


def _import_sd():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceDependencyError(
            'Voice mode needs extra packages. Install them with: pip install "nero[voice]"'
        ) from exc
    return sd


# ~320 ms of audio kept from BEFORE the VAD called it speech. Silero fires on
# the frame where speech is already audible, so the frame that tips it usually
# contains the first phoneme, and everything before it was being thrown away —
# clipped word onsets that whisper then has to guess at. Gemini's Live API
# calls the same idea `prefixPaddingMs`.
PREROLL_FRAMES = 10


class AudioSource:
    """One microphone stream, opened once and reused for the whole session.

    Opening an `InputStream` measured 112 ms on this machine, and a turn used
    to open two — one for recording, one for the barge-in monitor — which also
    meant two handles briefly contending for the same device. Kept open, a turn
    pays nothing.

    Blocking reads, one consumer at a time: the voice loop stops the barge-in
    monitor before recording starts, so the recorder and the monitor never read
    concurrently.
    """

    def __init__(self, sample_rate: int = RECORD_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._stream = None

    def read_frame(self):
        """One `VAD_FRAME`-sample mono float32 frame. Raises a mic error, never PortAudio's."""
        import numpy as np

        stream = self._open()
        try:
            data, overflowed = stream.read(VAD_FRAME)
        except Exception as exc:  # noqa: BLE001 — every device failure gets the friendly form
            raise _mic_error(exc) from exc
        if overflowed:
            # PortAudio dropped input because we did not read fast enough. This
            # flag was discarded before; unlogged, the loss only ever showed up
            # later as a mysteriously garbled transcript.
            logger.debug("microphone input overflowed - samples were dropped")
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def _open(self):
        if self._stream is not None:
            return self._stream
        sd = _import_sd()
        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate, channels=1, dtype="float32"
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001
            raise _mic_error(exc) from exc
        self._stream = stream
        return stream

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001 — releasing the mic must never raise
            logger.debug("input stream close failed", exc_info=True)


def record_until_enter(console, input_fn: Callable[[], str]):
    """Record mono float32 audio at 16 kHz until `input_fn` returns (Enter).

    Prints the recording banner, opens the default input device, and captures
    blocks on a background thread while the main thread blocks on `input_fn`.
    """
    import numpy as np

    sd = _import_sd()
    console.print("[bold red]●[/bold red] Recording — press Enter to stop")
    blocks: list = []
    errors: list[Exception] = []
    stop = threading.Event()

    def capture():
        try:
            with sd.InputStream(
                samplerate=RECORD_SAMPLE_RATE, channels=1, dtype="float32"
            ) as stream:
                while not stop.is_set():
                    data, _ = stream.read(_BLOCK)
                    blocks.append(data.copy())
        except sd.PortAudioError as exc:
            errors.append(exc)

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    try:
        input_fn()  # blocks until Enter
    finally:
        # Must run even on Ctrl+C, or the capture thread keeps reading and the
        # PortAudio input stream stays open after we've "exited".
        stop.set()
        worker.join(timeout=2)

    if errors:
        raise _mic_error(errors[0])
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks, axis=0).reshape(-1).astype(np.float32)


def record_until_silence(
    console,
    vad,
    *,
    silence_ms: int = 800,
    max_utterance_seconds: int = 180,
    wait_for_speech_seconds: int = 30,
    prefix=None,
    source: "AudioSource | None" = None,
) -> "np.ndarray":
    """Record until the speaker stops, using VAD rather than a keypress.

    Returns mono float32 at 16 kHz, empty if speech never started. `prefix` is
    prepended verbatim — barge-in passes the audio it already buffered so the
    interrupting word is not clipped. Exception: if no real speech follows
    `prefix`, `prefix` is discarded and empty audio is returned. This matters
    because barge-in can fire on Nero hearing its own voice through the
    speakers; without this, a spurious trigger with no real user speech would
    hand Nero's own words back to the STT engine as if the user had said
    them. When `prefix` is given, the wait uses `BARGE_IN_FOLLOWUP_SECONDS`
    instead of `wait_for_speech_seconds` — see that constant's comment.

    `source` is the session's shared microphone; without one a throwaway is
    opened and released for this call.
    """
    import numpy as np

    own_source = source is None
    source = source or AudioSource()
    vad.reset()
    console.print("[bold red]●[/bold red] Listening — I'll stop when you do")

    blocks: list = list([] if prefix is None else [np.asarray(prefix, dtype=np.float32).reshape(-1)])
    silent_needed = max(1, round(silence_ms / 1000 * RECORD_SAMPLE_RATE / VAD_FRAME))
    max_frames = round(max_utterance_seconds * RECORD_SAMPLE_RATE / VAD_FRAME)
    # Other half of the discard-on-timeout policy below: when a prefix is
    # supplied the caller (the barge-in monitor) already detected sustained
    # speech moments ago, so this is only confirming the user is still
    # talking, not waiting out a fresh user-initiated pause. Use the short
    # follow-up window instead of the long one, or a spurious trigger (Nero
    # hearing itself) strands the session for the full 30s.
    wait_seconds = BARGE_IN_FOLLOWUP_SECONDS if prefix is not None else wait_for_speech_seconds
    wait_frames = round(wait_seconds * RECORD_SAMPLE_RATE / VAD_FRAME)

    started = False
    silent_run = 0
    preroll: deque = deque(maxlen=PREROLL_FRAMES)
    window: deque = deque(maxlen=_INDICATOR_FRAMES)
    live = _start_indicator(console, window)
    try:
        for index in range(max_frames):
            frame = source.read_frame()
            speech = vad.is_speech(frame)
            _update_indicator(live, window, speech)
            if speech:
                if not started:
                    # The frame that tripped the VAD is rarely the first frame
                    # of the word; hand whisper what came just before it.
                    blocks.extend(preroll)
                    preroll.clear()
                started = True
                silent_run = 0
                blocks.append(frame.copy())
            elif started:
                silent_run += 1
                blocks.append(frame.copy())
                if silent_run >= silent_needed:
                    break
            else:
                preroll.append(frame.copy())
                if index >= wait_frames:
                    # Discard `prefix` here too. A spurious barge-in (Nero
                    # hearing itself) puts Nero's own speech in `prefix`; with
                    # no real speech following it, returning it would feed
                    # Nero's words back to the STT as if the user said them.
                    return np.zeros(0, dtype=np.float32)
    finally:
        _stop_indicator(live)
        if own_source:
            source.close()

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks, axis=0).reshape(-1).astype(np.float32)


# How long speech must persist before it counts as a barge-in. This is the
# self-hearing guard: Nero's own output leaking into the mic produces brief
# blips, not sustained speech. Deliberately a constant, not config — if this
# number needs tuning the heuristic is wrong and the honest fix is
# `voice.barge_in = false`, not a knob.
BARGE_IN_SUSTAINED_MS = 400

# How long `record_until_silence` waits for real speech after a barge-in
# handoff (`prefix` given), instead of `wait_for_speech_seconds`. The caller
# already detected sustained speech moments ago to trigger the handoff, so
# this only needs to confirm the user is still talking -- if they are not,
# the trigger was Nero hearing itself, and waiting the full user-initiated
# window would strand the session for nothing. Deliberately a constant, not
# config, for the same reason as BARGE_IN_SUSTAINED_MS above.
BARGE_IN_FOLLOWUP_SECONDS = 2.0


def listen_for_barge_in(vad, on_detect, stop, on_error=None, source=None) -> threading.Thread:
    """Watch the mic during playback; call `on_detect(prefix)` once on barge-in.

    `prefix` is the audio buffered since speech began, handed forward so the
    interrupting words are not lost when recording takes over.

    `source` is the session's shared microphone; without one a throwaway is
    opened and released for this watch.

    Never raises into the caller: a microphone failure disables barge-in for the
    session but must not end Nero's reply.
    """
    import numpy as np

    needed = max(1, round(BARGE_IN_SUSTAINED_MS / 1000 * RECORD_SAMPLE_RATE / VAD_FRAME))
    own_source = source is None
    mic = source or AudioSource()

    def watch():
        try:
            vad.reset()
            run: list = []
            while not stop.is_set():
                frame = mic.read_frame()
                if vad.is_speech(frame):
                    run.append(frame.copy())
                    if len(run) >= needed:
                        on_detect(np.concatenate(run, axis=0).reshape(-1))
                        return
                else:
                    run.clear()
        except Exception as exc:  # noqa: BLE001 — barge-in is optional, the reply is not
            logger.debug("barge-in monitor stopped", exc_info=True)
            if on_error is not None:
                on_error(exc)
        finally:
            if own_source:
                mic.close()

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


# Name-substring patterns for the default OUTPUT device that indicate
# built-in laptop/desktop speakers, across the platforms this project
# supports. Matched case-insensitively as substrings of the device name.
_BUILTIN_SPEAKER_PATTERNS = (
    "macbook air speakers",
    "macbook pro speakers",
    "imac speakers",
    "built-in output",
    "built-in audio",
    "internal speakers",
    "speakers (realtek",
)


def output_is_builtin_speakers() -> bool:
    """Best-effort guess: is the default OUTPUT device built-in speakers?

    A name heuristic, not a real capability check — it knows a handful of
    English device-name shapes (see `_BUILTIN_SPEAKER_PATTERNS`) and nothing
    else. It will miss non-English device names, external speakers with a
    generic name, and HDMI/monitor audio pretending to be something else.
    `voice.force_barge_in` is the escape hatch for anyone this heuristic
    gets wrong.

    Fails OPEN: on any error (no sounddevice, no default device, headless
    machine, whatever) this returns False so barge-in stays enabled rather
    than silently disabling a feature the user asked for.
    """
    try:
        sd = _import_sd()
        device = sd.query_devices(kind="output")
        name = str(device["name"]).lower()
    except Exception:  # noqa: BLE001 — any failure must fail open, not raise
        return False
    return any(pattern in name for pattern in _BUILTIN_SPEAKER_PATTERNS)


def _mic_error(exc: Exception) -> Exception:
    if "permission" in str(exc).lower():
        return MicPermissionError(
            "Microphone access was denied by the OS. On macOS, grant it in "
            "System Settings → Privacy & Security → Microphone, then retry."
        )
    return MicUnavailableError(
        "No usable microphone was found. Check that an input device is connected "
        "and accessible."
    )


def play(audio, sample_rate: int) -> None:
    sd = _import_sd()
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except VoiceDependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 — any device failure is a playback error
        raise PlaybackError(f"Audio playback failed: {exc}") from exc


class _StreamWriter:
    """Writes audio chunks to ONE output stream, reused for the whole turn.

    `sd.play()/sd.wait()` measured ~130 ms of device open + teardown per call
    on this machine. Paid per sentence, that is audible dead air between every
    sentence Nero speaks. One `OutputStream`, opened on the first chunk and
    kept, costs 10 ms once.

    Chunks are written in `_WRITE_BLOCK` slices so a barge-in cuts in within a
    block instead of at the end of the current sentence.
    """

    _WRITE_BLOCK = 2048  # ~85 ms at 24 kHz — the granularity stop_now() can cut at

    def __init__(self, stop: threading.Event):
        self._stop = stop
        self._stream = None
        self._rate: int | None = None

    def __call__(self, audio, sample_rate: int) -> None:
        import numpy as np

        stream = self._open(sample_rate)
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        try:
            for start in range(0, data.size, self._WRITE_BLOCK):
                if self._stop.is_set():
                    return
                stream.write(data[start : start + self._WRITE_BLOCK])
        except Exception as exc:  # noqa: BLE001 — any device failure is a playback error
            if self._stop.is_set():
                # An abort() from stop_now() races the write; that is the
                # interrupt working, not a fault to report.
                return
            raise PlaybackError(f"Audio playback failed: {exc}") from exc

    def _open(self, sample_rate: int):
        if self._stream is not None and self._rate == sample_rate:
            return self._stream
        self.close()
        sd = _import_sd()
        try:
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            stream.start()
        except VoiceDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PlaybackError(f"Audio playback failed: {exc}") from exc
        self._stream, self._rate = stream, sample_rate
        return stream

    def abort(self) -> None:
        """Drop audio already handed to the device — barge-in, not end of turn.

        The stream is closed rather than left stopped: writing to an aborted
        stream is not portable, and re-opening costs 10 ms.
        """
        stream = self._stream
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:  # noqa: BLE001 — never let a device quirk mask the interrupt
            logger.debug("output stream abort failed", exc_info=True)
        self.close()

    def close(self) -> None:
        stream, self._stream, self._rate = self._stream, None, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001 — teardown must never raise into a turn
            logger.debug("output stream close failed", exc_info=True)


class Player:
    """Speaks sentences as they arrive, synthesizing ahead of the speaker.

    Two threads, deliberately: a synthesis thread drives the TTS engine and
    pushes finished audio onto a small bounded queue, and a playback thread
    writes that queue to the device. They overlap, which is the whole point —
    the previous single-threaded `async for ... : await play(...)` shape could
    not start synthesizing sentence N+1 until sentence N had finished playing,
    so every sentence boundary cost a full synthesis (0.8-5.9 s measured) of
    silence.

    The queue is bounded: a long reply must not synthesize itself entirely into
    memory ahead of the speaker. Measured synthesis RTF is ~0.75, so one chunk
    of lead sustains continuous speech and three absorbs a long sentence
    arriving behind a short one.

    `close` signals end-of-turn; `join` drains and re-raises any error from
    either thread.
    """

    _SENTINEL = None

    _POLL = 0.2  # seconds; bounds how long a thread sits in a queue operation

    PREFETCH_CHUNKS = 3

    def __init__(self, tts, sample_rate: int, play_fn: Callable | None = None):
        self._tts = tts
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._writer = _StreamWriter(self._stop) if play_fn is None else None
        self._play = play_fn or self._writer
        self._queue: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue(maxsize=self.PREFETCH_CHUNKS)
        self._thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._yielded: list[str] = []
        self._spoken = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="nero-tts")
        self._play_thread = threading.Thread(
            target=self._playback, daemon=True, name="nero-playback"
        )
        self._thread.start()
        self._play_thread.start()

    def enqueue(self, sentence: str) -> None:
        self._queue.put(sentence)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)

    def join(self, timeout: float | None = None) -> None:
        for thread in (self._thread, self._play_thread):
            if thread is not None:
                thread.join(timeout)
        if self._error is not None:
            raise self._error

    def spoken_text(self) -> str:
        """The sentences that actually reached the speaker, in order.

        A sentence interrupted halfway counts as spoken (sentence-level
        granularity is the accepted ceiling).
        """
        return " ".join(self._yielded[: self._spoken])

    def stop_now(self) -> None:
        """Cut playback off immediately — barge-in, not end-of-turn.

        `shutdown()` drains politely; this stops the device mid-chunk AND drops
        audio already synthesized but not yet spoken, which is the difference
        between interrupting Nero and waiting out the sentences it has queued.
        """
        self._stop.set()
        try:
            sd = _import_sd()
            sd.stop()
        except Exception:  # noqa: BLE001 — never let a device quirk mask the interrupt
            logger.debug("sd.stop() failed during barge-in", exc_info=True)
        if self._writer is not None:
            self._writer.abort()
        self._drop_pending_audio()
        self._queue.put(self._SENTINEL)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop both threads and wait, bounded. Idempotent and safe to call from
        a `finally` even if the turn already ended normally.

        Without this, an aborted turn (Ctrl+C, auth/network error) leaves a
        thread parked on an empty queue forever — which also parks an asyncio
        default-executor worker, and ThreadPoolExecutor's atexit hook then joins
        that worker at interpreter shutdown, hanging the whole process on exit.
        """
        self._stop.set()
        self._queue.put(self._SENTINEL)
        self._drop_pending_audio()
        for thread in (self._thread, self._play_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout)
                if thread.is_alive():
                    logger.debug("%s did not stop within %.1fs", thread.name, timeout)
        if self._writer is not None:
            self._writer.close()

    def _drop_pending_audio(self) -> None:
        """Discard synthesized-but-unspoken chunks and unpark the playback thread."""
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                break

    def _next_item(self):
        """Blocking queue read that always returns within `_POLL` seconds.

        A plain `queue.get()` would park an executor worker forever when a turn
        is aborted; polling lets the thread notice `_stop` and unwind so the
        process can exit cleanly.
        """
        while True:
            if self._stop.is_set():
                return self._SENTINEL
            try:
                return self._queue.get(timeout=self._POLL)
            except queue.Empty:
                continue

    def _put_audio(self, item) -> None:
        """Hand a synthesized chunk to the playback thread, honouring backpressure.

        Bounded on purpose (see the class docstring); the poll is what lets a
        `stop_now()` unpark a synthesis thread blocked on a full queue.
        """
        while not self._stop.is_set():
            try:
                self._audio.put(item, timeout=self._POLL)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            asyncio.run(self._consume())
        except Exception as exc:  # noqa: BLE001 — surfaced on join()
            self._error = exc
        finally:
            self._put_audio(self._SENTINEL)

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()

        async def sentences():
            while True:
                item = await loop.run_in_executor(None, self._next_item)
                if item is self._SENTINEL:
                    return
                self._yielded.append(item)
                yield item

        async for audio in self._tts.synthesize_stream(sentences()):
            if self._stop.is_set():
                # An engine that yields ahead (batches/prefetches, like a real
                # streaming TTS could) would otherwise keep handing over every
                # chunk it already produced after stop_now() fires.
                break
            await loop.run_in_executor(None, self._put_audio, audio)

    def _playback(self) -> None:
        while True:
            if self._stop.is_set():
                return
            try:
                audio = self._audio.get(timeout=self._POLL)
            except queue.Empty:
                continue
            if audio is self._SENTINEL:
                return
            if self._stop.is_set():
                return
            # Counted before playing, so `spoken_text()` includes the sentence
            # currently coming out of the speaker — that is what the user heard.
            self._spoken += 1
            try:
                self._play(audio, self._sample_rate)
            except Exception as exc:  # noqa: BLE001 — surfaced on join()
                if self._error is None:
                    self._error = exc
                return

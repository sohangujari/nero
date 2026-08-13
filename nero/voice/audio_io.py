from __future__ import annotations

import asyncio
import logging  # DEBUG(hang)
import queue
import threading
import time  # DEBUG(hang)
from collections import deque
from collections.abc import Callable

from nero.voice.errors import (
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
    VoiceDependencyError,
)

logger = logging.getLogger("nero.voice.audio")  # DEBUG(hang)

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
    """
    import numpy as np

    sd = _import_sd()
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
    window: deque = deque(maxlen=_INDICATOR_FRAMES)
    live = _start_indicator(console, window)
    try:
        with sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            for index in range(max_frames):
                data, _ = stream.read(VAD_FRAME)
                frame = np.asarray(data, dtype=np.float32).reshape(-1)
                speech = vad.is_speech(frame)
                _update_indicator(live, window, speech)
                if speech:
                    started = True
                    silent_run = 0
                    blocks.append(frame.copy())
                elif started:
                    silent_run += 1
                    blocks.append(frame.copy())
                    if silent_run >= silent_needed:
                        break
                elif index >= wait_frames:
                    # Discard `prefix` here too. A spurious barge-in (Nero
                    # hearing itself) puts Nero's own speech in `prefix`; with
                    # no real speech following it, returning it would feed
                    # Nero's words back to the STT as if the user said them.
                    return np.zeros(0, dtype=np.float32)
    except sd.PortAudioError as exc:
        raise _mic_error(exc) from exc
    finally:
        _stop_indicator(live)

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


def listen_for_barge_in(vad, on_detect, stop, on_error=None) -> threading.Thread:
    """Watch the mic during playback; call `on_detect(prefix)` once on barge-in.

    `prefix` is the audio buffered since speech began, handed forward so the
    interrupting words are not lost when recording takes over.

    Never raises into the caller: a microphone failure disables barge-in for the
    session but must not end Nero's reply.
    """
    import numpy as np

    needed = max(1, round(BARGE_IN_SUSTAINED_MS / 1000 * RECORD_SAMPLE_RATE / VAD_FRAME))

    def watch():
        try:
            sd = _import_sd()
            vad.reset()
            run: list = []
            with sd.InputStream(
                samplerate=RECORD_SAMPLE_RATE, channels=1, dtype="float32"
            ) as stream:
                while not stop.is_set():
                    data, _ = stream.read(VAD_FRAME)
                    frame = np.asarray(data, dtype=np.float32).reshape(-1)
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

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


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
        # DEBUG(hang) STAGE 6: opening the output stream (blocks until done below)
        logger.debug("STAGE 6: opening playback stream @%dHz", sample_rate)
        _t_play = time.monotonic()  # DEBUG(hang)
        sd.play(audio, sample_rate)
        sd.wait()
        # DEBUG(hang) STAGE 7: playback finished
        logger.debug("STAGE 7: playback finished in %.2fs", time.monotonic() - _t_play)
    except VoiceDependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 — any device failure is a playback error
        raise PlaybackError(f"Audio playback failed: {exc}") from exc


class Player:
    """Speaks sentences as they arrive, on a background thread.

    Fed sentence-by-sentence from the LLM stream via `enqueue`; a dedicated
    thread synthesizes and plays each so speaking overlaps generation (cloud
    providers). `close` signals end-of-turn; `join` drains and re-raises any
    error that occurred on the thread.
    """

    _SENTINEL = None

    _POLL = 0.2  # seconds; bounds how long a pool worker sits in queue.get()

    def __init__(self, tts, sample_rate: int, play_fn: Callable | None = None):
        self._tts = tts
        self._sample_rate = sample_rate
        self._play = play_fn or play
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._yielded: list[str] = []
        self._spoken = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, sentence: str) -> None:
        self._queue.put(sentence)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
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

        `shutdown()` drains politely; this stops the device mid-chunk, which is
        the difference between interrupting Nero and waiting for the sentence.
        """
        self._stop.set()
        try:
            sd = _import_sd()
            sd.stop()
        except Exception:  # noqa: BLE001 — never let a device quirk mask the interrupt
            logger.debug("sd.stop() failed during barge-in", exc_info=True)
        self._queue.put(self._SENTINEL)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the playback thread and wait, bounded. Idempotent and safe to
        call from a `finally` even if the turn already ended normally.

        Without this, an aborted turn (Ctrl+C, auth/network error) leaves the
        thread parked on an empty queue forever — which also parks an asyncio
        default-executor worker, and ThreadPoolExecutor's atexit hook then joins
        that worker at interpreter shutdown, hanging the whole process on exit.
        """
        self._stop.set()
        self._queue.put(self._SENTINEL)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.debug("playback thread did not stop within %.1fs", timeout)

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

    def _run(self) -> None:
        try:
            asyncio.run(self._consume())
        except Exception as exc:  # noqa: BLE001 — surfaced on join()
            self._error = exc

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        _t_start = time.monotonic()  # DEBUG(hang)
        _chunks = 0  # DEBUG(hang)

        async def sentences():
            while True:
                logger.debug("STAGE 5a: playback thread waiting on queue")  # DEBUG(hang)
                item = await loop.run_in_executor(None, self._next_item)
                if item is self._SENTINEL:
                    logger.debug("STAGE 5b: end-of-turn sentinel received")  # DEBUG(hang)
                    return
                # DEBUG(hang) sentence handed to the TTS engine
                logger.debug("STAGE 5c: synthesizing %r", item)
                self._yielded.append(item)
                yield item

        async for audio in self._tts.synthesize_stream(sentences()):
            if self._stop.is_set():
                # An engine that yields ahead (batches/prefetches, like a real
                # streaming TTS could) would otherwise keep playing every chunk
                # it already produced after stop_now() fires, defeating barge-in.
                break
            _chunks += 1  # DEBUG(hang)
            if _chunks == 1:  # DEBUG(hang) STAGE 5: first audio chunk out of TTS
                logger.debug(
                    "STAGE 5: FIRST audio chunk from TTS after %.2fs",
                    time.monotonic() - _t_start,
                )
            else:
                logger.debug("STAGE 5: audio chunk #%d from TTS", _chunks)  # DEBUG(hang)
            self._spoken += 1
            await loop.run_in_executor(None, self._play, audio, self._sample_rate)
            logger.debug("STAGE 7c: chunk #%d dispatched+played", _chunks)  # DEBUG(hang)

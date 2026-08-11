from __future__ import annotations

import asyncio
import logging  # DEBUG(hang)
import queue
import threading
import time  # DEBUG(hang)
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
    interrupting word is not clipped.
    """
    import numpy as np

    sd = _import_sd()
    vad.reset()
    console.print("[bold red]●[/bold red] Listening — I'll stop when you do")

    blocks: list = list([] if prefix is None else [np.asarray(prefix, dtype=np.float32).reshape(-1)])
    silent_needed = max(1, round(silence_ms / 1000 * RECORD_SAMPLE_RATE / VAD_FRAME))
    max_frames = round(max_utterance_seconds * RECORD_SAMPLE_RATE / VAD_FRAME)
    wait_frames = round(wait_for_speech_seconds * RECORD_SAMPLE_RATE / VAD_FRAME)

    started = False
    silent_run = 0
    try:
        with sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            for index in range(max_frames):
                data, _ = stream.read(VAD_FRAME)
                frame = np.asarray(data, dtype=np.float32).reshape(-1)
                speech = vad.is_speech(frame)
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
                    return np.zeros(0, dtype=np.float32)
    except sd.PortAudioError as exc:
        raise _mic_error(exc) from exc

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks, axis=0).reshape(-1).astype(np.float32)


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
                yield item

        async for audio in self._tts.synthesize_stream(sentences()):
            _chunks += 1  # DEBUG(hang)
            if _chunks == 1:  # DEBUG(hang) STAGE 5: first audio chunk out of TTS
                logger.debug(
                    "STAGE 5: FIRST audio chunk from TTS after %.2fs",
                    time.monotonic() - _t_start,
                )
            else:
                logger.debug("STAGE 5: audio chunk #%d from TTS", _chunks)  # DEBUG(hang)
            await loop.run_in_executor(None, self._play, audio, self._sample_rate)
            logger.debug("STAGE 7c: chunk #%d dispatched+played", _chunks)  # DEBUG(hang)

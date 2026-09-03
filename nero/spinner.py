"""A one-character spinner for the dead time before a reply's first token.

Providers routinely take 20-40 s to first byte on a queued free tier, and a
session that starts with a full history pays that twice (compaction, then the
turn itself). With nothing on screen that is indistinguishable from a hang --
which is exactly how it was reported. Drawn after the already-printed
"Nero> " prefix and erased with backspaces, so the reply then starts exactly
where it would have.
"""

import itertools
import sys
import threading

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
INTERVAL = 0.1


class Spinner:
    """Idempotent start/stop. A no-op when the stream is not a terminal, so
    piped output and captured test runs stay byte-identical."""

    def __init__(self, file=None):
        self._file = file or sys.stdout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drawn = False

    def start(self) -> None:
        if self._thread is not None:
            return
        if not getattr(self._file, "isatty", bool)():
            return
        self._thread = threading.Thread(target=self._spin, daemon=True, name="nero-spinner")
        self._thread.start()

    def stop(self) -> None:
        """Safe to call from the streaming callback on every chunk."""
        if self._thread is None:
            return
        self._stop.set()
        # Joined, not just signalled: the caller writes reply text to this same
        # stream on the next line of code, and a frame landing between the two
        # would leave a stray glyph mid-word.
        self._thread.join(timeout=1)
        self._thread = None
        if self._drawn:
            self._write(" \b")

    def _spin(self) -> None:
        for frame in itertools.cycle(FRAMES):
            # Waits before the first draw, so a fast turn never flickers.
            if self._stop.wait(INTERVAL):
                return
            self._drawn = True
            self._write(frame + "\b")

    def _write(self, text: str) -> None:
        try:
            self._file.write(text)
            self._file.flush()
        except Exception:  # noqa: BLE001 — a progress hint may never break a turn
            pass

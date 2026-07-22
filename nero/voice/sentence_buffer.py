from __future__ import annotations

_BOUNDARIES = ".!?"
DEFAULT_MAX_LEN = 200


class SentenceBuffer:
    """Buffers streamed text chunks and emits complete sentences.

    Splits on . ! ? so TTS gets whole sentences (word-by-word audio is broken,
    whole-response audio kills the latency benefit). A max-length fallback keeps
    a run-on response (no punctuation) from buffering forever — it breaks at the
    last word boundary within the window.
    """

    def __init__(self, max_len: int = DEFAULT_MAX_LEN):
        self._buf = ""
        self._max_len = max_len

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            cut = self._cut_index()
            if cut is None:
                break
            sentence = self._buf[: cut + 1].strip()
            self._buf = self._buf[cut + 1 :]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        return rest

    def _cut_index(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if ch in _BOUNDARIES:
                return i
        if len(self._buf) >= self._max_len:
            window = self._buf[: self._max_len]
            space = window.rfind(" ")
            return space if space > 0 else self._max_len - 1
        return None

from __future__ import annotations

_BOUNDARIES = ".!?"
# Clause boundaries count only for the FIRST segment of a reply — see below.
_CLAUSE_BOUNDARIES = ",;:"
DEFAULT_MAX_LEN = 200

# Kokoro's synthesis cost is roughly proportional to the text: ~830 ms for a
# five-word sentence, 5.9 s for a 25-word one (measured). Time-to-first-audio
# is therefore set by how long the FIRST segment is, so the first segment is
# cut short — at a clause boundary, or at FIRST_MAX_LEN — and everything after
# it uses ordinary sentence granularity, where longer chunks sound better and
# the speaker is already busy anyway.
FIRST_MAX_LEN = 60
# Below this, a clause cut produces a standalone "So," or "Well," which sounds
# worse than the latency it saves. "Sure thing," (11) clears it; "So," does not.
FIRST_MIN_LEN = 8


class SentenceBuffer:
    """Buffers streamed text chunks and emits complete sentences.

    Splits on . ! ? so TTS gets whole sentences (word-by-word audio is broken,
    whole-response audio kills the latency benefit). A max-length fallback keeps
    a run-on response (no punctuation) from buffering forever — it breaks at the
    last word boundary within the window.

    The first segment is cut more eagerly than the rest (see FIRST_MAX_LEN):
    it is the only one the user waits on in silence.
    """

    def __init__(self, max_len: int = DEFAULT_MAX_LEN):
        self._buf = ""
        self._max_len = max_len
        self._first = True

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
                self._first = False
        return out

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._first = False
        return rest

    def _cut_index(self) -> int | None:
        limit = min(self._max_len, FIRST_MAX_LEN) if self._first else self._max_len
        for i, ch in enumerate(self._buf):
            if ch in _BOUNDARIES:
                return i
            if self._first and ch in _CLAUSE_BOUNDARIES and i + 1 >= FIRST_MIN_LEN:
                return i
        if len(self._buf) >= limit:
            window = self._buf[:limit]
            space = window.rfind(" ")
            return space if space > 0 else limit - 1
        return None

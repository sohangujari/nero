from __future__ import annotations

import re

# Markdown is written to be *seen*. Read aloud it is not merely ugly, it is
# slow: measured against Kokoro, one **bold** pair costs +2.7 s and a
# [link](url) costs +4.8 s because the URL itself gets pronounced. That is the
# long stall people hear at a comma or a colon -- not the punctuation, the
# markup sitting beside it. Backticks Kokoro already ignores; they are stripped
# anyway so a code span reads as its contents.
_MARKDOWN = [
    (re.compile(r"```[\s\S]*?```"), " "),                      # fenced code block
    (re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.M), ""),  # horizontal rule
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),            # image -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),             # link -> link text
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),              # heading marker
    (re.compile(r"^\s{0,3}>\s?", re.M), ""),                   # blockquote marker
    (re.compile(r"^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s+", re.M), ""),  # list marker
    (re.compile(r"(\*\*|__|~~)(.+?)\1", re.S), r"\2"),          # bold / strikethrough
    (re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)", re.S), r"\2"),  # italic
    (re.compile(r"`+"), ""),                                   # code span ticks
    # Whatever survived -- an unmatched pair split across two stream chunks,
    # say. An asterisk is never worth pronouncing. Underscores are left alone
    # because they appear inside identifiers Nero genuinely does say.
    (re.compile(r"\*+"), " "),
]
_WHITESPACE = re.compile(r"\s+")


def speakable(text: str) -> str:
    """Markdown as a person would read it aloud, not as a screen renders it."""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    return _WHITESPACE.sub(" ", text).strip()


_BOUNDARIES = ".!?"
# Clause boundaries count only for the FIRST segment of a reply — see below.
_CLAUSE_BOUNDARIES = ",;:"
DEFAULT_MAX_LEN = 200

# Kokoro's synthesis cost is roughly proportional to the text: ~830 ms for a
# five-word sentence, 5.9 s for a 25-word one (measured). Time-to-first-audio
# is therefore set by how long the FIRST segment is, so the first segment is
# cut short — at a clause boundary, or at FIRST_MAX_LEN.
FIRST_MAX_LEN = 60
# Below this, a clause cut produces a standalone "So," or "Well," which sounds
# worse than the latency it saves. "Sure thing," (11) clears it; "So," does not.
FIRST_MIN_LEN = 8

# Measured Kokoro real-time factor: 0.44-0.54 on an M3. 0.6 leaves margin.
SYNTH_RTF = 0.6


def catch_up_limit(spoken_chars: int, first_len: int) -> int:
    """The longest next segment that synthesis can finish before the speaker
    runs out of the audio already banked.

    Playback starts once segment 1 is synthesized, so at any point the speaker
    has `spoken_chars` of audio ahead of it and synthesis is `SYNTH_RTF` times
    faster than speech. Solving "next segment is ready before the queue empties"
    for its length gives spoken_chars * (1 - rtf)/rtf + first_len.

    This is the bug the fixed FIRST_MAX_LEN missed: it capped segment 1 only, so
    a 27-character opener ("Sure, I can help with that.") could be followed by a
    73-character sentence that took 1.87 s to synthesize against 1.49 s of
    banked audio — a hitch of dead air in the middle of the reply, every time.
    The allowance grows fast, so it only binds during the first second or two.
    """
    return int(spoken_chars * (1 - SYNTH_RTF) / SYNTH_RTF) + first_len


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
        # Characters emitted so far this reply, and the length of segment 1 —
        # together these are the speaker's lead (see catch_up_limit).
        self._spoken_chars = 0
        self._first_len = 0

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            cut = self._cut_index()
            if cut is None:
                break
            sentence = speakable(self._buf[: cut + 1])
            self._buf = self._buf[cut + 1 :]
            if sentence:
                out.append(sentence)
                self._record(sentence)
        return out

    def flush(self) -> str:
        rest = speakable(self._buf)
        self._buf = ""
        if rest:
            self._record(rest)
        return rest

    def _record(self, sentence: str) -> None:
        if self._first:
            self._first_len = len(sentence)
            self._first = False
        self._spoken_chars += len(sentence)

    def _limit(self) -> int:
        """How long this segment may be before it starves the speaker."""
        if self._first:
            return min(self._max_len, FIRST_MAX_LEN)
        allowance = max(
            FIRST_MAX_LEN, catch_up_limit(self._spoken_chars, self._first_len)
        )
        return min(self._max_len, allowance)

    def _cut_index(self) -> int | None:
        limit = self._limit()
        for i, ch in enumerate(self._buf):
            if i >= limit:
                # Past what the speaker's lead can cover. A sentence boundary
                # further out is no good to us: waiting for it is exactly the
                # stall this limit exists to prevent, so fall through and cut
                # inside the window instead.
                break
            if ch in _BOUNDARIES:
                if not self._closes_a_sentence(i):
                    continue
                if ch == "." and self._is_enumerator(i):
                    # Break before the item's number, not after it: each item
                    # becomes its own utterance, which both reads naturally and
                    # keeps chunks short enough for synthesis to stay ahead of
                    # the speaker.
                    line_start = self._buf.rfind("\n", 0, i)
                    if line_start > 0:
                        return line_start
                    continue
                return i
            if self._first and ch in _CLAUSE_BOUNDARIES and i + 1 >= FIRST_MIN_LEN:
                return i
        if len(self._buf) >= limit:
            return self._best_cut_within(limit)
        return None

    def _best_cut_within(self, limit: int) -> int:
        """Where to break text that has no sentence end inside the window.

        A clause boundary is the natural place — "…warm and humid," reads as a
        phrase, where the last-space fallback gives "…humid, around" and leaves
        "thirty-one degrees" stranded. Searched backwards, so the cut is as late
        as the window allows; clause characters are only trusted here, never
        mid-scan, because a ':' is far more often part of "https://" than the
        end of a phrase.
        """
        window = self._buf[:limit]
        for i in range(len(window) - 1, FIRST_MIN_LEN - 2, -1):
            if window[i] in _CLAUSE_BOUNDARIES and window[i + 1 : i + 2].isspace():
                return i
        space = window.rfind(" ")
        return space if space > 0 else limit - 1

    def _closes_a_sentence(self, index: int) -> bool:
        """A boundary character only ends a sentence when whitespace follows.

        Without this the "." in "example.com" cut a URL in half -- and since a
        half-URL no longer matches the link pattern, speakable() could not strip
        it and Kokoro pronounced it, at +4.8 s. Decimals ("3.14") and
        abbreviations ("e.g.") were cut the same way. A boundary at the very end
        of the buffer is still taken: the next character is genuinely unknown
        there, and waiting for it would stall the reply.
        """
        following = self._buf[index + 1 : index + 2]
        return not following or following.isspace()

    def _is_enumerator(self, index: int) -> bool:
        """True for the "." of a list marker ("1.", "2."), which does not end a
        sentence. Without this each item's number was cut off as its own
        segment and spoken alone -- a full stop where a list should flow."""
        line = self._buf[:index].rpartition("\n")[2].strip(" \t*->")
        return line.isdigit() and len(line) <= 3

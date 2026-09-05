"""The semantic half of recall: an optional, local, on-your-machine embedder.

Keyword search alone answers about 86% of long-conversation recall questions.
Fusing it with vector search takes that to about 95% — the single largest gain
of any component measured on LongMemEval (2026). This module supplies the
vectors.

Everything here is optional and probed, never assumed. No ollama server, no
`nomic-embed-text`, or no numpy, and `available` is False — recall falls back
to keyword-only and nothing else changes. That is why it is built on ollama
rather than a new dependency: Nero already integrates with ollama, it runs on
the user's own machine (no quota, no network round-trip, no per-turn cost),
and the model is 274 MB against the hundreds a torch stack would add.

Measured on an M3: 66 ms to embed one query, 8.9 ms per text in a batch of 32.
"""

import logging
import struct

import httpx

from nero.llm import ollama

logger = logging.getLogger("nero.memory")

MODEL = "nomic-embed-text"
DIMS = 768
# One query on the turn's critical path; writes are batched and happen after
# the reply is already on screen, where latency is invisible.
TIMEOUT = 5.0

# Cosine below which a "nearest" neighbour is not actually a neighbour.
#
# Vector search has no concept of "nothing matched" -- it always returns a
# top-k, so without a floor every "what is 2+2" drags four unrelated old
# exchanges into the prompt. Measured against a real transcript with this
# model: queries that should recall score 0.60-0.81, queries that should not
# score 0.40-0.47. 0.55 sits in the gap with margin on both sides. The number
# is calibrated to nomic-embed-text's scale and would need re-measuring for a
# different model.
SIMILARITY_FLOOR = 0.55


def pack(vector) -> bytes:
    """A vector as a compact float32 blob — 3 KB per turn, not 15 KB of JSON."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_all(blobs):
    """`blobs` as one (n, dims) float32 numpy array, ready to score in a single
    matrix multiply. Requires numpy; callers guard on `Embedder.available`."""
    import numpy as np

    return np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), -1)


class Embedder:
    """Text to vectors via a local ollama server, or nothing at all.

    `available` is probed once and cached: a missing server must cost one
    failed connection per session, not one per turn.
    """

    def __init__(self, base_url: str = ollama.BASE_URL, model: str = MODEL, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._enabled = enabled
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        if not self._enabled:
            return False
        try:
            import numpy  # noqa: F401 — scoring needs it; without it, stay off
        except ImportError:
            logger.debug("semantic recall off: numpy is not installed")
            return False
        if self.embed(["probe"], _probing=True) is None:
            logger.debug("semantic recall off: no %s at %s", self.model, self.base_url)
            return False
        return True

    def embed(self, texts: list[str], _probing: bool = False) -> list[bytes] | None:
        """`texts` as packed float32 blobs, or None if the embedder can't serve
        them. Never raises — recall is an optimisation, not part of the turn."""
        if not texts or (not _probing and not self.available):
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            vectors = response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.debug("embedding failed: %s", exc)
            return None
        if len(vectors) != len(texts):
            return None
        return [pack(vector) for vector in vectors]

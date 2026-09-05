"""Nero's memory: a small live window, plus retrieval over everything else.

Every turn re-sends what Nero is given, and a provider spends most of its
time-to-first-token *prefilling* that prompt. Two bad answers exist: send the
whole conversation (slow, and it grows forever) or send a fixed window of the
last N messages (constant, but it pays full price for N messages that are
usually irrelevant, and still forgets anything older). This module does
neither.

What reaches the model each turn:

    system prompt + facts      stable across the session, so it caches
    recalled exchanges         top-k, retrieved for THIS message
    the last few messages      recency, for "it" / "that one" to resolve
    the user's message

Measured over a 120-turn conversation, asked a question whose answer is 100
turns back (llama3.2 on an M3). All three policies answer it:

    whole conversation     242 messages   7,285 tokens   141.7 s to first token
    fixed 60-msg window     32 messages     936 tokens    12.9 s
    window + retrieval      14 messages     588 tokens     6.8 s

Retrieval itself costs 31 ms over 600 indexed turns.

The window is small because it only has to carry the live thread; everything
older is *retrieved*, not carried. Retrieval fuses two rankings with
Reciprocal Rank Fusion:

  - keyword (SQLite FTS5 + Porter stemming) — about 86% recall on its own
  - semantic (a local embedding model, `nero/memory/embeddings.py`) — optional

Fusing them is worth about +9 points, the largest single gain measured on
LongMemEval (2026), and RRF needs no score calibration between the two: it
only reads ranks, so a bm25 score and a cosine score never have to be made
comparable. With no embedder the semantic list is simply empty and RRF
degrades to keyword-only.

Why not a knowledge graph: building one costs several LLM calls per turn,
which is the exact cost this module exists to remove. Durable, structured
knowledge already has a cheaper home — FactStore, which the model writes with
`remember_fact` and which goes into the system prompt every turn.

ponytail: retrieval over raw turns, with no LLM extraction or consolidation
step. Ceiling — contradictions are not resolved (both "my favourite colour is
blue" and a later correction can be recalled together), and long exchanges are
recalled whole rather than as distilled facts. Upgrade path is an extraction
pass writing into FactStore; the seam is `recall_block`.
"""

import re

# Recall is capped in characters, not just in rows: three long exchanges could
# otherwise undo the very trimming this module does.
MAX_BLOCK_CHARS = 800
MAX_LINE_CHARS = 300
# Exchanges, not messages. Small on purpose: the point of retrieval is to
# beat a big window on tokens, not to rebuild one out of search results.
RECALL_LIMIT = 4

_TERM = re.compile(r"[a-z0-9']{3,}")
# Small on purpose. bm25 already discounts common words; this list only stops
# a query like "what is the thing" from being all-stopwords-no-signal.
_STOPWORDS = frozenset(
    """about after all also and any are but can did does for from get got had has
    have how into its just like make more not now one out say see she some tell than
    that the their them then there these they this those was were what when where
    which who why will with would you your""".split()
)


def find_cut_point(messages: list[dict], keep_recent: int) -> int | None:
    """The largest `cut` that both leaves a clean boundary and keeps at least
    `keep_recent` messages: messages[cut] is a user message and
    messages[cut - 1] is an assistant message carrying no tool_calls.

    `keep_recent` is what stops trimming from eating the whole conversation.
    Without it the largest valid cut is almost always "everything but the last
    message", so the model would lose the live thread it is mid-way through.

    Scanning from the end means a cut point that would split a tool-call
    sequence is skipped automatically: right after an assistant message with
    tool_calls comes a `tool` role message, never `user`, so that candidate
    fails and the search keeps walking left until it clears the whole
    tool_calls/tool group. Returns None if no valid boundary exists at all —
    trimming must be skipped rather than guess.
    """
    for cut in range(len(messages) - keep_recent, 0, -1):
        before, after = messages[cut - 1], messages[cut]
        if (
            after.get("role") == "user"
            and before.get("role") == "assistant"
            and not before.get("tool_calls")
        ):
            return cut
    return None


def trim_to_window(messages: list[dict], threshold: int) -> int:
    """Drop the oldest exchanges from `messages` in place once it passes
    `threshold`. Returns how many went; 0 means nothing was touched.

    Trimming in one block down to half the threshold, rather than a message or
    two per turn, is deliberate: the surviving prefix then stays byte-identical
    for many turns, which is what lets a provider's prompt cache hit.
    """
    if not threshold or len(messages) <= threshold:
        return 0
    cut = find_cut_point(messages, keep_recent=max(10, threshold // 2))
    if cut is None:
        return 0
    del messages[:cut]
    return cut


def query_terms(text: str) -> list[str]:
    """The searchable words in `text`: lowercased, de-duplicated, stopwords and
    one- and two-letter fragments dropped, capped so one rambling message can't
    turn into a 200-clause query."""
    terms = [term for term in _TERM.findall(text.lower()) if term not in _STOPWORDS]
    return list(dict.fromkeys(terms))[:12]


def fts_query(text: str) -> str:
    """`text` as an FTS5 OR-of-terms query, or "" if it has no usable terms.

    Every term is quoted, so a user's `-`, `*`, `NEAR` or stray `"` is matched
    as a literal word instead of being parsed as query syntax.
    """
    return " OR ".join(f'"{term}"' for term in query_terms(text))


RRF_K = 60  # the standard damping constant; larger flattens the rank curve
# Query terms a keyword-only hit must share before it counts as a memory.
# bm25 ranks matches against each other but has no absolute floor, so a single
# incidental word -- "function" stemming into "that functionality" -- was
# enough to drag an unrelated exchange into the prompt.
MIN_TERM_OVERLAP = 2


def rrf(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: merge ranked id lists into one ranking.

    Each list contributes 1/(k + rank) to an id's score, so agreement between
    rankings beats a strong showing in only one — and nothing has to know how
    to compare a bm25 score with a cosine similarity, which is exactly the
    problem that makes score-level fusion fragile.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: -scores[key])


def _covers_enough(terms: list[str], exchange: list[tuple[str, str]]) -> bool:
    """Whether a keyword-only hit shares enough of the query to be worth
    recalling. A one- or two-term query has nothing to spare, so it is taken at
    face value; a longer one has to match more than a single passing word."""
    if len(terms) < MIN_TERM_OVERLAP + 1:
        return True
    haystack = " ".join(content for _role, content in exchange).lower()
    return sum(term in haystack for term in terms) >= MIN_TERM_OVERLAP


def recall_block(store, text: str, messages: list[dict], limit: int = RECALL_LIMIT) -> str:
    """Past exchanges relevant to `text` that `messages` no longer holds,
    formatted to prefix the user's message — or "" when there is nothing worth
    adding.

    Carried on the user message rather than inserted at the front on purpose:
    the prompt's prefix stays stable, so recall can't invalidate a provider's
    cache of everything after it.
    """
    if store is None:
        return ""
    terms = query_terms(text)
    query = " OR ".join(f'"{term}"' for term in terms)
    # Each stream is asked for more than `limit` so RRF has room to disagree.
    keyword = store.search_keys(query, limit=limit * 2) if query else []
    semantic = store.search_semantic_keys(text, limit=limit * 2)
    if not keyword and not semantic:
        return ""
    strong = set(semantic)

    # Substring, not equality: an earlier turn's recall block is itself part of
    # a message's content, and matching on equality would re-recall it forever.
    live = [m.get("content") for m in messages if isinstance(m.get("content"), str)]
    lines: list[str] = []
    budget = MAX_BLOCK_CHARS
    for key, exchange in store.exchanges(rrf(keyword, semantic)[: limit * 2]):
        # The semantic stream has its own floor (embeddings.SIMILARITY_FLOOR);
        # this is the keyword stream's.
        if key not in strong and not _covers_enough(terms, exchange):
            continue
        rendered = []
        for role, content in exchange:
            content = content.strip()
            if not content or any(content in existing for existing in live):
                rendered = []
                break
            if len(content) > MAX_LINE_CHARS:
                content = content[:MAX_LINE_CHARS].rstrip() + "\u2026"
            rendered.append(f"{'user' if role == 'user' else 'assistant'}: {content}")
        if not rendered:
            continue
        cost = sum(len(line) for line in rendered)
        if cost > budget:
            break
        budget -= cost
        lines.extend(rendered)
        if len(lines) >= limit * 2:
            break
    if not lines:
        return ""
    # Two findings, both measured against a 3B local model, which is the
    # hardest case:
    #
    # Tagged, not headed. "[Possibly relevant earlier conversation]" above a
    # transcript reads exactly like a chat log the user pasted, and models
    # answered it as one ("I see you're sharing a previous conversation
    # snippet!"). The tag pairs with the system prompt, which says what it is.
    #
    # `user:`/`assistant:` for the speakers. "they said:"/"you said:" made the
    # model lose track of whose preference it was reading ("blue is my
    # favourite colour"); the role words it was trained on cost 0 of 6 such
    # slips against 2 and 3 for the friendlier-sounding alternatives.
    return "<memory>\n" + "\n".join(lines) + "\n</memory>\n\n"

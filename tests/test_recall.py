"""Bounded context window + keyword recall (nero/memory/recall.py), and the
wiring that makes both loops use them.

The cut-boundary rule is the risky part: a tool-call sequence split across
the cut breaks the provider contract (an assistant `tool_calls` message
whose `tool` results get dropped). Every boundary test here is built around
proving that never happens.
"""

import io

from rich.console import Console

from nero.core.chat_loop import ChatLoop
from nero.memory.history_store import HistoryStore
from nero.memory.recall import find_cut_point, fts_query, recall_block, rrf, trim_to_window


def quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def assistant_tool_call(call_id="call_1", name="get_weather"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def tool_result(call_id="call_1", content="sunny"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestFindCutPoint:
    def test_plain_alternating_turns_cuts_at_the_very_last_user_message(self):
        messages = [user("hi"), assistant("hello"), user("how are you"), assistant("fine"), user("bye")]
        # Largest valid cut: message after is 'user', message before is a
        # plain assistant message with no tool_calls.
        assert find_cut_point(messages, keep_recent=1) == 4

    def test_no_boundary_returns_none(self):
        # Every message is 'assistant' or 'tool' — no 'user' message to land
        # after any cut, so trimming must be skipped rather than guess.
        messages = [assistant_tool_call(), tool_result()]
        assert find_cut_point(messages, keep_recent=1) is None

    def test_does_not_split_a_straddling_tool_call_sequence(self):
        # The naive "cut at the very last user message" boundary would land
        # right after assistant_tool_call() and before tool_result() — the
        # message after that cut is 'tool', not 'user', so it must be
        # rejected and the search must fall back to an earlier valid cut
        # that keeps the pair together.
        messages = [
            user("earlier"),
            assistant("ok"),
            user("what's the weather"),
            assistant_tool_call(),
            tool_result(),
        ]
        cut = find_cut_point(messages, keep_recent=1)
        assert cut is not None
        # Whichever side of the cut the pair lands on, it must land together.
        dropped, kept = messages[:cut], messages[cut:]
        straddles = any(m.get("tool_calls") for m in dropped) and any(
            m.get("role") == "tool" for m in kept
        )
        assert not straddles
        # Concretely: the only valid boundary here is right before
        # "what's the weather", keeping the tool_call+result pair intact.
        assert cut == 2

    def test_multiple_tool_rounds_all_survive_together(self):
        messages = [
            user("first"),
            assistant("ok"),
            user("do two things"),
            assistant_tool_call("call_1", "get_weather"),
            tool_result("call_1", "sunny"),
            assistant_tool_call("call_2", "open_app"),
            tool_result("call_2", "opened"),
        ]
        # Every candidate cut inside the two tool rounds is invalid (the
        # message before is either 'user' or an assistant with tool_calls),
        # so the search backs off to before the whole cluster — both rounds
        # land intact in the kept tail.
        cut = find_cut_point(messages, keep_recent=1)
        assert cut == 2
        kept = messages[cut:]
        assert kept == messages[2:]  # both tool_calls/tool pairs intact together


    def test_keep_recent_protects_the_live_thread(self):
        """Without a recent window the largest valid cut is 'everything but the
        last message' — the model would lose the conversation it is mid-way
        through and see only a summary."""
        messages = []
        for i in range(10):
            messages.extend([user(f"q{i}"), assistant(f"a{i}")])
        messages.append(user("latest"))

        assert find_cut_point(messages, keep_recent=1) == 20
        cut = find_cut_point(messages, keep_recent=10)
        assert cut is not None and cut <= len(messages) - 10
        assert len(messages) - cut >= 10

    def test_keep_recent_larger_than_history_finds_no_cut(self):
        messages = [user("hi"), assistant("hello"), user("bye")]
        assert find_cut_point(messages, keep_recent=10) is None



class TestTrimToWindow:
    def test_disabled_at_zero(self):
        messages = [user("hi")] * 10
        assert trim_to_window(messages, 0) == 0
        assert len(messages) == 10

    def test_below_threshold_is_a_noop(self):
        messages = [user("hi"), assistant("ok")]
        assert trim_to_window(messages, 40) == 0
        assert len(messages) == 2

    def test_fires_past_threshold_and_keeps_the_tail_intact(self):
        messages = []
        for i in range(25):
            messages.extend([user(f"msg{i}"), assistant(f"reply{i}")])
        original = [dict(m) for m in messages]

        dropped = trim_to_window(messages, threshold=40)

        assert dropped > 0
        assert messages == original[dropped:]  # a prefix went; nothing else moved
        assert len(messages) >= 20  # keep_recent = threshold // 2

    def test_no_valid_boundary_leaves_messages_untouched(self):
        messages = [assistant_tool_call(), tool_result()] * 25
        original = [dict(m) for m in messages]
        assert trim_to_window(messages, threshold=40) == 0
        assert messages == original

    def test_repeated_trims_hold_the_window_flat(self):
        """The point of the whole module: a session that runs forever must not
        send a prompt that grows forever."""
        messages = []
        sizes = []
        for i in range(200):
            messages.extend([user(f"q{i}"), assistant(f"a{i}")])
            trim_to_window(messages, threshold=40)
            sizes.append(len(messages))
        assert max(sizes) <= 42  # threshold, plus the pair that tripped it


class TestFtsQuery:
    def test_terms_become_an_or_query(self):
        assert fts_query("ocean tides") == '"ocean" OR "tides"'

    def test_stopwords_and_short_words_are_dropped(self):
        assert fts_query("what is the ocean") == '"ocean"'

    def test_operator_characters_cannot_reach_fts5_as_syntax(self):
        # Unquoted, `NEAR`/`-`/`"` would be parsed as query syntax (or raise);
        # every term is quoted, so each is matched as a literal word.
        query = fts_query('rocket-fuel NEAR "quoted"')
        assert query == '"rocket" OR "fuel" OR "near" OR "quoted"'

    def test_nothing_usable_returns_empty(self):
        assert fts_query("is it?") == ""
        assert fts_query("") == ""


class TestHistorySearch:
    def make_store(self, tmp_path):
        return HistoryStore(tmp_path / "history.db", session_id="s1", max_turns=20)

    def keyword(self, store, query):
        """The flat (role, content) rows behind a keyword search."""
        return [row for _key, group in store.exchanges(store.search_keys(query)) for row in group]

    def test_finds_an_exchange_by_keyword(self, tmp_path):
        store = self.make_store(tmp_path)
        store.append_turn("what's my dog called", "Your dog is called Rex.")
        store.append_turn("what's the weather", "Sunny.")

        assert self.keyword(store, '"dog"') == [
            ("user", "what's my dog called"),
            ("assistant", "Your dog is called Rex."),
        ]

    def test_returns_both_halves_of_a_hit(self, tmp_path):
        # Matching only the answer must still bring back the question — a
        # stored reply with no prompt is not usable context.
        store = self.make_store(tmp_path)
        store.append_turn("and after that?", "We settled on Postgres.")
        assert [role for role, _ in self.keyword(store, '"postgres"')] == ["user", "assistant"]

    def test_stemming_matches_a_different_word_form(self, tmp_path):
        """The Porter tokenizer is what makes plain keyword recall worth
        keeping in the fusion at all."""
        store = self.make_store(tmp_path)
        store.append_turn("I am running late", "No problem.")
        assert self.keyword(store, '"run"')

    def test_a_query_fts5_rejects_returns_empty(self, tmp_path):
        store = self.make_store(tmp_path)
        store.append_turn("hi", "hello")
        assert store.search_keys('"unclosed') == []

    def test_rows_written_before_the_index_existed_are_still_found(self, tmp_path):
        """The index is added to a database users already have. Turns stored by
        an older version must not become invisible."""
        import sqlite3

        path = tmp_path / "history.db"
        legacy = sqlite3.connect(path)
        with legacy:
            legacy.execute(
                "CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            legacy.executemany(
                "INSERT INTO turns (session_id, role, content, created_at) VALUES (?,?,?,?)",
                [("s0", "user", "remember the kraken", "t0"),
                 ("s0", "assistant", "Noted.", "t0")],
            )
        legacy.close()

        assert self.keyword(HistoryStore(path, "s1"), '"kraken"') == [
            ("user", "remember the kraken"),
            ("assistant", "Noted."),
        ]

    def test_an_index_built_with_the_wrong_tokenizer_is_rebuilt(self, tmp_path):
        """Upgrading must not leave a stemless index in place, silently
        answering every later search with the wrong matcher."""
        import sqlite3

        path = tmp_path / "history.db"
        old = sqlite3.connect(path)
        with old:
            old.execute(
                "CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            old.execute("INSERT INTO turns (session_id, role, content, created_at) "
                        "VALUES ('s0','user','I am running late','t0')")
            old.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(content, "
                        "content='turns', content_rowid='id')")
            old.execute("INSERT INTO turns_fts (turns_fts) VALUES ('rebuild')")
        old.close()

        store = HistoryStore(path, "s1")
        assert store.search_keys('"run"'), "the un-stemmed index was not rebuilt"

    def test_clear_empties_the_index_too(self, tmp_path):
        store = self.make_store(tmp_path)
        store.append_turn("remember the kraken", "Noted.")
        store.clear()
        assert store.search_keys('"kraken"') == []

    def test_semantic_search_is_empty_without_an_embedder(self, tmp_path):
        store = self.make_store(tmp_path)
        store.append_turn("remember the kraken", "Noted.")
        assert store.search_semantic_keys("sea monster") == []


class TestSemanticSearch:
    """The embedder is faked: these prove the plumbing, not the model."""

    class FakeEmbedder:
        """One dimension per keyword — enough for cosine to order things."""

        available = True
        VOCAB = ["colour", "dog", "weather"]

        def embed(self, texts):
            from nero.memory.embeddings import pack

            out = []
            for text in texts:
                lowered = text.lower()
                out.append(pack([1.0 if w in lowered else 0.0 for w in self.VOCAB]))
            return out

    def make_store(self, tmp_path):
        store = HistoryStore(tmp_path / "history.db", "s1", embedder=self.FakeEmbedder())
        store.append_turn("my favourite colour", "Blue.")
        store.append_turn("my dog", "Rex.")
        return store

    def test_finds_the_nearest_exchange(self, tmp_path):
        store = self.make_store(tmp_path)
        rows = store.exchanges(store.search_semantic_keys("colour"))
        assert rows and rows[0][1] == [("user", "my favourite colour"), ("assistant", "Blue.")]

    def test_an_unrelated_query_recalls_nothing(self, tmp_path):
        """The floor is the difference between retrieval and noise: argsort
        always has a best match, so without it every "what is 2+2" drags old
        exchanges into the prompt."""
        store = self.make_store(tmp_path)
        assert store.search_semantic_keys("nothing in the vocabulary at all") == []

    def test_a_zero_vector_query_does_not_divide_by_zero(self, tmp_path):
        store = self.make_store(tmp_path)
        assert store.search_semantic_keys("nothing in the vocabulary") is not None

    def test_writes_are_embedded_as_they_land(self, tmp_path):
        store = self.make_store(tmp_path)
        connection = store._connect()
        try:
            count = connection.execute("SELECT COUNT(*) FROM turn_vectors").fetchone()[0]
        finally:
            connection.close()
        assert count == 4  # two exchanges, two rows each

    def test_a_transcript_that_predates_the_embedder_is_backfilled(self, tmp_path):
        path = tmp_path / "history.db"
        HistoryStore(path, "s1").append_turn("my dog", "Rex.")  # no embedder

        store = HistoryStore(path, "s1", embedder=self.FakeEmbedder())
        assert store.search_semantic_keys("dog") == []  # nothing embedded yet
        store.append_turn("unrelated", "ok")  # a later turn backfills the batch
        assert store.exchanges(store.search_semantic_keys("dog"))[0][1][0][1] == "my dog"

    def test_a_broken_embedder_never_breaks_the_turn(self, tmp_path):
        class Broken(self.FakeEmbedder):
            def embed(self, texts):
                raise RuntimeError("model gone")

        store = HistoryStore(tmp_path / "history.db", "s1", embedder=Broken())
        store.append_turn("my dog", "Rex.")  # must not raise
        assert store.search_semantic_keys("dog") == []


class TestKeywordRelevanceFloor:
    """bm25 ranks hits against each other but has no absolute floor, so a
    single passing word used to be enough to recall an unrelated exchange."""

    def make_store(self, tmp_path):
        store = HistoryStore(tmp_path / "history.db", session_id="s1")
        store.append_turn(
            "can you clear our history",
            "I can't do that — that functionality isn't available to me.",
        )
        return store

    def test_a_long_query_sharing_one_word_is_rejected(self, tmp_path):
        # "function" stems into "functionality" and nothing else matches.
        store = self.make_store(tmp_path)
        assert recall_block(store, "write a python function to sort a list", []) == ""

    def test_a_long_query_sharing_two_words_is_kept(self, tmp_path):
        store = self.make_store(tmp_path)
        assert recall_block(store, "can you clear the history for me", [])

    def test_a_short_query_is_taken_at_face_value(self, tmp_path):
        # Two terms have nothing to spare; demanding two of two would make
        # short questions unable to recall anything at all.
        store = self.make_store(tmp_path)
        assert recall_block(store, "clear history", [])


class TestMemoryFraming:
    """What the block looks like decides whether the model talks about it.

    The reported bug: recall was rendered as a headed you:/Nero: transcript on
    the user's own message, so the model answered it as a pasted chat log —
    "I see you're sharing a previous conversation snippet!".
    """

    def block(self, tmp_path):
        store = HistoryStore(tmp_path / "history.db", session_id="s1")
        store.append_turn("my fav colour is blue", "Got it — blue.")
        return recall_block(store, "what colour do I like", [])

    def test_is_tagged_as_memory_not_headed_as_a_transcript(self, tmp_path):
        block = self.block(tmp_path)
        assert block.startswith("<memory>\n")
        assert "</memory>" in block
        assert "earlier conversation" not in block.lower()

    def test_uses_role_words_the_model_was_trained_on(self, tmp_path):
        # "they said:"/"you said:" made a 3B model claim the user's preference
        # as its own ("blue is my favourite colour").
        block = self.block(tmp_path)
        assert "user: my fav colour is blue" in block
        assert "assistant: Got it — blue." in block
        assert "Nero:" not in block

    def test_sits_above_the_user_text_not_inside_it(self, tmp_path):
        # The block must end cleanly so the user's own words start a new line —
        # run together, the whole thing reads as one pasted document.
        assert self.block(tmp_path).endswith("</memory>\n\n")


class TestRrf:
    def test_appearing_in_both_rankings_wins(self):
        # "a" is found by both streams; "b" and "c" by one each, at the same
        # rank. Agreement is the signal RRF is actually built to reward.
        assert rrf(["a", "b"], ["a", "c"])[0] == "a"

    def test_a_high_rank_in_one_stream_still_beats_absence(self):
        # The semantic stream alone must be able to surface something keyword
        # search missed entirely — that is the whole point of fusing them.
        assert "z" in rrf(["a", "b", "c"], ["z"])

    def test_one_empty_ranking_degrades_to_the_other(self):
        assert rrf(["a", "b"], []) == ["a", "b"]

    def test_no_rankings_is_empty(self):
        assert rrf([], []) == []

    def test_never_compares_raw_scores(self):
        """Only ranks matter — the same order must fuse identically however
        wildly the two streams' underlying scores differ in scale."""
        assert rrf(["x", "y"], ["y", "x"]) == rrf(["x", "y"], ["y", "x"])


class TestRecallBlock:
    def make_store(self, tmp_path):
        store = HistoryStore(tmp_path / "history.db", session_id="s1")
        store.append_turn("my dog is called Rex", "Got it — Rex.")
        return store

    def test_no_store_is_no_block(self):
        assert recall_block(None, "anything", []) == ""

    def test_brings_back_a_trimmed_exchange(self, tmp_path):
        block = recall_block(self.make_store(tmp_path), "remind me about the dog", [])
        assert "Rex" in block
        assert block.startswith("<memory>")
        assert block.endswith("\n\n")  # sits above the user's own text

    def test_nothing_relevant_is_no_block(self, tmp_path):
        assert recall_block(self.make_store(tmp_path), "what is 2 + 2", []) == ""

    def test_never_repeats_what_the_live_window_still_holds(self, tmp_path):
        live = [user("my dog is called Rex"), assistant("Got it — Rex.")]
        assert recall_block(self.make_store(tmp_path), "tell me about the dog", live) == ""

    def test_never_re_recalls_its_own_earlier_block(self, tmp_path):
        store = self.make_store(tmp_path)
        block = recall_block(store, "the dog", [])
        assert block
        # The block is now part of a live user message, not a message of its
        # own — an equality check would miss it and recall the same rows again.
        live = [user(block + "the dog")]
        assert recall_block(store, "the dog again", live) == ""

    def test_block_is_capped(self, tmp_path):
        store = HistoryStore(tmp_path / "history.db", session_id="s1")
        for i in range(10):
            store.append_turn(f"kraken question {i} " + "x" * 500,
                              f"kraken answer {i} " + "y" * 500)
        block = recall_block(store, "kraken", [])
        assert 0 < len(block) < 1200  # MAX_BLOCK_CHARS plus the header


class FakeSendClient:
    provider = "claude"

    def send(self, messages, on_text):
        on_text("ok")
        messages.append(assistant("ok"))


def make_loop(inputs, compact_after_messages=0, seed_messages=None, history=None):
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    loop = ChatLoop(
        FakeSendClient(), console=quiet_console(), assistant_name="Nero",
        input_fn=next_input, compact_after_messages=compact_after_messages,
        history=history,
    )
    if seed_messages is not None:
        loop.messages = seed_messages
    return loop


def seeded(pairs):
    messages = []
    for i in range(pairs):
        messages.extend([user(f"msg{i}"), assistant(f"reply{i}")])
    return messages


class TestChatLoopWiring:
    def test_default_disabled_never_trims(self):
        loop = make_loop(["one more", "exit"], compact_after_messages=0,
                         seed_messages=seeded(25))
        loop.run()
        assert len(loop.messages) == 52  # 50 seeded + user + assistant

    def test_trims_and_says_so_past_threshold(self):
        console = quiet_console()
        loop = make_loop(["one more", "exit"], compact_after_messages=10,
                         seed_messages=seeded(20))
        loop.console = console
        loop.run()
        assert "trimmed" in console.file.getvalue()
        assert len(loop.messages) <= 12

    def test_trimming_costs_no_provider_call(self):
        """The bug this replaced: summarizing spent a whole extra round-trip
        before the reply even started."""
        calls = []

        class CountingClient(FakeSendClient):
            def send(self, messages, on_text):
                calls.append(messages[-1])
                super().send(messages, on_text)

            async def stream_chat(self, messages, tools):  # pragma: no cover
                raise AssertionError("trimming must never call the provider")

        loop = make_loop(["one more", "exit"], compact_after_messages=10,
                         seed_messages=seeded(20))
        loop.client = CountingClient()
        loop.run()
        assert len(calls) == 1

    def test_turn_rollback_still_works_after_trimming(self):
        class FailingClient(FakeSendClient):
            def send(self, messages, on_text):
                raise RuntimeError("boom")

        loop = make_loop(["trigger", "exit"], compact_after_messages=10,
                         seed_messages=seeded(20))
        loop.client = FailingClient()
        loop.run()
        # Trimming fired, the turn failed, and the rollback landed on the
        # post-trim state — never a mix of pre- and post-trim messages.
        assert len(loop.messages) <= 11
        assert loop.messages[-1]["role"] == "assistant"

    def test_a_recalled_exchange_reaches_the_model_but_not_the_history(self, tmp_path):
        history = HistoryStore(tmp_path / "history.db", session_id="s1")
        history.append_turn("my dog is called Rex", "Got it — Rex.")
        sent = []

        class CapturingClient(FakeSendClient):
            def send(self, messages, on_text):
                sent.append(messages[-1]["content"])
                super().send(messages, on_text)

        loop = make_loop(["remind me about the dog", "exit"], history=history)
        loop.messages = []  # ignore the seeded window; recall is what's under test
        loop.client = CapturingClient()
        loop.run()

        assert "Rex" in sent[0]
        # What gets persisted is what the user actually said.
        stored = history.exchanges(history.search_keys('"remind"'))
        assert stored[0][1][0] == ("user", "remind me about the dog")

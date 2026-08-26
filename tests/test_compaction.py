"""Session compaction (nero/core/chat_loop.py): find_compaction_cut,
summarize_messages, compact_messages, and ChatLoop wiring.

The cut-boundary rule is the risky part: a tool-call sequence split across
the cut breaks the provider contract (an assistant `tool_calls` message
whose `tool` results get dropped). Every boundary test here is built around
proving that never happens.
"""

import io

from rich.console import Console

from nero.core.chat_loop import ChatLoop, compact_messages, find_compaction_cut, summarize_messages


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


class TestFindCompactionCut:
    def test_plain_alternating_turns_cuts_at_the_very_last_user_message(self):
        messages = [user("hi"), assistant("hello"), user("how are you"), assistant("fine"), user("bye")]
        # Largest valid cut: message after is 'user', message before is a
        # plain assistant message with no tool_calls.
        assert find_compaction_cut(messages) == 4

    def test_no_boundary_returns_none(self):
        # Every message is 'assistant' or 'tool' — no 'user' message to land
        # after any cut, so compaction must be skipped rather than guess.
        messages = [assistant_tool_call(), tool_result()]
        assert find_compaction_cut(messages) is None

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
        cut = find_compaction_cut(messages)
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
        cut = find_compaction_cut(messages)
        assert cut == 2
        kept = messages[cut:]
        assert kept == messages[2:]  # both tool_calls/tool pairs intact together


class TestSummarizeMessages:
    class FakeClient:
        def __init__(self, chunks=None, error=None):
            self._chunks = chunks if chunks is not None else ["a summary"]
            self._error = error
            self.seen_messages = None
            self.seen_tools = None

        async def stream_chat(self, messages, tools):
            self.seen_messages = messages
            self.seen_tools = tools
            if self._error:
                raise self._error
            for chunk in self._chunks:
                yield chunk

    def test_returns_joined_text(self):
        client = self.FakeClient(chunks=["hello ", "world"])
        assert summarize_messages(client, [user("hi")]) == "hello world"

    def test_passes_no_tools(self):
        client = self.FakeClient()
        summarize_messages(client, [user("hi")])
        assert client.seen_tools == []

    def test_never_raises_on_failure(self):
        client = self.FakeClient(error=RuntimeError("boom"))
        assert summarize_messages(client, [user("hi")]) is None

    def test_empty_response_is_none(self):
        client = self.FakeClient(chunks=["   "])
        assert summarize_messages(client, [user("hi")]) is None


class TestCompactMessages:
    def test_disabled_at_zero(self):
        messages = [user("hi")] * 10
        assert compact_messages(messages, client=None, threshold=0) is None

    def test_below_threshold_is_a_noop(self):
        messages = [user("hi"), assistant("ok")]
        assert compact_messages(messages, client=None, threshold=40) is None

    def test_fires_past_threshold(self):
        messages = []
        for i in range(25):
            messages.append(user(f"msg{i}"))
            messages.append(assistant(f"reply{i}"))
        client = TestSummarizeMessages.FakeClient(chunks=["gist"])
        result = compact_messages(messages, client, threshold=40)
        assert result is not None
        new_messages, dropped = result
        assert dropped > 0
        assert new_messages[0] == {
            "role": "user",
            "content": "[Earlier conversation summary]\ngist",
        }
        # Nothing after the summary was split or lost relative to the tail.
        assert new_messages[1:] == messages[dropped:]

    def test_no_valid_boundary_skips_compaction(self):
        messages = [assistant_tool_call(), tool_result()] * 25
        client = TestSummarizeMessages.FakeClient()
        assert compact_messages(messages, client, threshold=40) is None

    def test_summary_failure_leaves_messages_untouched(self):
        messages = []
        for i in range(25):
            messages.append(user(f"msg{i}"))
            messages.append(assistant(f"reply{i}"))
        original = [dict(m) for m in messages]
        client = TestSummarizeMessages.FakeClient(error=RuntimeError("boom"))
        assert compact_messages(messages, client, threshold=40) is None
        assert messages == original  # untouched


class FakeSendClient:
    """A ChatLoop-facing client: `send` for the normal turn path,
    `stream_chat` for summarize_messages, both on the same object as real
    LLMClient does."""

    provider = "claude"

    def __init__(self, summary="a tidy summary"):
        self._summary = summary

    def send(self, messages, on_text):
        on_text("ok")
        messages.append(assistant("ok"))

    async def stream_chat(self, messages, tools):
        yield self._summary


def make_loop(inputs, compact_after_messages=0, seed_messages=None):
    client = FakeSendClient()
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    loop = ChatLoop(
        client, console=quiet_console(), assistant_name="Nero", input_fn=next_input,
        compact_after_messages=compact_after_messages,
    )
    if seed_messages is not None:
        loop.messages = seed_messages
    return loop, client


class TestChatLoopCompactionWiring:
    def test_default_disabled_never_compacts(self):
        seed = []
        for i in range(25):
            seed.append(user(f"msg{i}"))
            seed.append(assistant(f"reply{i}"))
        loop, _ = make_loop(["one more", "exit"], compact_after_messages=0, seed_messages=seed)
        loop.run()
        assert len(loop.messages) == 52  # 50 seeded + user + assistant, untouched

    def test_fires_and_prints_dim_line_past_threshold(self):
        seed = []
        for i in range(20):
            seed.append(user(f"msg{i}"))
            seed.append(assistant(f"reply{i}"))
        console = quiet_console()
        loop, _ = make_loop(["one more", "exit"], compact_after_messages=10, seed_messages=seed)
        loop.console = console
        loop.run()
        output = console.file.getvalue()
        assert "compacted" in output
        assert loop.messages[0]["content"].startswith("[Earlier conversation summary]")

    def test_turn_rollback_still_works_after_compaction(self):
        seed = []
        for i in range(20):
            seed.append(user(f"msg{i}"))
            seed.append(assistant(f"reply{i}"))

        class FailingClient(FakeSendClient):
            def send(self, messages, on_text):
                raise RuntimeError("boom")

        client = FailingClient()
        queue = ["trigger", "exit"]

        def next_input(prompt):
            if not queue:
                raise EOFError
            return queue.pop(0)

        loop = ChatLoop(
            client, console=quiet_console(), assistant_name="Nero", input_fn=next_input,
            compact_after_messages=10,
        )
        loop.messages = seed
        before_len = len(loop.messages)
        loop.run()
        # Compaction fired (over threshold), then the turn failed and rolled
        # back to just the post-compaction state (summary + kept tail),
        # never a mix of pre- and post-compaction messages.
        assert loop.messages[0]["content"].startswith("[Earlier conversation summary]")
        assert len(loop.messages) < before_len

import asyncio
import io
from types import SimpleNamespace

import httpx
import litellm
import pytest
from rich.console import Console

from nero.config.schema import LLMConfig
from nero.core.chat_loop import ChatLoop
from nero.llm.client import APOLOGY, LLMClient, PendingToolCall, RoundResult
from nero.skills.base import Skill, SkillMeta
from nero.skills.registry import SkillRegistry


def quiet_console() -> Console:
    """A Console that swallows ChatLoop's output.

    StringIO rather than os.devnull: nothing here reads the text back, so an
    in-memory buffer needs no file handle (the old `open(...)` calls were never
    closed) and names no per-platform device path. The hardcoded "/dev/null"
    this replaces failed every windows-latest CI run with FileNotFoundError.
    """
    return Console(file=io.StringIO(), force_terminal=False)


class StubTool(Skill):
    meta = SkillMeta(
        name="open_app",
        description="stub",
        input_schema={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
        requires_network=False,
        permission_tier="state_changing",
    )

    def __init__(self, result="done", error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._result


def make_client(provider="claude", model="claude-sonnet-5", tools=None, api_key="sk-test", registry=None):
    skills = tools if tools is not None else [StubTool(result="Opened Safari.")]
    return LLMClient(
        config=LLMConfig(provider=provider, model=model),
        assistant_name="Nero",
        registry=registry if registry is not None else SkillRegistry(skills),
        api_key=api_key,
    )


def pending(call_id="call_1", name="open_app", raw='{"app_name": "Safari"}', arguments=...):
    if arguments is ...:
        arguments = {"app_name": "Safari"}
    return PendingToolCall(id=call_id, name=name, raw_arguments=raw, arguments=arguments)


class TestModelString:
    @pytest.mark.parametrize(
        ("provider", "model", "expected"),
        [
            ("claude", "claude-sonnet-5", "claude-sonnet-5"),
            ("openai", "gpt-5", "gpt-5"),
            ("gemini", "gemini-2.5-pro", "gemini/gemini-2.5-pro"),
            ("ollama", "llama3.2:3b", "ollama/llama3.2:3b"),
            ("gemini", "gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),  # no double prefix
        ],
    )
    def test_litellm_model(self, provider, model, expected):
        assert make_client(provider=provider, model=model).litellm_model == expected

    def test_provider_exposed(self):
        assert make_client(provider="ollama", model="qwen3:8b").provider == "ollama"


class TestToolDefinitions:
    def test_openai_function_shape(self):
        stub = StubTool()
        definitions = make_client(tools=[stub])._tool_definitions()
        assert definitions == [
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "stub",
                    "parameters": stub.meta.input_schema,
                },
            }
        ]


class TestExecuteTool:
    def run(self, client, name, arguments):
        return asyncio.run(client._execute_tool(name, arguments))

    def test_success(self):
        assert self.run(make_client(), "open_app", {"app_name": "Safari"}) == "Opened Safari."

    def test_unknown_tool(self):
        result = self.run(make_client(), "format_disk", {})
        assert "Error" in result and "format_disk" in result

    def test_unparseable_arguments(self):
        result = self.run(make_client(), "open_app", None)
        assert "Error" in result and "JSON" in result

    def test_tool_exception_becomes_error_string(self):
        client = make_client(tools=[StubTool(error=RuntimeError("boom"))])
        result = self.run(client, "open_app", {})
        assert "Error" in result and "boom" in result


class TestPendingFromLitellmMessage:
    def message(self, calls):
        return SimpleNamespace(content=None, tool_calls=calls)

    def call(self, call_id, arguments):
        return SimpleNamespace(
            id=call_id, function=SimpleNamespace(name="open_app", arguments=arguments)
        )

    def test_missing_id_gets_fallback(self):
        result = make_client()._pending_from_message(self.message([self.call(None, "{}")]))
        assert result[0].id == "call_0"

    def test_malformed_arguments_become_none(self):
        result = make_client()._pending_from_message(
            self.message([self.call("c1", '{"app_name": "Saf')])
        )
        assert result[0].arguments is None
        assert result[0].raw_arguments == '{"app_name": "Saf'

    def test_valid_arguments_parsed(self):
        result = make_client()._pending_from_message(
            self.message([self.call("c1", '{"app_name": "Safari"}')])
        )
        assert result[0].arguments == {"app_name": "Safari"}


def fake_turns(client, turns):
    """Replace client.stream_chat with canned (chunks, RoundResult) turns."""
    remaining = list(turns)

    async def fake_stream(messages, tools):
        chunks, round_result = remaining.pop(0)
        client._last_round = round_result
        for chunk in chunks:
            yield chunk

    client.stream_chat = fake_stream
    return client


BLOB = '{"name": "open_app", "arguments":{"app_name": "Safari"}}'


class TestSendLoop:
    def test_text_only_turn_streams_live(self):
        client = make_client()
        fake_turns(client, [(["hi ", "there"], RoundResult(content="hi there", tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "hello"}]
        client.send(messages, on_text=collected.append)
        assert collected == ["hi ", "there"]  # streamed as chunks, not buffered
        assert messages[-1] == {"role": "assistant", "content": "hi there"}

    def test_structured_tool_call_turn(self):
        stub = StubTool(result="Opened Safari.")
        client = make_client(tools=[stub])
        fake_turns(
            client,
            [
                ([], RoundResult(content=None, tool_calls=[pending()])),
                (["Done!"], RoundResult(content="Done!", tool_calls=[])),
            ],
        )
        messages = [{"role": "user", "content": "open safari"}]
        client.send(messages, on_text=lambda t: None)
        assert stub.calls == [{"app_name": "Safari"}]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[1]["content"] == ""  # never raw JSON
        assert messages[1]["tool_calls"][0]["function"]["name"] == "open_app"
        assert messages[2]["tool_call_id"] == "call_1"
        assert messages[2]["content"] == "Opened Safari."

    def test_fallback_blob_executes_tool_and_never_reaches_user(self):
        stub = StubTool(result="Opened Safari.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        fake_turns(
            client,
            [
                # The upstream bug: tool call leaks through as raw JSON content.
                ([BLOB[:20], BLOB[20:]], RoundResult(content=BLOB, tool_calls=[])),
                (["Opening Safari now."], RoundResult(content="Opening Safari now.", tool_calls=[])),
            ],
        )
        collected, messages = [], [{"role": "user", "content": "open safari"}]
        client.send(messages, on_text=collected.append)
        shown = "".join(collected)
        assert '"name"' not in shown and "{" not in shown
        assert "Opening Safari now." in shown
        assert stub.calls == [{"app_name": "Safari"}]
        # History: structured pair + clean text, no malformed JSON anywhere.
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        for m in messages:
            if m["role"] == "assistant":
                assert '"name"' not in (m["content"] or "")
        assert messages[-1]["content"] == "Opening Safari now."

    def test_non_tool_json_reply_is_shown(self):
        client = make_client()
        reply = '{"greeting": "hi"}'
        fake_turns(client, [([reply], RoundResult(content=reply, tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "reply in JSON please"}]
        client.send(messages, on_text=collected.append)
        assert "".join(collected) == reply  # legitimate JSON content is not swallowed
        assert messages[-1] == {"role": "assistant", "content": reply}

    def test_broken_tool_shaped_json_gets_apology(self):
        client = make_client()
        garbled = '{"name": "open_app", "arguments":{"app_na'
        fake_turns(client, [([garbled], RoundResult(content=garbled, tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "open safari"}]
        client.send(messages, on_text=collected.append)
        shown = "".join(collected)
        assert garbled not in shown
        assert APOLOGY in shown
        assert messages[-1] == {"role": "assistant", "content": APOLOGY}

    def test_empty_response_gets_apology(self):
        client = make_client()
        fake_turns(client, [([], RoundResult(content=None, tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "hello"}]
        client.send(messages, on_text=collected.append)
        assert APOLOGY in "".join(collected)
        assert messages[-1] == {"role": "assistant", "content": APOLOGY}

    def test_issue_a_array_blob_executes_and_confirms(self):
        """`open calculator` → tool actually runs; user sees confirmation, never the array."""
        stub = StubTool(result="Opened Calculator.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        array_blob = '[{"name": "open_app", "arguments": {"app_name": "Calculator"}}]'
        fake_turns(
            client,
            [
                ([array_blob], RoundResult(content=array_blob, tool_calls=[])),
                (
                    ["Opening Calculator for you."],
                    RoundResult(content="Opening Calculator for you.", tool_calls=[]),
                ),
            ],
        )
        collected, messages = [], [{"role": "user", "content": "open calculator"}]
        client.send(messages, on_text=collected.append)
        shown = "".join(collected)
        assert "[" not in shown and '"name"' not in shown
        assert shown == "Opening Calculator for you."
        assert stub.calls == [{"app_name": "Calculator"}]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]

    def test_issue_b_filler_content_alongside_tool_call_is_suppressed(self):
        """Mixed content + tool_calls on ollama: the filler text is not shown."""
        stub = StubTool(result="Opened Safari.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        filler = "Hello! How can I assist you today? \U0001f60a"
        fake_turns(
            client,
            [
                ([filler], RoundResult(content=filler, tool_calls=[pending(arguments={"app_name": "Safari"})])),
                (["Safari is open."], RoundResult(content="Safari is open.", tool_calls=[])),
            ],
        )
        collected, messages = [], [{"role": "user", "content": "hi"}]
        client.send(messages, on_text=collected.append)
        shown = "".join(collected)
        assert filler not in shown
        assert shown == "Safari is open."
        assert stub.calls == [{"app_name": "Safari"}]

    def test_ollama_plain_text_round_is_shown_after_buffering(self):
        client = make_client(provider="ollama", model="llama3.2:3b", api_key=None)
        fake_turns(client, [(["Hey", " there!"], RoundResult(content="Hey there!", tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "hi"}]
        client.send(messages, on_text=collected.append)
        assert "".join(collected) == "Hey there!"
        assert messages[-1] == {"role": "assistant", "content": "Hey there!"}

    def test_verbatim_sequence_hi_then_open_calculator(self):
        """The bug report's sequence: clean output at each step."""
        stub = StubTool(result="Opened Calculator.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        array_blob = '[{"name": "open_app", "arguments": {"app_name": "Calculator"}}]'
        fake_turns(
            client,
            [
                (["Hey! How can I help?"], RoundResult(content="Hey! How can I help?", tool_calls=[])),
                ([array_blob], RoundResult(content=array_blob, tool_calls=[])),
                (["Opening Calculator for you."], RoundResult(content="Opening Calculator for you.", tool_calls=[])),
            ],
        )
        messages, outputs = [], []
        for user_text in ["hi", "open calculator"]:
            turn_output = []
            messages.append({"role": "user", "content": user_text})
            client.send(messages, on_text=turn_output.append)
            outputs.append("".join(turn_output))
        assert outputs[0] == "Hey! How can I help?"  # greeting: no tool call, no JSON
        assert outputs[1] == "Opening Calculator for you."
        assert stub.calls == [{"app_name": "Calculator"}]
        for m in messages:
            if m["role"] == "assistant":
                assert '"name"' not in (m["content"] or "")

    def test_malformed_blob_after_content_shows_text_not_json(self):
        """`hi` → greeting + malformed [empty-arg, junk {}] blob: greeting shown, blob gone."""
        stub = StubTool()
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        greeting = "Hello! How can I assist you today?"
        content = greeting + '\n[{"name": "open_app", "arguments": {"app_name": ""}}, {}\n]'
        fake_turns(client, [([content], RoundResult(content=content, tool_calls=[]))])
        collected, messages = [], [{"role": "user", "content": "hi"}]
        client.send(messages, on_text=collected.append)
        shown = "".join(collected)
        assert shown.strip() == greeting
        assert "{" not in shown and "[" not in shown
        assert stub.calls == []  # phantom call must not execute
        # History carries the clean greeting, never the blob.
        assert messages[-1]["role"] == "assistant"
        assert '"name"' not in messages[-1]["content"]
        assert messages[-1]["content"].strip() == greeting

    def test_malformed_structured_call_with_content_shows_content(self):
        """Structured call with an empty app_name is discarded; the text still shows."""
        stub = StubTool()
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        fake_turns(
            client,
            [
                (
                    ["Sure, which app?"],
                    RoundResult(
                        content="Sure, which app?",
                        tool_calls=[pending(name="open_app", raw='{"app_name": ""}', arguments={"app_name": ""})],
                    ),
                )
            ],
        )
        collected, messages = [], [{"role": "user", "content": "open the thing"}]
        client.send(messages, on_text=collected.append)
        assert "".join(collected) == "Sure, which app?"
        assert stub.calls == []
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[-1]["content"] == "Sure, which app?"

    def test_missing_arguments_structured_call_is_discarded(self):
        stub = StubTool()
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        fake_turns(
            client,
            [(["Which app should I open?"], RoundResult(
                content="Which app should I open?",
                tool_calls=[pending(name="open_app", raw="{}", arguments={})],
            ))],
        )
        collected, messages = [], [{"role": "user", "content": "open"}]
        client.send(messages, on_text=collected.append)
        assert "".join(collected) == "Which app should I open?"
        assert stub.calls == []

    def test_valid_call_still_executes_after_classifier(self):
        """Regression: the working calculator path must still fire."""
        stub = StubTool(result="Opened Calculator.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        fake_turns(
            client,
            [
                ([], RoundResult(content=None, tool_calls=[pending(arguments={"app_name": "Calculator"})])),
                (["Opening Calculator."], RoundResult(content="Opening Calculator.", tool_calls=[])),
            ],
        )
        collected, messages = [], [{"role": "user", "content": "open calculator"}]
        client.send(messages, on_text=collected.append)
        assert stub.calls == [{"app_name": "Calculator"}]
        assert "".join(collected) == "Opening Calculator."

    def test_system_prompt_gates_tool_use(self):
        prompt = make_client().system_prompt.lower()
        # Gating is preserved; wording changed when the general-purpose framing
        # was restored (see TestGeneralPurposeRegression).
        assert "only when the user explicitly asks for that action" in prompt
        assert "do not call any tool" in prompt

    def test_bug_report_repro_history_stays_clean(self):
        """hi → open_app → hi: no raw JSON ever persisted, no echo loop fuel."""
        stub = StubTool(result="Opened Terminal.")
        client = make_client(provider="ollama", model="llama3.2:3b", tools=[stub], api_key=None)
        blob = '{"name": "open_app", "arguments":{"app_name": "terminal"}}'
        fake_turns(
            client,
            [
                (["Hey! How can I help?"], RoundResult(content="Hey! How can I help?", tool_calls=[])),
                ([blob], RoundResult(content=blob, tool_calls=[])),
                (["Terminal is open."], RoundResult(content="Terminal is open.", tool_calls=[])),
                (["Still here!"], RoundResult(content="Still here!", tool_calls=[])),
            ],
        )
        messages = []
        for user_text in ["hi", "open_app", "hi"]:
            messages.append({"role": "user", "content": user_text})
            client.send(messages, on_text=lambda t: None)
        for m in messages:
            if m["role"] == "assistant":
                content = m["content"] or ""
                assert '"name"' not in content and '"arguments"' not in content
        assert stub.calls == [{"app_name": "terminal"}]


class TestOllamaPathWiring:
    def test_stream_chat_routes_ollama_off_litellm(self, monkeypatch):
        """The ollama provider must hit the native adapter, never litellm."""
        import nero.llm.client as client_module

        async def fail_acompletion(**kwargs):
            raise AssertionError("litellm.acompletion must not be called for ollama")

        monkeypatch.setattr(client_module.litellm, "acompletion", fail_acompletion)

        seen = {}

        async def fake_ollama_chat(base_url, model, messages, tools):
            seen.update(base_url=base_url, model=model, tools=tools)
            from nero.llm.ollama_adapter import OllamaChatResponse

            yield OllamaChatResponse(content="hello", tool_calls=None)

        monkeypatch.setattr(client_module, "ollama_chat", fake_ollama_chat)

        client = make_client(provider="ollama", model="qwen3:8b", api_key=None)

        async def run():
            return [t async for t in client.stream_chat([{"role": "user", "content": "hi"}], [])]

        chunks = asyncio.run(run())
        assert chunks == ["hello"]
        assert seen["model"] == "qwen3:8b"  # bare model name, no ollama/ prefix
        assert seen["base_url"].startswith("http://localhost:11434")
        assert client._last_round.content == "hello"

    def test_history_converted_to_ollama_format(self):
        client = make_client(provider="ollama", model="qwen3:8b", api_key=None)
        converted = client._to_ollama_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "open_app", "arguments": '{"app_name": "Safari"}'},
                    }
                ],
            }
        )
        # Ollama's native format wants dict arguments, not JSON strings.
        assert converted["tool_calls"][0]["function"]["arguments"] == {"app_name": "Safari"}
        tool_message = client._to_ollama_message(
            {"role": "tool", "tool_call_id": "call_0", "content": "Opened Safari."}
        )
        assert tool_message == {"role": "tool", "content": "Opened Safari."}


class FakeLLMClient:
    provider = "claude"

    def __init__(self, error: Exception | None = None):
        self._error = error
        self.send_calls: list[list[dict]] = []

    def send(self, messages, on_text):
        if self._error:
            raise self._error
        self.send_calls.append([dict(m) for m in messages])
        on_text("hi there")
        messages.append({"role": "assistant", "content": "hi there"})


def make_loop(inputs, client=None):
    client = client or FakeLLMClient()
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = quiet_console()
    loop = ChatLoop(client, console=console, assistant_name="Nero", input_fn=next_input)
    return loop, client


class FakeHistory:
    def __init__(self, seed=None):
        self._seed = seed or []
        self.appended = []

    def recent(self, limit=None):
        return list(self._seed)

    def append_turn(self, user, assistant):
        self.appended.append((user, assistant))


class TestChatLoopHistory:
    def _loop(self, inputs, client=None, history=None):
        client = client or FakeLLMClient()
        queue = list(inputs)

        def next_input(prompt):
            if not queue:
                raise EOFError
            return queue.pop(0)

        console = quiet_console()
        return ChatLoop(client, console=console, assistant_name="Nero",
                        input_fn=next_input, history=history), client

    def test_seeds_messages_from_history(self):
        seed = [{"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "reply"}]
        loop, _ = self._loop(["exit"], history=FakeHistory(seed))
        assert loop.messages == seed

    def test_no_history_starts_empty(self):
        loop, _ = self._loop(["exit"])
        assert loop.messages == []

    def test_successful_turn_is_appended(self):
        hist = FakeHistory()
        loop, _ = self._loop(["hello", "exit"], history=hist)
        loop.run()
        assert hist.appended == [("hello", "hi there")]

    def test_failed_turn_is_not_appended(self):
        hist = FakeHistory()
        loop, _ = self._loop(["boom", "exit"],
                             client=FakeLLMClient(error=RuntimeError("nope")),
                             history=hist)
        loop.run()
        assert hist.appended == []

    def test_max_rounds_fallthrough_not_appended(self):
        """send() can return normally without ever appending final assistant
        text (MAX_TOOL_ROUNDS exhausted) — messages[-1] is left as a tool
        result. That must never be persisted as the assistant's reply."""

        class MaxRoundsClient:
            provider = "claude"

            def send(self, messages, on_text):
                on_text("")
                messages.append(
                    {"role": "tool", "tool_call_id": "x", "content": "raw tool output"}
                )

        hist = FakeHistory()
        loop, _ = self._loop(["do it", "exit"], client=MaxRoundsClient(), history=hist)
        loop.run()
        assert hist.appended == []

    def test_tool_turn_persists_only_text_pair(self):
        """A completed tool turn appends assistant+tool_calls, then a tool
        result, then the final assistant text. Only the (user, final text)
        pair may be persisted — the intermediate tool-call scaffolding must
        never reach history. If append_turn ever switched to a fixed offset
        (e.g. messages[turn_start + 1]) instead of messages[-1], this test
        would catch it: that index would grab the tool_calls stub, not
        "Done!"."""

        class ToolTurnClient:
            provider = "claude"

            def send(self, messages, on_text):
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "x", "type": "function",
                         "function": {"name": "open_app", "arguments": '{"app_name": "Safari"}'}},
                    ]}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": "x", "content": "Opened Safari."}
                )
                on_text("Done!")
                messages.append({"role": "assistant", "content": "Done!"})

        hist = FakeHistory()
        loop, _ = self._loop(["open safari", "exit"], client=ToolTurnClient(), history=hist)
        loop.run()
        assert hist.appended == [("open safari", "Done!")]


class TestChatLoop:
    def test_exit_ends_loop(self):
        loop, client = make_loop(["exit"])
        loop.run()
        assert client.send_calls == []

    def test_quit_and_eof_end_loop(self):
        loop, _ = make_loop(["quit"])
        loop.run()
        loop, _ = make_loop([])  # immediate EOF
        loop.run()

    def test_turn_appends_history_and_calls_client(self):
        loop, client = make_loop(["open safari", "exit"])
        loop.run()
        assert len(client.send_calls) == 1
        assert client.send_calls[0][-1] == {"role": "user", "content": "open safari"}
        assert loop.messages[-1]["role"] == "assistant"

    def test_history_accumulates_across_turns(self):
        loop, client = make_loop(["one", "two", "exit"])
        loop.run()
        assert len(client.send_calls) == 2
        assert len(client.send_calls[1]) == 3

    def test_empty_input_skipped(self):
        loop, client = make_loop(["", "   ", "exit"])
        loop.run()
        assert client.send_calls == []

    def test_generic_error_survives_and_rolls_back(self):
        loop, _ = make_loop(["hello", "exit"], client=FakeLLMClient(error=RuntimeError("boom")))
        loop.run()  # must not raise
        assert loop.messages == []

    def test_httpx_connection_error_survives_and_rolls_back(self):
        loop, _ = make_loop(
            ["hello", "exit"], client=FakeLLMClient(error=httpx.ConnectError("refused"))
        )
        loop.run()
        assert loop.messages == []

    def test_auth_error_survives_and_rolls_back(self):
        error = litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="anthropic", model="claude-sonnet-5"
        )
        loop, _ = make_loop(["hello", "exit"], client=FakeLLMClient(error=error))
        loop.run()
        assert loop.messages == []

    def test_keyboard_interrupt_ends_cleanly(self):
        def raise_interrupt(prompt):
            raise KeyboardInterrupt

        console = quiet_console()
        loop = ChatLoop(
            FakeLLMClient(), console=console, assistant_name="Nero", input_fn=raise_interrupt
        )
        loop.run()  # must not raise


def four_skill_registry():
    from nero.skills.open_app.server import OpenAppSkill
    from nero.skills.open_website.server import OpenWebsiteSkill
    from nero.skills.play_music.server import PlayMusicSkill
    from nero.skills.registry import SkillRegistry
    from nero.skills.weather.server import WeatherSkill

    return SkillRegistry(
        [OpenAppSkill(), OpenWebsiteSkill(), WeatherSkill(), PlayMusicSkill()]
    )


# --- Regression: Nero is a general assistant that ALSO has a tool ---
class TestGeneralPurposeRegression:
    def test_prompt_states_general_capability_before_the_tool(self):
        """The tool must be framed as an addition, not as the whole job."""
        lowered = make_client().system_prompt.lower()
        assert "general-purpose" in lowered
        # General capabilities are named explicitly, so the model doesn't infer
        # that opening apps is its only job.
        for capability in ("maths", "facts", "jokes", "casual conversation"):
            assert capability in lowered, f"missing capability: {capability}"
        # And it is explicitly told not to self-limit to the tool.
        assert "never limited to one topic" in lowered
        assert "never say that you can only do one kind of task" in lowered
        # The general framing comes before the tool description.
        assert lowered.index("general-purpose") < lowered.index("a few tools")
        # Tool gating is still present.
        assert "only when the user explicitly asks for that action" in lowered

    def test_none_classification_passes_content_through_verbatim(self):
        """A plain answer must reach the user unmodified — no canned substitute."""
        answer = "100 plus 50 is 150."
        client = make_client()
        fake_turns(client, [([answer], RoundResult(content=answer, tool_calls=[]))])
        shown, messages = [], [{"role": "user", "content": "calculate 100 plus 50"}]
        client.send(messages, on_text=shown.append)
        assert "".join(shown) == answer
        assert messages[-1] == {"role": "assistant", "content": answer}
        assert APOLOGY not in "".join(shown)

    @pytest.mark.parametrize(
        "reply",
        [
            "100 plus 5 is 105.",
            "Publication tooling usually means typesetting and layout software.",
            "I'm doing well, thanks for asking! How can I help?",
            "Python was created by Guido van Rossum in 1991.",
            "Why did the chicken cross the road? To get to the other side.",
            "Photosynthesis converts light energy into chemical energy.",
        ],
    )
    def test_general_answers_are_not_swallowed(self, reply):
        """Varied general-knowledge answers all survive the classifier intact.

        Uses the full four-skill registry (not the single-skill default) so
        this exercises the fuller tool schema — the shape of prompt the
        Phase 3 regression actually broke under.
        """
        client = make_client(registry=four_skill_registry())
        fake_turns(client, [([reply], RoundResult(content=reply, tool_calls=[]))])
        shown = []
        client.send([{"role": "user", "content": "q"}], on_text=shown.append)
        assert "".join(shown) == reply

    def test_ollama_path_general_answer_survives_buffering(self):
        """The buffered ollama path must not eat plain answers either.

        Also uses the full four-skill registry: the original regression lived
        on this buffered ollama path, so it is the one that most needs
        full-schema coverage.
        """
        reply = "The Eiffel Tower is 330 metres tall."
        client = make_client(
            provider="ollama", model="phi4-mini", api_key=None, registry=four_skill_registry()
        )
        fake_turns(client, [([reply], RoundResult(content=reply, tool_calls=[]))])
        shown = []
        client.send([{"role": "user", "content": "how tall is it"}], on_text=shown.append)
        assert "".join(shown) == reply

    def test_tool_calling_still_works(self):
        """Regression check: restoring general chat must not break the tool."""
        stub = StubTool(result="Opened Calculator.")
        client = make_client(tools=[stub])
        fake_turns(
            client,
            [
                ([], RoundResult(content=None, tool_calls=[pending(raw='{"app_name": "Calculator"}',
                                                                  arguments={"app_name": "Calculator"})])),
                (["Calculator is open."], RoundResult(content="Calculator is open.", tool_calls=[])),
            ],
        )
        messages = [{"role": "user", "content": "open calculator"}]
        shown = []
        client.send(messages, on_text=shown.append)
        assert stub.calls == [{"app_name": "Calculator"}]
        assert "".join(shown) == "Calculator is open."


class TestSystemPromptFraming:
    def test_states_general_capability_first(self):
        prompt = make_client().system_prompt
        general = prompt.index("general-purpose")
        tools = prompt.index("tools")
        assert general < tools, "the general-assistant half must come first"

    def test_names_no_specific_skill(self):
        # Spec D4: per-skill detail lives in SkillMeta.description, not the
        # prompt, so prompt length stays constant as skills are added.
        prompt = make_client().system_prompt
        for name in ("open_app", "open_website", "get_weather", "play_music"):
            assert name not in prompt

    def test_forbids_narrowing_replies(self):
        assert "never limited" in make_client().system_prompt

    def test_forbids_tool_json_as_text(self):
        assert "never" in make_client().system_prompt
        assert "tool-call JSON" in make_client().system_prompt

    def test_stays_short(self):
        # The previous regression came from an over-long, over-specific prompt.
        # Small local models follow terse instructions far better.
        assert len(make_client().system_prompt) < 900


from nero.llm.ollama_adapter import OllamaModelError


class TestOllamaErrorMessaging:
    """A reachable Ollama refusing a model must not read as a dead server."""

    def _run(self, error, provider="ollama"):
        import io

        client = FakeLLMClient(error=error)
        client.provider = provider
        queue = ["what's up", "exit"]

        def next_input(prompt):
            if not queue:
                raise EOFError
            return queue.pop(0)

        buffer = io.StringIO()
        # Wide console: rich would otherwise wrap mid-command and break the
        # substring assertions below for reasons unrelated to the behaviour.
        console = Console(file=buffer, force_terminal=False, width=200)
        loop = ChatLoop(client, console=console, assistant_name="Nero", input_fn=next_input)
        loop.run()
        return buffer.getvalue(), loop

    def test_model_error_reports_the_model_not_the_server(self):
        out, _ = self._run(
            OllamaModelError(
                "Model 'gemma3n' isn't available locally. "
                "Pull it with `ollama pull gemma3n`, or pick another with `nero config`."
            )
        )
        assert "gemma3n" in out
        assert "ollama pull gemma3n" in out
        assert "Could not reach the model provider" not in out
        assert "make sure it's running" not in out
        # Must be handled by a dedicated branch, not the generic catch-all:
        # "Something went wrong" reads like a crash for a known, fixable state.
        assert "Something went wrong" not in out

    def test_connection_failure_still_suggests_starting_ollama(self):
        out, _ = self._run(httpx.ConnectError("connection refused"))
        assert "Could not reach the model provider" in out
        assert "ollama serve" in out

    def test_model_error_rolls_back_the_failed_turn(self):
        # Same contract as every other recoverable error: no orphan user turn.
        _, loop = self._run(OllamaModelError("Model 'gemma3n' isn't available locally."))
        assert loop.messages == []

    def test_loop_survives_a_model_error(self):
        out, _ = self._run(OllamaModelError("Model 'gemma3n' isn't available locally."))
        assert "signing off" in out


class TestAllSkillsFire:
    """Locks both halves at once: every skill still reachable, and ordinary
    questions still answered. The Phase 3 regression broke the second half while
    the first half kept working, so neither is sufficient alone."""

    def _registry(self):
        return four_skill_registry()

    def test_all_four_skills_are_offered_to_the_model(self):
        client = LLMClient(
            config=LLMConfig(provider="claude", model="claude-sonnet-5"),
            assistant_name="Nero",
            registry=self._registry(),
            api_key="sk-test",
        )
        names = {d["function"]["name"] for d in client._tool_definitions()}
        assert names == {"open_app", "open_website", "get_weather", "play_music"}

    def test_every_skill_description_is_non_empty(self):
        for skill in self._registry().available():
            assert skill.meta.description.strip(), skill.meta.name

    def test_every_skill_declares_a_tier_and_network_flag(self):
        for skill in self._registry().available():
            assert skill.meta.permission_tier in (
                "read_only", "state_changing", "destructive"
            )
            assert isinstance(skill.meta.requires_network, bool)

    def test_network_flags_match_the_spec(self):
        flags = {
            skill.meta.name: skill.meta.requires_network
            for skill in self._registry().available()
        }
        assert flags == {
            "open_app": False,
            "open_website": True,
            "get_weather": True,
            "play_music": False,
        }

    def test_every_network_skill_has_an_offline_message(self):
        for skill in self._registry().available():
            if skill.meta.requires_network:
                assert skill.meta.offline_message, skill.meta.name


class TestOllamaFullJsonReplyIsNoise:
    """Structural leak guard: on the ollama path, if a reply is entirely a JSON
    object/array AND tools were offered this turn, it is tool-call protocol noise
    the model narrated instead of calling — never show it. Replaces chasing each
    fabricated shape (function/result, status/message, escaped quotes, ...).
    """

    def _ollama_client(self, tools=None):
        return make_client(
            provider="ollama", model="phi4-mini",
            tools=tools if tools is not None else [StubTool(result="ok")],
            api_key=None,
        )

    def _run(self, client, content):
        fake_turns(client, [([content], RoundResult(content=content, tool_calls=[]))])
        shown, messages = [], [{"role": "user", "content": "open youtube"}]
        client.send(messages, on_text=shown.append)
        return "".join(shown), messages

    def test_fabricated_result_json_is_not_shown(self):
        shown, messages = self._run(
            self._ollama_client(),
            '[{"type": "function/result", "result": "YouTube opened"}]',
        )
        assert "function" not in shown and "YouTube opened" not in shown
        assert shown == APOLOGY
        assert messages[-1] == {"role": "assistant", "content": APOLOGY}

    def test_status_message_json_is_not_shown(self):
        shown, _ = self._run(
            self._ollama_client(),
            '{"status": "success", "message": "Opening YouTube"}',
        )
        assert shown == APOLOGY

    def test_json_with_leading_junk_is_not_shown(self):
        shown, _ = self._run(
            self._ollama_client(), '{}\n\n\n[{"status": "success"}]'
        )
        assert shown == APOLOGY

    def test_genuine_prose_answer_still_shows(self):
        shown, _ = self._run(self._ollama_client(), "Tokyo is the capital of Japan.")
        assert shown == "Tokyo is the capital of Japan."

    def test_json_reply_is_kept_when_no_tools_offered(self):
        # If Nero offered no tools, a JSON reply is the user's legitimate answer.
        shown, _ = self._run(
            self._ollama_client(tools=[]), '{"city": "Tokyo", "pop": 37400068}'
        )
        assert shown == '{"city": "Tokyo", "pop": 37400068}'

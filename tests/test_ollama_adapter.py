import asyncio
import json

import pytest

from nero.llm.ollama_adapter import (
    OllamaChatResponse,
    ToolCallRequest,
    _try_parse_json_tool_call,
    extract_tool_call,
    ollama_chat,
)


class TestFallbackParser:
    def test_parses_bug_report_blob(self):
        blob = '{"name": "open_app", "arguments":{"app_name": "terminal"}}'
        result = _try_parse_json_tool_call(blob)
        assert result == ToolCallRequest(name="open_app", arguments={"app_name": "terminal"})

    def test_parses_with_surrounding_whitespace(self):
        blob = '  \n{"name": "open_app", "arguments": {}}\n'
        assert _try_parse_json_tool_call(blob) is not None

    def test_plain_text_is_not_a_tool_call(self):
        assert _try_parse_json_tool_call("Hello! How can I help?") is None

    def test_non_tool_json_is_not_a_tool_call(self):
        assert _try_parse_json_tool_call('{"greeting": "hi"}') is None

    def test_broken_json_is_not_a_tool_call(self):
        assert _try_parse_json_tool_call('{"name": "open_app", "arguments":{"app') is None

    def test_array_wrapped_tool_call_is_parsed(self):
        # The Issue A leak shape: some models emit the tool_calls list itself.
        blob = '[{"name": "open_app", "arguments": {"app_name": "Calculator"}}]'
        result = _try_parse_json_tool_call(blob)
        assert result == ToolCallRequest(name="open_app", arguments={"app_name": "Calculator"})

    def test_array_takes_first_valid_entry(self):
        blob = '[{"junk": 1}, {"name": "open_app", "arguments": {}}]'
        result = _try_parse_json_tool_call(blob)
        assert result is not None and result.name == "open_app"

    def test_empty_or_junk_arrays_rejected(self):
        assert _try_parse_json_tool_call("[]") is None
        assert _try_parse_json_tool_call('["name", "arguments"]') is None
        assert _try_parse_json_tool_call('[{"name": 5, "arguments": {}}]') is None

    def test_none_and_empty_content(self):
        assert _try_parse_json_tool_call(None) is None
        assert _try_parse_json_tool_call("") is None

    def test_stringified_arguments_are_coerced(self):
        blob = '{"name": "open_app", "arguments": "{\\"app_name\\": \\"Safari\\"}"}'
        result = _try_parse_json_tool_call(blob)
        assert result == ToolCallRequest(name="open_app", arguments={"app_name": "Safari"})

    def test_non_dict_arguments_rejected(self):
        assert _try_parse_json_tool_call('{"name": "open_app", "arguments": 5}') is None


class TestExtractToolCall:
    def test_structured_field_wins(self):
        response = OllamaChatResponse(
            content='{"name": "wrong", "arguments": {}}',
            tool_calls=[ToolCallRequest(name="open_app", arguments={"app_name": "Safari"})],
        )
        assert extract_tool_call(response).name == "open_app"

    def test_fallback_from_content(self):
        response = OllamaChatResponse(
            content='{"name": "open_app", "arguments": {"app_name": "Safari"}}',
            tool_calls=None,
        )
        result = extract_tool_call(response)
        assert result is not None and result.arguments == {"app_name": "Safari"}

    def test_neither(self):
        assert extract_tool_call(OllamaChatResponse(content="hi there", tool_calls=None)) is None


class FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeStreamContext:
    def __init__(self, lines, calls, method, url, payload):
        self._lines = lines
        calls.append({"method": method, "url": url, "payload": payload})

    async def __aenter__(self):
        return FakeStreamResponse(self._lines)

    async def __aexit__(self, *args):
        return False


class FakeAsyncClient:
    def __init__(self, lines, calls):
        self._lines = lines
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None):
        return FakeStreamContext(self._lines, self._calls, method, url, json)


@pytest.fixture
def fake_ollama(monkeypatch):
    def install(lines):
        calls = []
        monkeypatch.setattr(
            "httpx.AsyncClient", lambda **kwargs: FakeAsyncClient(lines, calls)
        )
        return calls

    return install


def collect(base_url="http://localhost:11434", model="qwen3:8b", messages=None, tools=None):
    async def run():
        return [
            r
            async for r in ollama_chat(base_url, model, messages or [], tools or [])
        ]

    return asyncio.run(run())


class TestOllamaChat:
    def test_streams_content_and_parses_native_tool_calls(self, fake_ollama):
        calls = fake_ollama(
            [
                json.dumps({"message": {"role": "assistant", "content": "Hel"}, "done": False}),
                json.dumps({"message": {"content": "lo"}, "done": False}),
                "",  # blank keep-alive line must be skipped
                json.dumps(
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "open_app", "arguments": {"app_name": "Safari"}}}
                            ],
                        },
                        "done": True,
                    }
                ),
            ]
        )
        tools = [{"type": "function", "function": {"name": "open_app"}}]
        responses = collect(tools=tools)
        assert [r.content for r in responses if r.content] == ["Hel", "lo"]
        tool_bearing = [r for r in responses if r.tool_calls]
        assert len(tool_bearing) == 1
        assert tool_bearing[0].tool_calls[0] == ToolCallRequest(
            name="open_app", arguments={"app_name": "Safari"}
        )
        # Hits the native chat endpoint with tools in the payload — no LiteLLM.
        request = calls[0]
        assert request["url"] == "http://localhost:11434/api/chat"
        assert request["payload"]["model"] == "qwen3:8b"
        assert request["payload"]["stream"] is True
        assert request["payload"]["tools"] == tools
        # Tool selection is a decision task — low temperature for consistency.
        assert request["payload"]["options"]["temperature"] == 0.2

    def test_custom_base_url(self, fake_ollama):
        calls = fake_ollama([json.dumps({"message": {"content": "ok"}, "done": True})])
        collect(base_url="http://otherhost:9999")
        assert calls[0]["url"] == "http://otherhost:9999/api/chat"

    def test_no_tools_key_when_empty(self, fake_ollama):
        calls = fake_ollama([json.dumps({"message": {"content": "ok"}, "done": True})])
        collect(tools=[])
        assert "tools" not in calls[0]["payload"]


from nero.llm.ollama_adapter import ToolCallOutcome, classify_tool_call


def valid_if_nonempty_app_name(call):
    """Stand-in schema validator: open_app requires a non-empty app_name."""
    if call.name != "open_app":
        return False
    app = call.arguments.get("app_name")
    return isinstance(app, str) and bool(app.strip())


NAMES = {"open_app"}


def classify(content):
    return classify_tool_call(content, NAMES, valid_if_nonempty_app_name)


class TestClassifyToolCall:
    def test_valid_single_object(self):
        outcome, call, _ = classify('{"name": "open_app", "arguments": {"app_name": "Calculator"}}')
        assert outcome is ToolCallOutcome.VALID
        assert call.arguments == {"app_name": "Calculator"}

    def test_valid_array(self):
        outcome, call, _ = classify('[{"name": "open_app", "arguments": {"app_name": "Safari"}}]')
        assert outcome is ToolCallOutcome.VALID and call.name == "open_app"

    def test_malformed_empty_arg_value(self):
        outcome, call, _ = classify('{"name": "open_app", "arguments": {"app_name": ""}}')
        assert outcome is ToolCallOutcome.MALFORMED
        assert call is None

    def test_malformed_junk_array_element_embedded_after_text(self):
        content = (
            "Hello! How can I assist you today?\n"
            '[{"name": "open_app", "arguments": {"app_name": ""}}, {}\n]'
        )
        outcome, call, cleaned = classify(content)
        assert outcome is ToolCallOutcome.MALFORMED
        assert call is None
        assert cleaned.strip() == "Hello! How can I assist you today?"
        assert "{" not in cleaned and "[" not in cleaned

    def test_malformed_missing_arguments_key(self):
        outcome, call, _ = classify('{"name": "open_app"}')
        assert outcome is ToolCallOutcome.MALFORMED
        assert call is None

    def test_unknown_tool_blob_is_malformed(self):
        outcome, _, _ = classify('{"name": "delete_everything", "arguments": {}}')
        assert outcome is ToolCallOutcome.MALFORMED

    def test_legit_json_with_name_field_is_not_a_tool_call(self):
        # A "name" field in ordinary JSON content must not be over-suppressed.
        content = '{"name": "Alice", "age": 30}'
        outcome, call, cleaned = classify(content)
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == content

    def test_plain_text_is_none(self):
        outcome, _, cleaned = classify("Hello! How can I help you today?")
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == "Hello! How can I help you today?"

    def test_none_and_empty(self):
        assert classify(None)[0] is ToolCallOutcome.NONE
        assert classify("")[0] is ToolCallOutcome.NONE

    def test_truncated_tool_shape_is_malformed(self):
        outcome, call, cleaned = classify('{"name": "open_app", "arguments":{"app_na')
        assert outcome is ToolCallOutcome.MALFORMED
        assert cleaned == ""

    def test_valid_takes_precedence_over_junk_in_array(self):
        content = '[{}, {"name": "open_app", "arguments": {"app_name": "Terminal"}}]'
        outcome, call, _ = classify(content)
        assert outcome is ToolCallOutcome.VALID
        assert call.arguments == {"app_name": "Terminal"}


class TestOpenAIWireFormatToolCalls:
    """phi4-mini emits tool calls as text in OpenAI format, sometimes with
    unbalanced brackets. Both must still be recognised and executed."""

    def test_nested_function_shape_is_coerced(self):
        blob = '{"type": "function", "function": {"name": "open_app", "arguments": {"app_name": "Safari"}}}'
        outcome, call, _ = classify(blob)
        assert outcome is ToolCallOutcome.VALID
        assert call.name == "open_app"
        assert call.arguments == {"app_name": "Safari"}

    def test_nested_function_in_array(self):
        blob = '[{"type": "function", "function": {"name": "open_app", "arguments": {"app_name": "Calendar"}}}]'
        outcome, call, _ = classify(blob)
        assert outcome is ToolCallOutcome.VALID
        assert call.arguments == {"app_name": "Calendar"}

    def test_real_phi4_mini_unbalanced_output_is_salvaged(self):
        """Verbatim capture from phi4-mini: note the missing '}' before ']'."""
        blob = '[{"type": "function", "function": {"name": "open_app", "arguments": {"app_name": "Calculator"}}]}'
        outcome, call, _ = classify(blob)
        assert outcome is ToolCallOutcome.VALID, "unbalanced OpenAI-format call must be salvaged"
        assert call.name == "open_app"
        assert call.arguments == {"app_name": "Calculator"}

    def test_salvage_still_rejects_empty_app_name(self):
        blob = '[{"type": "function", "function": {"name": "open_app", "arguments": {"app_name": ""}}]}'
        outcome, call, _ = classify(blob)
        assert outcome is ToolCallOutcome.MALFORMED
        assert call is None

    def test_salvage_does_not_fire_on_ordinary_prose(self):
        outcome, _, cleaned = classify("Tokyo is the capital of Japan.")
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == "Tokyo is the capital of Japan."

import asyncio
import json

import httpx
import pytest

from nero.llm.ollama_adapter import (
    OllamaChatResponse,
    OllamaModelError,
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
    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        pass

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeStreamContext:
    def __init__(self, lines, calls, method, url, payload, status_code=200, body=b""):
        self._lines = lines
        self._status_code = status_code
        self._body = body
        calls.append({"method": method, "url": url, "payload": payload})

    async def __aenter__(self):
        return FakeStreamResponse(self._lines, self._status_code, self._body)

    async def __aexit__(self, *args):
        return False


class FakeAsyncClient:
    def __init__(self, lines, calls, status_code=200, body=b"", connect_error=None):
        self._lines = lines
        self._calls = calls
        self._status_code = status_code
        self._body = body
        self._connect_error = connect_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None):
        if self._connect_error is not None:
            raise self._connect_error
        return FakeStreamContext(
            self._lines, self._calls, method, url, json, self._status_code, self._body
        )


@pytest.fixture
def fake_ollama(monkeypatch):
    def install(lines):
        calls = []
        monkeypatch.setattr(
            "httpx.AsyncClient", lambda **kwargs: FakeAsyncClient(lines, calls)
        )
        return calls

    return install


@pytest.fixture
def fake_ollama_failure(monkeypatch):
    """Install a fake client that fails: either an HTTP error response, or a
    transport-level connection error. The two must stay distinguishable."""

    def install(status_code=200, body=b"", connect_error=None):
        calls = []
        monkeypatch.setattr(
            "httpx.AsyncClient",
            lambda **kwargs: FakeAsyncClient(
                [], calls, status_code, body, connect_error
            ),
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


class TestModelVersusConnectionErrors:
    """Ollama answering with an error body is NOT the server being unreachable.

    Regression: `raise_for_status()` raised httpx.HTTPStatusError, which
    subclasses httpx.HTTPError — the same class the chat loop catches for
    genuine connection failures. A 404 for a mistyped model therefore surfaced
    as "Could not reach the model provider... make sure it's running", while
    Ollama was demonstrably running and serving other models fine.
    """

    def test_missing_model_raises_model_error_not_http_error(self, fake_ollama_failure):
        fake_ollama_failure(
            status_code=404, body=b'{"error":"model \'gemma3n\' not found"}'
        )
        with pytest.raises(OllamaModelError):
            collect(model="gemma3n")

    def test_missing_model_error_is_not_an_httpx_error(self, fake_ollama_failure):
        # The heart of the bug: if this inherits from httpx.HTTPError, the chat
        # loop's connection-failure branch swallows it again.
        fake_ollama_failure(
            status_code=404, body=b'{"error":"model \'gemma3n\' not found"}'
        )
        try:
            collect(model="gemma3n")
        except OllamaModelError as exc:
            assert not isinstance(exc, httpx.HTTPError)
        else:
            pytest.fail("expected OllamaModelError")

    def test_missing_model_message_names_the_model_and_the_pull_command(
        self, fake_ollama_failure
    ):
        fake_ollama_failure(
            status_code=404, body=b'{"error":"model \'gemma3n\' not found"}'
        )
        with pytest.raises(OllamaModelError) as caught:
            collect(model="gemma3n")
        message = str(caught.value)
        assert "gemma3n" in message
        assert "ollama pull gemma3n" in message
        assert "running" not in message.lower()

    def test_unusable_model_relays_ollamas_own_explanation(self, fake_ollama_failure):
        # A pulled but embedding-only model (nomic-embed-text) 400s on /api/chat.
        # No list-models preflight can catch this — only the response can.
        fake_ollama_failure(
            status_code=400, body=b'{"error":"\\"nomic-embed-text\\" does not support chat"}'
        )
        with pytest.raises(OllamaModelError) as caught:
            collect(model="nomic-embed-text")
        message = str(caught.value)
        assert "does not support chat" in message
        assert "nomic-embed-text" in message

    def test_error_response_without_parseable_body_still_explains(
        self, fake_ollama_failure
    ):
        fake_ollama_failure(status_code=500, body=b"<html>gateway blew up</html>")
        with pytest.raises(OllamaModelError) as caught:
            collect(model="gemma3")
        message = str(caught.value)
        assert "gemma3" in message
        assert "500" in message

    def test_connection_failure_still_raises_an_httpx_error(self, fake_ollama_failure):
        # The other half of the fix: a genuinely unreachable server must keep
        # raising an httpx error so the "is Ollama running?" hint still fires.
        fake_ollama_failure(connect_error=httpx.ConnectError("connection refused"))
        with pytest.raises(httpx.ConnectError):
            collect(model="gemma3")

    def test_connection_failure_is_not_a_model_error(self, fake_ollama_failure):
        fake_ollama_failure(connect_error=httpx.ConnectError("connection refused"))
        try:
            collect(model="gemma3")
        except httpx.ConnectError as exc:
            assert not isinstance(exc, OllamaModelError)
        else:
            pytest.fail("expected httpx.ConnectError")

    def test_successful_response_is_unaffected(self, fake_ollama):
        fake_ollama([json.dumps({"message": {"content": "hi"}, "done": True})])
        responses = collect(model="gemma3")
        assert [r.content for r in responses] == ["hi"]


class TestFabricatedToolResults:
    """Small models sometimes narrate a tool RESULT as JSON text.

    Captured verbatim from phi4-mini on "open YouTube". The shape carries
    neither "name" nor "arguments", so it used to slip past _is_tool_attempt,
    classify as NONE, and get printed to the user as if it were the reply.
    """

    NAMES = {"open_app", "open_website", "get_weather", "play_music"}

    def classify(self, content):
        return classify_tool_call(content, self.NAMES, lambda call: True)

    def test_fabricated_result_is_not_shown_to_the_user(self):
        blob = '[{"type": "function", "result": true, "message": "YouTube has been opened"}]'
        outcome, call, cleaned = self.classify(blob)
        assert call is None
        assert '"result"' not in cleaned
        assert "YouTube has been opened" not in cleaned

    def test_fabricated_result_after_real_text_keeps_the_text(self):
        blob = (
            'Opening YouTube for you. '
            '[{"type": "function", "result": true, "message": "done"}]'
        )
        _outcome, _call, cleaned = self.classify(blob)
        assert "Opening YouTube for you." in cleaned
        assert '"result"' not in cleaned

    def test_bare_function_envelope_is_not_shown(self):
        # Same family: a "type": "function" wrapper with no call inside it.
        blob = '{"type": "function", "status": "ok"}'
        _outcome, _call, cleaned = self.classify(blob)
        assert '"function"' not in cleaned

    def test_definition_echo_is_still_stripped(self):
        # The other phi4-mini failure shape — echoes the tool definition back.
        blob = (
            '[{"type": "function", "function": {"name": "open_website", '
            '"description": "Open a website.", "parameters": {}}}]'
        )
        outcome, call, cleaned = self.classify(blob)
        assert outcome is ToolCallOutcome.MALFORMED
        assert call is None
        assert "open_website" not in cleaned

    def test_ordinary_json_answer_is_still_shown(self):
        # Must not over-trigger: a genuine JSON answer has no tool-call shape.
        blob = '{"population": 37400068, "city": "Tokyo"}'
        outcome, _call, cleaned = self.classify(blob)
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == blob

    def test_plain_prose_is_untouched(self):
        outcome, _call, cleaned = self.classify("The capital of Japan is Tokyo.")
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == "The capital of Japan is Tokyo."


class TestBracketResidue:
    """Stripping a tool blob must not leave punctuation posing as the reply.

    An unbalanced blob (phi4-mini drops closing brackets often) gets matched
    from its inner `{`, leaving the orphaned `[` behind. Showing a lone bracket
    as Nero's answer is the same defect as showing the JSON, just smaller.
    """

    NAMES = {"open_app", "open_website"}

    def classify(self, content):
        return classify_tool_call(content, self.NAMES, lambda call: True)

    def test_orphaned_open_bracket_is_not_the_reply(self):
        blob = '[{"type": "function", "function": {"name": "open_website", "arguments": {}}}'
        _outcome, _call, cleaned = self.classify(blob)
        assert cleaned.strip() == ""

    def test_orphaned_close_bracket_is_not_the_reply(self):
        blob = '{"name": "open_app", "arguments": {"app_name": "Safari"}}]'
        _outcome, _call, cleaned = self.classify(blob)
        assert cleaned.strip() == ""

    def test_real_text_around_a_blob_survives(self):
        blob = 'Sure, opening it. [{"name": "open_app", "arguments": {"app_name": "Safari"}}]'
        _outcome, _call, cleaned = self.classify(blob)
        assert "Sure, opening it." in cleaned

    def test_prose_with_brackets_is_untouched(self):
        text = "Use the list [1, 2, 3] for that."
        outcome, _call, cleaned = self.classify(text)
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == text


class TestEscapedToolJson:
    """phi4-mini sometimes emits the blob with backslash-escaped quotes.

    Captured verbatim on "open YouTube". The marker check looked for `"name"`,
    but the raw text contains `\\"name\\"`, so the region was never recognised
    as protocol noise and printed as the reply.
    """

    NAMES = {"open_app", "open_website"}

    def classify(self, content):
        return classify_tool_call(content, self.NAMES, lambda call: True)

    def test_escaped_definition_echo_is_not_shown(self):
        blob = (
            '{\\"type\\": \\"function\\", \\"function\\": '
            '{\\"name\\": \\"open_website\\", \\"parameters\\": {}}}'
        )
        _outcome, _call, cleaned = self.classify(blob)
        assert "open_website" not in cleaned
        assert "function" not in cleaned

    def test_escaped_result_is_not_shown(self):
        blob = '[{\\"type\\": \\"function\\", \\"result\\": true}]'
        _outcome, _call, cleaned = self.classify(blob)
        assert "function" not in cleaned

    def test_prose_mentioning_a_quoted_word_is_untouched(self):
        text = 'The word \\"function\\" comes from Latin.'
        outcome, _call, cleaned = self.classify(text)
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == text


class TestFunctionTypeVariants:
    """The `type` value itself varies: "function", "function/result", ...

    Captured verbatim on "open YouTube". Requiring the value to equal
    "function" exactly let every variant through.
    """

    NAMES = {"open_app", "open_website"}

    def classify(self, content):
        return classify_tool_call(content, self.NAMES, lambda call: True)

    def test_function_slash_result_variant_is_not_shown(self):
        blob = '[{"type": "function/result", "result": "Opening YouTube in your browser"}]'
        _outcome, _call, cleaned = self.classify(blob)
        assert "Opening YouTube" not in cleaned
        assert "function" not in cleaned

    def test_function_call_variant_is_not_shown(self):
        blob = '[{"type": "function_call", "status": "done"}]'
        _outcome, _call, cleaned = self.classify(blob)
        assert "function" not in cleaned

    def test_unrelated_type_field_is_untouched(self):
        # A JSON answer whose "type" means something else must still show.
        text = '{"type": "mammal", "name_of_animal": "otter"}'
        outcome, _call, cleaned = self.classify(text)
        assert outcome is ToolCallOutcome.NONE
        assert cleaned == text

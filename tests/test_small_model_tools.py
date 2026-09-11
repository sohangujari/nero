"""Making skills work on a local model that cannot gate them.

Two independent problems, measured on llama3.2 and fixed separately:

1. It calls a tool on almost every message, including "hi". Asking it *first*
   whether the message wants an action is a question it answers correctly
   (20/20 measured), so the decision is taken before the turn.
2. It sends every argument as a string — `{"mute": "true"}` for a boolean — and
   strict validation threw the call away. That alone was every one of three
   failed `set_volume` calls.
"""

import httpx
import pytest

from nero.config.schema import NeroConfig
from nero.llm import ollama
from nero.llm.client import _last_user_text
from nero.skills.registry import build_registry, clean


@pytest.fixture
def registry():
    return build_registry(NeroConfig())


class _Reply:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


class TestActionGate:
    def _answer(self, monkeypatch, content):
        monkeypatch.setattr("httpx.post", lambda *a, **k: _Reply(content))
        return ollama.wants_action("llama3.2", "anything")

    def test_act_means_offer_the_tools(self, monkeypatch):
        assert self._answer(monkeypatch, "ACT") is True

    def test_talk_means_hold_them_back(self, monkeypatch):
        assert self._answer(monkeypatch, "TALK") is False

    def test_it_survives_a_chatty_answer(self, monkeypatch):
        """Small models narrate. "Answer: ACT" is still an answer."""
        assert self._answer(monkeypatch, "Answer: ACT\n") is True

    def test_an_answer_that_is_neither_word_decides_nothing(self, monkeypatch):
        assert self._answer(monkeypatch, "I'm not sure what you mean") is None

    def test_a_server_that_is_down_decides_nothing(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr("httpx.post", boom)
        assert ollama.wants_action("llama3.2", "hi") is None


class TestWhichTurnsAreGated:
    """The gate costs a round trip, so it runs only where it is needed."""

    def _client(self, registry, provider="ollama", model="llama3.2"):
        from nero.llm.client import LLMClient

        config = NeroConfig().llm.model_copy(update={"provider": provider, "model": model})
        return LLMClient(config, "Nero", registry)

    def _tools_offered(self, monkeypatch, client, registry, wants):
        seen = []

        def gate(name, message, base_url=None):
            seen.append(message)
            return wants

        monkeypatch.setattr("nero.llm.ollama.wants_action", gate)
        offered = client._gated_tools(
            [{"role": "user", "content": "hi"}], registry.tool_definitions()
        )
        return offered, seen

    def test_a_misfiring_model_loses_its_tools_on_a_chat_turn(self, registry, monkeypatch):
        offered, seen = self._tools_offered(
            monkeypatch, self._client(registry), registry, wants=False
        )
        assert offered == []
        assert seen == ["hi"]

    def test_a_misfiring_model_keeps_them_on_an_action_turn(self, registry, monkeypatch):
        offered, _seen = self._tools_offered(
            monkeypatch, self._client(registry), registry, wants=True
        )
        assert offered

    def test_an_undecided_gate_leaves_the_tools_in_place(self, registry, monkeypatch):
        """Withholding tools from a real request makes a small model narrate an
        action it never took, which is worse than a visible wrong call."""
        offered, _seen = self._tools_offered(
            monkeypatch, self._client(registry), registry, wants=None
        )
        assert offered

    def test_a_well_behaved_local_model_is_never_gated(self, registry, monkeypatch):
        monkeypatch.setattr(
            "nero.llm.ollama.wants_action",
            lambda *a, **k: pytest.fail("gated a model that gates itself"),
        )
        client = self._client(registry, model="qwen3")
        assert client._gated_tools([{"role": "user", "content": "hi"}],
                                   registry.tool_definitions())

    def test_a_cloud_model_is_never_gated(self, registry, monkeypatch):
        monkeypatch.setattr(
            "nero.llm.ollama.wants_action",
            lambda *a, **k: pytest.fail("gated a cloud model"),
        )
        client = self._client(registry, provider="gemini", model="gemini-3.5-flash-lite")
        assert client._gated_tools([{"role": "user", "content": "hi"}],
                                   registry.tool_definitions())


class TestWhatTheGateIsAsked:
    """Recall and learned procedures ride on the front of the user message.
    Classifying those instead of the question would classify the wrong thing."""

    def test_a_carried_memory_block_is_stripped(self):
        content = "<memory>\nuser: hello\nassistant: hi\n</memory>\n\nopen calculator"
        assert _last_user_text([{"role": "user", "content": content}]) == "open calculator"

    def test_a_carried_playbook_is_stripped(self):
        content = "<playbook>\nHow to deploy:\n1. build\n</playbook>\n\nhi"
        assert _last_user_text([{"role": "user", "content": content}]) == "hi"

    def test_a_plain_message_is_left_alone(self):
        assert _last_user_text([{"role": "user", "content": "mute it"}]) == "mute it"

    def test_the_latest_user_message_is_the_one_classified(self):
        messages = [
            {"role": "user", "content": "open calculator"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "thanks"},
        ]
        assert _last_user_text(messages) == "thanks"


class TestArgumentCoercion:
    """`{"mute": "true"}` is a perfectly clear call badly encoded. Refusing it
    turned "mute it" into "I didn't quite catch that"."""

    def _schema(self, registry, name):
        return registry.get(name).meta.input_schema

    def test_a_quoted_boolean_becomes_a_boolean(self, registry):
        assert clean(self._schema(registry, "set_volume"), {"mute": "true"}) == {"mute": True}

    def test_a_quoted_false_becomes_false(self, registry):
        assert clean(self._schema(registry, "set_volume"), {"mute": "false"}) == {"mute": False}

    def test_a_quoted_integer_becomes_an_integer(self, registry):
        assert clean(self._schema(registry, "set_volume"), {"level": "30"}) == {"level": 30}

    def test_a_negative_number_keeps_its_sign(self, registry):
        assert clean(self._schema(registry, "set_volume"), {"change": "-20"}) == {"change": -20}

    def test_the_user_s_words_around_a_number_are_dropped(self, registry):
        """The model quoting "30 percent" back is still unambiguously 30."""
        assert clean(self._schema(registry, "set_volume"), {"level": "30 percent"}) == {
            "level": 30
        }

    def test_a_value_that_is_not_a_number_is_left_to_fail(self, registry):
        """Coercion must not invent a plausible argument out of a wrong one."""
        assert clean(self._schema(registry, "set_volume"), {"level": "loud"}) == {
            "level": "loud"
        }
        assert registry.validate("set_volume", {"level": "loud"}) is False

    def test_strings_stay_strings(self, registry):
        assert clean(self._schema(registry, "open_app"), {"app_name": "Calculator"}) == {
            "app_name": "Calculator"
        }

    def test_a_value_already_the_right_type_is_untouched(self, registry):
        assert clean(self._schema(registry, "set_volume"), {"level": 30, "mute": True}) == {
            "level": 30,
            "mute": True,
        }


class TestValidateAndExecuteAgree:
    """The client validates a call before running it. If validate said no where
    execute would have said yes, the call is discarded as malformed and the
    skill that could have handled it never sees it."""

    @pytest.mark.parametrize(
        "arguments",
        [
            {"mute": "true"},
            {"level": "30"},
            {"change": "-20"},
            {"level": "30", "mute": "false"},
        ],
    )
    def test_a_stringly_typed_call_is_accepted(self, registry, arguments):
        assert registry.validate("set_volume", arguments) is True

    def test_an_unparseable_call_is_still_refused(self, registry):
        assert registry.validate("set_volume", {"level": "as loud as possible"}) is False

    def test_a_missing_required_argument_is_still_refused(self, registry):
        assert registry.validate("open_app", {}) is False


class TestOllamaSpeedOptions:
    """Two measured costs Nero used to pay by default: a thinking model
    reasoning before every reply, and ollama unloading the model after five
    minutes idle."""

    def _payload(self, monkeypatch, **overrides):
        import nero.llm.ollama_adapter as adapter

        captured = {}

        class _Stream:
            status_code = 200

            async def aiter_lines(self):
                yield '{"message": {"content": "hi"}, "done": true}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, _method, _url, json):
                captured.update(json)
                return _Stream()

        monkeypatch.setattr(adapter.httpx, "AsyncClient", lambda **k: _Client())

        async def drain():
            async for _ in adapter.ollama_chat(
                "http://x", "qwen3.5:2b", [{"role": "user", "content": "hi"}], [], **overrides
            ):
                pass

        import asyncio

        asyncio.run(drain())
        return captured

    def test_thinking_is_switched_off_explicitly(self, monkeypatch):
        """Not merely omitted: a thinking-capable model thinks unless told not
        to, and Nero throws the reasoning away."""
        assert self._payload(monkeypatch)["think"] is False

    def test_thinking_can_be_asked_for(self, monkeypatch):
        assert self._payload(monkeypatch, think=True)["think"] is True

    def test_keep_alive_is_sent_when_set(self, monkeypatch):
        assert self._payload(monkeypatch, keep_alive="15m")["keep_alive"] == "15m"

    def test_keep_alive_is_omitted_when_empty(self, monkeypatch):
        """An empty value means "leave ollama's own default alone", which is
        not the same as sending an empty string."""
        assert "keep_alive" not in self._payload(monkeypatch, keep_alive="")

    def test_the_defaults_favour_a_responsive_assistant(self):
        from nero.config.schema import LLMConfig

        config = LLMConfig()
        assert config.think is False
        assert config.keep_alive == "15m"

    def test_the_client_passes_both_through(self, monkeypatch, registry):
        from nero.config.schema import LLMConfig
        from nero.llm.client import LLMClient
        import nero.llm.client as client_module

        seen = {}

        async def fake(base_url, model, messages, tools, think=None, keep_alive=None):
            seen.update(think=think, keep_alive=keep_alive)
            from nero.llm.ollama_adapter import OllamaChatResponse

            yield OllamaChatResponse(content="hi", tool_calls=None)

        monkeypatch.setattr(client_module, "ollama_chat", fake)
        config = LLMConfig(provider="ollama", model="qwen3.5:2b", think=True, keep_alive="1h")
        client = LLMClient(config, "Nero", registry)
        client.send([{"role": "user", "content": "hi"}], lambda _t: None)
        assert seen == {"think": True, "keep_alive": "1h"}


class TestGateAgainstThinkingModels:
    def test_the_gate_disables_thinking(self):
        """It budgets four tokens for the answer, and a thinking model spends
        them on reasoning — which is why it scored 0/10 against qwen3.5 and
        gemma4 before this."""
        import inspect

        from nero.llm import ollama

        source = inspect.getsource(ollama.wants_action)
        assert '"think": False' in source

"""v1.5.7: a single fallback model used when the primary model fails
transiently mid-turn."""
import io

import httpx
import litellm
import pytest
from pydantic import ValidationError
from rich.console import Console

from nero.config.schema import LLMConfig
from nero.core.chat_loop import ChatLoop


def quiet_console(width=200) -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=width)


class FakeClient:
    """Records calls; raises `error` (once if `raise_once`, else every call)
    and otherwise appends a plain assistant reply."""

    def __init__(self, provider="claude", model="claude-sonnet-5", error=None,
                 reply="ok", raise_once=False):
        self.provider = provider
        self.model = model
        self._error = error
        self._reply = reply
        self._raise_once = raise_once
        self._raised = False
        self.send_calls: list[list[dict]] = []

    def send(self, messages, on_text):
        self.send_calls.append([dict(m) for m in messages])
        if self._error is not None and not (self._raise_once and self._raised):
            self._raised = True
            raise self._error
        on_text(self._reply)
        messages.append({"role": "assistant", "content": self._reply})


class PartialThenFailClient(FakeClient):
    """Simulates a stream that appended partial turn state before failing —
    the rollback must strip this before the fallback ever sees it."""

    def send(self, messages, on_text):
        self.send_calls.append([dict(m) for m in messages])
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "open_app", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "partial"})
        raise self._error


def make_loop(inputs, client, fallback_client=None, console=None):
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = console or quiet_console()
    loop = ChatLoop(
        client, console=console, assistant_name="Nero", input_fn=next_input,
        fallback_client=fallback_client,
    )
    return loop, console


def connect_error():
    return httpx.ConnectError("refused")


def rate_limit_error():
    return litellm.exceptions.RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-5"
    )


def auth_error():
    return litellm.exceptions.AuthenticationError(
        message="bad key", llm_provider="anthropic", model="claude-sonnet-5"
    )


def not_found_error():
    return litellm.exceptions.NotFoundError(
        message="no such model", llm_provider="anthropic", model="claude-sonnet-5"
    )


class TestFallbackFires:
    def test_transient_primary_with_fallback_uses_it(self):
        primary = FakeClient(provider="claude", model="claude-sonnet-5", error=connect_error())
        fallback = FakeClient(provider="openai", model="gpt-5", reply="fallback reply")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        out = console.file.getvalue()

        assert fallback.send_calls == [[{"role": "user", "content": "hi"}]]
        assert "fallback reply" in out
        assert "retrying with gpt-5 via openai" in out
        # History has ONE clean turn — no leftover primary-attempt artifacts.
        assert loop.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "fallback reply"},
        ]

    def test_rate_limit_error_is_treated_as_transient(self):
        """RateLimitError is a new member of the transient set, not in the
        existing outer connection-branch tuple."""
        primary = FakeClient(error=rate_limit_error())
        fallback = FakeClient(provider="openai", model="gpt-5", reply="fb")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        assert fallback.send_calls
        assert "fb" in console.file.getvalue()

    def test_messages_rollback_before_the_fallback_call(self):
        """A failed stream that appended partial tool/assistant turns must not
        leak them into the retry — the fallback sees exactly [..., user]."""
        primary = PartialThenFailClient(error=connect_error())
        fallback = FakeClient(provider="openai", model="gpt-5", reply="fb")
        loop, _ = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        assert fallback.send_calls == [[{"role": "user", "content": "hi"}]]


class TestNonTransientErrorsSkipFallback:
    def test_auth_error_does_not_use_the_fallback(self):
        primary = FakeClient(error=auth_error())
        fallback = FakeClient(provider="openai", model="gpt-5")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        assert fallback.send_calls == []
        assert "rejected" in console.file.getvalue()
        assert loop.messages == []

    def test_not_found_error_does_not_use_the_fallback(self):
        primary = FakeClient(error=not_found_error())
        fallback = FakeClient(provider="openai", model="gpt-5")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        assert fallback.send_calls == []
        assert "no model called" in console.file.getvalue()


class TestNoFallbackConfigured:
    def test_transient_error_without_a_fallback_is_unchanged(self):
        """Regression: byte-identical existing copy when no fallback exists."""
        primary = FakeClient(error=connect_error())
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=None)
        loop.run()
        out = console.file.getvalue()
        assert "Could not reach the model provider." in out
        assert "retrying with" not in out
        assert loop.messages == []


class TestBothFail:
    def test_both_transient_prints_notice_then_the_existing_copy(self):
        primary = FakeClient(error=connect_error())
        fallback = FakeClient(provider="openai", model="gpt-5", error=connect_error())
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_client=fallback)
        loop.run()
        out = console.file.getvalue()
        assert out.count("retrying with") == 1
        assert "Could not reach the model provider." in out
        assert loop.messages == []


class TestStatelessNextTurn:
    def test_the_next_turn_uses_the_primary_again(self):
        primary = FakeClient(error=connect_error(), reply="primary reply", raise_once=True)
        fallback = FakeClient(provider="openai", model="gpt-5", reply="fallback reply")
        loop, console = make_loop(
            ["hi", "again", "exit"], client=primary, fallback_client=fallback
        )
        loop.run()
        out = console.file.getvalue()
        assert len(primary.send_calls) == 2
        assert len(fallback.send_calls) == 1
        assert "fallback reply" in out
        assert "primary reply" in out


class TestSchema:
    def test_fallback_fields_default_to_none(self):
        config = LLMConfig()
        assert config.fallback_provider is None
        assert config.fallback_model is None

    def test_accepts_a_valid_provider_literal(self):
        config = LLMConfig(fallback_provider="openai", fallback_model="gpt-5")
        assert config.fallback_provider == "openai"
        assert config.fallback_model == "gpt-5"

    def test_rejects_a_junk_provider(self):
        with pytest.raises(ValidationError):
            LLMConfig(fallback_provider="not-a-real-provider", fallback_model="x")

    def test_a_pre_v157_config_still_loads(self):
        loaded = LLMConfig.model_validate({"provider": "claude", "model": "claude-sonnet-5"})
        assert loaded.fallback_provider is None
        assert loaded.fallback_model is None

    def test_a_half_set_pair_is_representable(self):
        """No cross-field validator: half-set is legal and inert, surfaced by
        a warning elsewhere, not rejected here."""
        config = LLMConfig(fallback_provider="openai", fallback_model=None)
        assert config.fallback_provider == "openai"
        assert config.fallback_model is None


from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


def _fallback_manager(tmp_path, fallback_provider="openai", fallback_model="gpt-5"):
    manager = ConfigManager(config_dir=tmp_path)
    manager.save(
        NeroConfig.model_validate(
            {
                "llm": {
                    "provider": "claude",
                    "model": "claude-sonnet-5",
                    "fallback_provider": fallback_provider,
                    "fallback_model": fallback_model,
                }
            }
        )
    )
    return manager


class TestCliWiring:
    def test_both_fields_set_builds_a_fallback_client_from_its_own_key(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _fallback_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        isolate_keyring[("nero", "openai_api_key")] = "sk-fallback"

        captured = {}

        class FakeChatLoop:
            def __init__(self, client, **kwargs):
                captured["client"] = client
                captured["fallback_client"] = kwargs.get("fallback_client")

            def run(self):
                pass

        monkeypatch.setattr(cli, "ChatLoop", FakeChatLoop)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0

        fallback_client = captured["fallback_client"]
        assert fallback_client is not None
        assert fallback_client.provider == "openai"
        assert fallback_client.model == "gpt-5"
        # Resolved from the FALLBACK provider's own keyring entry — never the
        # primary's key.
        assert fallback_client.api_key == "sk-fallback"
        assert fallback_client.api_key != "sk-primary"

    def test_missing_fallback_key_warns_but_still_starts(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _fallback_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        # No key stored for openai — the fallback provider.
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        assert "Fallback configured but" in result.output
        assert "disabled this" in result.output


class TestConfigSurfaces:
    def test_config_table_shows_off_by_default(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "show"])
        assert "off" in result.output

    def test_config_table_shows_the_pair_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _fallback_manager(tmp_path))
        result = runner.invoke(cli.app, ["config", "show"])
        assert "openai/gpt-5" in result.output

    def test_setting_one_half_warns_and_does_not_mutate_the_other(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(
            cli.app, ["config", "set", "llm.fallback_provider", "openai"]
        )
        assert result.exit_code == 0
        assert "needs both" in result.output
        assert "inert" in result.output
        loaded = manager.load()
        assert loaded.llm.fallback_provider == "openai"
        assert loaded.llm.fallback_model is None

    def test_a_fallback_identical_to_the_primary_warns(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())  # default llm.provider=claude, llm.model=claude-sonnet-5
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        runner.invoke(cli.app, ["config", "set", "llm.fallback_provider", "claude"])
        result = runner.invoke(
            cli.app, ["config", "set", "llm.fallback_model", "claude-sonnet-5"]
        )
        assert "same as the primary" in result.output

    def test_a_complete_different_pair_warns_of_nothing(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        runner.invoke(cli.app, ["config", "set", "llm.fallback_provider", "openai"])
        result = runner.invoke(cli.app, ["config", "set", "llm.fallback_model", "gpt-5"])
        assert "currently inert" not in result.output
        assert "same as the primary" not in result.output

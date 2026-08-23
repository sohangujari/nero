"""v1.5.7: a single fallback model used when the primary model fails
transiently mid-turn. v1.5.8 generalizes this to a priority-ordered fallback
chain, plus a model blacklist/whitelist that constrain AUTOMATIC selection
only (explicit config set always wins, warn-only)."""
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


def flat(text: str) -> str:
    """Collapse whitespace/newlines so a rich-console line wrap doesn't split
    a substring a test wants to assert on."""
    return " ".join(text.split())


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


def make_loop(inputs, client, fallback_clients=None, console=None):
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = console or quiet_console()
    loop = ChatLoop(
        client, console=console, assistant_name="Nero", input_fn=next_input,
        fallback_clients=fallback_clients,
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
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
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
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
        loop.run()
        assert fallback.send_calls
        assert "fb" in console.file.getvalue()

    def test_messages_rollback_before_the_fallback_call(self):
        """A failed stream that appended partial tool/assistant turns must not
        leak them into the retry — the fallback sees exactly [..., user]."""
        primary = PartialThenFailClient(error=connect_error())
        fallback = FakeClient(provider="openai", model="gpt-5", reply="fb")
        loop, _ = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
        loop.run()
        assert fallback.send_calls == [[{"role": "user", "content": "hi"}]]


class TestNonTransientErrorsSkipFallback:
    def test_auth_error_does_not_use_the_fallback(self):
        primary = FakeClient(error=auth_error())
        fallback = FakeClient(provider="openai", model="gpt-5")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
        loop.run()
        assert fallback.send_calls == []
        assert "rejected" in console.file.getvalue()
        assert loop.messages == []

    def test_not_found_error_does_not_use_the_fallback(self):
        primary = FakeClient(error=not_found_error())
        fallback = FakeClient(provider="openai", model="gpt-5")
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
        loop.run()
        assert fallback.send_calls == []
        assert "no model called" in console.file.getvalue()


class TestNoFallbackConfigured:
    def test_transient_error_without_a_fallback_is_unchanged(self):
        """Regression: byte-identical existing copy when the chain is empty."""
        primary = FakeClient(error=connect_error())
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=None)
        loop.run()
        out = console.file.getvalue()
        assert "Could not reach the model provider." in out
        assert "retrying with" not in out
        assert loop.messages == []


class TestBothFail:
    def test_both_transient_prints_notice_then_the_existing_copy(self):
        primary = FakeClient(error=connect_error())
        fallback = FakeClient(provider="openai", model="gpt-5", error=connect_error())
        loop, console = make_loop(["hi", "exit"], client=primary, fallback_clients=[fallback])
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
            ["hi", "again", "exit"], client=primary, fallback_clients=[fallback]
        )
        loop.run()
        out = console.file.getvalue()
        assert len(primary.send_calls) == 2
        assert len(fallback.send_calls) == 1
        assert "fallback reply" in out
        assert "primary reply" in out


class TestFallbackChain:
    """v1.5.8: the chain is walked in order; first success wins."""

    def test_chain_of_two_first_fails_second_succeeds(self):
        fb1 = FakeClient(provider="openai", model="gpt-5", error=connect_error())
        fb2 = FakeClient(provider="gemini", model="gemini-3-pro", reply="second reply")
        primary = FakeClient(error=connect_error())
        loop, console = make_loop(
            ["hi", "exit"], client=primary, fallback_clients=[fb1, fb2]
        )
        loop.run()
        out = console.file.getvalue()

        assert fb1.send_calls == [[{"role": "user", "content": "hi"}]]
        assert fb2.send_calls == [[{"role": "user", "content": "hi"}]]
        assert out.count("retrying with") == 2
        assert "second reply" in out
        # One clean history turn — no leftover attempts from fb1.
        assert loop.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "second reply"},
        ]

    def test_all_chain_entries_fail_transiently(self):
        """Exceptions from the LAST attempt propagate to the existing outer
        handlers unchanged; rollback leaves history clean."""
        primary = FakeClient(error=connect_error())
        fb1 = FakeClient(provider="openai", model="gpt-5", error=connect_error())
        fb2 = FakeClient(provider="gemini", model="gemini-3-pro", error=connect_error())
        loop, console = make_loop(
            ["hi", "exit"], client=primary, fallback_clients=[fb1, fb2]
        )
        loop.run()
        out = console.file.getvalue()
        assert out.count("retrying with") == 2
        assert "Could not reach the model provider." in out
        assert loop.messages == []


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

    def test_chain_and_lists_default_to_empty(self):
        config = LLMConfig()
        assert config.fallback_chain == []
        assert config.model_blacklist == []
        assert config.model_whitelist == []

    def test_chain_entry_accepts_provider_slash_model(self):
        config = LLMConfig(fallback_chain=["openai/gpt-5", "gemini/gemini-3-pro"])
        assert config.fallback_chain == ["openai/gpt-5", "gemini/gemini-3-pro"]

    def test_chain_entry_rejects_a_bad_provider_half(self):
        with pytest.raises(ValidationError):
            LLMConfig(fallback_chain=["not-a-real-provider/some-model"])

    def test_chain_entry_splits_on_the_first_slash_only(self):
        """Model ids may contain '/' (replicate, huggingface); the provider
        half never does, so splitting on the FIRST '/' is correct."""
        config = LLMConfig(fallback_chain=["replicate/meta/llama-3"])
        provider, _, model = config.fallback_chain[0].partition("/")
        assert provider == "replicate"
        assert model == "meta/llama-3"

    def test_lists_have_no_validator_an_inert_id_is_legal(self):
        config = LLMConfig(model_blacklist=["nothing-matches-this"])
        assert config.model_blacklist == ["nothing-matches-this"]


from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


def _manager_with_llm(tmp_path, **llm_overrides):
    manager = ConfigManager(config_dir=tmp_path)
    llm = {"provider": "claude", "model": "claude-sonnet-5", **llm_overrides}
    manager.save(NeroConfig.model_validate({"llm": llm}))
    return manager


def _fallback_manager(tmp_path, fallback_provider="openai", fallback_model="gpt-5"):
    return _manager_with_llm(
        tmp_path, fallback_provider=fallback_provider, fallback_model=fallback_model
    )


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
                captured["fallback_clients"] = kwargs.get("fallback_clients")

            def run(self):
                pass

        monkeypatch.setattr(cli, "ChatLoop", FakeChatLoop)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0

        fallback_clients = captured["fallback_clients"]
        assert len(fallback_clients) == 1
        fallback_client = fallback_clients[0]
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
        out = flat(result.output)
        assert "Fallback configured but" in out
        assert "disabled this session" in out


class TestCliChainWiring:
    """v1.5.8: the ordered chain, filtered by blacklist/whitelist, resolved
    into clients at startup."""

    def _fake_chat_loop(self, monkeypatch, captured):
        class FakeChatLoop:
            def __init__(self, client, **kwargs):
                captured["client"] = client
                captured["fallback_clients"] = kwargs.get("fallback_clients")

            def run(self):
                pass

        monkeypatch.setattr(cli, "ChatLoop", FakeChatLoop)

    def test_empty_chain_and_scalar_pair_behaves_like_v157(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _fallback_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        isolate_keyring[("nero", "openai_api_key")] = "sk-fallback"
        captured = {}
        self._fake_chat_loop(monkeypatch, captured)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        clients = captured["fallback_clients"]
        assert len(clients) == 1
        assert (clients[0].provider, clients[0].model) == ("openai", "gpt-5")

    def test_both_chain_and_scalar_set_chain_wins_with_startup_note(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _manager_with_llm(
            tmp_path,
            fallback_provider="mistral",
            fallback_model="mistral-large",
            fallback_chain=["openai/gpt-5"],
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        isolate_keyring[("nero", "openai_api_key")] = "sk-fallback"
        captured = {}
        self._fake_chat_loop(monkeypatch, captured)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        clients = captured["fallback_clients"]
        assert len(clients) == 1
        assert (clients[0].provider, clients[0].model) == ("openai", "gpt-5")
        assert "chain wins" in flat(result.output)

    def test_blacklisted_chain_entry_is_skipped(self, monkeypatch, tmp_path, isolate_keyring):
        manager = _manager_with_llm(
            tmp_path,
            fallback_chain=["openai/gpt-5", "gemini/gemini-3-pro"],
            model_blacklist=["gpt-5"],
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        isolate_keyring[("nero", "gemini_api_key")] = "sk-gemini"
        captured = {}
        self._fake_chat_loop(monkeypatch, captured)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        clients = captured["fallback_clients"]
        assert len(clients) == 1
        assert (clients[0].provider, clients[0].model) == ("gemini", "gemini-3-pro")
        assert "blacklisted" in flat(result.output)

    def test_whitelist_drops_a_non_listed_chain_entry(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _manager_with_llm(
            tmp_path,
            fallback_chain=["openai/gpt-5", "gemini/gemini-3-pro"],
            model_whitelist=["gemini-3-pro"],
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        isolate_keyring[("nero", "gemini_api_key")] = "sk-gemini"
        captured = {}
        self._fake_chat_loop(monkeypatch, captured)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        clients = captured["fallback_clients"]
        assert len(clients) == 1
        assert (clients[0].provider, clients[0].model) == ("gemini", "gemini-3-pro")
        assert "whitelist" in flat(result.output)

    def test_entry_with_missing_key_is_skipped_remaining_entries_intact(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = _manager_with_llm(
            tmp_path,
            fallback_chain=["openai/gpt-5", "gemini/gemini-3-pro"],
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "anthropic_api_key")] = "sk-primary"
        # No key stored for openai; gemini has one.
        isolate_keyring[("nero", "gemini_api_key")] = "sk-gemini"
        captured = {}
        self._fake_chat_loop(monkeypatch, captured)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        clients = captured["fallback_clients"]
        assert len(clients) == 1
        assert (clients[0].provider, clients[0].model) == ("gemini", "gemini-3-pro")
        assert "disabled this session" in flat(result.output)
        # Startup didn't block: exit_code == 0 and a client for the surviving
        # entry was still built (asserted above).


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

    def test_config_table_shows_chain_and_lists_when_set(self, monkeypatch, tmp_path):
        manager = _manager_with_llm(
            tmp_path,
            fallback_chain=["openai/gpt-5", "gemini/gemini-3-pro"],
            model_blacklist=["gpt-5"],
            model_whitelist=["claude-sonnet-5"],
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "show"])
        out = flat(result.output)
        assert "openai/gpt-5, gemini/gemini-3-pro" in out
        assert "gpt-5" in out
        assert "claude-sonnet-5" in out

    def test_config_table_shows_placeholder_when_chain_and_lists_empty(
        self, monkeypatch, tmp_path
    ):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "show"])
        out = flat(result.output)
        assert "Fallback Chain" in out
        assert "Model Blacklist" in out
        assert "Model Whitelist" in out
        assert out.count("—") == 3

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


class TestBlacklistWhitelistWarnOnly:
    """Priority/lists constrain AUTOMATIC selection only — an explicit
    `config set llm.model` always wins, warn-only."""

    def test_setting_a_blacklisted_model_warns_but_sets_it_anyway(
        self, monkeypatch, tmp_path
    ):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig.model_validate({"llm": {"model_blacklist": ["gpt-5"]}}))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gpt-5"])
        assert result.exit_code == 0
        out = flat(result.output)
        assert "model_blacklist" in out
        assert "explicit choice stands" in out
        loaded = manager.load()
        assert loaded.llm.model == "gpt-5"
        assert loaded.llm.model_blacklist == ["gpt-5"]  # untouched by the set

    def test_setting_a_non_whitelisted_model_warns_but_sets_it_anyway(
        self, monkeypatch, tmp_path
    ):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(
            NeroConfig.model_validate({"llm": {"model_whitelist": ["claude-sonnet-5"]}})
        )
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gpt-5"])
        assert result.exit_code == 0
        assert "model_whitelist" in flat(result.output)
        assert manager.load().llm.model == "gpt-5"

    def test_setting_an_unlisted_model_warns_of_nothing(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gpt-5"])
        assert "explicit choice stands" not in result.output



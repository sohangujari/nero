"""v1.5.4: a custom OpenAI-compatible endpoint, configured by base URL."""
from typing import get_args

import pytest
from pydantic import ValidationError

from nero.config.schema import LLMConfig, Provider
from nero.llm import providers


class TestSchema:
    def test_base_url_defaults_to_none(self):
        assert LLMConfig().base_url is None

    def test_a_config_without_base_url_still_loads(self):
        """extra='forbid' plus a new field: files written before v1.5.4 must
        keep loading, which is the whole migration story."""
        loaded = LLMConfig.model_validate({"provider": "claude", "model": "claude-sonnet-5"})
        assert loaded.base_url is None

    def test_a_non_http_url_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="custom", model="x", base_url="localhost:1234")

    def test_a_trailing_slash_is_stripped(self):
        config = LLMConfig(provider="custom", model="x", base_url="http://localhost:1234/v1/")
        assert config.base_url == "http://localhost:1234/v1"

    def test_a_stale_url_on_another_provider_is_representable(self):
        """The guard lives in LLMClient.api_base, not here. This state must be
        constructible or the leak test in Task 3 cannot be written at all."""
        config = LLMConfig(provider="claude", model="claude-sonnet-5", base_url="http://stale")
        assert config.base_url == "http://stale"


class TestTable:
    def test_custom_is_in_the_table_and_the_literal(self):
        assert "custom" in providers.names()
        assert get_args(Provider) == providers.names()

    def test_custom_is_the_only_key_optional_provider(self):
        assert [p.name for p in providers.PROVIDERS if p.key_optional] == ["custom"]

    def test_custom_can_still_store_a_key(self):
        """key_optional asks whether a MISSING key is fatal, not whether one
        can be stored — a Together endpoint needs a key like any cloud provider."""
        assert providers.get("custom").keyring_entry == "custom_api_key"

    def test_custom_has_no_catalog(self):
        info = providers.get("custom")
        assert info.models == ()
        assert info.default_model is None
        assert info.catalog_key == ""
        assert providers.catalog_models("custom") == []


import asyncio

from nero.llm.client import LLMClient
from nero.skills.registry import SkillRegistry


def _client(provider, model, base_url=None):
    return LLMClient(
        config=LLMConfig(provider=provider, model=model, base_url=base_url),
        assistant_name="Nero",
        registry=SkillRegistry([]),
    )


class TestApiBaseGuard:
    def test_custom_exposes_its_endpoint(self):
        client = _client("custom", "llama-3.1-8b", "http://localhost:1234/v1")
        assert client.api_base == "http://localhost:1234/v1"

    def test_a_stale_url_never_leaks_into_another_provider(self):
        """Nothing clears base_url on a provider switch, so a real config
        reaches this state. The guard is on the provider, not on the field
        merely being set."""
        assert _client("claude", "claude-sonnet-5", "http://stale").api_base is None

    def test_custom_without_a_url_is_none(self):
        assert _client("custom", "llama-3.1-8b").api_base is None

    def test_the_model_is_exposed_like_the_provider_is(self):
        """Task 7's error messages read this off the client by name, the same
        way the chat loop already reads `provider`."""
        assert _client("custom", "llama-3.1-8b").model == "llama-3.1-8b"


class TestCustomModelString:
    def test_a_bare_model_gets_the_openai_prefix(self):
        assert _client("custom", "llama-3.1-8b").litellm_model == "openai/llama-3.1-8b"

    def test_a_model_named_after_a_provider_is_still_prefixed(self):
        """Together really serves "openai/gpt-oss-120b" — the groq shortlist
        carries that exact id. The startswith guard used elsewhere would see
        the prefix already present and pass it through, LiteLLM would strip it,
        and the endpoint would receive the bare name and 404. LiteLLM strips
        exactly one prefix, so doubling it is correct."""
        client = _client("custom", "openai/gpt-oss-120b")
        assert client.litellm_model == "openai/openai/gpt-oss-120b"


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _completion_kwargs(monkeypatch, client):
    """Run one litellm round against a stubbed transport and return the kwargs."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _EmptyStream()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("litellm.stream_chunk_builder", lambda chunks, messages=None: None)

    async def drain():
        return [text async for text in client._litellm_chat([], [])]

    asyncio.run(drain())
    return captured


class TestApiBaseReachesLitellm:
    def test_custom_sends_api_base(self, monkeypatch):
        client = _client("custom", "llama-3.1-8b", "http://localhost:1234/v1")
        assert _completion_kwargs(monkeypatch, client)["api_base"] == "http://localhost:1234/v1"

    def test_a_named_provider_sends_no_api_base(self, monkeypatch):
        client = _client("claude", "claude-sonnet-5", "http://stale")
        assert "api_base" not in _completion_kwargs(monkeypatch, client)


from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


def _custom_manager(tmp_path, base_url="http://localhost:1234/v1"):
    manager = ConfigManager(config_dir=tmp_path)
    manager.save(
        NeroConfig.model_validate(
            {"llm": {"provider": "custom", "model": "llama-3.1-8b", "base_url": base_url}}
        )
    )
    return manager


class TestStartupGate:
    def test_a_keyless_custom_endpoint_starts(self, monkeypatch, tmp_path):
        """LM Studio and llama.cpp need no key; a missing one must not be fatal.
        EOF on stdin ends the chat loop immediately."""
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path))
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        assert "API key" not in result.output

    def test_a_custom_endpoint_without_a_url_exits_with_the_right_message(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path, None))
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "Endpoint URL" in result.output
        assert "API key" not in result.output  # the wrong diagnosis for this failure

    def test_custom_never_reaches_the_ollama_preflight(self, monkeypatch, tmp_path):
        """Before this task, "keyless" meant "is ollama", so a keyless custom
        endpoint was told to run `ollama serve`."""
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path))

        def fail(model):
            raise AssertionError(f"_ollama_preflight called with {model!r}")

        monkeypatch.setattr(cli, "_ollama_preflight", fail)
        assert runner.invoke(cli.app, []).exit_code == 0

"""v1.5.4: a custom OpenAI-compatible endpoint, configured by base URL."""
from typing import get_args

import pytest
from pydantic import ValidationError

from nero.config.schema import LLMConfig, Provider
from nero.hardware.detector import HardwareSpecs
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

    def test_a_bare_scheme_is_rejected(self):
        """startswith("http://") passes for "http://" itself, and rstrip("/")
        then reduces it to the unusable "http:" — reject before that happens."""
        with pytest.raises(ValidationError):
            LLMConfig(provider="custom", model="x", base_url="http://")

    def test_a_bare_https_scheme_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="custom", model="x", base_url="https://")

    def test_a_stale_url_on_another_provider_is_representable(self):
        """The guard lives in LLMClient.api_base, not here. This state must be
        constructible or the leak test in Task 3 cannot be written at all."""
        config = LLMConfig(provider="claude", model="claude-sonnet-5", base_url="http://stale")
        assert config.base_url == "http://stale"

    def test_custom_anthropic_accepts_a_base_url(self):
        config = LLMConfig(
            provider="custom_anthropic", model="kimi-k2.5",
            base_url="https://api.moonshot.ai/anthropic",
        )
        assert config.base_url == "https://api.moonshot.ai/anthropic"


class TestTable:
    def test_custom_is_in_the_table_and_the_literal(self):
        assert "custom" in providers.names()
        assert get_args(Provider) == providers.names()

    def test_the_custom_rows_are_the_only_key_optional_providers(self):
        assert [p.name for p in providers.PROVIDERS if p.key_optional] == [
            "custom", "custom_anthropic",
        ]

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

    def test_custom_anthropic_row_facts(self):
        info = providers.get("custom_anthropic")
        assert info.prefix == "anthropic/"
        assert info.keyring_entry == "custom_anthropic_api_key"
        assert info.key_optional
        assert info.models == ()
        assert info.catalog_key == ""

    def test_custom_providers_set_matches_the_table(self):
        assert providers.CUSTOM_PROVIDERS == {"custom", "custom_anthropic"}
        for name in providers.CUSTOM_PROVIDERS:
            assert name in providers.names()

    def test_custom_anthropic_is_last(self):
        """test_provider_menu.py picks providers by ordinal; new rows append last."""
        assert providers.names()[-1] == "custom_anthropic"


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


class TestAnthropicGuardAndModel:
    def test_custom_anthropic_exposes_its_endpoint(self):
        client = _client(
            "custom_anthropic", "kimi-k2.5", "https://api.moonshot.ai/anthropic"
        )
        assert client.api_base == "https://api.moonshot.ai/anthropic"

    def test_a_stale_url_still_never_leaks_into_a_named_provider(self):
        """The frozenset widens which providers USE the URL; named providers
        must stay exactly as guarded as before."""
        assert _client("claude", "claude-sonnet-5", "http://stale").api_base is None

    def test_a_bare_model_gets_the_anthropic_prefix(self):
        """NOTE: passes even before this task via the generic table-prefix path
        (custom_anthropic doesn't hit the old == "custom" branch). Kept as a
        regression guard; the discriminating test is the double-prefix one."""
        assert _client("custom_anthropic", "kimi-k2.5").litellm_model == "anthropic/kimi-k2.5"

    def test_an_anthropic_named_model_is_still_prefixed(self):
        """Same one-strip logic as the Together case: LiteLLM strips exactly
        one prefix, so a model genuinely named anthropic/… must be doubled."""
        client = _client("custom_anthropic", "anthropic/claude-sonnet-5")
        assert client.litellm_model == "anthropic/anthropic/claude-sonnet-5"

    def test_custom_anthropic_sends_api_base(self, monkeypatch):
        client = _client(
            "custom_anthropic", "kimi-k2.5", "https://api.moonshot.ai/anthropic"
        )
        kwargs = _completion_kwargs(monkeypatch, client)
        assert kwargs["api_base"] == "https://api.moonshot.ai/anthropic"


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


class TestCustomModelPicker:
    def test_free_text_writes_the_model(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Menu row 3 -> picker row 2 (type a name) -> the name -> finish.
        runner.invoke(cli.app, ["config"], input="3\n2\nmixtral-8x7b\n\n")
        assert manager.load().llm.model == "mixtral-8x7b"

    def test_fetching_offers_the_endpoint_models(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli.openai_compat, "fetch_models",
            lambda url, key=None: (url, ["alpha", "beta"]),
        )
        # Menu row 3 -> picker row 1 (fetch) -> model row 2 -> finish.
        runner.invoke(cli.app, ["config"], input="3\n1\n2\n\n")
        assert manager.load().llm.model == "beta"

    def test_a_corrected_url_is_offered_and_stored(self, monkeypatch, tmp_path):
        """LM Studio's reported address lacks the /v1 the API lives at."""
        manager = _custom_manager(tmp_path, "http://localhost:1234")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli.openai_compat, "fetch_models",
            lambda url, key=None: (f"{url}/v1", ["alpha"]),
        )
        # Fetch -> accept the correction (blank takes the yes default) -> model 1.
        runner.invoke(cli.app, ["config"], input="3\n1\n\n1\n\n")
        assert manager.load().llm.base_url == "http://localhost:1234/v1"
        assert manager.load().llm.model == "alpha"

    def test_a_failed_fetch_falls_back_to_free_text(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli.openai_compat, "fetch_models", lambda url, key=None: (url, [])
        )
        result = runner.invoke(cli.app, ["config"], input="3\n1\ntyped-by-hand\n\n")
        assert manager.load().llm.model == "typed-by-hand"
        assert result.exit_code == 0

    def test_the_litellm_catalog_is_not_offered(self, monkeypatch, tmp_path):
        """A custom endpoint's models come from the endpoint. Offering OpenAI's
        catalog for a vLLM box would be worse than offering nothing."""
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="3\n2\nx\n\n")
        assert "LiteLLM" not in result.output


class TestCustomSetup:
    def test_first_run_with_custom_completes(self, monkeypatch, tmp_path):
        """`nero`, not `nero config`, is what runs first-time setup — and only
        when no config file exists yet; the `config` callback goes straight to
        the menu. Pre-task this path saves model=None and dies in validation.

        Requires `from nero.hardware.detector import HardwareSpecs` at the top
        of the file; all four fields are required by the model.
        """
        manager = ConfigManager(config_dir=tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli,
            "detect_hardware",
            lambda: HardwareSpecs(ram_gb=16.0, cpu_cores=8, os="Darwin", has_ollama=False),
        )
        # Provider row 14 (custom) -> URL -> model row 2 (type) -> name -> blank
        # key. Setup then starts the chat loop; EOF on stdin ends it cleanly.
        result = runner.invoke(
            cli.app, [], input="14\nhttp://localhost:1234/v1\n2\nllama-3.1-8b\n\n"
        )
        assert result.exit_code == 0
        assert "Traceback" not in result.output
        assert "localhost:1234/v1" in result.output
        loaded = manager.load()
        assert loaded.llm.provider == "custom"
        assert loaded.llm.base_url == "http://localhost:1234/v1"
        assert loaded.llm.model == "llama-3.1-8b"

    def test_switching_away_keeps_the_url_and_does_not_raise(self, monkeypatch, tmp_path):
        """Nothing clears base_url on a switch: it persists inertly and comes
        back if the user returns to custom. The guard is LLMClient.api_base."""
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 2 -> provider row 1 (claude) -> keep the default model -> key -> finish.
        result = runner.invoke(cli.app, ["config"], input="2\n1\n\nsk-test\n\n")
        assert result.exit_code == 0
        loaded = manager.load()
        assert loaded.llm.provider == "claude"
        assert loaded.llm.base_url == "http://localhost:1234/v1"

    def test_a_malformed_url_warns_and_keeps_the_menu_alive(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 16 -> a URL with no scheme -> finish.
        result = runner.invoke(cli.app, ["config"], input="16\nlocalhost:1234\n\n")
        assert result.exit_code == 0
        assert "Traceback" not in result.output
        assert manager.load().llm.base_url == "http://localhost:1234/v1"  # unchanged


class TestEndpointRow:
    def test_row_16_shows_for_custom(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "Endpoint URL" in result.stdout

    def test_row_16_is_absent_for_a_named_provider(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "Endpoint URL" not in result.stdout

    def test_row_16_writes_the_url(self, monkeypatch, tmp_path):
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        runner.invoke(cli.app, ["config"], input="16\nhttp://10.0.0.5:8000/v1\n\n")
        assert manager.load().llm.base_url == "http://10.0.0.5:8000/v1"


class TestSurfaces:
    def test_config_show_lists_the_endpoint_for_custom(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path))
        result = runner.invoke(cli.app, ["config", "show"])
        assert "http://localhost:1234/v1" in result.output

    def test_config_show_omits_the_endpoint_for_a_named_provider(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        assert "Endpoint URL" not in runner.invoke(cli.app, ["config", "show"]).output

    def test_setting_an_inert_url_warns_but_keeps_it(self, monkeypatch, tmp_path):
        """Warn only, never mutate — the same rule _warn_if_model_mismatched
        follows. The value is kept because switching to custom brings it back."""
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(
            cli.app, ["config", "set", "llm.base_url", "http://localhost:1234/v1"]
        )
        assert result.exit_code == 0
        assert "custom" in result.output
        assert manager.load().llm.base_url == "http://localhost:1234/v1"

    def test_no_warning_when_the_provider_is_custom(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path))
        result = runner.invoke(
            cli.app, ["config", "set", "llm.base_url", "http://10.0.0.5:8000/v1"]
        )
        assert "will not be used" not in result.output

    def test_no_ownership_warning_for_a_custom_endpoint_model(self, monkeypatch, tmp_path):
        """openai/gpt-oss-120b is a legitimate id on a custom endpoint (it's the
        Together example in the double-prefix comment) even though it sits in
        groq's curated list — a custom endpoint can serve any model id, so
        curated-list ownership means nothing there."""
        monkeypatch.setattr(cli, "ConfigManager", lambda: _custom_manager(tmp_path))
        result = runner.invoke(
            cli.app, ["config", "set", "llm.model", "openai/gpt-oss-120b"]
        )
        assert result.exit_code == 0
        assert "belongs to" not in result.output


def _anthropic_manager(tmp_path, base_url="https://api.moonshot.ai/anthropic"):
    manager = ConfigManager(config_dir=tmp_path)
    manager.save(
        NeroConfig.model_validate(
            {"llm": {"provider": "custom_anthropic", "model": "kimi-k2.5",
                     "base_url": base_url}}
        )
    )
    return manager


class TestAnthropicStartupGate:
    def test_a_keyless_anthropic_endpoint_starts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path))
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        assert "API key" not in result.output

    def test_an_anthropic_endpoint_without_a_url_exits_with_the_right_message(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path, None))
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "Endpoint URL" in result.output
        assert "API key" not in result.output


class TestAnthropicPicker:
    def test_the_anthropic_fetcher_is_used(self, monkeypatch, tmp_path, isolate_keyring):
        manager = _anthropic_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        isolate_keyring[("nero", "custom_anthropic_api_key")] = "sk-a"

        def wrong_fetcher(url, key=None):
            raise AssertionError("OpenAI-dialect fetch_models called for custom_anthropic")

        seen_keys = {}

        def fake_anthropic_fetch(url, key=None):
            seen_keys["key"] = key
            return (url, ["kimi-k2.5", "kimi-latest"])

        monkeypatch.setattr(cli.openai_compat, "fetch_models", wrong_fetcher)
        monkeypatch.setattr(cli.openai_compat, "fetch_models_anthropic", fake_anthropic_fetch)
        # Menu row 3 -> picker row 1 (fetch) -> model row 2 -> finish.
        runner.invoke(cli.app, ["config"], input="3\n1\n2\n\n")
        assert manager.load().llm.model == "kimi-latest"
        assert seen_keys["key"] == manager.get_api_key("custom_anthropic") == "sk-a"

    def test_a_corrected_anthropic_url_is_offered_and_stored(self, monkeypatch, tmp_path):
        """A pasted OpenAI-style …/v1 base gets the suffix stripped."""
        manager = _anthropic_manager(tmp_path, "http://localhost:8080/v1")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli.openai_compat, "fetch_models_anthropic",
            lambda url, key=None: (url.removesuffix("/v1"), ["m1"]),
        )
        # Fetch -> accept the correction (blank takes the yes default) -> model 1.
        runner.invoke(cli.app, ["config"], input="3\n1\n\n1\n\n")
        assert manager.load().llm.base_url == "http://localhost:8080"
        assert manager.load().llm.model == "m1"


class TestAnthropicSetupAndSurfaces:
    def test_first_run_with_custom_anthropic_completes(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = ConfigManager(config_dir=tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(
            cli, "detect_hardware",
            lambda: HardwareSpecs(ram_gb=16.0, cpu_cores=8, os="Darwin", has_ollama=False),
        )
        # Provider row 15 (custom_anthropic, appended last) -> URL -> model row 2
        # (type a name) -> name -> a key. EOF then ends the chat loop.
        result = runner.invoke(
            cli.app, [],
            input="15\nhttps://api.moonshot.ai/anthropic\n2\nkimi-k2.5\nsk-anth\n",
        )
        assert result.exit_code == 0
        assert "Traceback" not in result.output
        assert "no /v1" in result.output
        loaded = manager.load()
        assert loaded.llm.provider == "custom_anthropic"
        assert loaded.llm.base_url == "https://api.moonshot.ai/anthropic"
        assert loaded.llm.model == "kimi-k2.5"
        assert isolate_keyring == {("nero", "custom_anthropic_api_key"): "sk-anth"}

    def test_row_16_shows_for_custom_anthropic(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path))
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "Endpoint URL" in result.stdout

    def test_config_show_lists_the_endpoint_for_custom_anthropic(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path))
        result = runner.invoke(cli.app, ["config", "show"])
        assert "https://api.moonshot.ai/anthropic" in result.output

    def test_no_inert_warning_under_custom_anthropic(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path))
        result = runner.invoke(
            cli.app, ["config", "set", "llm.base_url", "https://api.z.ai/api/anthropic"]
        )
        assert "will not be used" not in result.output

    def test_no_ownership_warning_under_custom_anthropic(self, monkeypatch, tmp_path):
        """claude-sonnet-5 sits in claude's curated list, but a custom endpoint
        can serve any model id — ownership means nothing there."""
        monkeypatch.setattr(cli, "ConfigManager", lambda: _anthropic_manager(tmp_path))
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "claude-sonnet-5"])
        assert "belongs to" not in result.output

    def test_switching_between_dialects_keeps_the_url(self, monkeypatch, tmp_path):
        """One slot, shared: moving custom -> custom_anthropic offers the same
        URL as the prompt default and nothing clears it."""
        manager = _custom_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 2 -> provider row 15 (custom_anthropic) -> accept URL default ->
        # model row 2 -> name -> blank key -> finish.
        result = runner.invoke(cli.app, ["config"], input="2\n15\n\n2\nkimi-k2.5\n\n\n")
        assert result.exit_code == 0
        loaded = manager.load()
        assert loaded.llm.provider == "custom_anthropic"
        assert loaded.llm.base_url == "http://localhost:1234/v1"

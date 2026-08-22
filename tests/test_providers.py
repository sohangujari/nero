from typing import get_args

from nero.config.schema import Provider
from nero.llm import providers


class TestTable:
    def test_every_provider_has_a_unique_name_and_label(self):
        names = [p.name for p in providers.PROVIDERS]
        assert len(names) == len(set(names))
        assert all(p.label for p in providers.PROVIDERS)

    def test_literal_matches_table_exactly(self):
        """The Literal is hand-written because pydantic needs it at the type
        level. This test is the only thing stopping the two from drifting."""
        assert get_args(Provider) == providers.names()

    def test_keyless_providers_are_ollama_and_bedrock(self):
        keyless = [p.name for p in providers.PROVIDERS if p.keyring_entry is None]
        assert keyless == ["ollama", "bedrock"]

    def test_every_cloud_provider_has_curated_models(self):
        # Two providers have no fixed catalog to curate: ollama's model comes
        # from hardware detection, and a custom endpoint's from the endpoint.
        no_catalog = {"ollama", "custom", "custom_anthropic"}
        for info in providers.PROVIDERS:
            if info.name in no_catalog:
                assert info.models == ()
                assert info.default_model is None
            else:
                assert len(info.models) >= 2, info.name
                assert info.default_model == info.models[0]

    def test_defaults_for_existing_providers_are_unchanged(self):
        """Phase 6 must not move anyone's model out from under them."""
        assert providers.get("claude").default_model == "claude-sonnet-5"
        assert providers.get("openai").default_model == "gpt-5"
        assert providers.get("gemini").default_model == "gemini-2.5-pro"

    def test_get_rejects_unknown(self):
        import pytest

        with pytest.raises(KeyError):
            providers.get("grok")


class TestCatalog:
    def test_returns_bare_names_for_a_prefixed_provider(self):
        models = providers.catalog_models("qwen")
        assert models, "dashscope should have tool-capable chat models"
        assert all(not m.startswith("dashscope/") for m in models)
        assert "qwen3-max" in models

    def test_excludes_non_chat_models(self):
        """The mode filter is load-bearing for openai, which lists 37 tool-capable
        realtime/responses models. minimax's speech models cannot prove this — they
        lack supports_function_calling, so the first filter already removed them."""
        models = providers.catalog_models("openai")
        assert models, "openai should have tool-capable chat models"
        assert "gpt-4o-realtime-preview" not in models
        assert "codex-mini-latest" not in models
        assert "gpt-5" in models

    def test_excludes_speech_models(self):
        """MiniMax's catalog is mostly speech synthesis. This passes via the
        supports_function_calling filter, not the mode filter — see
        test_excludes_non_chat_models for the mode filter's own guard."""
        assert not any(m.startswith("speech-") for m in providers.catalog_models("minimax"))

    def test_excludes_models_without_tool_support(self):
        """Groq ships safety classifiers that are chat-shaped but toolless."""
        assert not any("prompt-guard" in m for m in providers.catalog_models("groq"))

    def test_results_are_sorted_and_deduplicated(self):
        models = providers.catalog_models("deepseek")
        assert models == sorted(set(models))

    def test_unknown_provider_returns_empty(self):
        assert providers.catalog_models("nonesuch") == []

    def test_excludes_self_prefixed_models(self):
        """openrouter/auto's bare form still starts with "openrouter/", so the
        startswith guard in litellm_model would leave it half-qualified. Dropped
        from the curated list for this reason; the catalog must drop it too."""
        models = providers.catalog_models("openrouter")
        assert models, "openrouter should have tool-capable chat models"
        assert not any(m.startswith("openrouter/") for m in models)

    def test_import_failure_degrades_to_empty(self, monkeypatch):
        """A broken/absent litellm must cost the user free-text entry, never a
        crash in the config menu."""
        import builtins

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert providers.catalog_models("mistral") == []


class TestExpansionTable:
    def test_the_five_rows_are_appended_last_in_order(self):
        assert providers.names()[-7:] == (
            "custom", "custom_anthropic", "bedrock", "huggingface", "cohere",
            "perplexity", "replicate",
        )

    def test_bedrock_is_keyless_with_a_region_flavored_shortlist(self):
        info = providers.get("bedrock")
        assert info.keyring_entry is None
        assert info.key_optional is False
        assert info.prefix == "bedrock/"
        assert info.catalog_key == "bedrock"
        assert info.default_model.startswith("us.anthropic.")

    def test_cohere_catalog_key_is_cohere_chat(self):
        """LiteLLM's model_cost files command models under "cohere_chat"."""
        assert providers.get("cohere").catalog_key == "cohere_chat"

    def test_cohere_catalog_is_reachable(self):
        assert providers.catalog_models("cohere") != []

    def test_the_keyed_rows_have_service_named_entries(self):
        assert providers.get("huggingface").keyring_entry == "huggingface_api_key"
        assert providers.get("cohere").keyring_entry == "cohere_api_key"
        assert providers.get("perplexity").keyring_entry == "perplexity_api_key"
        assert providers.get("replicate").keyring_entry == "replicate_api_key"

    def test_replicate_curated_ids_are_nested_but_not_self_prefixed(self):
        for model in providers.get("replicate").models:
            assert "/" in model
            assert not model.startswith("replicate/")

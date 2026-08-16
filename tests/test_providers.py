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

    def test_ollama_is_the_only_keyless_provider(self):
        keyless = [p.name for p in providers.PROVIDERS if p.keyring_entry is None]
        assert keyless == ["ollama"]

    def test_every_cloud_provider_has_curated_models(self):
        for info in providers.PROVIDERS:
            if info.name == "ollama":
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
        """MiniMax's catalog is mostly speech synthesis — mode != chat."""
        assert not any(m.startswith("speech-") for m in providers.catalog_models("minimax"))

    def test_excludes_models_without_tool_support(self):
        """Groq ships safety classifiers that are chat-shaped but toolless."""
        assert not any("prompt-guard" in m for m in providers.catalog_models("groq"))

    def test_results_are_sorted_and_deduplicated(self):
        models = providers.catalog_models("deepseek")
        assert models == sorted(set(models))

    def test_unknown_provider_returns_empty(self):
        assert providers.catalog_models("nonesuch") == []

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

"""The provider and model rows of `nero config`. Every test drives stdin
through a pipe, which is exactly the non-TTY path ui.pick must support."""

import pytest
from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


@pytest.fixture
def manager(monkeypatch, tmp_path, isolate_audit_log):
    m = ConfigManager(config_dir=tmp_path)
    m.save(NeroConfig())
    monkeypatch.setattr(cli, "ConfigManager", lambda: m)
    return m


class TestProviderRow:
    def test_menu_lists_every_provider_label(self, manager):
        # Row 2 opens the provider picker; blank answer leaves it, blank exits.
        result = runner.invoke(cli.app, ["config"], input="2\n\n\n")
        assert "Mistral" in result.stdout
        assert "Moonshot (Kimi)" in result.stdout
        assert "Z.AI (GLM)" in result.stdout

    def test_choosing_a_provider_sets_it_and_its_default_model(self, manager, monkeypatch):
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "test-key")
        # Row 2 -> pick "mistral" by its position in PROVIDERS (5th) -> exit.
        runner.invoke(cli.app, ["config"], input="2\n5\n\n\n")
        loaded = manager.load()
        assert loaded.llm.provider == "mistral"
        assert loaded.llm.model == "mistral-large-latest"

    def test_blank_answer_leaves_the_provider_alone(self, manager):
        runner.invoke(cli.app, ["config"], input="2\n\n\n")
        assert manager.load().llm.provider == "claude"


class TestModelRow:
    def test_curated_models_are_offered(self, manager):
        # Row 3 opens the model picker for the current provider (claude).
        result = runner.invoke(cli.app, ["config"], input="3\n\n\n")
        assert "claude-opus-5" in result.stdout
        assert "Type a model name" in result.stdout

    def test_choosing_a_curated_model_saves_it(self, manager):
        # Row 3 -> option 2 is claude-opus-5 -> exit.
        runner.invoke(cli.app, ["config"], input="3\n2\n\n")
        assert manager.load().llm.model == "claude-opus-5"

    def test_typing_a_model_name_saves_it(self, manager):
        # Row 3 -> the "Type a model name…" row is last (5th for claude:
        # 3 curated + catalog + custom) -> then the free-text answer.
        runner.invoke(cli.app, ["config"], input="3\n5\nclaude-fable-5\n\n")
        assert manager.load().llm.model == "claude-fable-5"

    def test_catalog_row_offers_the_full_list(self, manager):
        # Row 3 -> option 4 is "Show all …" -> blank leaves it -> exit.
        result = runner.invoke(cli.app, ["config"], input="3\n4\n\n\n")
        assert "claude-haiku-4-5-20251001" in result.stdout

    def test_catalog_failure_degrades_to_free_text(self, manager, monkeypatch):
        monkeypatch.setattr(cli.providers, "catalog_models", lambda name: [])
        # Row 4 is the catalog row; an empty catalog degrades it to free text.
        runner.invoke(cli.app, ["config"], input="3\n4\nclaude-opus-5\n\n")
        assert manager.load().llm.model == "claude-opus-5"


class TestOllamaModelRow:
    def test_ollama_uses_free_text_not_a_cloud_list(self, manager):
        config = manager.load()
        config.llm.provider = "ollama"
        config.llm.model = "qwen3:8b"
        manager.save(config)
        runner.invoke(cli.app, ["config"], input="3\nphi4-mini\n\n")
        assert manager.load().llm.model == "phi4-mini"

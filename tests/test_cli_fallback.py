

class TestOllamaFallbackIsChecked:
    """A chain entry naming a model that was never pulled is worse than no
    entry at all: it looks like a safety net right up to the moment the primary
    fails and it turns out `ollama/llama3.2` was only ever `llama3.2:1b`."""

    def resolve(self, monkeypatch, model, pulled, up=True):
        from nero.cli import _resolve_fallback_client
        from nero.config.manager import ConfigManager
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        monkeypatch.setattr("nero.cli.ollama.reachable", lambda *a, **k: up)
        monkeypatch.setattr("nero.cli.ollama.has_model", lambda name, **k: name in pulled)
        monkeypatch.setattr("nero.cli.ollama.list_models", lambda *a, **k: pulled)
        config = NeroConfig()
        return _resolve_fallback_client(
            ConfigManager(), config, build_registry(config), "ollama", model
        )

    def test_a_missing_model_is_dropped_with_a_reason(self, monkeypatch):
        client, warning = self.resolve(monkeypatch, "llama3.2", ["llama3.2:1b"])
        assert client is None
        assert "llama3.2" in warning and "llama3.2:1b" in warning

    def test_a_pulled_model_resolves(self, monkeypatch):
        client, warning = self.resolve(monkeypatch, "llama3.2:1b", ["llama3.2:1b"])
        assert client is not None and warning is None

    def test_ollama_being_down_is_not_treated_as_a_missing_model(self, monkeypatch):
        """That is a separate problem, and the entry may work fine once it is
        running — dropping it here would disable the fallback permanently."""
        client, warning = self.resolve(monkeypatch, "llama3.2", [], up=False)
        assert client is not None and warning is None

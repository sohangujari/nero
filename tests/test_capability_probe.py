"""Ollama tool-capability warning on model switch (Phase 4 follow-up)."""
from typer.testing import CliRunner

from nero import cli
from nero.config.schema import NeroConfig

runner = CliRunner()


def _manager(tmp_path, provider, model):
    from nero.config.manager import ConfigManager
    m = ConfigManager(config_dir=tmp_path)
    cfg = NeroConfig()
    cfg.llm.provider = provider
    cfg.llm.model = model
    m.save(cfg)
    return m


class TestSupportsTools:
    def test_true_when_capabilities_list_tools(self, monkeypatch):
        from nero.llm import ollama
        monkeypatch.setattr(
            "httpx.post",
            lambda *a, **k: _FakeResp({"capabilities": ["completion", "tools"]}),
        )
        assert ollama.supports_tools("phi4-mini") is True

    def test_false_when_tools_absent(self, monkeypatch):
        from nero.llm import ollama
        monkeypatch.setattr(
            "httpx.post",
            lambda *a, **k: _FakeResp({"capabilities": ["completion", "vision"]}),
        )
        assert ollama.supports_tools("gemma3") is False

    def test_none_when_ollama_unreachable(self, monkeypatch):
        import httpx
        from nero.llm import ollama

        def boom(*a, **k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr("httpx.post", boom)
        assert ollama.supports_tools("gemma3") is None


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestWarning:
    def test_warns_on_switch_to_non_tool_model(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path, "ollama", "gemma3")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr("nero.llm.ollama.supports_tools", lambda m: False)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gemma3"])
        assert "no tool-calling support" in result.stdout
        assert "phi4-mini" in result.stdout

    def test_silent_for_tool_capable_model(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path, "ollama", "phi4-mini")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr("nero.llm.ollama.supports_tools", lambda m: True)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "phi4-mini"])
        assert "no tool-calling support" not in result.stdout

    def test_silent_for_cloud_provider(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path, "claude", "claude-sonnet-5")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # supports_tools must not even be consulted for a cloud provider.
        monkeypatch.setattr(
            "nero.llm.ollama.supports_tools",
            lambda m: (_ for _ in ()).throw(AssertionError("probed a cloud model")),
        )
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gpt-5"])
        assert "no tool-calling support" not in result.stdout

    def test_silent_when_capability_undeterminable(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path, "ollama", "gemma3")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr("nero.llm.ollama.supports_tools", lambda m: None)
        result = runner.invoke(cli.app, ["config", "set", "llm.model", "gemma3"])
        assert "no tool-calling support" not in result.stdout

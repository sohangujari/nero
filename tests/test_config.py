import keyring.errors
import pytest
from pydantic import ValidationError

from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import AssistantConfig, LLMConfig, NeroConfig
from nero.llm import providers


class TestSchema:
    def test_defaults(self):
        config = NeroConfig()
        assert config.assistant.name == "Nero"
        assert config.llm.provider == "claude"
        assert config.llm.model == "claude-sonnet-5"

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"assistant": {"name": "X", "voice": "loud"}})
        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"telemetry": {}})

    def test_nested_models(self):
        config = NeroConfig.model_validate(
            {"assistant": {"name": "Jarvis"}, "llm": {"model": "claude-opus-4-8"}}
        )
        assert isinstance(config.assistant, AssistantConfig)
        assert isinstance(config.llm, LLMConfig)
        assert config.assistant.name == "Jarvis"
        assert config.llm.provider == "claude"

    def test_provider_literal(self):
        for provider in ("claude", "openai", "gemini", "ollama", "mistral", "glm", "qwen"):
            assert NeroConfig.model_validate({"llm": {"provider": provider}}).llm.provider == provider
        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"llm": {"provider": "nonesuch"}})

    def test_hardware_defaults_to_undetected(self):
        hardware = NeroConfig().hardware
        assert hardware.detected_ram_gb is None
        assert hardware.detected_cpu_cores is None
        assert hardware.recommended_local_model is None


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(config_dir=tmp_path)


class TestConfigManager:
    def test_load_without_file_returns_defaults(self, manager):
        assert not manager.exists()
        config = manager.load()
        assert config == NeroConfig()

    def test_save_load_roundtrip(self, manager):
        config = NeroConfig()
        config.assistant.name = "Jarvis"
        manager.save(config)
        assert manager.exists()
        assert manager.load().assistant.name == "Jarvis"

    def test_config_file_never_contains_api_key(self, manager):
        manager.save(NeroConfig())
        text = manager.config_path.read_text()
        assert "api_key" not in text
        assert "sk-ant" not in text

    def test_set_value_persists(self, manager):
        manager.set_value("assistant.name", "Jarvis")
        assert manager.load().assistant.name == "Jarvis"
        manager.set_value("llm.model", "claude-opus-4-8")
        loaded = manager.load()
        assert loaded.llm.model == "claude-opus-4-8"
        assert loaded.assistant.name == "Jarvis"

    def test_set_value_provider_validated(self, manager):
        manager.set_value("llm.provider", "ollama")
        assert manager.load().llm.provider == "ollama"
        with pytest.raises(ConfigError):
            manager.set_value("llm.provider", "grok")

    def test_set_value_hardware_coerces_types(self, manager):
        manager.set_value("hardware.detected_ram_gb", "16.2")
        manager.set_value("hardware.detected_cpu_cores", "8")
        loaded = manager.load()
        assert loaded.hardware.detected_ram_gb == 16.2
        assert loaded.hardware.detected_cpu_cores == 8

    def test_set_value_unknown_key_raises(self, manager):
        with pytest.raises(ConfigError):
            manager.set_value("assistant.bogus", "x")
        with pytest.raises(ConfigError):
            manager.set_value("nope", "x")
        assert not manager.exists()

    def test_load_invalid_file_raises_config_error(self, manager):
        manager.config_dir.mkdir(parents=True, exist_ok=True)
        manager.config_path.write_text("assistant:\n  voice: loud\n")
        with pytest.raises(ConfigError) as exc_info:
            manager.load()
        assert str(manager.config_path) in str(exc_info.value)

    def test_api_keys_stored_per_provider(self, manager, monkeypatch):
        store = {}
        monkeypatch.setattr(
            "keyring.set_password", lambda svc, user, val: store.__setitem__((svc, user), val)
        )
        monkeypatch.setattr("keyring.get_password", lambda svc, user: store.get((svc, user)))
        assert manager.get_api_key("claude") is None
        manager.set_api_key("claude", "sk-ant-test-1234")
        manager.set_api_key("openai", "sk-oai-test-5678")
        manager.set_api_key("gemini", "gm-test-9012")
        assert store == {
            ("nero", "anthropic_api_key"): "sk-ant-test-1234",
            ("nero", "openai_api_key"): "sk-oai-test-5678",
            ("nero", "gemini_api_key"): "gm-test-9012",
        }
        assert manager.get_api_key("claude") == "sk-ant-test-1234"
        assert manager.get_api_key("openai") == "sk-oai-test-5678"

    def test_ollama_never_has_a_key(self, manager, monkeypatch):
        monkeypatch.setattr("keyring.get_password", lambda svc, user: "should-not-be-read")
        assert manager.get_api_key("ollama") is None
        with pytest.raises(ConfigError):
            manager.set_api_key("ollama", "anything")

    def test_unusable_keyring_reads_as_not_set(self, manager, monkeypatch):
        """Linux CI runners (and a locked/absent keychain anywhere) have no backend.

        Reading must degrade to "no key", not blow up: `config`/`config show`
        render the key through here, so an escaping KeyringError takes down the
        whole menu.
        """
        def boom(svc, user):
            raise keyring.errors.NoKeyringError("No recommended backend was available")

        monkeypatch.setattr("keyring.get_password", boom)
        assert manager.get_api_key("claude") is None

    def test_unusable_keyring_still_fails_loudly_on_write(self, manager, monkeypatch):
        """Writing is not the same call. A silently swallowed set_api_key would
        report success and lose the user's key."""
        def boom(svc, user, val):
            raise keyring.errors.NoKeyringError("No recommended backend was available")

        monkeypatch.setattr("keyring.set_password", boom)
        with pytest.raises(keyring.errors.KeyringError):
            manager.set_api_key("claude", "sk-ant-test-1234")

    def test_provider_needs_key(self, manager):
        assert manager.provider_needs_key("claude") is True
        assert manager.provider_needs_key("openai") is True
        assert manager.provider_needs_key("gemini") is True
        assert manager.provider_needs_key("ollama") is False

    def test_mask_api_key(self):
        assert ConfigManager.mask_api_key("sk-ant-api03-abcdefgh1234") == "sk-ant-...1234"
        assert ConfigManager.mask_api_key("someotherlongkey9876") == "...9876"
        assert ConfigManager.mask_api_key("tiny") == "****"


from typer.testing import CliRunner

import nero.cli as cli

runner = CliRunner()


from nero.hardware.detector import HardwareSpecs

FAKE_SPECS = HardwareSpecs(ram_gb=16.0, cpu_cores=8, os="Darwin", has_ollama=True)


@pytest.fixture
def detected(monkeypatch):
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_SPECS)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ConfigManager", lambda: ConfigManager(config_dir=tmp_path))
    store = {}
    monkeypatch.setattr(
        "keyring.set_password", lambda svc, user, val: store.__setitem__((svc, user), val)
    )
    monkeypatch.setattr("keyring.get_password", lambda svc, user: store.get((svc, user)))
    return tmp_path, store


class TestCLI:
    def test_version(self):
        result = runner.invoke(cli.app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_config_set_valid(self, cli_env):
        tmp_path, _ = cli_env
        result = runner.invoke(cli.app, ["config", "set", "assistant.name", "Jarvis"])
        assert result.exit_code == 0
        assert ConfigManager(config_dir=tmp_path).load().assistant.name == "Jarvis"

    def test_config_set_invalid_key(self, cli_env):
        tmp_path, _ = cli_env
        result = runner.invoke(cli.app, ["config", "set", "assistant.bogus", "x"])
        assert result.exit_code == 1
        assert "Unknown config key" in result.output
        assert not (tmp_path / "config.yaml").exists()

    def test_config_show_defaults(self, cli_env):
        result = runner.invoke(cli.app, ["config", "show"])
        assert result.exit_code == 0
        assert "Nero" in result.output
        assert "claude-sonnet-5" in result.output
        assert "not set" in result.output

    def test_config_show_masks_api_key(self, cli_env):
        _, store = cli_env
        store[("nero", "anthropic_api_key")] = "sk-ant-api03-abcdefgh1234"
        result = runner.invoke(cli.app, ["config", "show"])
        assert "sk-ant-...1234" in result.output
        assert "abcdefgh" not in result.output

    def test_chat_without_api_key_gives_guidance(self, cli_env):
        tmp_path, _ = cli_env
        ConfigManager(config_dir=tmp_path).save(NeroConfig())
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "nero config" in result.output
        assert "Traceback" not in result.output

    def test_first_run_setup_saves_key_config_and_hardware(self, cli_env, detected):
        tmp_path, store = cli_env
        # Input: Enter accepts the default provider (claude), Enter again keeps
        # the default model in the model picker, then the API key.
        # After setup the chat loop starts; EOF on stdin ends it cleanly.
        result = runner.invoke(cli.app, [], input="\n\nsk-ant-first-run-key\n")
        assert "Nero is ready" in result.output
        assert store[("nero", "anthropic_api_key")] == "sk-ant-first-run-key"
        loaded = ConfigManager(config_dir=tmp_path).load()
        assert loaded.hardware.detected_ram_gb == 16.0
        assert loaded.hardware.recommended_local_model == "qwen3:8b"

    def test_detect_persists_hardware_without_touching_provider(self, cli_env, detected):
        tmp_path, _ = cli_env
        result = runner.invoke(cli.app, ["detect"])
        assert result.exit_code == 0
        assert "qwen3:8b" in result.output
        loaded = ConfigManager(config_dir=tmp_path).load()
        assert loaded.hardware.detected_ram_gb == 16.0
        assert loaded.hardware.detected_cpu_cores == 8
        assert loaded.hardware.recommended_local_model == "qwen3:8b"
        assert loaded.llm.provider == "claude"

    def test_ollama_unreachable_gives_clear_message(self, cli_env, monkeypatch):
        tmp_path, _ = cli_env
        ConfigManager(config_dir=tmp_path).save(
            NeroConfig.model_validate({"llm": {"provider": "ollama", "model": "qwen3:8b"}})
        )
        monkeypatch.setattr("nero.llm.ollama.reachable", lambda **kwargs: False)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "ollama serve" in result.output
        assert "Traceback" not in result.output

    def test_ollama_ready_never_mentions_api_key(self, cli_env, monkeypatch):
        tmp_path, _ = cli_env
        ConfigManager(config_dir=tmp_path).save(
            NeroConfig.model_validate({"llm": {"provider": "ollama", "model": "qwen3:8b"}})
        )
        monkeypatch.setattr("nero.llm.ollama.reachable", lambda **kwargs: True)
        monkeypatch.setattr("nero.llm.ollama.has_model", lambda name, **kwargs: True)
        result = runner.invoke(cli.app, [])  # EOF ends the chat loop immediately
        assert result.exit_code == 0
        assert "listening" in result.output
        assert "API key" not in result.output

    def test_missing_cloud_key_redirects_to_config(self, cli_env):
        tmp_path, _ = cli_env
        ConfigManager(config_dir=tmp_path).save(
            NeroConfig.model_validate({"llm": {"provider": "openai", "model": "gpt-5"}})
        )
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "openai" in result.output
        assert "nero config" in result.output


class TestKeyringAcrossProviders:
    def test_every_cloud_provider_needs_a_key(self, manager):
        for info in providers.PROVIDERS:
            expected = info.name not in {"ollama", "bedrock"}
            assert manager.provider_needs_key(info.name) is expected, info.name

    def test_keys_round_trip_per_provider(self, manager, monkeypatch):
        store = {}
        monkeypatch.setattr(
            "keyring.set_password", lambda service, entry, value: store.__setitem__((service, entry), value)
        )
        monkeypatch.setattr("keyring.get_password", lambda service, entry: store.get((service, entry)))
        for info in providers.PROVIDERS:
            if info.keyring_entry is None:
                continue
            manager.set_api_key(info.name, f"key-for-{info.name}")
        for info in providers.PROVIDERS:
            if info.keyring_entry is None:
                continue
            assert manager.get_api_key(info.name) == f"key-for-{info.name}"

    def test_existing_entry_names_are_preserved(self, manager, monkeypatch):
        """A key stored before Phase 6 must still be found — no migration."""
        monkeypatch.setattr(
            "keyring.get_password",
            lambda service, entry: "old-key" if (service, entry) == ("nero", "anthropic_api_key") else None,
        )
        assert manager.get_api_key("claude") == "old-key"

    def test_friendly_names_map_to_upstream_entries(self):
        assert providers.get("qwen").keyring_entry == "dashscope_api_key"
        assert providers.get("glm").keyring_entry == "zai_api_key"
        assert providers.get("kimi").keyring_entry == "moonshot_api_key"

    def test_keyless_provider_rejects_a_key(self, manager):
        with pytest.raises(ConfigError):
            manager.set_api_key("ollama", "nope")

    def test_unknown_provider_has_no_key(self, manager):
        assert manager.provider_needs_key("nonesuch") is False
        assert manager.get_api_key("nonesuch") is None

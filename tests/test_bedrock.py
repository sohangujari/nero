"""v1.5.6: AWS Bedrock (ambient credentials + region) and friends."""
import asyncio

from nero.config.schema import LLMConfig
from nero.llm.client import LLMClient
from nero.skills.registry import SkillRegistry


class TestSchema:
    def test_aws_region_defaults_to_none(self):
        assert LLMConfig().aws_region is None

    def test_a_pre_v156_config_still_loads(self):
        loaded = LLMConfig.model_validate({"provider": "claude", "model": "claude-sonnet-5"})
        assert loaded.aws_region is None

    def test_a_stale_region_on_another_provider_is_representable(self):
        """The guard lives in LLMClient.aws_region, not here — the same
        inert-persistence rule llm.base_url follows."""
        config = LLMConfig(provider="claude", model="claude-sonnet-5", aws_region="us-east-1")
        assert config.aws_region == "us-east-1"


def _client(provider, model, aws_region=None):
    return LLMClient(
        config=LLMConfig(provider=provider, model=model, aws_region=aws_region),
        assistant_name="Nero",
        registry=SkillRegistry([]),
    )


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _completion_kwargs(monkeypatch, client):
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


class TestRegionGuard:
    def test_bedrock_exposes_its_region(self):
        assert _client("bedrock", "amazon.nova-pro-v1:0", "eu-west-1").aws_region == "eu-west-1"

    def test_a_stale_region_never_leaks_into_another_provider(self):
        assert _client("claude", "claude-sonnet-5", "us-east-1").aws_region is None

    def test_bedrock_without_a_region_is_none(self):
        assert _client("bedrock", "amazon.nova-pro-v1:0").aws_region is None

    def test_the_region_reaches_litellm(self, monkeypatch):
        client = _client("bedrock", "amazon.nova-pro-v1:0", "eu-west-1")
        assert _completion_kwargs(monkeypatch, client)["aws_region_name"] == "eu-west-1"

    def test_no_region_kwarg_without_a_config_region(self, monkeypatch):
        """LiteLLM must be left to resolve the environment itself."""
        client = _client("bedrock", "amazon.nova-pro-v1:0")
        assert "aws_region_name" not in _completion_kwargs(monkeypatch, client)

    def test_a_named_provider_sends_no_region_kwarg(self, monkeypatch):
        client = _client("claude", "claude-sonnet-5", "us-east-1")
        assert "aws_region_name" not in _completion_kwargs(monkeypatch, client)


class TestModelString:
    def test_bedrock_models_get_the_bedrock_prefix(self):
        client = _client("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert client.litellm_model == "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_a_replicate_nested_id_is_prefixed_exactly_once(self):
        config = LLMConfig(provider="replicate", model="openai/gpt-5")
        client = LLMClient(config=config, assistant_name="Nero", registry=SkillRegistry([]))
        assert client.litellm_model == "replicate/openai/gpt-5"


from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()

_AWS_ENV = ("AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION")


def _bedrock_manager(tmp_path, aws_region="us-east-1"):
    manager = ConfigManager(config_dir=tmp_path)
    manager.save(
        NeroConfig.model_validate(
            {"llm": {"provider": "bedrock",
                     "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                     "aws_region": aws_region}}
        )
    )
    return manager


def _clear_aws_env(monkeypatch):
    for name in _AWS_ENV:
        monkeypatch.delenv(name, raising=False)


class TestBedrockStartupGate:
    def test_bedrock_with_region_and_credentials_starts(self, monkeypatch, tmp_path):
        """No API key exists or is needed; EOF ends the chat loop cleanly."""
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: True)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 0
        assert "API key" not in result.output

    def test_bedrock_without_a_region_exits_naming_the_region(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path, None))
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: True)
        _clear_aws_env(monkeypatch)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "region" in result.output.lower()
        assert "API key" not in result.output

    def test_an_env_region_satisfies_the_gate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path, None))
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: True)
        _clear_aws_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        assert runner.invoke(cli.app, []).exit_code == 0

    def test_bedrock_without_credentials_exits_pointing_at_aws_configure(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: False)
        result = runner.invoke(cli.app, [])
        assert result.exit_code == 1
        assert "aws configure" in result.output
        assert "API key" not in result.output

    def test_bedrock_never_reaches_the_ollama_preflight(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: True)

        def fail(model):
            raise AssertionError(f"_ollama_preflight called with {model!r}")

        monkeypatch.setattr(cli, "_ollama_preflight", fail)
        assert runner.invoke(cli.app, []).exit_code == 0


from nero.hardware.detector import HardwareSpecs


class TestBedrockFlows:
    def test_switching_to_bedrock_writes_region_and_model_and_asks_no_key(
        self, monkeypatch, tmp_path
    ):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 2 -> provider row 16 (bedrock) -> accept region default (us-east-1)
        # -> keep the default model (blank) -> finish. A key prompt would
        # consume input that isn't there and derail the sequence.
        result = runner.invoke(cli.app, ["config"], input="2\n16\n\n\n\n")
        assert result.exit_code == 0
        loaded = manager.load()
        assert loaded.llm.provider == "bedrock"
        assert loaded.llm.aws_region == "us-east-1"
        assert loaded.llm.model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert "AWS credentials" in result.output
        assert "API key" not in result.output

    def test_first_run_with_bedrock_completes(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(cli, "_bedrock_credentials_present", lambda: True)
        monkeypatch.setattr(
            cli, "detect_hardware",
            lambda: HardwareSpecs(ram_gb=16.0, cpu_cores=8, os="Darwin", has_ollama=False),
        )
        # Provider row 16 (bedrock) -> region eu-west-1 -> keep default model
        # (blank) -> chat loop EOF.
        result = runner.invoke(cli.app, [], input="16\neu-west-1\n\n\n")
        assert result.exit_code == 0
        assert "Traceback" not in result.output
        loaded = manager.load()
        assert loaded.llm.provider == "bedrock"
        assert loaded.llm.aws_region == "eu-west-1"
        assert loaded.llm.model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_row_16_shows_and_edits_the_region_under_bedrock(self, monkeypatch, tmp_path):
        manager = _bedrock_manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="16\nap-southeast-2\n\n")
        assert "AWS Region" in result.stdout
        assert manager.load().llm.aws_region == "ap-southeast-2"

    def test_row_16_is_absent_for_a_named_provider(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert "AWS Region" not in result.stdout

    def test_row_4_under_bedrock_talks_credentials_not_locally(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        result = runner.invoke(cli.app, ["config"], input="4\n\n")
        assert "AWS credentials" in result.output
        assert "runs locally" not in result.output

    def test_config_show_lists_the_region(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        assert "us-east-1" in runner.invoke(cli.app, ["config", "show"]).output


class TestRegionSurfaces:
    def test_setting_an_inert_region_warns_but_keeps_it(self, monkeypatch, tmp_path):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config", "set", "llm.aws_region", "us-east-1"])
        assert result.exit_code == 0
        assert "bedrock" in result.output
        assert manager.load().llm.aws_region == "us-east-1"

    def test_no_warning_when_the_provider_is_bedrock(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "ConfigManager", lambda: _bedrock_manager(tmp_path))
        result = runner.invoke(cli.app, ["config", "set", "llm.aws_region", "eu-west-1"])
        assert "will not be used" not in result.output


class TestKeyedRowRepresentative:
    def test_switching_to_perplexity_lands_the_key_in_its_own_entry(
        self, monkeypatch, tmp_path, isolate_keyring
    ):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 2 -> provider row 19 (perplexity) -> keep default model (blank)
        # -> key -> finish.
        result = runner.invoke(cli.app, ["config"], input="2\n19\n\npplx-test\n\n")
        assert result.exit_code == 0
        loaded = manager.load()
        assert loaded.llm.provider == "perplexity"
        assert loaded.llm.model == "sonar-pro"
        assert isolate_keyring == {("nero", "perplexity_api_key"): "pplx-test"}

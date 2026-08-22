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

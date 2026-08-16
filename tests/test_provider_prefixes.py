"""Every provider's curated models must survive the trip to a LiteLLM model
string. The nested cases (OpenRouter, Groq) are the ones that bite."""

import pytest

from nero.config.schema import LLMConfig
from nero.llm import providers
from nero.llm.client import LLMClient
from nero.skills.registry import SkillRegistry


def _client(provider: str, model: str) -> LLMClient:
    return LLMClient(
        config=LLMConfig(provider=provider, model=model),
        assistant_name="Nero",
        registry=SkillRegistry([]),
    )


class TestLitellmModel:
    @pytest.mark.parametrize("info", providers.PROVIDERS, ids=lambda i: i.name)
    def test_every_curated_model_gets_its_prefix(self, info):
        for model in info.models:
            resolved = _client(info.name, model).litellm_model
            assert resolved == f"{info.prefix}{model}"

    def test_bare_providers_are_left_alone(self):
        assert _client("claude", "claude-sonnet-5").litellm_model == "claude-sonnet-5"
        assert _client("openai", "gpt-5").litellm_model == "gpt-5"

    def test_nested_openrouter_model_keeps_both_segments(self):
        resolved = _client("openrouter", "anthropic/claude-sonnet-4.6").litellm_model
        assert resolved == "openrouter/anthropic/claude-sonnet-4.6"

    def test_nested_groq_model_keeps_both_segments(self):
        assert _client("groq", "openai/gpt-oss-120b").litellm_model == "groq/openai/gpt-oss-120b"

    def test_an_already_prefixed_model_is_not_double_prefixed(self):
        assert _client("qwen", "dashscope/qwen3-max").litellm_model == "dashscope/qwen3-max"

    def test_friendly_name_maps_to_the_upstream_prefix(self):
        assert _client("qwen", "qwen3-max").litellm_model == "dashscope/qwen3-max"
        assert _client("glm", "glm-5").litellm_model == "zai/glm-5"
        assert _client("kimi", "kimi-k2.6").litellm_model == "moonshot/kimi-k2.6"

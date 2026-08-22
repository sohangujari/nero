"""v1.5.6: AWS Bedrock (ambient credentials + region) and friends."""
from nero.config.schema import LLMConfig


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

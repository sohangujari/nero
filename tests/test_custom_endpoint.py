"""v1.5.4: a custom OpenAI-compatible endpoint, configured by base URL."""
from typing import get_args

import pytest
from pydantic import ValidationError

from nero.config.schema import LLMConfig, Provider
from nero.llm import providers


class TestSchema:
    def test_base_url_defaults_to_none(self):
        assert LLMConfig().base_url is None

    def test_a_config_without_base_url_still_loads(self):
        """extra='forbid' plus a new field: files written before v1.5.4 must
        keep loading, which is the whole migration story."""
        loaded = LLMConfig.model_validate({"provider": "claude", "model": "claude-sonnet-5"})
        assert loaded.base_url is None

    def test_a_non_http_url_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="custom", model="x", base_url="localhost:1234")

    def test_a_trailing_slash_is_stripped(self):
        config = LLMConfig(provider="custom", model="x", base_url="http://localhost:1234/v1/")
        assert config.base_url == "http://localhost:1234/v1"

    def test_a_stale_url_on_another_provider_is_representable(self):
        """The guard lives in LLMClient.api_base, not here. This state must be
        constructible or the leak test in Task 3 cannot be written at all."""
        config = LLMConfig(provider="claude", model="claude-sonnet-5", base_url="http://stale")
        assert config.base_url == "http://stale"


class TestTable:
    def test_custom_is_in_the_table_and_the_literal(self):
        assert "custom" in providers.names()
        assert get_args(Provider) == providers.names()

    def test_custom_is_the_only_key_optional_provider(self):
        assert [p.name for p in providers.PROVIDERS if p.key_optional] == ["custom"]

    def test_custom_can_still_store_a_key(self):
        """key_optional asks whether a MISSING key is fatal, not whether one
        can be stored — a Together endpoint needs a key like any cloud provider."""
        assert providers.get("custom").keyring_entry == "custom_api_key"

    def test_custom_has_no_catalog(self):
        info = providers.get("custom")
        assert info.models == ()
        assert info.default_model is None
        assert info.catalog_key == ""
        assert providers.catalog_models("custom") == []

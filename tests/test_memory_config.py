import pytest
from pydantic import ValidationError

from nero.config.schema import MemoryConfig, NeroConfig


class TestDefaults:
    def test_memory_defaults(self):
        cfg = NeroConfig()
        assert cfg.memory.enabled is True
        assert cfg.memory.max_history_turns == 20

    def test_roundtrips_through_full_config(self):
        data = NeroConfig().model_dump()
        assert data["memory"]["max_history_turns"] == 20
        assert NeroConfig.model_validate(data).memory.enabled is True


class TestValidation:
    def test_rejects_unknown_key(self):
        with pytest.raises(ValidationError):
            MemoryConfig.model_validate({"enabled": True, "bogus": 1})

    def test_accepts_string_bool_and_int(self):
        # ConfigManager.set_value writes strings; pydantic coerces.
        m = MemoryConfig.model_validate({"enabled": "false", "max_history_turns": "5"})
        assert m.enabled is False
        assert m.max_history_turns == 5


class TestSetValue:
    def test_nested_keys(self, tmp_path):
        from nero.config.manager import ConfigManager

        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        assert manager.set_value("memory.enabled", "false").memory.enabled is False
        assert manager.set_value("memory.max_history_turns", "8").memory.max_history_turns == 8

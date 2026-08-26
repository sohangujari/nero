import pytest
from pydantic import ValidationError

from nero.config.schema import MemoryConfig, NeroConfig


class TestDefaults:
    def test_memory_defaults(self):
        cfg = NeroConfig()
        assert cfg.memory.enabled is True
        assert cfg.memory.max_history_turns == 20
        assert cfg.memory.notes_dir is None
        assert cfg.memory.notes_max_bytes == 2_000_000
        assert cfg.memory.compact_after_messages == 40

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

    def test_rejects_negative_max_history_turns(self):
        # -1 flows into HistoryStore(max_turns=-1) -> SQLite `LIMIT -2` -> no
        # limit at all, dumping the entire history table into model context.
        with pytest.raises(ValidationError):
            MemoryConfig.model_validate({"max_history_turns": -1})

    def test_accepts_zero_and_default_max_history_turns(self):
        assert MemoryConfig.model_validate({"max_history_turns": 0}).max_history_turns == 0
        assert MemoryConfig.model_validate({"max_history_turns": 20}).max_history_turns == 20

    def test_accepts_zero_compact_after_messages(self):
        assert MemoryConfig.model_validate({"compact_after_messages": 0}).compact_after_messages == 0

    def test_rejects_negative_compact_after_messages(self):
        with pytest.raises(ValidationError):
            MemoryConfig.model_validate({"compact_after_messages": -1})

    def test_rejects_non_positive_notes_max_bytes(self):
        with pytest.raises(ValidationError):
            MemoryConfig.model_validate({"notes_max_bytes": 0})


class TestSetValue:
    def test_nested_keys(self, tmp_path):
        from nero.config.manager import ConfigManager

        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        assert manager.set_value("memory.enabled", "false").memory.enabled is False
        assert manager.set_value("memory.max_history_turns", "8").memory.max_history_turns == 8

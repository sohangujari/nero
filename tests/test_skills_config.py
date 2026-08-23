import pytest
from pydantic import ValidationError

from nero.config.schema import NeroConfig, SkillsConfig, SkillToggles


class TestDefaults:
    def test_mode_defaults_to_online(self):
        assert NeroConfig().mode == "online"

    def test_all_skills_enabled_by_default(self):
        toggles = NeroConfig().skills.enabled
        assert toggles.open_app is True
        assert toggles.open_website is True
        assert toggles.get_weather is True
        assert toggles.play_music is True

    def test_read_only_new_skills_default_enabled(self):
        toggles = NeroConfig().skills.enabled
        assert toggles.read_file is True
        assert toggles.fetch_web_page is True

    def test_destructive_new_skills_default_disabled(self):
        toggles = NeroConfig().skills.enabled
        assert toggles.write_file is False
        assert toggles.edit_file is False
        assert toggles.delete_path is False
        assert toggles.move_path is False

    def test_execution_skills_default_disabled(self):
        toggles = NeroConfig().skills.enabled
        assert toggles.run_shell is False
        assert toggles.git_command is False
        assert toggles.run_python is False
        assert toggles.run_javascript is False

    def test_weather_location_defaults_to_none(self):
        assert NeroConfig().skills.weather.default_location is None

    def test_roundtrips_through_full_config(self):
        data = NeroConfig().model_dump()
        assert data["mode"] == "online"
        assert data["skills"]["enabled"]["get_weather"] is True
        assert NeroConfig.model_validate(data).skills.weather.default_location is None


class TestValidation:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"mode": "airplane"})

    def test_rejects_unknown_skill_name(self):
        with pytest.raises(ValidationError):
            SkillToggles.model_validate({"open_app": True, "hack_mainframe": True})

    def test_rejects_unknown_skills_section_key(self):
        with pytest.raises(ValidationError):
            SkillsConfig.model_validate({"bogus": {}})

    def test_toggles_accept_string_booleans(self):
        # ConfigManager.set_value writes strings; pydantic's lax mode coerces.
        assert SkillToggles.model_validate({"open_app": "false"}).open_app is False


class TestSetValue:
    def test_nested_skill_toggle(self, tmp_path):
        from nero.config.manager import ConfigManager

        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        config = manager.set_value("skills.enabled.get_weather", "false")
        assert config.skills.enabled.get_weather is False
        assert manager.load().skills.enabled.get_weather is False

    def test_mode(self, tmp_path):
        from nero.config.manager import ConfigManager

        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        assert manager.set_value("mode", "offline").mode == "offline"

    def test_rejects_bad_mode(self, tmp_path):
        from nero.config.manager import ConfigError, ConfigManager

        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        with pytest.raises(ConfigError):
            manager.set_value("mode", "airplane")

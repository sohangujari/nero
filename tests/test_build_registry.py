"""`build_registry` (nero/skills/registry.py) wired end to end.

No test here calls WeatherSkill.execute() without replacing `_fetch` first
— building the registry alone must never touch the network.
"""

from nero.config.schema import NeroConfig
from nero.core.audit_log import AuditLog


def make_config(mode="online", **enabled_overrides):
    config = NeroConfig()
    config.mode = mode
    for name, value in enabled_overrides.items():
        setattr(config.skills.enabled, name, value)
    return config


class TestOfflineAndDisabledGating:
    def test_tool_definitions_hide_network_skills_and_disabled_skill(self):
        from nero.skills.registry import build_registry

        config = make_config(mode="offline", open_app=False)
        registry = build_registry(config)
        names = {d["function"]["name"] for d in registry.tool_definitions()}
        # open_website, get_weather, and fetch_web_page require the network ->
        # hidden offline. open_app is explicitly disabled -> hidden regardless
        # of mode. write_file/edit_file/delete_path/move_path default disabled.
        # play_music and read_file are local and enabled -> what's left standing.
        assert names == {"play_music", "read_file"}

    def test_known_names_lists_every_registered_skill(self):
        from nero.skills.registry import build_registry

        config = make_config(mode="offline", open_app=False)
        registry = build_registry(config)
        assert registry.known_names() == {
            "open_app",
            "open_website",
            "get_weather",
            "play_music",
            "read_file",
            "write_file",
            "edit_file",
            "delete_path",
            "move_path",
            "fetch_web_page",
        }


class TestAuditWiring:
    def test_audit_log_passed_through_records_a_call(self, tmp_path):
        import asyncio

        from nero.skills.registry import build_registry

        config = make_config()
        audit = AuditLog(tmp_path / "audit.db")
        registry = build_registry(config, audit=audit)

        # play_music is local (no network) but still drives a real per-platform
        # controller (subprocess/pynput) unless one is injected — stub it out
        # so this test only proves the audit wiring, not platform media control.
        play_music = registry.get("play_music")
        play_music._controller = type("Stub", (), {"control": lambda self, action: "ok"})()

        asyncio.run(registry.execute("play_music", {"action": "play"}, provider="claude"))

        entries = audit.recent()
        assert len(entries) == 1
        assert entries[0].skill_name == "play_music"


class TestFilesAndWebWiring:
    def test_all_six_new_skills_register(self):
        from nero.skills.registry import build_registry

        registry = build_registry(make_config())
        for name in (
            "read_file", "write_file", "edit_file", "delete_path", "move_path",
            "fetch_web_page",
        ):
            assert registry.get(name) is not None

    def test_destructive_ones_default_disabled(self):
        from nero.skills.registry import build_registry

        registry = build_registry(make_config())
        for name in ("write_file", "edit_file", "delete_path", "move_path"):
            assert registry.is_enabled(name) is False

    def test_read_only_ones_default_enabled(self):
        from nero.skills.registry import build_registry

        registry = build_registry(make_config())
        for name in ("read_file", "fetch_web_page"):
            assert registry.is_enabled(name) is True

    def test_destructive_file_skill_refused_with_no_confirm_callback(self, tmp_path):
        import asyncio

        from nero.skills.registry import build_registry

        config = make_config(write_file=True)  # enabled, but no confirm wired
        registry = build_registry(config)
        target = tmp_path / "f.txt"
        result = asyncio.run(
            registry.execute("write_file", {"path": str(target), "content": "x"})
        )
        assert "declined" in result
        assert not target.exists()


class TestWeatherWiring:
    def test_weather_skill_receives_configured_default_location(self):
        from nero.skills.registry import build_registry

        config = make_config()
        config.skills.weather.default_location = "Oslo"
        registry = build_registry(config)
        weather = registry.get("get_weather")
        assert weather._default_location == "Oslo"

    def test_weather_skill_receives_the_on_location_resolved_callback(self):
        from nero.skills.registry import build_registry

        config = make_config()
        seen = []
        registry = build_registry(config, on_location_resolved=seen.append)
        weather = registry.get("get_weather")
        # Bound methods compare equal (not `is`, a fresh wrapper each access).
        assert weather._on_location_resolved == seen.append

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
        # open_website and get_weather require the network -> hidden offline.
        # open_app is explicitly disabled -> hidden regardless of mode.
        # play_music is local and enabled -> the only one left standing.
        assert names == {"play_music"}

    def test_known_names_still_lists_all_four_skills(self):
        from nero.skills.registry import build_registry

        config = make_config(mode="offline", open_app=False)
        registry = build_registry(config)
        assert registry.known_names() == {
            "open_app",
            "open_website",
            "get_weather",
            "play_music",
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

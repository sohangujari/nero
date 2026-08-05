import asyncio

from nero.config.schema import NeroConfig
from nero.core.audit_log import AuditLog

from nero import cli

GEOCODE_RESPONSE = {
    "results": [{"name": "Oslo", "country": "Norway", "latitude": 59.91, "longitude": 10.75}]
}

FORECAST_RESPONSE = {
    "current": {
        "temperature_2m": 14.2,
        "apparent_temperature": 12.8,
        "weather_code": 61,
        "wind_speed_10m": 11.0,
    },
    "daily": {
        "temperature_2m_max": [16.0, 18.5],
        "temperature_2m_min": [9.0, 11.0],
        "weather_code": [61, 0],
    },
}


class ExplodingManager:
    """A ConfigManager stand-in whose set_value always fails like a real
    read-only config dir or full disk would."""

    def set_value(self, key_path, value):
        raise OSError("Read-only file system")


def run(coro):
    return asyncio.run(coro)


class TestRememberLocationIsBestEffort:
    def test_config_write_failure_does_not_corrupt_a_successful_result(self, tmp_path):
        # Builds the real registry wiring (cli._build_registry), then breaks
        # only the config write, to prove a save failure can't turn a
        # successful weather lookup into an error the user sees.
        config = NeroConfig()
        registry = cli._build_registry(ExplodingManager(), config)
        skill = registry.get("get_weather")

        queue = [GEOCODE_RESPONSE, FORECAST_RESPONSE]

        async def fake_fetch(url, params):
            return queue.pop(0)

        skill._fetch = fake_fetch

        result = run(registry.execute("get_weather", {"location": "Oslo"}, provider="claude"))
        assert "Error" not in result
        assert "Oslo" in result

    def test_config_write_failure_does_not_corrupt_the_audit_entry(self, isolate_audit_log):
        # The registry's audit entry is built from the same result string the
        # user gets — so if the report survives, the audited summary must too.
        # `isolate_audit_log` (autouse) points default_audit_path at a tmp_path
        # file, so this exercises cli._build_registry's real audit wiring.
        config = NeroConfig()
        registry = cli._build_registry(ExplodingManager(), config)
        skill = registry.get("get_weather")

        queue = [GEOCODE_RESPONSE, FORECAST_RESPONSE]

        async def fake_fetch(url, params):
            return queue.pop(0)

        skill._fetch = fake_fetch

        run(registry.execute("get_weather", {"location": "Oslo"}, provider="claude"))
        entries = AuditLog(isolate_audit_log).recent()
        assert len(entries) == 1
        assert "Error" not in entries[0].result_summary
        assert "Oslo" in entries[0].result_summary

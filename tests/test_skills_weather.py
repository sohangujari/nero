import asyncio

import httpx
import pytest

from nero.skills.weather.server import WeatherSkill, describe_code

GEOCODE_RESPONSE = {
    "results": [
        {"name": "Oslo", "country": "Norway", "latitude": 59.91, "longitude": 10.75}
    ]
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


def make_skill(default_location=None, on_location_resolved=None, responses=None):
    skill = WeatherSkill(
        default_location=default_location, on_location_resolved=on_location_resolved
    )
    queue = list(responses if responses is not None else [GEOCODE_RESPONSE, FORECAST_RESPONSE])
    skill.requests = []

    async def fake_fetch(url, params):
        skill.requests.append((url, params))
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    skill._fetch = fake_fetch
    return skill


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


class TestMeta:
    def test_metadata(self):
        skill = WeatherSkill()
        assert skill.meta.name == "get_weather"
        assert skill.meta.requires_network is True
        assert skill.meta.permission_tier == "read_only"
        assert "internet connection" in skill.meta.offline_message
        assert skill.meta.input_schema["required"] == []


class TestDescribeCode:
    def test_known_codes(self):
        assert describe_code(0) == "clear sky"
        assert describe_code(61) == "light rain"
        assert describe_code(95) == "a thunderstorm"

    def test_unknown_code_is_generic(self):
        assert describe_code(1234) == "unsettled weather"


class TestSuccess:
    def test_reports_current_conditions(self):
        result = run(make_skill(), location="Oslo")
        assert "Oslo" in result
        assert "14" in result
        assert "light rain" in result

    def test_includes_tomorrow(self):
        assert "omorrow" in run(make_skill(), location="Oslo")

    def test_geocodes_the_requested_location(self):
        skill = make_skill()
        run(skill, location="Oslo")
        assert skill.requests[0][1]["name"] == "Oslo"

    def test_uses_default_location_when_none_given(self):
        skill = make_skill(default_location="Oslo")
        run(skill)
        assert skill.requests[0][1]["name"] == "Oslo"

    def test_explicit_location_beats_default(self):
        skill = make_skill(default_location="Paris")
        run(skill, location="Oslo")
        assert skill.requests[0][1]["name"] == "Oslo"


class TestLocationMemory:
    def test_first_explicit_location_is_remembered(self):
        # No default set yet — the first success seeds it.
        saved = []
        run(make_skill(on_location_resolved=saved.append), location="Oslo")
        assert saved == ["Oslo"]

    def test_explicit_location_does_not_overwrite_an_existing_default(self):
        # A deliberately-set default must survive a one-off "weather in X?".
        saved = []
        run(make_skill(default_location="London", on_location_resolved=saved.append),
            location="Tokyo")
        assert saved == []

    def test_default_location_is_not_re_saved(self):
        saved = []
        run(make_skill(default_location="Oslo", on_location_resolved=saved.append))
        assert saved == []

    def test_failure_is_not_remembered(self):
        saved = []
        skill = make_skill(on_location_resolved=saved.append, responses=[{"results": []}])
        run(skill, location="Nowhereville")
        assert saved == []


class TestErrors:
    def test_no_location_and_no_default_asks(self):
        skill = make_skill()
        result = run(skill)
        assert skill.requests == []
        assert "ask the user" in result.lower()

    def test_unknown_place(self):
        result = run(make_skill(responses=[{"results": []}]), location="Nowhereville")
        assert "couldn't find" in result.lower()
        assert "Nowhereville" in result

    def test_missing_results_key(self):
        result = run(make_skill(responses=[{}]), location="Nowhereville")
        assert "couldn't find" in result.lower()

    def test_network_error_on_geocode(self):
        skill = make_skill(responses=[httpx.ConnectError("no route")])
        result = run(skill, location="Oslo")
        assert "couldn't reach the weather service" in result.lower()
        assert "ConnectError" not in result

    def test_network_error_on_forecast(self):
        skill = make_skill(responses=[GEOCODE_RESPONSE, httpx.ReadTimeout("slow")])
        result = run(skill, location="Oslo")
        assert "couldn't reach the weather service" in result.lower()

    def test_malformed_forecast_payload(self):
        skill = make_skill(responses=[GEOCODE_RESPONSE, {"current": {}}])
        result = run(skill, location="Oslo")
        assert "couldn't read" in result.lower()

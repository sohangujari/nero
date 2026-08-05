from collections.abc import Callable

import httpx

from nero.skills.base import Skill, SkillMeta

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# WMO weather interpretation codes, phrased to drop into a sentence.
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a thunderstorm with heavy hail",
}


def describe_code(code) -> str:
    return WMO_CODES.get(code, "unsettled weather")


class WeatherSkill(Skill):
    meta = SkillMeta(
        name="get_weather",
        description=(
            "Get the current weather and a short forecast for a place. Use this "
            "when the user asks about the weather, temperature, or whether it will "
            "rain. If the user names a place, pass it as `location`; otherwise omit "
            "it to use their saved default location."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "A city or place name, e.g. 'Oslo' or 'Paris, France'.",
                }
            },
            "required": [],
        },
        requires_network=True,
        permission_tier="read_only",
        offline_message=(
            "Weather needs an internet connection, and you're in offline mode right now."
        ),
    )

    def __init__(
        self,
        default_location: str | None = None,
        on_location_resolved: Callable[[str], None] | None = None,
    ):
        self._default_location = default_location
        # Injected rather than importing ConfigManager, so the skill stays a pure
        # unit with no knowledge of how Nero persists anything.
        self._on_location_resolved = on_location_resolved

    async def _fetch(self, url: str, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def execute(self, **kwargs) -> str:
        requested = str(kwargs.get("location") or "").strip()
        location = requested or (self._default_location or "").strip()
        if not location:
            return (
                "No default location is set. Ask the user which city they want the "
                "weather for, then call this skill again with that location."
            )
        try:
            place = await self._geocode(location)
            if place is None:
                return f"I couldn't find a place called {location!r}."
            report = await self._report(place)
        except httpx.HTTPError:
            # Never leak a raw transport error to the user.
            return "I couldn't reach the weather service just now. Try again in a moment."
        except (KeyError, IndexError, TypeError, ValueError):
            return "I couldn't read the forecast that came back from the weather service."
        # Seed the default from the first explicit location the user names, once
        # the lookup has succeeded — but never overwrite a default they set
        # deliberately, so a one-off "weather in Tokyo?" doesn't move their home.
        if (
            requested
            and not self._default_location
            and self._on_location_resolved is not None
        ):
            self._on_location_resolved(requested)
        return report

    async def _geocode(self, location: str) -> dict | None:
        payload = await self._fetch(
            GEOCODE_URL, {"name": location, "count": 1, "format": "json"}
        )
        results = payload.get("results") or []
        return results[0] if results else None

    async def _report(self, place: dict) -> str:
        payload = await self._fetch(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
                ),
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "forecast_days": 2,
                "timezone": "auto",
            },
        )
        current = payload["current"]
        daily = payload["daily"]
        label = ", ".join(part for part in (place.get("name"), place.get("country")) if part)
        return (
            f"Right now in {label}: {describe_code(current['weather_code'])}, "
            f"{current['temperature_2m']:.0f}°C "
            f"(feels like {current['apparent_temperature']:.0f}°C), "
            f"wind {current['wind_speed_10m']:.0f} km/h. "
            f"Today {daily['temperature_2m_min'][0]:.0f} to "
            f"{daily['temperature_2m_max'][0]:.0f}°C. "
            f"Tomorrow: {describe_code(daily['weather_code'][1])}, "
            f"{daily['temperature_2m_min'][1]:.0f} to "
            f"{daily['temperature_2m_max'][1]:.0f}°C."
        )


SKILL = WeatherSkill()

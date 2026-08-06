"""Prompt-context builders for Robin's conversation system prompt. Ported from
recover/prompt_context/conversation.py, recover/prompt_context/prompt_settings.py, and the
context helpers in recover/shared/utils.py.

DB-backed pieces are replaced with either plain parameters or graceful local-file fallbacks:
  - Weather uses the public, keyless Open-Meteo API directly and defaults to a Boston-area
    lat/lon (same default the original used) if you don't pass your own location.
  - Schedule reads a bundled dummy calendar.json ("hardcoded dummy calendar for
    development/testing" upstream) next to this file. Point CALENDAR_PATH at your own file
    of the same shape to use real data; a missing/invalid file degrades to an empty schedule.
  - Personal profile reads an optional local JSON file per user id from a `profiles/`
    directory next to this file. No files are bundled (upstream's were real-looking patient
    data); missing is fine, it degrades to an empty profile.
  - Step count has no default source (upstream calls a Northeastern-internal Garmin/ubiwell
    proxy tied to per-patient DB metadata) -- pass your own `steps_provider(external_id)`
    callable to engine.process_turn()/this module's context builder if you want it filled in.
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).parent
CALENDAR_PATH = HERE / "calendar.json"
PROFILES_DIR = HERE / "profiles"
SYSTEM_CAPABILITIES_PATH = HERE / "system_capabilities.txt"

EASTERN_TZ = ZoneInfo("America/New_York")
DEFAULT_LOCATION = {"lat": 42.34, "lon": -71.09}  # matches upstream's hardcoded default

WEATHER_CODE_MAP = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow fall", 73: "moderate snow fall", 75: "heavy snow fall", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}

SHARED_PROMPT_FEATURE_DEFAULTS: Dict[str, bool] = {
    "include_temporal_context": True,
    "include_weather": True,
    "include_personal_profile": True,
    "include_steps": True,
    "include_quote_of_day": False,
}

POSITIVE_QUOTES = [
    "Small steps each day add up to big changes.",
    "You are stronger than you think and capable of steady progress.",
    "Every day is a new chance to do one kind thing for yourself.",
    "Progress, not perfection, is what moves life forward.",
    "A calm breath and a hopeful thought can reset your whole day.",
    "You have already made it through hard days before.",
    "Your effort today matters, even when it feels small.",
    "Kindness to yourself is a powerful daily habit.",
]


def get_prompt_feature_flags() -> Dict[str, bool]:
    return dict(SHARED_PROMPT_FEATURE_DEFAULTS)


def quote_of_the_day() -> str:
    day_index = datetime.utcnow().timetuple().tm_yday
    return POSITIVE_QUOTES[day_index % len(POSITIVE_QUOTES)]


def format_number_for_tts(number) -> str:
    if number is None:
        return "unknown"
    number = int(number)
    if number == 0:
        return "zero"

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def two_digits(n):
        if n < 20:
            return ones[n]
        if n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
        return ""

    def three_digits(n):
        if n < 100:
            return two_digits(n)
        return ones[n // 100] + " hundred" + (" " + two_digits(n % 100) if n % 100 != 0 else "")

    if number < 1000:
        return three_digits(number)
    if number < 1000000:
        thousands, remainder = number // 1000, number % 1000
        result = three_digits(thousands) + " thousand"
        if remainder > 0:
            result += " " + three_digits(remainder)
        return result
    return f"{number:,}"


def steps_info_for_conversation(total_steps) -> str:
    if total_steps is None:
        return "Step count data is not available at this time."
    return f"{format_number_for_tts(total_steps)} steps"


def get_temporal_context() -> Dict[str, str]:
    now_est = datetime.now(EASTERN_TZ)
    return {
        "current_day": now_est.strftime("%A"),
        "current_date": now_est.strftime("%B %d, %Y"),
        "current_time": now_est.strftime("%I:%M %p"),
    }


def _weather_description(weather_code) -> str:
    return WEATHER_CODE_MAP.get(weather_code, "unknown weather condition")


def _format_current_weather_summary(temperature, weather_code) -> str:
    return f"{round(temperature)} degrees Fahrenheit and {_weather_description(weather_code)}"


def _build_hourly_forecast_entries(data, *, now_est: datetime, limit: int = 6):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    weather_codes = hourly.get("weather_code", [])
    precip_probs = hourly.get("precipitation_probability", [])

    entries = []
    for index, time_value in enumerate(times):
        try:
            forecast_dt = datetime.fromisoformat(time_value)
        except Exception:
            continue
        if forecast_dt.tzinfo is None:
            forecast_dt = forecast_dt.replace(tzinfo=EASTERN_TZ)
        forecast_dt = forecast_dt.astimezone(EASTERN_TZ)
        if forecast_dt < now_est:
            continue

        entries.append({
            "time": forecast_dt.strftime("%Y-%m-%d %I:%M %p"),
            "date": forecast_dt.strftime("%Y-%m-%d"),
            "label": forecast_dt.strftime("%I %p").lstrip("0"),
            "temp_f": round(temperatures[index]) if index < len(temperatures) else None,
            "condition": _weather_description(weather_codes[index] if index < len(weather_codes) else None),
            "precip_prob": (
                round(precip_probs[index])
                if index < len(precip_probs) and precip_probs[index] is not None else None
            ),
        })
        if len(entries) >= limit:
            break
    return entries


def _build_daily_forecast_entries(data, *, limit: int = 3):
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    weather_codes = daily.get("weather_code", [])

    entries = []
    for index, date_value in enumerate(dates[:limit]):
        try:
            forecast_dt = datetime.fromisoformat(date_value)
        except Exception:
            continue
        entries.append({
            "date": date_value,
            "day_name": forecast_dt.strftime("%A"),
            "high_f": round(highs[index]) if index < len(highs) else None,
            "low_f": round(lows[index]) if index < len(lows) else None,
            "condition": _weather_description(weather_codes[index] if index < len(weather_codes) else None),
        })
    return entries


def get_weather_context(location_coordinates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    coords = location_coordinates or DEFAULT_LOCATION
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={coords['lat']}&longitude={coords['lon']}"
        "&current=temperature_2m,weather_code"
        "&hourly=temperature_2m,weather_code,precipitation_probability"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min"
        "&temperature_unit=fahrenheit&timezone=America%2FNew_York&forecast_days=4"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        now_est = datetime.now(EASTERN_TZ)
        return {
            "current_summary": _format_current_weather_summary(current["temperature_2m"], current["weather_code"]),
            "hourly_forecast": _build_hourly_forecast_entries(data, now_est=now_est),
            "daily_forecast": _build_daily_forecast_entries(data),
        }
    except Exception:
        return {"current_summary": "Weather data unavailable.", "hourly_forecast": [], "daily_forecast": []}


def load_calendar_data() -> Dict[str, Any]:
    empty = {"weeks": [], "sourceEvents": [], "locations": []}
    if not CALENDAR_PATH.exists():
        return empty
    try:
        return json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return empty


def get_calendar_data_by_date(date_str: str, calendar_data: Dict[str, Any]) -> str:
    try:
        target_day = datetime.strptime(date_str, "%Y-%m-%d").day
        events_for_day = []
        for week in calendar_data.get("weeks", []):
            for day in week.get("daysOfWeek", []):
                if day.get("dayOfMonth") == target_day:
                    events_for_day = day.get("events", [])
                    break
            if events_for_day:
                break

        schedule_items = []
        for event in events_for_day:
            source_event = next(
                (e for e in calendar_data.get("sourceEvents", []) if e.get("eventId") == event.get("sourceEventId")),
                None,
            )
            if not source_event:
                continue
            defaults = source_event.get("defaults", {})
            time = event.get("startTime", defaults.get("startTime", ""))
            time = time.replace("a.m.", "AM").replace("p.m.", "PM").replace("a.m", "AM").replace("p.m", "PM")
            location_id = event.get("locationId", defaults.get("locationId", ""))
            location = next(
                (loc.get("name", "") for loc in calendar_data.get("locations", []) if loc.get("locationId") == location_id),
                "",
            )
            schedule_items.append({"time": time, "activity": source_event.get("name", ""), "location": location})

        return json.dumps({"date": date_str, "schedule": schedule_items}, indent=2)
    except Exception:
        return json.dumps({"date": date_str, "schedule": []}, indent=2)


def get_schedule_for_today() -> str:
    date_str = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d")
    return get_calendar_data_by_date(date_str, load_calendar_data())


def get_user_profile(external_id: str) -> Optional[Dict[str, Any]]:
    path = PROFILES_DIR / f"{external_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_personal_profile_context(external_id: str, speaker_profile_key: Optional[str] = None) -> str:
    profile_data = get_user_profile(external_id) or {}
    if speaker_profile_key and isinstance(profile_data, dict):
        speaker_profile = (profile_data.get("speaker_profiles") or {}).get(speaker_profile_key)
        if isinstance(speaker_profile, dict):
            profile_data = speaker_profile
    return json.dumps(profile_data, indent=2)


def load_system_capabilities_text() -> str:
    if not SYSTEM_CAPABILITIES_PATH.exists():
        return ""
    try:
        return SYSTEM_CAPABILITIES_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def build_conversation_prompt_context(
    external_id: str,
    *,
    speaker_profile_key: Optional[str] = None,
    location_coordinates: Optional[Dict[str, float]] = None,
    steps_provider: Optional[Callable[[str], Optional[int]]] = None,
) -> Dict[str, Any]:
    """Same shape as recover/prompt_context/conversation.py's build_conversation_prompt_context."""
    flags = get_prompt_feature_flags()
    context: Dict[str, Any] = {}

    if flags["include_temporal_context"]:
        context.update(get_temporal_context())

    if flags["include_weather"]:
        weather = get_weather_context(location_coordinates)
        context["weather_info"] = weather["current_summary"]
        context["weather_hourly"] = weather["hourly_forecast"]
        context["weather_daily_forecast"] = weather["daily_forecast"]

    if flags["include_personal_profile"]:
        context["personal_data_profile"] = get_personal_profile_context(external_id, speaker_profile_key)

    if flags["include_steps"]:
        total_steps = steps_provider(external_id) if steps_provider else None
        context["steps_info"] = steps_info_for_conversation(total_steps)

    if flags["include_quote_of_day"]:
        context["positive_quote_of_day"] = quote_of_the_day()

    context["system_capabilities"] = load_system_capabilities_text()
    return context

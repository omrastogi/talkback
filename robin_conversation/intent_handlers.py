"""Non-conversational intent replies. Ported from recover/intent_handlers/
{capabilities.py,delete_message.py,end_conversation.py,weather.py,schedule.py}.

reminder.py and affirmation.py from the original aren't ported here -- reminders call out
to a separate microservice this package doesn't have (see engine.py's pluggable
`reminder_handler` hook instead), and the only *live* affirmation gate (delete-message
confirmation) is small enough to live directly in engine.py alongside its in-memory
history handling.
"""
import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .llm import chat_model_id, inference as llm_inference
from .prompt_context import (
    get_calendar_data_by_date,
    get_weather_context,
    load_calendar_data,
    load_system_capabilities_text,
)


def build_delete_message_confirmation() -> str:
    return "Are you sure you want to delete the last message?"


END_CONVERSATION_MESSAGES = (
    "Goodbye. I'll talk to you later.",
    "Talk to you later. Goodbye.",
    "Goodbye for now. Talk soon.",
)


def build_end_conversation_message() -> str:
    return random.choice(END_CONVERSATION_MESSAGES)


def build_capabilities_message() -> str:
    capabilities_text = load_system_capabilities_text()
    if not capabilities_text.strip():
        return (
            "I can chat with you, share weather and schedule updates, "
            "and give gentle wellness encouragement."
        )

    prompt = f"""Summarize these system capabilities into one short spoken sentence.
Rules:
- Maximum 16 words.
- Plain text only.
- Start with "I can help with".
- Mention only core supported capabilities, not boundaries.

Capabilities:
{capabilities_text}
"""
    try:
        response = llm_inference(
            [{"role": "user", "content": prompt}], model=chat_model_id(), operation_name="capabilities_response"
        )
        message = (response or "").strip()
        if message:
            return message
    except Exception:
        pass

    return "I can help with conversation, reminders, weather updates, schedule guidance, and gentle encouragement."


def build_weather_message(*, user_message: str, location_coordinates: Optional[Dict[str, float]] = None) -> str:
    context = get_weather_context(location_coordinates)
    current_weather = context.get("current_summary", "Weather data unavailable.")
    hourly_weather = context.get("hourly_forecast", [])
    daily_weather = context.get("daily_forecast", [])

    prompt = f"""You are Robin, a friendly and concise voice assistant.
Answer the user's weather question using only the provided weather data.

User question:
{user_message}

Weather context:
- Current weather: {current_weather}
- Hourly weather forecast: {hourly_weather}
- Daily weather forecast: {daily_weather}

Rules:
- Keep the answer short, usually one or two brief sentences.
- Answer directly. Do not say things like "Let me check."
- For questions about now, use current weather.
- For questions about later today, this afternoon, this evening, or tonight, use hourly forecast.
- For questions about tomorrow or upcoming days, use daily forecast.
- Do not mention schedule or unrelated information.
- Do not invent details not present in the weather context.
- Use plain text only.
- Output only the final spoken answer.
"""
    try:
        response = llm_inference(
            [{"role": "user", "content": prompt}], model=chat_model_id(), operation_name="weather_response"
        )
        message = (response or "").strip()
        if message:
            return message
    except Exception:
        pass

    return current_weather


def build_schedule_message(*, user_message: str) -> str:
    eastern = ZoneInfo("America/New_York")
    now_est = datetime.now(eastern)
    today_str = now_est.strftime("%Y-%m-%d")
    tomorrow_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
    calendar = load_calendar_data()
    today_schedule = get_calendar_data_by_date(today_str, calendar)
    tomorrow_schedule = get_calendar_data_by_date(tomorrow_str, calendar)

    prompt = f"""You are Robin, a friendly and concise voice assistant.
Answer the user's schedule or calendar question using only the provided schedule data.

User question:
{user_message}

Schedule context:
- Today's schedule: {today_schedule}
- Tomorrow's schedule: {tomorrow_schedule}

Rules:
- Keep the answer short, usually one or two brief sentences.
- Answer directly. Do not say things like "Let me check."
- Mention only one to three key items first, not the whole schedule.
- If there is more to share, ask whether the user wants to hear more.
- If the user asks about tomorrow, use tomorrow's schedule. Otherwise use today's schedule unless the question clearly asks for another day.
- Do not mention weather or unrelated information.
- Do not invent activities not present in the schedule data.
- Use plain text only.
- Output only the final spoken answer.
"""
    try:
        response = llm_inference(
            [{"role": "user", "content": prompt}], model=chat_model_id(), operation_name="schedule_response"
        )
        message = (response or "").strip()
        if message:
            return message
    except Exception:
        pass

    return "You have a few things planned today. Would you like to hear more?"

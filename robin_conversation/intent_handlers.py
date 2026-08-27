"""Non-conversational intent replies. Ported from recover/intent_handlers/
{capabilities.py,delete_message.py,end_conversation.py,weather.py,schedule.py}.

reminder.py and affirmation.py from the original aren't ported here -- reminders call out
to a separate microservice this package doesn't have (see engine.py's pluggable
`reminder_handler` hook instead), and the only *live* affirmation gate (delete-message
confirmation) is small enough to live directly in engine.py alongside its in-memory
history handling.
"""
import json
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


# --- Timers and alarms ------------------------------------------------------------------
# These two don't produce a reply from a knowledge source like weather/schedule do: they
# produce a CLIENT ACTION. The device (RobinVoice D5) owns the actual countdown -- it fires
# the tablet's stock Clock app via AlarmClock intents -- so all this does is turn the spoken
# request into the control frame the client validates, plus the spoken confirmation. Nothing
# is scheduled server-side; if the socket drops, no timer is left orphaned here.
#
# Frame contract (agreed with the client; it re-validates and drops malformed args):
#   {"type": "set_timer", "seconds": <positive int>, "label": <str, optional>}
#   {"type": "set_alarm", "hour": <0-23>, "minutes": <0-59>, "label": <str, optional>}

MAX_TIMER_SECONDS = 24 * 3600      # a day; anything longer is a misparse, not a request
# Labels the model likes to echo back from the request itself. They are not what the timer is
# FOR, and reading them aloud gives "Timer set for two minutes for timer."
_GENERIC_LABELS = {"timer", "alarm", "timers", "alarms", "countdown", "reminder"}
# Phrases our own confirmations use -- the only evidence that something is actually running.
_SET_CONFIRMED_MARKERS = ("timer set", "alarm set", "i've set")


def _spoken_duration(seconds: int) -> str:
    """120 -> 'two minutes'. Spoken-style, because this text goes to TTS, not a screen."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen",
             20: "twenty", 30: "thirty", 45: "forty-five"}

    def unit(n, name):
        return f"{words.get(n, n)} {name}{'' if n == 1 else 's'}"

    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    parts = [unit(n, name) for n, name in ((hours, "hour"), (minutes, "minute"), (secs, "second")) if n]
    return " and ".join(parts) if parts else "zero seconds"


def _spoken_clock_time(hour: int, minutes: int) -> str:
    """(19, 30) -> '7:30 PM'. 12-hour, because that is how the confirmation is spoken."""
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minutes:02d} {suffix}" if minutes else f"{display_hour} {suffix}"


_TIMER_EXTRACTION_PROMPT = """Extract EVERY timer and alarm the user asked for in this message.

What Robin said immediately before (may be empty): "{previous}"
User message: "{message}"

Return a JSON object only, with one key "items": a list. One entry per timer or alarm the
user asked for -- a single request yields a list of one; "3 minutes and another for 7" yields
two. Each entry has:
- "kind": "timer" for a relative countdown, "alarm" for a clock time.
- "seconds": total seconds as an integer (timer only).
- "hour": hour in 24-hour form, 0-23, and "minutes": 0-59 (alarm only).
- "label": what it is for, a short lowercase noun phrase, or null if not stated.

Rules:
- Resolve spoken numbers ("two minutes" -> 120, "half an hour" -> 1800).
- CRITICAL: only fill in seconds/hour/minutes if the user ACTUALLY STATED a duration or a
  clock time. If they did not state one, omit the field entirely. Never guess a "typical"
  duration from the label -- "a timer for eggs" states no duration, so omit seconds. Setting
  a silently-wrong timer is far worse than asking how long.
- For an alarm with no am/pm stated, choose the interpretation a person most likely means
  (7 -> 07:00, "seven at night" -> 19:00).
- "label" is what the item is FOR, in the user's own words. Take it from phrasings like
  "a timer for the pasta" -> "pasta", "call it tea" -> "tea", "my gym alarm" -> "gym".
- Never invent a label the user did not say, and never use the words "timer" or "alarm"
  themselves as a label: "set a timer for two minutes" has NO label.
- If the user gives a NAMING INSTRUCTION covering several items -- "name them A, B, C",
  "name them alphabetically", "call them one, two, three" -- apply it across the items in
  the order the user listed them. The instruction usually comes once, at the end, and
  applies to EVERY item in the request. Speech recognition often mangles it ("name at name
  it name them"); follow the intent anyway.
  Example: "timers for 3, 7 and 8 minutes, name them alphabetically" ->
    {{"items": [{{"kind":"timer","seconds":180,"label":"A"}},
               {{"kind":"timer","seconds":420,"label":"B"}},
               {{"kind":"timer","seconds":480,"label":"C"}}]}}
"""


def _validate_item(item: Any) -> Optional[Dict[str, Any]]:
    """One extracted entry -> a validated client frame, or None. Strict on purpose: a
    malformed frame the client drops is recoverable, a well-formed WRONG one is not."""
    if not isinstance(item, dict):
        return None
    label = item.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else None
    if label and label.lower() in _GENERIC_LABELS:
        label = None

    if item.get("kind") == "timer":
        try:
            seconds = int(item["seconds"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 < seconds <= MAX_TIMER_SECONDS:
            return None
        frame = {"type": "set_timer", "seconds": seconds}
    elif item.get("kind") == "alarm":
        try:
            hour, minutes = int(item["hour"]), int(item.get("minutes") or 0)
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= hour <= 23 and 0 <= minutes <= 59):
            return None
        frame = {"type": "set_alarm", "hour": hour, "minutes": minutes}
    else:
        return None

    if label:
        frame["label"] = label
    return frame


def _extract_timer_args(user_message: str, previous: str = "") -> tuple:
    """One LLM call -> (list of validated client frames, is_correction)."""
    raw = llm_inference(
        [{"role": "user", "content": _TIMER_EXTRACTION_PROMPT.format(
            message=user_message, previous=previous)}],
        model=chat_model_id(),
        response_format={"type": "json_object"},
        operation_name="timer_extraction",
    )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return [], False
    if not isinstance(parsed, dict):
        return [], False
    items = parsed.get("items")
    if not isinstance(items, list):
        return [], bool(parsed.get("is_correction"))
    return ([f for f in (_validate_item(i) for i in items) if f],
            bool(parsed.get("is_correction")))


def _phrase(frame: Dict[str, Any]) -> str:
    label = frame.get("label")
    if frame["type"] == "set_timer":
        dur = _spoken_duration(frame["seconds"])
        return f"{label} for {dur}" if label else f"a timer for {dur}"
    when = _spoken_clock_time(frame["hour"], frame["minutes"])
    return f"{label} at {when}" if label else f"an alarm at {when}"


def build_timer_action(*, user_message: str, intent: str = "set_timer",
                       previous_reply: str = "") -> tuple:
    """(spoken reply, list of client frames). One frame PER requested timer/alarm -- the
    device handles concurrent timers, so "3 minutes and another for 7" must not silently
    drop the second one.

    An empty list means no duration/time was stated ("can you set a timer?", "a timer for
    eggs"). That answer AFFIRMS the capability and asks for the missing piece rather than
    guessing: the old wording sounded like Robin had misheard, and a guessed duration is the
    worst failure this feature has."""
    frames, is_correction = _extract_timer_args(user_message, previous_reply)
    # Only a reply that actually CONFIRMED setting something can be corrected. Without this,
    # a clarifying question ("how long would you like it for?") followed by "no, thirty
    # seconds" was read as a correction and claimed an earlier timer that never existed.
    if is_correction and not any(marker in (previous_reply or "").lower()
                                 for marker in _SET_CONFIRMED_MARKERS):
        is_correction = False

    if is_correction:
        # A running timer cannot be cancelled (see build_show_action). Setting the corrected
        # one and staying silent would leave the WRONG timer running too -- which is what
        # happened live: "no, only thirty seconds" produced 90s AND 30s. So set the new one
        # if we got it, then say plainly the old one is still running and open the screen.
        if frames:
            phrases = [_phrase(f) for f in frames]
            joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}" if len(phrases) > 1 else phrases[0]
            return (f"I've set {joined}, but I can't cancel the earlier one — "
                    "here's your timer screen so you can remove it."), frames + [{"type": "show_timers"}]
        return ("I can't change a timer once it's running — here's your timer screen "
                "so you can remove it and we'll set a new one."), [{"type": "show_timers"}]

    if not frames:
        if intent == "set_alarm":
            return "Yes, I can set an alarm for you. What time should I set it for?", []
        return "Yes, I can set a timer for you. How long would you like it for?", []

    if len(frames) == 1:
        frame = frames[0]
        for_label = f" for {frame['label']}" if frame.get("label") else ""
        if frame["type"] == "set_timer":
            return f"Timer set for {_spoken_duration(frame['seconds'])}{for_label}.", frames
        return f"Alarm set for {_spoken_clock_time(frame['hour'], frame['minutes'])}{for_label}.", frames

    phrases = [_phrase(f) for f in frames]
    joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"I've set {joined}.", frames


def build_show_action(*, intent: str) -> tuple:
    """(spoken reply, [frame]) for show_timers / show_alarms.

    Deliberately NOT a cancel: Samsung Clock's public AlarmClock intents can create but never
    cancel (DISMISS_TIMER is ignored for running timers, DISMISS_ALARM by label is ignored
    outright -- verified on the Tab A9+), so a cancel frame would be a lie. Opening the Clock
    screen is the honest capability, and the reply says plainly that Robin cannot do it
    itself. Unlike set_*, this frame DOES take over the screen -- that is the point."""
    # Lead with the ACTION, not the limitation. The old wording opened with "I can't...",
    # which users heard as "this feature does not exist" even though the screen was opening
    # in front of them. Still honest -- it never claims to have read or cancelled anything.
    if intent == "show_alarms":
        return ("Opening your alarm screen now — you can see and change your alarms there."
                ), [{"type": "show_alarms"}]
    return ("Opening your timer screen now — you can see and stop your timers there."
            ), [{"type": "show_timers"}]

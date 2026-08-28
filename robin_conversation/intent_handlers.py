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

# Repeating alarms. The client is strict on purpose: a present-but-invalid days list rejects
# the whole frame, because a repeating alarm silently downgraded to a one-off is a missed
# wake-up. So we normalise here and refuse to send anything we could not fully resolve.
DAY_ORDER = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
DAY_FULL = {"sun": "Sunday", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
            "thu": "Thursday", "fri": "Friday", "sat": "Saturday"}
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri")
_WEEKEND = ("sat", "sun")
_DAY_ALIASES = {}
for _abbr, _full in DAY_FULL.items():
    _DAY_ALIASES[_abbr] = _abbr
    _DAY_ALIASES[_full.lower()] = _abbr
    _DAY_ALIASES[_full.lower() + "s"] = _abbr          # "mondays"


def normalize_days(raw):
    """Any day list -> canonical week-ordered abbreviations, or None if ANY entry is
    unrecognised. None means 'do not send a days field' -- never a silent partial list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out = set()
    for item in raw:
        if not isinstance(item, str):
            return None
        key = item.strip().lower()
        if key in ("weekday", "weekdays"):
            out.update(_WEEKDAYS)
        elif key in ("weekend", "weekends"):
            out.update(_WEEKEND)
        elif key in ("daily", "everyday", "every day", "all"):
            out.update(DAY_ORDER)
        elif key in _DAY_ALIASES:
            out.add(_DAY_ALIASES[key])
        else:
            return None                                # unrecognised -> refuse the whole list
    return [d for d in DAY_ORDER if d in out] or None


def spoken_days(days):
    """['mon','wed','fri'] -> 'Monday, Wednesday and Friday'. Shorthands read naturally."""
    if not days:
        return ""
    dset = set(days)
    if dset == set(DAY_ORDER):
        return "every day"
    if dset == set(_WEEKDAYS):
        return "every weekday"
    if dset == set(_WEEKEND):
        return "weekends"
    names = [DAY_FULL[d] for d in DAY_ORDER if d in dset]
    if len(names) == 1:
        return f"every {names[0]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


MAX_TIMER_SECONDS = 24 * 3600      # a day; anything longer is a misparse, not a request
# Labels the model likes to echo back from the request itself. They are not what the timer is
# FOR, and reading them aloud gives "Timer set for two minutes for timer."
_GENERIC_LABELS = {"timer", "alarm", "timers", "alarms", "countdown", "reminder",
                   # recurrence words the model likes to mistake for a purpose
                   "weekday", "weekdays", "weekend", "weekends", "daily", "weekly",
                   "every day", "every week", "monday", "tuesday", "wednesday",
                   "thursday", "friday", "saturday", "sunday"}
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

The user's earlier request in this same exchange (may be empty): "{prior}"
What Robin said immediately before (may be empty): "{previous}"
User message: "{message}"

If the earlier request is present, Robin asked for ONE missing detail and the user message is
the answer. Combine them into a single item: the earlier request supplies everything except
the detail just given (days, label, which kind), the user message supplies that detail.
"Alarm for Monday, Tuesday and Wednesday" + "at eight thirty" is ONE alarm at 08:30 with
days ["mon","tue","wed"] -- never a one-off that drops the days.

Return a JSON object only, with one key "items": a list. One entry per timer or alarm the
user asked for -- a single request yields a list of one; "3 minutes and another for 7" yields
two. Each entry has:
- "kind": "timer" for a relative countdown, "alarm" for a clock time.
- "seconds": total seconds as an integer (timer only).
- "hour": hour in 24-hour form, 0-23, and "minutes": 0-59 (alarm only).
- "label": what it is for, a short lowercase noun phrase, or null if not stated.
- "days": for a REPEATING alarm, the days it should fire, as lowercase 3-letter names from
  sun, mon, tue, wed, thu, fri, sat. Omit entirely for a one-off.
  Resolve shorthands yourself: "every weekday" -> ["mon","tue","wed","thu","fri"];
  "weekends" -> ["sat","sun"]; "every day"/"daily" -> all seven;
  "Monday, Wednesday and Friday" -> ["mon","wed","fri"]; "every Tuesday" -> ["tue"].
- "recurrence_requested": true if the user asked for it to repeat at all; omit otherwise.

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
- Recurrence words are NEVER labels. "every weekday", "on Mondays", "daily", "every week"
  describe WHEN it repeats, not what it is for -- set recurrence_requested instead and leave
  label null.
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
        days = normalize_days(item.get("days"))
        if days:
            frame["days"] = days
        elif item.get("days") or item.get("recurrence_requested"):
            # They asked for a repeat but we could not resolve the days. Sending a one-off
            # would be a missed wake-up dressed up as success, so refuse the item and let
            # build_timer_action ask which days.
            return None
    else:
        return None

    if label:
        frame["label"] = label
    return frame


def _extract_timer_args(user_message: str, previous: str = "", prior: str = "") -> tuple:
    """One LLM call -> (frames, is_correction, recurrence_requested, missing_piece)."""
    raw = llm_inference(
        [{"role": "user", "content": _TIMER_EXTRACTION_PROMPT.format(
            message=user_message, previous=previous, prior=prior)}],
        model=chat_model_id(),
        response_format={"type": "json_object"},
        operation_name="timer_extraction",
    )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return [], False, False, None
    if not isinstance(parsed, dict):
        return [], False, False, None
    items_raw = parsed.get("items")
    items = items_raw
    # The model puts this flag at the top level OR on the item, depending on the phrasing;
    # accept either rather than depending on it choosing one.
    recurrence = bool(parsed.get("recurrence_requested")) or any(
        isinstance(i, dict) and i.get("recurrence_requested")
        for i in (items if isinstance(items, list) else []))
    if not isinstance(items, list):
        return [], bool(parsed.get("is_correction")), recurrence, None
    frames = [f for f in (_validate_item(i) for i in items) if f]
    # Identical frames in one turn are never intentional -- they are the model splitting a
    # single repeating request into one item per day. Collapse them.
    deduped, seen = [], set()
    for f in frames:
        key = json.dumps(f, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    missing = None
    if not deduped and isinstance(items_raw, list) and items_raw:
        first = items_raw[0] if isinstance(items_raw[0], dict) else {}
        if first.get("kind") == "alarm":
            has_time = first.get("hour") is not None
            has_days = normalize_days(first.get("days")) is not None
            # Days given but no clock time -> asking "which days?" (as it used to) ignores
            # what the user just said. Ask for the missing half.
            missing = "time" if not has_time else ("days" if not has_days else None)
        else:
            missing = "duration"
    return deduped, bool(parsed.get("is_correction")), recurrence, missing


def _phrase(frame: Dict[str, Any]) -> str:
    label = frame.get("label")
    if frame["type"] == "set_timer":
        dur = _spoken_duration(frame["seconds"])
        return f"{label} for {dur}" if label else f"a timer for {dur}"
    when = _spoken_clock_time(frame["hour"], frame["minutes"])
    if frame.get("days"):
        when = f"{when} {spoken_days(frame['days'])}"
    return f"{label} at {when}" if label else f"an alarm at {when}"


def build_timer_action(*, user_message: str, intent: str = "set_timer",
                       previous_reply: str = "", prior_request: str = "") -> tuple:
    """(spoken reply, list of client frames). One frame PER requested timer/alarm -- the
    device handles concurrent timers, so "3 minutes and another for 7" must not silently
    drop the second one.

    An empty list means no duration/time was stated ("can you set a timer?", "a timer for
    eggs"). That answer AFFIRMS the capability and asks for the missing piece rather than
    guessing: the old wording sounded like Robin had misheard, and a guessed duration is the
    worst failure this feature has."""
    frames, is_correction, recurrence, missing = _extract_timer_args(
        user_message, previous_reply, prior_request)
    if not frames and missing == "time":
        return "What time should that alarm go off?", []
    if not frames and missing == "days":
        return "Which days should that alarm go off on?", []
    # Only a reply that actually CONFIRMED setting something can be corrected. Without this,
    # a clarifying question ("how long would you like it for?") followed by "no, thirty
    # seconds" was read as a correction and claimed an earlier timer that never existed.
    if is_correction and not any(marker in (previous_reply or "").lower()
                                 for marker in _SET_CONFIRMED_MARKERS):
        is_correction = False

    if is_correction:
        # Setting the corrected one and staying silent leaves the WRONG timer running too --
        # live, "no, only thirty seconds" produced 90s AND 30s. Cancelling is now possible
        # (the device owns the clock), but this turn does not know the old timer's id, so
        # say it is still there and invite the cancel, which is one short sentence away.
        if frames:
            phrases = [_phrase(f) for f in frames]
            joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}" if len(phrases) > 1 else phrases[0]
            return (f"I've set {joined}. The earlier one is still running — "
                    "say cancel it and I will."), frames
        return ("Tell me the new time and I'll set it — the earlier one is still running, "
                "so say cancel it if you want it gone."), []

    if not frames:
        if intent == "set_alarm":
            return "Yes, I can set an alarm for you. What time should I set it for?", []
        return "Yes, I can set a timer for you. How long would you like it for?", []

    if len(frames) == 1:
        frame = frames[0]
        for_label = f" for {frame['label']}" if frame.get("label") else ""
        if frame["type"] == "set_timer":
            return f"Timer set for {_spoken_duration(frame['seconds'])}{for_label}.", frames
        when = _spoken_clock_time(frame["hour"], frame["minutes"])
        on_days = f" {spoken_days(frame['days'])}" if frame.get("days") else ""
        return f"Alarm set for {when}{on_days}{for_label}.", frames

    phrases = [_phrase(f) for f in frames]
    joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"I've set {joined}.", frames

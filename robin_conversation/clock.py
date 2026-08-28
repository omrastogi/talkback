"""Reasoning over the client-owned clock (contract v2).

The device is the clock: it owns timers, alarms, ringing, and persistence, and pushes its
whole state here as `clock_state` frames. This module never schedules anything -- it only
(a) renders that state for the LLM so Robin can ANSWER from ground truth, (b) resolves a
spoken cancel into a match spec the client can act on, and (c) speaks the result.

Design rule, learned the hard way today: never state something we have not verified. Every
answer here is derived from the last clock_state the client sent, and a cancel that matches
nothing is reported as such rather than guessed at. `id` is preferred for matching because
the client already assigns stable ids -- resolving once, here, avoids two resolvers
disagreeing (see the label/seconds ambiguity noted in the contract discussion).
"""
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .intent_handlers import (_spoken_clock_time, _spoken_duration, normalize_days,
                              spoken_days)
from .llm import chat_model_id, inference as llm_inference


# Day names / shorthands, used to decide whether the user actually asked to change the days.
_DAY_WORDS = re.compile(r"\b(mon|tues|wednes|thurs|fri|satur|sun)day s?\b".replace(" ", "")
                        + r"|\bweekdays?\b|\bweekends?\b|\bdaily\b|\bevery (?:day|week)\b",
                        re.IGNORECASE)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _timers(state) -> List[Dict[str, Any]]:
    return list((state or {}).get("timers") or [])


def _alarms(state) -> List[Dict[str, Any]]:
    return list((state or {}).get("alarms") or [])


def ringing(state) -> Optional[Dict[str, Any]]:
    return (state or {}).get("ringing")


def is_empty(state) -> bool:
    return not _timers(state) and not _alarms(state)


def describe_timer(t: Dict[str, Any], now_ms: Optional[int] = None) -> str:
    """'the tea timer, four minutes left' -- always REMAINING time, never the original
    duration, because that is what someone cooking actually asked for."""
    now_ms = now_ms if now_ms is not None else _now_ms()
    label = t.get("label")
    remaining = max(0, int(round(((t.get("ends_at") or 0) - now_ms) / 1000)))
    head = f"the {label} timer" if label else "a timer"
    return f"{head}, {_spoken_duration(remaining)} left" if remaining else f"{head}, finished"


def describe_alarm(a: Dict[str, Any]) -> str:
    label = a.get("label")
    when = _spoken_clock_time(int(a.get("hour", 0)), int(a.get("minutes") or 0))
    if a.get("days"):
        when = f"{when} {spoken_days(a['days'])}"
    return f"the {label} alarm at {when}" if label else f"the alarm at {when}"


_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten"}


def name_timer(t: Dict[str, Any]) -> str:
    """Name only, no remaining time -- 'Cancelled the tea timer, four minutes left' reads as
    if the timer is somehow still running."""
    label = t.get("label")
    return f"the {label} timer" if label else "the timer"


def format_clock_context(state) -> str:
    """The state as plain lines for the LLM prompt. Ids are included so the model can refer
    to a specific item, but they are never spoken."""
    if state is None:
        return "Clock state unknown (the device has not reported yet)."
    now_ms = _now_ms()
    lines = []
    for t in _timers(state):
        remaining = max(0, int(round(((t.get("ends_at") or 0) - now_ms) / 1000)))
        lines.append(f"- timer id={t.get('id')} label={t.get('label') or 'none'} "
                     f"remaining_seconds={remaining} total_seconds={t.get('seconds')}")
    for a in _alarms(state):
        days = a.get("days")
        lines.append(f"- alarm id={a.get('id')} label={a.get('label') or 'none'} "
                     f"time={int(a.get('hour', 0)):02d}:{int(a.get('minutes') or 0):02d} "
                     f"repeats={spoken_days(days) if days else 'one-off'}")
    r = ringing(state)
    if r:
        lines.append(f"- RINGING NOW: {json.dumps(r)}")
    if not lines:
        return "The user currently has no timers and no alarms set."
    return "The user's current timers and alarms (this is ground truth, answer from it):\n" + "\n".join(lines)


def _join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


_QUERY_PROMPT = """Answer the user's question about their timers and alarms.

{context}

User question: "{message}"

Rules:
- Answer ONLY from the state above. Never invent a timer or alarm that is not listed.
- If an alarm repeats, say which days ("half past eight on Monday, Wednesday and Friday").
- If the state lists none, say plainly that they have none set.
- One or two short spoken sentences. Say times and durations naturally ("four minutes left",
  "half past eight"), never as raw numbers of seconds and never as an id.
- Do not offer to cancel anything unless the user asked.
"""


def build_clock_query_reply(*, user_message: str, state) -> str:
    """Answer 'when's my next alarm?' / 'how long left?' from the device's own state."""
    if state is None:
        return "I can't see your timers just now — let me know if you'd like me to set one."
    if is_empty(state) and not ringing(state):
        return "You don't have any timers or alarms set right now."
    return (llm_inference(
        [{"role": "user", "content": _QUERY_PROMPT.format(
            context=format_clock_context(state), message=user_message)}],
        model=chat_model_id(), operation_name="clock_query") or "").strip()


_CANCEL_PROMPT = """Work out exactly which timer or alarm the user wants cancelled.

{context}

User message: "{message}"

Return a JSON object only:
- "kind": "timer" or "alarm".
- "id": the id of the single item to cancel, copied EXACTLY from the state above.
- "all": true instead of "id" if the user wants every timer (or every alarm) cancelled.
- "id": null if nothing in the state matches what they asked for.

Rules:
- Only ever return an id that appears verbatim in the state above. Never invent one.
- If the user is ambiguous and several items match equally, return "id": null.
- "cancel the tea timer" -> the timer whose label is tea. "delete the 8:30 alarm" -> the
  alarm at 20:30 or 08:30, whichever is listed.
- If the user is asking for ALL of them ("all my timers", "every alarm", "clear them all"),
  return "all": true -- never a single id. Speech recognition mangles these requests, so
  honour the intent even when the sentence is garbled: cancelling ONE when they asked for
  ALL leaves timers running that they believe are off.
"""

# A single-id cancel when the user said "all" leaves the rest running while Robin reports
# success -- the user then gets an unexpected ring. Cheap deterministic backstop for when
# the model picks an id despite an "all"-shaped request (seen live with STT garble:
# "Can't install all my timers").
_ALL_WORDS = re.compile(r"\b(all|every|everything|them all)\b", re.IGNORECASE)


def spoken_count(n: int) -> str:
    return _COUNT_WORDS.get(int(n), str(n))


def all_cancel_reply(count: int, noun: str) -> str:
    """Wording for an all-cancel, from the count the DEVICE reported. English prefers "both"
    for exactly two -- "cancelled all two timers" is understandable but reads like a machine,
    and this text is spoken aloud."""
    count = int(count or 0)
    if count == 0:
        return f"I didn't find any {noun}s to cancel."
    if count == 1:
        return f"Cancelled your {noun}."
    if count == 2:
        return f"Cancelled both {noun}s."
    return f"Cancelled all {spoken_count(count)} {noun}s."


def build_clock_cancel_action(*, user_message: str, state) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """(spoken reply, frames, extra). Resolves the utterance against the device's state and
    emits a cancel keyed by ID -- resolving once, here, rather than having the client
    re-resolve a label and possibly pick a different item. Matching nothing is SAID, never
    guessed.

    `extra` carries what the caller needs to speak from the device's `cancel_result` rather
    than from hope: awaits_cancel_result, the wording to use if the device reports failure,
    and (for an all-cancel) a template whose {count} the device fills in. The returned
    `reply` is only used when the device confirms ok."""
    if state is None:
        return ("I can't see your timers just now, so I don't want to cancel the wrong one. "
                "Give me a moment and try again."), [], {}
    if is_empty(state):
        return "You don't have any timers or alarms set right now.", [], {}

    raw = llm_inference(
        [{"role": "user", "content": _CANCEL_PROMPT.format(
            context=format_clock_context(state), message=user_message)}],
        model=chat_model_id(), response_format={"type": "json_object"},
        operation_name="clock_cancel_resolve")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    kind = parsed.get("kind")
    if not parsed.get("all") and _ALL_WORDS.search(user_message or ""):
        parsed["all"] = True
    items = _alarms(state) if kind == "alarm" else _timers(state)
    frame_type = "cancel_alarm" if kind == "alarm" else "cancel_timer"
    noun = "alarm" if kind == "alarm" else "timer"

    if parsed.get("all"):
        if not items:
            # 3-tuple: engine.py unpacks (reply, frames, extra) -- returning two crashed the
            # turn with ValueError on "cancel all my alarms" when only timers existed.
            return f"You don't have any {noun}s set right now.", [], {}
        return (all_cancel_reply(len(items), noun),      # provisional; the device's count wins
                [{"type": frame_type, "match": {"all": True}}],
                {"awaits_cancel_result": True,
                 # The spoken count comes from the device's cancel_result, not our snapshot.
                 "cancel_all_noun": noun,
                 "reply_on_fail": f"I couldn't cancel those {noun}s — have a look at the screen."})

    wanted = parsed.get("id")
    match = next((i for i in items if str(i.get("id")) == str(wanted)), None)
    if match is None:
        # Never guess. Say what they actually have so the next turn can be precise.
        described = _join([describe_timer(t) for t in _timers(state)]
                          + [describe_alarm(a) for a in _alarms(state)])
        if not described:
            return "You don't have any timers or alarms set right now.", [], {}
        return (f"I'm not sure which one you mean. Right now you have {described}. "
                "Which should I cancel?"), [], {}

    described = describe_alarm(match) if kind == "alarm" else name_timer(match)
    return (f"Cancelled {described}.",
            [{"type": frame_type, "match": {"id": match.get("id")}}],
            {"awaits_cancel_result": True,
             "reply_on_fail": f"I couldn't cancel {described} — it may have already finished."})


def build_stop_ring_action(*, state) -> Tuple[str, List[Dict[str, Any]]]:
    """Silence whatever is ringing. If nothing is, say so rather than sending a no-op."""
    if state is not None and not ringing(state):
        return "Nothing is ringing at the moment.", []
    return "Stopped.", [{"type": "stop_ring"}]


_MODIFY_PROMPT = """The user wants to CHANGE an existing timer or alarm, not delete it.

{context}

User message: "{message}"

Return a JSON object only:
- "kind": "timer" or "alarm".
- "id": the id of the item being changed, copied EXACTLY from the state above, or null if
  nothing matches.
- "hour" / "minutes": the alarm's time AFTER the change (unchanged values if they did not
  mention the time).
- "days": the FULL list of days the alarm should fire on after the change, lowercase 3-letter
  names. Apply the edit to the existing list: dropping Monday from ["mon","tue","fri"] gives
  ["tue","fri"]; adding Sunday to ["mon"] gives ["mon","sun"]. Omit for a one-off alarm.
- "seconds": for a timer, the new total duration in seconds.
- "label": the label after the change, or null.

Rules:
- Only ever return an id that appears verbatim in the state above.
- Carry over every field the user did NOT mention; only change what they asked to change.
- If several items match equally, return "id": null.
"""


def build_clock_modify_action(*, user_message: str, state) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """(spoken reply, frames, extra) for an edit.

    There is no edit frame in the contract, so an edit is cancel-then-recreate. The recreate
    is deliberately deferred until the device confirms the cancel actually applied (see
    `actions_after_ok`): sending both at once would leave the user with TWO alarms whenever
    the cancel failed, which is worse than the edit not happening.
    """
    if state is None or is_empty(state):
        return "You don't have any timers or alarms set right now.", [], {}

    raw = llm_inference(
        [{"role": "user", "content": _MODIFY_PROMPT.format(
            context=format_clock_context(state), message=user_message)}],
        model=chat_model_id(), response_format={"type": "json_object"},
        operation_name="clock_modify_resolve")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    kind = parsed.get("kind")
    items = _alarms(state) if kind == "alarm" else _timers(state)
    match = next((i for i in items if str(i.get("id")) == str(parsed.get("id"))), None)
    if match is None:
        described = _join([describe_timer(t) for t in _timers(state)]
                          + [describe_alarm(a) for a in _alarms(state)])
        return (f"I'm not sure which one you mean. Right now you have {described}. "
                "Which should I change?"), [], {}

    label = parsed.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else None
    if kind == "alarm":
        # dict.get(k, default) returns None when the key EXISTS and is null, which the model
        # emits for fields the user did not mention -- so fall back explicitly.
        raw_hour = parsed.get("hour")
        raw_min = parsed.get("minutes")
        try:
            hour = int(raw_hour if raw_hour is not None else match.get("hour"))
            if raw_min is not None:
                minutes = int(raw_min or 0)
            elif raw_hour is not None and int(raw_hour) != int(match.get("hour") or -1):
                minutes = 0          # "move it to 9" means 9 o'clock, not 9:30
            else:
                minutes = int(match.get("minutes") or 0)
        except (TypeError, ValueError):
            return "I didn't catch the new time — what should it be?", [], {}
        if not (0 <= hour <= 23 and 0 <= minutes <= 59):
            return "I didn't catch the new time — what should it be?", [], {}
        # If the user said nothing about days, the existing ones must survive verbatim.
        # The model dropped a day on "move my 8:30 alarm to 9" -- silently un-setting a
        # wake-up the user never mentioned is exactly the failure this feature must not have.
        if _DAY_WORDS.search(user_message or ""):
            days = normalize_days(parsed.get("days"))
        else:
            days = normalize_days(match.get("days"))
        new = {"type": "set_alarm", "hour": hour, "minutes": minutes}
        if days:
            new["days"] = days
        if label:
            new["label"] = label
        described_new = f"{_spoken_clock_time(hour, minutes)}" + (f" {spoken_days(days)}" if days else "")
        cancel = {"type": "cancel_alarm", "match": {"id": match.get("id")}}
        noun = "alarm"
    else:
        raw_secs = parsed.get("seconds")
        try:
            seconds = int(raw_secs if raw_secs is not None else match.get("seconds"))
        except (TypeError, ValueError):
            return "How long should that timer be now?", [], {}
        if not 0 < seconds <= 24 * 3600:
            return "How long should that timer be now?", [], {}
        new = {"type": "set_timer", "seconds": seconds}
        if label:
            new["label"] = label
        described_new = _spoken_duration(seconds)
        cancel = {"type": "cancel_timer", "match": {"id": match.get("id")}}
        noun = "timer"

    return (f"Changed the {noun} to {described_new}.", [cancel],
            {"awaits_cancel_result": True,
             "actions_after_ok": [new],
             "reply_on_fail": f"I couldn't change that {noun} — it may have already gone off."})

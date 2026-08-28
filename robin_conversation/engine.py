"""Turn-level orchestration: intent routing + the delete-confirmation sentinel gate, tying
llm.py/classifier.py/intent_handlers.py/prompt_context.py together. Ported from
recover/services/conversation/{engine.py,flow.py} with the DB (Patient/Conversation/
ConversationLog rows) replaced by a plain in-memory `history: list[dict]` you own and pass
in on every call -- nothing here persists anything.

Message dict shape: {"role": "user"|"assistant", "content": str,
                      "chain_of_thoughts": str (optional), "require_affirmation": bool (opt),
                      "reminder_active": bool (opt), "redacted": bool (opt),
                      "speaker_display_name": str (opt)}.

Reminders have no built-in implementation -- the original calls out to a separate
microservice this package doesn't have. Pass your own:
    reminder_handler(message: str, user_id: str, history: list[dict]) -> (reply: str, completed: bool)
if you want "remind me to..." requests handled; completed=False keeps routing subsequent
turns straight back to reminder_handler (an ongoing reminder dialog) until it returns True.
Without one, reminder-classified messages just fall through to the normal conversational
reply, same as any other unsupported capability.
"""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .asr_fixes import repair_transcript
from .classifier import classify_conversation_intent, is_affirmation_intent
from .clock import (
    build_clock_cancel_action,
    build_clock_modify_action,
    build_clock_query_reply,
    build_stop_ring_action,
    format_clock_context,
    ringing,
)
from .intent_handlers import (
    build_capabilities_message,
    build_timer_action,
    build_delete_message_confirmation,
    build_end_conversation_message,
    build_schedule_message,
    build_weather_message,
)
from .llm import conversation as llm_conversation
from .prompt_context import build_conversation_prompt_context

# When Robin has just asked "how long?" / "what time?", the user's next turn is the ANSWER --
# but the classifier only sees that turn's text, so a bare "thirty seconds" or "no, make it
# five" lands in `conversation` and the request stalls. These markers let the engine route the
# answer back to the timer flow, the same shape as the delete-confirmation gate below.
# "delete the 8:30 alarm" is a clock request, not a request to delete a chat message.
_CLOCK_WORDS = re.compile(r"\b(alarm|alarms|timer|timers)\b", re.IGNORECASE)

# "make it every weekday" right after "Alarm set for 7 AM." is a modification of what was
# just set, but on its own it looks like small talk and fell through to a refusal.
_MODIFY_OPENERS = re.compile(r"^\s*(no|nope|actually|make it|change it|instead|i meant|i said)\b",
                             re.IGNORECASE)
_SET_CONFIRMED = ("timer set", "alarm set", "i've set")
# Day words right after a confirmed set are almost always about THAT alarm ("make it every
# weekday", "didn't I ask for Monday Tuesday Wednesday"), but on their own they drag the
# classifier into schedule_query and the user gets their calendar read back.
_DAY_WORDS = re.compile(r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|\bweekdays?\b"
                        r"|\bweekends?\b|\bdaily\b|\bevery (?:day|week)\b", re.IGNORECASE)

# These must match the exact questions build_timer_action asks. Keep them in sync: rewording
# a question without updating its marker silently disables the follow-up gate, which is how
# "alarm for Monday, Tuesday, Wednesday" + "at eight thirty" lost its days.
TIMER_PENDING_MARKERS = ("how long would you like it for",
                         "how long should i set it for")
ALARM_PENDING_MARKERS = ("what time should i set it for",
                         "what time should that alarm go off",
                         "which days should that alarm go off")

REMINDER_CHAIN_MARKER = "REMINDER_AGENT"
DELETE_CONFIRMATION_CHAIN_MARKER = "DELETE_CONFIRMATION_PENDING"

ReminderHandler = Callable[[str, str, List[Dict[str, Any]]], Tuple[str, bool]]


def _redact_last_two_before(history: List[Dict[str, Any]], prev_index: int) -> None:
    """Ported from intent_handlers/affirmation.py::redact_last_two_before, operating on the
    in-memory history list instead of ConversationLog rows."""
    one_before = prev_index - 1
    if one_before <= 0:
        return
    for message in history[max(0, one_before - 2):one_before]:
        if not message.get("redacted"):
            message["content"] = "REDACTED"
            message["redacted"] = True


def _resolve_active_reminder(
    history: List[Dict[str, Any]], content: str, user_id: str, reminder_handler: Optional[ReminderHandler]
) -> Optional[str]:
    last_message = history[-1] if history else None
    if not (
        reminder_handler
        and last_message
        and last_message.get("chain_of_thoughts") == REMINDER_CHAIN_MARKER
        and last_message.get("reminder_active")
    ):
        return None

    history.append({"role": "user", "content": content})
    reply, completed = reminder_handler(content, user_id, history)
    history.append({
        "role": "assistant",
        "content": reply,
        "chain_of_thoughts": REMINDER_CHAIN_MARKER,
        "reminder_active": not completed,
    })
    return reply


def _resolve_pending_followup(history: List[Dict[str, Any]], content: str, user_id: str) -> Optional[str]:
    """Ported from services/conversation/flow.py::resolve_pending_followup -- only the
    delete-confirmation gate, the sole one actually wired up in the original app."""
    last_message = history[-1] if history else None
    if not (
        last_message
        and last_message.get("require_affirmation")
        and DELETE_CONFIRMATION_CHAIN_MARKER in (last_message.get("chain_of_thoughts") or "")
    ):
        return None

    if is_affirmation_intent(content, user_id):
        history.append({"role": "user", "content": content})
        _redact_last_two_before(history, len(history) - 2)
        reply = "I have deleted the last message. Is there anything else I can help you with?"
        history.append({"role": "assistant", "content": reply})
        return reply

    # Not a yes. The user has almost always moved on to a real request -- live, "can you
    # delete the alarm for eight thirty please" arrived here and was consumed by a bland
    # "Is there anything else I can help you with?", throwing the request away and costing
    # the user two more turns. Clear the gate and let normal routing handle this turn.
    last_message["require_affirmation"] = False
    return None


def _run_default_conversation_flow(history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    llm_messages = [
        {
            "role": m["role"],
            "content": (
                f"[speaker: {m['speaker_display_name']}] {m['content']}"
                if m["role"] == "user" and m.get("speaker_display_name")
                else m["content"]
            ),
        }
        for m in history
    ]
    reply = llm_conversation(llm_messages, "", context=context)
    history.append({"role": "assistant", "content": reply})
    return reply


def process_turn(
    history: List[Dict[str, Any]],
    content: str,
    *,
    user_id: str = "user",
    context: Optional[Dict[str, Any]] = None,
    location_coordinates: Optional[Dict[str, float]] = None,
    reminder_handler: Optional[ReminderHandler] = None,
    clock_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Route one user turn. Mutates `history` in place (appends the user turn and the
    assistant reply) and returns {"reply": str, "intent": str, "should_end_session": bool},
    plus "client_actions": list[dict] on the timer/alarm intents -- a control frame the caller
    is expected to forward to the device, which owns the actual countdown (see
    intent_handlers.build_timer_action). Absent on every other intent.

    `context` is passed straight to llm.conversation()/build_system_prompt(); if omitted,
    it's auto-built via prompt_context.build_conversation_prompt_context(user_id) -- real
    weather, a bundled dummy schedule/capabilities text, no profile/steps unless you wire
    those up (see prompt_context.py).
    """
    # Repair known speech-recognition mishearings once, up front, so classification,
    # extraction and cancel-resolution all see the same corrected text. "what announced do I
    # have" was being answered with the day's calendar instead of the user's alarms.
    content = repair_transcript(content)

    if context is None:
        context = build_conversation_prompt_context(user_id, location_coordinates=location_coordinates)
    # The device owns the clock and pushes its whole state up; put it in the prompt so an
    # ordinary conversational turn can refer to a running timer without a dedicated intent.
    if clock_state is not None:
        context = dict(context)
        context["clock_state"] = format_clock_context(clock_state)

    reminder_reply = _resolve_active_reminder(history, content, user_id, reminder_handler)
    if reminder_reply is not None:
        return {"reply": reminder_reply, "intent": "reminder", "should_end_session": False}

    followup_reply = _resolve_pending_followup(history, content, user_id)
    if followup_reply is not None:
        return {"reply": followup_reply, "intent": "delete_message", "should_end_session": False}

    intent = classify_conversation_intent(content, user_id)

    if intent == "delete_message" and _CLOCK_WORDS.search(content or ""):
        intent = "clock_cancel"

    # A modification of the thing just set belongs in the timer flow, which knows how to
    # treat it as a correction, not in ordinary conversation.
    _prev = (history[-1]["content"] if history and history[-1].get("role") == "assistant" else "").lower()
    if (intent in ("conversation", "affirmation", "schedule_query")
            and any(marker in _prev for marker in _SET_CONFIRMED)
            and (_MODIFY_OPENERS.search(content or "") or _DAY_WORDS.search(content or ""))):
        intent = "set_alarm" if "alarm set" in _prev else "set_timer"

    # While something is actually ringing, a bare "stop"/"okay" is aimed at the noise. The
    # classifier cannot know that -- only the device state does.
    if ringing(clock_state) and intent in ("end_conversation", "conversation", "affirmation"):
        intent = "stop_ring"

    # Robin asked for the missing duration/time on the previous turn -> treat this turn as the
    # answer, whatever the classifier made of it on its own.
    last_reply = (history[-1]["content"] if history and history[-1]["role"] == "assistant" else "").lower()
    prior_request = ""
    if intent != "end_conversation":
        if any(marker in last_reply for marker in TIMER_PENDING_MARKERS + ALARM_PENDING_MARKERS):
            intent = "set_timer" if any(m in last_reply for m in TIMER_PENDING_MARKERS) else "set_alarm"
            # The answer to "what time?" carries only the time -- the DAYS were in the turn
            # before it. Without this, "alarm for Monday Tuesday Wednesday" + "at eight
            # thirty" produced a one-off 8:30 alarm and silently lost the days.
            for message in reversed(history[:-1]):
                if message.get("role") == "user":
                    prior_request = message.get("content", "")
                    break

    if intent == "reminder" and reminder_handler is not None:
        history.append({"role": "user", "content": content})
        reply, completed = reminder_handler(content, user_id, history)
        history.append({
            "role": "assistant",
            "content": reply,
            "chain_of_thoughts": REMINDER_CHAIN_MARKER,
            "reminder_active": not completed,
        })
        return {"reply": reply, "intent": intent, "should_end_session": False}

    # Matches process_conversation_turn: the user's turn is logged once here, then routed
    # (the reminder branch above logs its own turn instead, same as the original).
    history.append({"role": "user", "content": content})

    if intent == "delete_message":
        reply = build_delete_message_confirmation()
        history.append({
            "role": "assistant",
            "content": reply,
            "chain_of_thoughts": DELETE_CONFIRMATION_CHAIN_MARKER,
            "require_affirmation": True,
        })
        return {"reply": reply, "intent": intent, "should_end_session": False}

    if intent == "stop_ring":
        reply, client_actions = build_stop_ring_action(state=clock_state)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": client_actions}

    if intent == "clock_modify":
        reply, client_actions, extra = build_clock_modify_action(user_message=content, state=clock_state)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": client_actions, **extra}

    if intent == "clock_cancel":
        reply, client_actions, extra = build_clock_cancel_action(user_message=content, state=clock_state)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": client_actions, **extra}

    if intent == "clock_query":
        reply = build_clock_query_reply(user_message=content, state=clock_state)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": []}

    if intent in ("set_timer", "set_alarm"):
        # history[-1] is the user turn just appended; [-2] is Robin's previous line, which is
        # what makes "no, only thirty seconds" recognisable as a correction rather than a
        # second timer.
        previous_reply = history[-2]["content"] if len(history) >= 2 else ""
        reply, client_actions = build_timer_action(user_message=content, intent=intent,
                                                   previous_reply=previous_reply,
                                                   prior_request=prior_request)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": client_actions}

    if intent == "capabilities_query":
        reply = build_capabilities_message()
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False}

    if intent == "weather_query":
        reply = build_weather_message(user_message=content, location_coordinates=location_coordinates)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False}

    if intent == "schedule_query":
        reply = build_schedule_message(user_message=content)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False}

    if intent == "end_conversation":
        reply = build_end_conversation_message()
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": True}

    reply = _run_default_conversation_flow(history, context)
    return {"reply": reply, "intent": intent, "should_end_session": False}

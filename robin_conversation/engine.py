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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .classifier import classify_conversation_intent, is_affirmation_intent
from .clock import (
    build_clock_cancel_action,
    build_clock_query_reply,
    build_stop_ring_action,
    format_clock_context,
    ringing,
)
from .intent_handlers import (
    build_capabilities_message,
    build_show_action,
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
TIMER_PENDING_MARKERS = ("how long would you like it for",)
ALARM_PENDING_MARKERS = ("what time should i set it for",)

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

    history.append({"role": "user", "content": content})
    if is_affirmation_intent(content, user_id):
        _redact_last_two_before(history, len(history) - 2)
        reply = "I have deleted the last message. Is there anything else I can help you with?"
    else:
        reply = "Is there anything else I can help you with?"
    history.append({"role": "assistant", "content": reply})
    return reply


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

    # While something is actually ringing, a bare "stop"/"okay" is aimed at the noise. The
    # classifier cannot know that -- only the device state does.
    if ringing(clock_state) and intent in ("end_conversation", "conversation", "affirmation"):
        intent = "stop_ring"

    # Robin asked for the missing duration/time on the previous turn -> treat this turn as the
    # answer, whatever the classifier made of it on its own.
    last_reply = (history[-1]["content"] if history and history[-1]["role"] == "assistant" else "").lower()
    if intent not in ("show_timers", "show_alarms", "end_conversation"):
        if any(marker in last_reply for marker in TIMER_PENDING_MARKERS):
            intent = "set_timer"
        elif any(marker in last_reply for marker in ALARM_PENDING_MARKERS):
            intent = "set_alarm"

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

    if intent in ("show_timers", "show_alarms"):
        reply, client_actions = build_show_action(intent=intent)
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "intent": intent, "should_end_session": False,
                "client_actions": client_actions}

    if intent in ("set_timer", "set_alarm"):
        # history[-1] is the user turn just appended; [-2] is Robin's previous line, which is
        # what makes "no, only thirty seconds" recognisable as a correction rather than a
        # second timer.
        previous_reply = history[-2]["content"] if len(history) >= 2 else ""
        reply, client_actions = build_timer_action(user_message=content, intent=intent,
                                                   previous_reply=previous_reply)
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

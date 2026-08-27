"""Robin's intent router. Extracted from recover/intent/classifier.py with the Flask
app/recover-config coupling removed -- the only dependency is this package's own
llm.inference(). Logic (intent definitions, precedence, few-shots, prompt) is unchanged.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .llm import inference as llm_inference, intent_model_id

logger = logging.getLogger(__name__)

INTENT_FLAGS_PATH = Path(__file__).with_name("intent_feature_flags.json")
DEFAULT_PRECEDENCE = [
    "end_conversation",
    "show_timers",
    "show_alarms",
    "set_timer",
    "set_alarm",
    "reminder",
    "delete_message",
    "weather_query",
    "schedule_query",
    "capabilities_query",
    "conversation",
    "affirmation",
]

INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "affirmation": {
        "description": (
            "User gives an affirmative response (for example yes, yup, sure, "
            "okay) in response to a confirmation-style prompt."
        ),
        "few_shots": [
            {"input": "yes", "output": True},
            {"input": "i'm here", "output": True},
            {"input": "no", "output": False},
            {"input": "i do not know", "output": False},
            {"input": "", "output": False},
        ],
    },
    "show_timers": {
        "description": (
            "User wants to see, check, list, count, change, stop, cancel, or delete a TIMER "
            "(or all timers) that already exists -- including simply asking what timers are "
            "running. Creating a new timer is set_timer, not this."
        ),
        "few_shots": [
            {"input": "cancel my timer", "output": True},
            {"input": "remove all timers", "output": True},
            {"input": "stop the pasta timer", "output": True},
            {"input": "what timers do i have", "output": True},
            {"input": "show me my timers", "output": True},
            {"input": "how long is left on my timer", "output": True},
            {"input": "check my timers", "output": True},
            {"input": "how many timers are running", "output": True},
            {"input": "do i have any timers going", "output": True},
            {"input": "set a timer for five minutes", "output": False},
            {"input": "cancel my alarm", "output": False},
        ],
    },
    "show_alarms": {
        "description": (
            "User wants to see, check, list, change, turn off, cancel, or delete an ALARM (or "
            "all alarms) that already exists -- including simply asking what alarms are set. "
            "Creating a new alarm is set_alarm, not this."
        ),
        "few_shots": [
            {"input": "cancel my alarm", "output": True},
            {"input": "turn off my morning alarm", "output": True},
            {"input": "what alarms do i have set", "output": True},
            {"input": "delete all my alarms", "output": True},
            {"input": "check my alarms", "output": True},
            {"input": "what time is my alarm set for", "output": True},
            {"input": "set an alarm for 7 am", "output": False},
            {"input": "cancel my timer", "output": False},
        ],
    },
    "set_timer": {
        "description": (
            "User asks to start a countdown for a RELATIVE duration -- a timer for some "
            "number of seconds, minutes, or hours from now. "
            "A request tied to a clock time (\"at 7 am\") is set_alarm, not set_timer."
        ),
        "few_shots": [
            {"input": "set a timer for two minutes", "output": True},
            {"input": "timer for 10 minutes", "output": True},
            {"input": "give me 30 seconds", "output": True},
            {"input": "start a 5 minute timer for my tea", "output": True},
            {"input": "can you time an hour for me", "output": True},
            {"input": "wake me up at 7 am", "output": False},
            {"input": "remind me to take my pills at 7 pm", "output": False},
            {"input": "what time is it", "output": False},
        ],
    },
    "set_alarm": {
        "description": (
            "User asks to be alerted at a specific CLOCK TIME (for example 7 am, 19:30, "
            "half past six). A relative countdown is set_timer, not set_alarm."
        ),
        "few_shots": [
            {"input": "set an alarm for 7 am", "output": True},
            {"input": "wake me up at half past six", "output": True},
            {"input": "alarm at 19:30 for the gym", "output": True},
            {"input": "set a timer for five minutes", "output": False},
            {"input": "what is on my calendar tomorrow", "output": False},
        ],
    },
    "reminder": {
        "description": (
            "User explicitly asks to create, schedule, edit, or set a reminder, "
            "OR asks to view, list, or query their existing reminders, "
            "OR asks to delete or remove a reminder. "
            "Questions about weather, schedule, calendar, plans, or general information are not reminder intent."
        ),
        "few_shots": [
            {"input": "remind me to take my pills at 7 pm", "output": True},
            {"input": "set a reminder for my appointment tomorrow", "output": True},
            {"input": "change my reminder to 8 pm", "output": True},
            {"input": "what reminders do I have", "output": True},
            {"input": "do I have any reminders", "output": True},
            {"input": "what reminder do I have related to hydration", "output": True},
            {"input": "show me my reminders", "output": True},
            {"input": "delete the drink water reminder", "output": True},
            {"input": "remove my morning medication reminder", "output": True},
            {"input": "cancel all my reminders", "output": True},
            {"input": "what is the weather", "output": False},
            {"input": "what's on my calendar", "output": False},
            {"input": "tell me my schedule for tomorrow", "output": False},
            {"input": "hello robin", "output": False},
        ],
    },
    "delete_message": {
        "description": (
            "User asks to delete the last message, delete a conversation message, "
            "or remove their recent chat content."
        ),
        "few_shots": [
            {"input": "delete my last message", "output": True},
            {"input": "can you erase what i just said", "output": True},
            {"input": "delete this reminder", "output": False},
            {"input": "set a reminder for later", "output": False},
        ],
    },
    "end_conversation": {
        "description": (
            "User clearly signals they want to stop, end, or close this conversation."
        ),
        "few_shots": [
            {"input": "goodbye", "output": True},
            {"input": "that's all for now", "output": True},
            {"input": "thanks, remind me at 7 pm", "output": False},
            {"input": "delete my last message", "output": False},
        ],
    },
    "capabilities_query": {
        "description": (
            "User explicitly asks about Robin's abilities, features, or what Robin can help with. "
            "Requests for actual information or content should not be classified as capabilities_query."
        ),
        "few_shots": [
            {"input": "what can you do", "output": True},
            {"input": "what are your capabilities", "output": True},
            {"input": "how can you help me", "output": True},
            {"input": "tell me the time", "output": False},
            {"input": "what's on my calendar", "output": False},
        ],
    },
    "weather_query": {
        "description": (
            "User explicitly asks about weather conditions, forecast, rain, snow, sunshine, temperature, "
            "or other weather details for now, later today, or upcoming days."
        ),
        "few_shots": [
            {"input": "what's the weather tomorrow", "output": True},
            {"input": "is it going to rain later today", "output": True},
            {"input": "will it snow tonight", "output": True},
            {"input": "what can you do", "output": False},
            {"input": "what's on my calendar", "output": False},
        ],
    },
    "schedule_query": {
        "description": (
            "User explicitly asks about their calendar, schedule, agenda, plans, or activities for today, later today, "
            "tomorrow, or another upcoming day."
        ),
        "few_shots": [
            {"input": "what's on my calendar", "output": True},
            {"input": "what's on my schedule tomorrow", "output": True},
            {"input": "do I have any plans later today", "output": True},
            {"input": "what's the weather tomorrow", "output": False},
            {"input": "what can you do", "output": False},
        ],
    },
    "conversation": {
        "description": (
            "Default conversation intent. This includes ordinary questions, requests for information, "
            "small talk, companionship, and user requests that do not clearly match another special intent."
        ),
        "few_shots": [
            {"input": "tell me the time", "output": True},
            {"input": "what's on my calendar", "output": False},
            {"input": "how's the weather", "output": False},
            {"input": "what can you do", "output": False},
            {"input": "delete my last message", "output": False},
        ],
    },
}


def _load_intent_feature_flags() -> Dict[str, Any]:
    default_payload: Dict[str, Any] = {"default": {}, "users": {}}
    if not INTENT_FLAGS_PATH.exists():
        return default_payload
    try:
        payload = json.loads(INTENT_FLAGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_payload
        payload.setdefault("default", {})
        payload.setdefault("users", {})
        return payload
    except Exception as exc:
        logger.error("EVT_INTENT_FLAGS_LOAD_FAILED path=%s error=%s", INTENT_FLAGS_PATH, exc)
        return default_payload


def _is_intent_enabled_for_user(user_id: str, intent_name: str) -> bool:
    payload = _load_intent_feature_flags()
    default_flags = payload.get("default", {})
    user_flags = payload.get("users", {}).get(user_id, {})

    if isinstance(user_flags, dict) and intent_name in user_flags:
        return bool(user_flags[intent_name])
    if isinstance(default_flags, dict) and intent_name in default_flags:
        return bool(default_flags[intent_name])
    return True


def _parse_intent_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        raise TypeError(f"Unexpected intent response type: {type(response).__name__}")

    response_text = response.strip()
    if not response_text:
        raise ValueError("Empty intent response")

    decoder = json.JSONDecoder()
    json_start = response_text.find("{")
    if json_start == -1:
        raise json.JSONDecodeError("No JSON object found", response_text, 0)

    parsed, _ = decoder.raw_decode(response_text, idx=json_start)
    if not isinstance(parsed, dict):
        raise ValueError("Intent response JSON was not an object")
    return parsed


def classify_primary_intent(
    message: str,
    user_id: str,
    intent_names: List[str],
    precedence: List[str] | None = None,
) -> str:
    enabled_intents = [
        intent_name for intent_name in intent_names
        if _is_intent_enabled_for_user(user_id, intent_name)
    ]
    if not enabled_intents:
        return "none"

    ordered_precedence = precedence or DEFAULT_PRECEDENCE
    precedence_for_enabled = [name for name in ordered_precedence if name in enabled_intents]
    precedence_for_enabled.extend(
        [name for name in enabled_intents if name not in precedence_for_enabled]
    )

    intent_sections = []
    for intent_name in enabled_intents:
        definition = INTENT_DEFINITIONS.get(intent_name)
        if not definition:
            continue
        few_shots = "\n".join(
            f'- input: "{shot["input"]}" -> {intent_name}={str(bool(shot["output"])).lower()}'
            for shot in definition["few_shots"]
        )
        intent_sections.append(
            f"""Intent: {intent_name}
Description: {definition["description"]}
Few shots:
{few_shots}"""
        )

    if not intent_sections:
        return "none"

    intent_blocks = "\n".join(intent_sections)

    prompt = f"""You are an intent classifier.
Classify the user message into exactly one intent label from the allowed labels.

User message: "{message}"

Intents to classify:
{intent_blocks}

Precedence (highest to lowest, use when multiple intents could match):
{", ".join(precedence_for_enabled)}

Rules:
- Return a JSON object only.
- Return one key named "intent".
- The "intent" value must be one of: {", ".join([f'"{name}"' for name in enabled_intents])}, "none".
- Return "show_timers" / "show_alarms" when the user wants to see, check, list, change, stop or cancel a timer/alarm that ALREADY exists, rather than create a new one.
- Return "set_timer" when the user asks to count down a relative duration (minutes/seconds/hours from now).
- Return "set_alarm" when the user asks to be alerted at a specific clock time.
- Only return "reminder" when the user is explicitly asking to create, set, change, or manage a reminder, or asking to view, list, or query their reminders, or asking to delete or remove a reminder.
- Return "weather_query" when the user is explicitly asking about weather, forecast, rain, snow, sunshine, or temperature.
- Return "schedule_query" when the user is explicitly asking about their calendar, schedule, agenda, plans, or activities.
- Only return "capabilities_query" when the user is explicitly asking what Robin can do or how Robin can help.
- Return "conversation" for ordinary information requests, time questions, and general chat.
- If unsure, return "conversation".
"""

    try:
        response = llm_inference(
            [{"role": "user", "content": prompt}],
            model=intent_model_id(),
            response_format={"type": "json_object"},
            operation_name="intent_classification",
        )
        response_json = _parse_intent_response(response)
        intent = response_json.get("intent", "none")
        if intent in enabled_intents:
            return intent
        return "none"
    except Exception as exc:
        logger.error(
            "EVT_INTENT_CLASSIFY_ERROR user_id=%s error=%s raw_response=%r",
            user_id, exc, response if "response" in locals() else None,
        )

    return "none"


def is_affirmation_intent(message: str, user_id: str) -> bool:
    return (
        classify_primary_intent(message, user_id, ["affirmation"], precedence=["affirmation"])
        == "affirmation"
    )


def classify_routing_intent(message: str, user_id: str) -> str:
    return classify_primary_intent(
        message,
        user_id,
        ["conversation", "capabilities_query", "weather_query", "schedule_query", "show_timers", "show_alarms", "set_timer", "set_alarm", "reminder", "delete_message", "end_conversation"],
        precedence=["end_conversation", "show_timers", "show_alarms", "set_timer", "set_alarm", "reminder", "delete_message", "weather_query", "schedule_query", "capabilities_query", "conversation"],
    )


def classify_conversation_intent(message: str, user_id: str) -> str:
    return classify_routing_intent(message, user_id)

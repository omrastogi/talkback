"""Robin's core conversation turn: system-prompt template + an OpenAI-compatible chat
completion. Extracted from recover/llm/llm_utils.py with the Flask app, recover/config.py,
DB-backed prompt context, and AWS Bedrock support all removed -- this module has no
dependency beyond the `openai` package, so it can be dropped into any project.

Config is read from plain environment variables instead of recover/config.py, on every
call rather than once at import time (so setting them after importing this package still
works):
  OPENAI_API_KEY   required.
  OPENAI_BASE_URL  optional; point this at any OpenAI-compatible gateway (e.g. a local
                   PARCS-style endpoint) instead of api.openai.com.
  CHAT_MODEL_ID    optional, default "gpt-4o-mini".
  INTENT_MODEL_ID  optional, default CHAT_MODEL_ID.

conversation() no longer looks up weather/schedule/profile from a database; pass whatever
you want substituted into the prompt via the `context` dict (see build_system_prompt for
the recognized keys). Omitted keys are simply left out of the prompt.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)


def chat_model_id() -> str:
    return os.environ.get("CHAT_MODEL_ID", "gpt-4o-mini")


def intent_model_id() -> str:
    return os.environ.get("INTENT_MODEL_ID", chat_model_id())


def _client() -> openai.OpenAI:
    """Built fresh per call (cheap: no network I/O) instead of once at import time, so
    setting OPENAI_API_KEY/OPENAI_BASE_URL after importing this module still works -- e.g.
    a script that imports the package first and configures env vars in main()."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return openai.OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))


def gpt_inference(messages: List[Dict[str, Any]], stop=None, model: str | None = None, **argv):
    model = model or chat_model_id()
    client = _client()
    # Newer models (gpt-5.x, o1, etc.) use max_completion_tokens and don't support 'stop'.
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=512, **argv
        )
    else:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=512, stop=stop, **argv
        )
    return response.choices[0].message.content


def inference(messages: List[Dict[str, Any]], stop=None, **argv):
    """Thin logging wrapper around gpt_inference, mirroring llm_utils.py's inference()."""
    operation_name = argv.pop("operation_name", "unspecified")
    model = argv.get("model", chat_model_id())
    logger.info(
        "EVT_LLM_REQUEST operation=%s model=%s message_count=%s",
        operation_name, model, len(messages),
    )
    try:
        response = gpt_inference(messages, stop=stop, **argv)
        logger.info(
            "EVT_LLM_RESPONSE operation=%s model=%s response_chars=%s",
            operation_name, model, len(response or ""),
        )
        return response
    except Exception as exc:
        logger.error("EVT_LLM_FAILED operation=%s model=%s error=%s", operation_name, model, exc)
        raise


def build_system_prompt(*, context: Optional[Dict[str, Any]] = None, previous_reports_text: str = "") -> str:
    """Byte-for-byte copy of llm_utils.py's _build_conversation_system_prompt."""
    context = context or {}
    context_lines = []
    if "current_day" in context:
        context_lines.append(f"Current day: {context['current_day']}")
    if "current_date" in context:
        context_lines.append(f"Current date: {context['current_date']}")
    if "current_time" in context:
        context_lines.append(f"Current time: {context['current_time']}")
    if "weather_info" in context:
        context_lines.append(f"Weather information: {context['weather_info']}")
    if "weather_hourly" in context:
        context_lines.append(f"Hourly weather forecast: {context['weather_hourly']}")
    if "weather_daily_forecast" in context:
        context_lines.append(f"Daily weather forecast: {context['weather_daily_forecast']}")
    if "personal_data_profile" in context:
        context_lines.append(f"Personal data profile: {context['personal_data_profile']}")
    if "steps_info" in context:
        context_lines.append(f"Step count info: {context['steps_info']}")
    if "positive_quote_of_day" in context:
        context_lines.append(f"Positive quote of the day: {context['positive_quote_of_day']}")
    if "clock_state" in context:
        context_lines.append(f"Timers and alarms on the user's device: {context['clock_state']}")
    if "system_capabilities" in context:
        context_lines.append(f"System capabilities and boundaries: {context['system_capabilities']}")

    context_block = "\n".join(context_lines) if context_lines else "None"
    return f"""You are Robin, a friendly, empathetic AI companion supporting a person living alone in an assisted living facility in the United States.

Behavior:
- Speak respectfully, warmly, and naturally.
- Use plain text only (no markdown), since output is spoken by text-to-speech.
- Keep responses very short, usually one or two brief sentences.
- Default to about 10 to 30 words unless the user explicitly asks for more detail.
- If the answer could be long (for example schedule details, weather details, lists, explanations, or summaries), give only a small subset of the most important information first.
- For long informational answers, never read out the full list by default. Share one to three key items, then ask if the user wants to hear more.
- End with a short, gentle follow-up question when appropriate.
- Always use AM and PM (without dots) when mentioning times.
- Answer directly. Do not narrate your process or say things like "Let me check," "I see," "Looking at your schedule," or similar setup phrases.
- Use current weather for questions about now.
- Use hourly weather forecast for questions about later today, this afternoon, this evening, or tonight.
- Use daily weather forecast for questions about tomorrow or upcoming days.
- Do not invent forecast details that are not present in the provided weather context.
- Do not append control tokens.
- Only claim abilities that are explicitly listed in the configured system capabilities.
- Do not pretend to complete actions Robin cannot actually perform.
- Do not say you will create reminders, send notifications, contact people, or take external actions unless that capability is explicitly available and the request is being handled by a dedicated workflow.
- CRITICAL: timers and alarms are set by a separate dedicated workflow, NOT by this reply. In
  this reply you must NEVER say a timer or alarm has been set, is starting, or is running --
  saying so when no timer was actually created is the worst failure this assistant can make.
  If the user seems to want one, ask them to say the duration or time plainly instead.
- If the user asks for something outside supported capabilities, give a short refusal and redirect to what Robin can help with.
- If the user shares personal details, memories, visitors, or relationships during the conversation, treat those statements as valid conversational context unless there is a clear safety reason not to.
- Do not bluntly contradict the user by citing the profile as authority on personal facts like relationships, friends, or visitors.
- If profile information and the user's current statement seem inconsistent, respond gently, play along supportively, and ask a soft follow-up question instead of correcting them.

Continuity:
- Previous conversation summary: {previous_reports_text}
- The full message history for this same conversation is provided in the chat messages below.
- Use the earlier messages in this same conversation to resolve context, pronouns, follow-up questions, and references like "that", "it", "tomorrow", or "the one I mentioned".
- When the user's latest message depends on something said earlier in this same conversation, answer using that earlier conversation context instead of treating the message as standalone.

Configured context for this user:
{context_block}
"""


def conversation(messages: List[Dict[str, str]], previous_reports_text: str = "", context: Optional[Dict[str, Any]] = None) -> str:
    """The same turn logic as recover/llm/llm_utils.py::conversation() -- minus the DB-backed
    context lookup. `messages` is the full role/content history including the latest user
    turn; `context` optionally supplies prompt placeholders (weather, schedule, profile...)."""
    sys_prompt = build_system_prompt(context=context, previous_reports_text=previous_reports_text)
    return inference(
        [{"role": "system", "content": sys_prompt}, *messages],
        model=chat_model_id(),
        operation_name="conversation_response",
    )

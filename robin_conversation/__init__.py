"""Robin's conversation logic, self-sufficient: intent classification, the LLM turn, the
delete-confirmation sentinel gate, and the reminder/weather/schedule/capabilities/
end-conversation intent replies -- everything services/conversation/{engine.py,flow.py} +
intent_handlers/ + intent/classifier.py + llm/llm_utils.py did, minus Flask, the database,
and AWS Bedrock.

No Flask, no database, no AWS Bedrock -- depends on `openai` and `requests` only. See
llm.py for required environment variables (OPENAI_API_KEY, optionally OPENAI_BASE_URL,
CHAT_MODEL_ID, INTENT_MODEL_ID).

Typical usage:
    from robin_conversation import process_turn

    history = []
    result = process_turn(history, "what's the weather tomorrow?", user_id="u1")
    print(result["reply"], result["intent"])
"""
from .classifier import (
    classify_conversation_intent,
    classify_routing_intent,
    is_affirmation_intent,
)
from .engine import process_turn
from .llm import build_system_prompt, conversation, inference
from .prompt_context import build_conversation_prompt_context

__all__ = [
    "process_turn",
    "classify_conversation_intent",
    "classify_routing_intent",
    "is_affirmation_intent",
    "conversation",
    "inference",
    "build_system_prompt",
    "build_conversation_prompt_context",
]

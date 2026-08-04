"""Port of recover/alexa.py::conversation() — the transport-agnostic RECOVER turn logic —
adapted for the local voice-server.

Original lives at E:/PARCS/robin-ca-mirror/backend/recover/{alexa.py, openai_utils.py}. That
version is wired to Flask + SQLAlchemy (Patient/ConversationLog rows), a Chroma vector DB, and
live weather/calendar lookups. None of that exists here, so the port keeps the *decision logic*
verbatim and swaps the infrastructure for plug points:

  - conversation state  -> an in-memory list of message dicts (one per WebSocket connection)
  - patient history RAG -> optional `rag(user_text) -> str` callback (default: no history)
  - weather/schedule/profile -> a `ctx` dict the caller supplies (default: today's day + blanks)

The two sentinel side-paths (DELETE_MESSAGE confirm/redact, require_affirmation gate) and the
CONVERSATION_END goodbye keyword are ported as-is. The system prompt is the byte-for-byte copy
in robin_prompt.txt. The OpenAI-compatible `client` and `model` are passed in, so this reuses
the voice-server's configured backend/key — no hardcoded credentials.

Message dict shape: {"role": "user"|"assistant", "content": str,
                     "require_affirmation": bool (opt), "redacted": bool (opt)}.
"""
import json
import os
from datetime import datetime

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robin_prompt.txt")
with open(_PROMPT_PATH, encoding="utf-8") as _fh:
    SYSTEM_PROMPT = _fh.read()

# Placeholders the prompt expects. The caller can override any of these via `ctx`; whatever is
# left uses these blanks so the prompt never ships an unfilled {token}. current_day is the one
# value we can compute locally (original used America/New_York).
_PLACEHOLDERS = ("current_day", "weather_info", "schedule",
                 "personal_data_profile", "no_schedule_reason", "patient_history_reports")


def _default_ctx():
    try:
        from zoneinfo import ZoneInfo
        day = datetime.now(ZoneInfo("America/New_York")).strftime("%A")
    except Exception:                      # zoneinfo/tzdata missing -> local time is close enough
        day = datetime.now().strftime("%A")
    return {"current_day": day, "weather_info": "", "schedule": "",
            "personal_data_profile": "", "no_schedule_reason": "", "patient_history_reports": ""}


def build_system_prompt(ctx=None, previous_reports_text=""):
    """Substitute the prompt's {placeholders}. previous_reports_text fills patient_history_reports
    (the RAG slot); ctx overrides any other placeholder."""
    filled = _default_ctx()
    if previous_reports_text:
        filled["patient_history_reports"] = previous_reports_text
    if ctx:
        filled.update({k: v for k, v in ctx.items() if k in _PLACEHOLDERS})
    prompt = SYSTEM_PROMPT
    for key in _PLACEHOLDERS:
        prompt = prompt.replace("{" + key + "}", str(filled[key]))
    return prompt


def _llm_text(client, model, messages, *, stream=True, **kw):
    """One OpenAI-compatible chat completion -> reply text.

    Streams and accumulates by default, mirroring server.py's working default path: the PARCS
    gemma gateway returned empty content on the non-streaming main call, while streaming is the
    mode proven to work against it. No max_tokens cap (the default path uses none). The tiny
    affirmation classifier passes stream=False — that short non-streaming call already works on
    gemma and keeps response_format=json_object simple."""
    if not stream:
        resp = client.chat.completions.create(model=model, messages=messages, **kw)
        return resp.choices[0].message.content or ""
    parts = []
    for chunk in client.chat.completions.create(model=model, messages=messages, stream=True, **kw):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def check_affirmation_intent(client, model, message):
    """LLM yes/no classifier for the two sentinel paths (ported from alexa.py)."""
    prompt = (
        "You are an intent classifier.\n"
        "The user maybe answering the question: 'Hey <user_name>, are you around?' or "
        "'Are you sure you want to delete the last message?'.\n"
        f"The user response is: '{message}'.\n"
        "If the message means YES (affirmative intent, e.g. 'yes', 'here', 'yup'), return only True.\n"
        "If the message means NO, unsure, or unrelated, return only False. If the user doesn't say "
        "anything, return only False.\n"
        'Respond with a JSON object with a single key "affirmation" and value True or False.'
    )
    raw = _llm_text(client, model, [{"role": "user", "content": prompt}],
                    stream=False, response_format={"type": "json_object"})
    return bool(json.loads(raw)["affirmation"])


def redact_last_two_before(history, prev_index):
    """Mark the two messages before the message one-before `prev_index` as REDACTED — the
    exchange preceding a delete request (ported from alexa.py::redact_last_two_before)."""
    if len(history) < 3 or prev_index <= 0:
        return
    one_before = prev_index - 1                 # the user's "delete my last message" turn
    if one_before <= 0:
        return
    for m in history[max(0, one_before - 2):one_before]:
        if not m.get("redacted"):
            m["content"] = "REDACTED"
            m["redacted"] = True


def conversation(history, content, *, client, model, ctx=None, rag=None):
    """Port of alexa.py::conversation(). Mutates `history` in place (appends the user turn and the
    assistant reply) and returns the assistant reply string. `rag(user_text) -> str` optionally
    supplies retrieved patient-history context; `ctx` optionally supplies prompt placeholders.

    Reply may contain the sentinel keyword CONVERSATION_END (caller decides when to end the
    session); DELETE_MESSAGE is handled here across two turns.
    """
    last = history[-1] if history else None

    # --- sentinel 1: previous turn asked to confirm a deletion ---
    if last and "DELETE_MESSAGE" in last["content"]:
        history.append({"role": "user", "content": content})
        if check_affirmation_intent(client, model, content):
            redact_last_two_before(history, len(history) - 2)   # -2 = the DELETE_MESSAGE prompt
            reply = "I have deleted the last message. Is there anything else I can help you with?"
        else:
            reply = "Is there anything else I can help you with?"
        history.append({"role": "assistant", "content": reply})
        return reply

    # --- sentinel 2: previous turn was an "are you around?" affirmation gate ---
    if last and last.get("require_affirmation"):
        if not check_affirmation_intent(client, model, content):
            return "Goodbye! CONVERSATION_END"
        last["require_affirmation"] = False
        return last["content"]

    # --- normal turn ---
    history.append({"role": "user", "content": content})
    previous_reports_text = rag(content) if rag else ""
    sys_prompt = build_system_prompt(ctx, previous_reports_text)
    messages = [{"role": "system", "content": sys_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    reply = _llm_text(client, model, messages)
    history.append({"role": "assistant", "content": reply})
    return reply


# --- self-check: no network, a scripted fake client stands in for OpenAI ---
def _selftest():
    class _Resp:                                    # non-streaming response
        def __init__(self, c): self.choices = [type("Ch", (), {"message": type("M", (), {"content": c})})]
    class _Chunk:                                   # one streaming delta
        def __init__(self, c): self.choices = [type("Ch", (), {"delta": type("D", (), {"content": c})})]
    class FakeClient:
        """Streams queued replies (main call); affirmation calls (stream=False, json_object)
        return a queued yes/no non-streamed."""
        def __init__(self, replies, affirms):
            self.replies, self.affirms = list(replies), list(affirms)
            self.chat = type("C", (), {"completions": self})()
        def create(self, **kw):
            if kw.get("response_format", {}).get("type") == "json_object":
                return _Resp(json.dumps({"affirmation": self.affirms.pop(0)}))
            reply = self.replies.pop(0)
            return [_Chunk(reply)] if kw.get("stream") else _Resp(reply)

    # 1) placeholders all filled
    p = build_system_prompt({"weather_info": "sunny"}, "prev summary")
    assert "{" not in p or "{token}" not in p
    for k in _PLACEHOLDERS:
        assert "{" + k + "}" not in p, k
    assert "sunny" in p and "prev summary" in p

    # 2) normal turn appends user + assistant, returns reply
    h = []
    r = conversation(h, "hello", client=FakeClient(["Hi there!"], []), model="m")
    assert r == "Hi there!" and len(h) == 2 and h[0]["content"] == "hello"

    # 3) CONVERSATION_END passes through untouched
    r = conversation(h, "bye", client=FakeClient(["Take care! CONVERSATION_END"], []), model="m")
    assert "CONVERSATION_END" in r

    # 4) affirmation gate: last turn requires affirmation, user declines -> goodbye
    h2 = [{"role": "assistant", "content": "Are you around?", "require_affirmation": True}]
    r = conversation(h2, "no", client=FakeClient([], [False]), model="m")
    assert r == "Goodbye! CONVERSATION_END", r

    # 5) delete path: confirm -> redacts the two turns before the delete request
    h3 = [{"role": "user", "content": "old A"}, {"role": "assistant", "content": "old B"},
          {"role": "user", "content": "delete my last message"},
          {"role": "assistant", "content": "Are you sure? DELETE_MESSAGE"}]
    r = conversation(h3, "yes", client=FakeClient(["done"], [True]), model="m")
    assert h3[0]["content"] == "REDACTED" and h3[1]["content"] == "REDACTED", h3
    assert h3[2]["content"] == "delete my last message"        # the request itself is kept
    assert "deleted" in r
    print("robin_convo selftest OK")


if __name__ == "__main__":
    _selftest()

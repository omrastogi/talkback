"""Repair the speech recogniser's systematic mishearings before routing.

Parakeet mangles "alarm(s)" roughly 29% of the time in this deployment -- observed forms in
one evening's logs: "anarms", "announc", "announced", "announce", "a numb". Those land as
ordinary English and the intent classifier reasonably reads "what announced do I have" as a
question about the day's plans, so the user gets their calendar read back instead of their
alarms. Few-shot examples were not enough: the garbled sentence genuinely looks like a
schedule question, so the model kept choosing schedule_query.

Fixing the transcript once, deterministically, is more reliable than asking the model to see
through the error at every downstream step (classification, extraction, cancel resolution).

Only rewrites forms that are either not English at all ("anarms", "announc") or are pinned by
surrounding words to the alarm sense ("set an announced for seven" / "what announced do I
have"), so a genuine "the doctor announced the results" is left alone.
"""
import re

# parakeet-tdt-1.1b (current model) mishears "alarm(s)" differently from the 0.6b it
# replaced: "arms", "amounts", "a number", "norm". Every one of those is a real English word,
# and one of them is a BODY PART -- in a post-op patient assistant "my arms hurt" must never
# become "my alarms hurt". So each rule below is pinned by surrounding words that only occur
# in the clock sense, never by the bare noun.
_FIXES = (
    (re.compile(r"\barms\b(?=\s+(?:do|does|did)\s+(?:i|we|you)\s+have)", re.I), "alarms"),
    (re.compile(r"\barms\b(?=\s+(?:at|for)\s+\d)", re.I), "alarms"),
    (re.compile(r"(?<=\bset an\s)arms?\b", re.I), "alarm"),
    (re.compile(r"(?<=\bset a\s)norm\b", re.I), "alarm"),
    (re.compile(r"\bamounts\b(?=\s+(?:do|does|did)\s+(?:i|we|you)\s+have)", re.I), "alarms"),
    (re.compile(r"(?<=\ball these\s)amounts\b", re.I), "alarms"),
    (re.compile(r"(?<=\ball\s)amounts\b(?=\s|$)", re.I), "alarms"),
    (re.compile(r"\ba number\b(?=\s+(?:for|at)\s+(?:\d|eight|nine|ten|seven|six))", re.I), "an alarm"),
    # Not words in any context -- always the alarm noun.
    (re.compile(r"\banarms\b", re.I), "alarms"),
    (re.compile(r"\banarm\b", re.I), "alarm"),
    (re.compile(r"\bannounc\b", re.I), "alarm"),
    (re.compile(r"\bnarms\b", re.I), "alarms"),
    # Real words -- only rewrite when the surrounding phrasing fixes the meaning.
    (re.compile(r"\ba numb\b(?=\s+(?:for|at|set|on)\b)", re.I), "an alarm"),
    (re.compile(r"\bannounces?\b(?=\s+(?:do|does|did)\s+(?:i|we|you)\s+have)", re.I), "alarms"),
    (re.compile(r"\bannounced\b(?=\s+(?:do|does|did)\s+(?:i|we|you)\s+have)", re.I), "alarms"),
    (re.compile(r"(?<=\bset an\s)announce[ds]?\b", re.I), "alarm"),
    (re.compile(r"(?<=\badd an\s)announce[ds]?\b", re.I), "alarm"),
    (re.compile(r"(?<=\bmy\s)announce[ds]?\b(?!\s+(?:that|the)\b)", re.I), "alarms"),
)


def repair_transcript(text: str) -> str:
    """Apply the known mishearing fixes. Returns the text unchanged when nothing matches."""
    if not text:
        return text
    for pattern, replacement in _FIXES:
        text = pattern.sub(replacement, text)
    return text


if __name__ == "__main__":   # repo convention: each module self-checks
    cases = [
        ("Can you tell me what all announced do I have?", "alarms"),
        ("Um, what uh anarms do I have today?", "alarms"),
        ("Uh set an announc for eight thirty in the morning.", "alarm"),
        ("Can you add a numb for six thirty pm?", "an alarm"),
        ("What alarms do I have?", "alarms"),
    ]
    cases += [
        ("what arms do we have", "alarms"),
        ("can you delete all these amounts", "alarms"),
        ("can you add in a number for eight thirty", "an alarm"),
        ("can you set a norm for six twenty pm today", "alarm"),
        ("can you tell me what all amounts do i have", "alarms"),
    ]
    for text, expect in cases:
        got = repair_transcript(text)
        assert expect in got.lower(), (text, got)

    # A post-op patient talking about their body must never be rewritten into clock-speak.
    for safe in ("my arms hurt a lot today",
                 "both my arms feel weak",
                 "the amounts on my prescription changed",
                 "is that the normal amount"):
        assert repair_transcript(safe) == safe, repair_transcript(safe)
    # Must NOT touch a legitimate use of the verb.
    untouched = "The doctor announced the results this morning."
    assert repair_transcript(untouched) == untouched, repair_transcript(untouched)
    assert repair_transcript("She announced that she was leaving") == "She announced that she was leaving"
    print("asr_fixes selftest OK")

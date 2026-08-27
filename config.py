"""Shared config for every stage: .env loading, system prompt, LLM backend selection.

One place for the three things that used to be copy-pasted across the stage scripts:
the minimal .env parser, the assistant system prompt, and the openai|parcs backend
switch. `parcs` is the lab's OpenAI-compatible gateway serving gemma.

Importing this module sets PYTORCH_CUDA_ALLOC_CONF *before* torch is imported anywhere
in the process. The GPU stages (`server.py`, `stream.py`) rely on that: they `import
config` ahead of `import torch`, which lets every import stay at the top of the file
while still guaranteeing the alloc setting lands before CUDA initialises.
"""
import os

# Must precede any torch/CUDA import — expandable segments cut fragmentation so Parakeet
# + Kokoro both stay resident on a 4 GB card. setdefault: a value from the shell wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_PROMPT = (
    "You are a warm, careful voice assistant for personal health support. You help people "
    "understand symptoms, medications, and healthy habits, and figure out when to seek care. "
    "You are not a doctor: never diagnose, prescribe, or give a definitive medical verdict, and "
    "say so when asked to. Reply in one or two short, spoken-style sentences — plain words, no "
    "lists or markdown, numbers and units said naturally — and ask one clarifying question when "
    "you need more detail. If the user describes an emergency such as chest pain, trouble "
    "breathing, signs of a stroke, severe bleeding, or thoughts of self-harm, tell them to call "
    "their local emergency number right now. When you are unsure, say so plainly and recommend "
    "seeing a licensed clinician."
)
PARCS_DEFAULT_URL = "https://gateway.parcs.northeastern.edu/llm/api/v1"


def load_env():
    """Load .env (this dir, then parent) into os.environ. Minimal KEY=VALUE parser so the
    harness needs no python-dotenv. setdefault, so anything already in the shell wins."""
    for path in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.removeprefix("export ").strip().split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_backend(backend=None, model=None):
    """Resolve (base_url, api_key, model) for backend 'openai' or 'parcs'.

    backend/model None -> LLM_BACKEND env (default 'parcs') / the backend's default model.
    base_url None means OpenAI's own endpoint. Loads .env first so keys are available.
    """
    load_env()
    backend = backend or os.environ.get("LLM_BACKEND", "parcs")
    if backend == "openai":
        return None, os.environ.get("OPENAI_API_KEY"), model or "gpt-4o-mini"
    return (os.environ.get("PARCS_BASE_URL", PARCS_DEFAULT_URL),
            os.environ.get("PARCS_API_KEY"),
            model or os.environ.get("PARCS_MODEL", "gemma4:12b:fast"))


def make_client(backend=None, model=None, async_=False):
    """Return (client, model): an (Async)OpenAI client pointed at the chosen backend."""
    base_url, api_key, model = resolve_backend(backend, model)
    from openai import AsyncOpenAI, OpenAI
    cls = AsyncOpenAI if async_ else OpenAI
    return cls(base_url=base_url, api_key=api_key), model


# --- VAD / tap-to-talk turn-taking -----------------------------------------------------
# Runtime config (env, load_env() first so .env and the shell both work). Defaults live in
# vad.turn.TurnParams — the single source of truth — and env only overrides them, so there is
# no second copy of the numbers to drift.
TURN_MODE_DEFAULT = "tap"                # "tap" (VAD endpointing) | "hold" (button-up = EOU)
VAD_PROVIDER_OVERRIDE_DEFAULT = "auto"   # "auto" | "cuda" | "cpu"

_VAD_ENV = {   # TurnParams field -> (env var, caster)
    "vad_speech_threshold": ("VAD_SPEECH_THRESHOLD", float),
    "vad_silence_threshold": ("VAD_SILENCE_THRESHOLD", float),
    "vad_onset_frames": ("VAD_ONSET_FRAMES", int),
    "vad_hangover_ms": ("VAD_HANGOVER_MS", int),
    "vad_prespeech_ms": ("VAD_PRESPEECH_MS", int),
    "vad_max_utterance_s": ("VAD_MAX_UTTERANCE_S", float),
    "vad_arm_timeout_s": ("VAD_ARM_TIMEOUT_S", float),
}


def turn_mode():
    load_env()
    return os.environ.get("TURN_MODE", TURN_MODE_DEFAULT)


def api_key():
    """Shared secret for WebSocket auth (ROBIN_API_KEY). None means auth is disabled --
    the caller is responsible for warning about that, since only it knows if this is dev."""
    load_env()
    return os.environ.get("ROBIN_API_KEY") or None


def vad_provider_override():
    load_env()
    return os.environ.get("VAD_PROVIDER_OVERRIDE", VAD_PROVIDER_OVERRIDE_DEFAULT)


def vad_params():
    """Build a vad.turn.TurnParams from env overrides (imported lazily so config stays
    importable-before-torch and cheap on boxes without the vad package on the path)."""
    load_env()
    import dataclasses

    from vad.turn import TurnParams
    overrides = {field: cast(os.environ[env])
                 for field, (env, cast) in _VAD_ENV.items() if env in os.environ}
    return dataclasses.replace(TurnParams(), **overrides) if overrides else TurnParams()


if __name__ == "__main__":   # self-check: backend resolution picks the right url + model
    os.environ.setdefault("OPENAI_API_KEY", "test")
    base, _, mdl = resolve_backend("openai")
    assert base is None and mdl == "gpt-4o-mini", (base, mdl)
    base, _, mdl = resolve_backend("parcs", "gemma4:12b:fast")
    assert base and "parcs" in base and mdl == "gemma4:12b:fast", (base, mdl)
    # vad config: defaults come through untouched, and an env override lands as the right type.
    assert turn_mode() == "tap" and vad_provider_override() == "auto"
    assert vad_params().vad_hangover_ms == 600
    os.environ["VAD_HANGOVER_MS"] = "450"
    assert vad_params().vad_hangover_ms == 450
    del os.environ["VAD_HANGOVER_MS"]
    print("config OK")

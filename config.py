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
            model or os.environ.get("PARCS_MODEL", "gemma4:12b"))


def make_client(backend=None, model=None, async_=False):
    """Return (client, model): an (Async)OpenAI client pointed at the chosen backend."""
    base_url, api_key, model = resolve_backend(backend, model)
    from openai import AsyncOpenAI, OpenAI
    cls = AsyncOpenAI if async_ else OpenAI
    return cls(base_url=base_url, api_key=api_key), model


if __name__ == "__main__":   # self-check: backend resolution picks the right url + model
    os.environ.setdefault("OPENAI_API_KEY", "test")
    base, _, mdl = resolve_backend("openai")
    assert base is None and mdl == "gpt-4o-mini", (base, mdl)
    base, _, mdl = resolve_backend("parcs", "gemma4:12b")
    assert base and "parcs" in base and mdl == "gemma4:12b", (base, mdl)
    print("config OK")

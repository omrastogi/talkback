"""VAD + turn-taking for tap-to-talk endpointing (transport/ingest layer only).

Keep this __init__ light: it exports the pure state machine (stdlib only) but does NOT
import silero.py / provider.py, which pull in onnxruntime. Import those explicitly where
the GPU/ORT dependency is actually wanted, so `import vad` stays cheap and never fails on
a box without onnxruntime.
"""
from .turn import Event, State, TurnMachine, TurnParams  # noqa: F401

"""Push-to-talk realtime voice server: browser mic -> STT -> Robin's conversation engine -> sentence TTS -> browser.

Models load once at import (both resident on the 4 GB card). The LLM turn is remote (blocking
robin_conversation.process_turn(), run in a thread); Parakeet + Kokoro are blocking too, so they
also run in threads to keep the event loop free.

    conda run -n voice uvicorn server:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000 in the Windows browser
"""
import math
import os

import config   # importing sets PYTORCH_CUDA_ALLOC_CONF before torch loads (see config.py)

import argparse
import asyncio
import collections
import dataclasses
import datetime
import functools
import hmac
import io
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = "af_heart"
# Played once ahead of a proactive utterance: Robin speaking unprompted needs a moment of
# warning, or the first words land before the user has looked up.
CHIME_PATH = os.path.join(HERE, "chime.mp3")

# Per-turn audio + server.log live together under log/; each log line references the
# turn's wav files by path so you can open the audio straight from the log.
LOG_DIR = os.path.join(HERE, "log")
os.makedirs(LOG_DIR, exist_ok=True)
# Client playback telemetry: one JSONL file per session, one line per turn. Kept separate
# from the audio/server logs so it's easy to feed to scripts/analyze_telemetry.py.
TELEMETRY_DIR = os.path.join(HERE, "logs", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
# VAD / turn-taking instrumentation: raw per-stage timestamps + the startup provider bench,
# one JSONL per session, alongside the existing bench matrix so VAD placement is visible.
BENCH_DIR = os.path.join(HERE, "log", "bench")
os.makedirs(BENCH_DIR, exist_ok=True)
logging.getLogger().setLevel(logging.WARNING)          # mute chatty libraries (NeMo, httpx, ...)
log = logging.getLogger("voice")
log.setLevel(logging.INFO)
log.propagate = False                                  # keep server.log to our lines only
_fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
for _h in (logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "server.log"))):
    _h.setFormatter(_fmt)
    log.addHandler(_h)


def _f(x):
    return f"{x:.2f}" if x is not None else "n/a"


def _append_bench(name, obj):
    """Append one JSONL record to log/bench/<name>.jsonl. Best-effort — instrumentation must
    never break the audio path. Stores RAW timestamps; deltas are derived at analysis time."""
    try:
        with open(os.path.join(BENCH_DIR, f"{name}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
    except Exception as e:                                  # noqa: BLE001 — best-effort logging only
        log.warning("bench write failed: %s", e)


def _load_stt():
    import nemo.collections.asr as nemo_asr
    # Load on CPU first, then move — NeMo's direct-to-GPU restore OOMs the 4 GB card.
    # parakeet-tdt-1.1b over 0.6b-v3: benchmarked on 210 logged utterances from real
    # sessions, it is both FASTER (0.035s vs 0.053s per utterance) and more accurate on this
    # deployment's problem word -- "alarm" correct 29/36 vs 26/36 -- and it recovered two
    # quiet clips the 0.6b dropped to an empty transcript, i.e. two user requests that
    # silently vanished. Whisper large-v3 scored 30/36 but costs 6x the latency.
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b", map_location="cpu")
    if torch.cuda.is_available():
        m = m.to("cuda")
    m.eval()
    return m


config.load_env()

# --- LLM backend: --backend openai|parcs (or LLM_BACKEND env). parse_known_args so this
# also works under `uvicorn server:app` (uvicorn's own argv is simply ignored). ---
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--backend", choices=["openai", "parcs"],
                 default=os.environ.get("LLM_BACKEND", "parcs"))
_ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
_ap.add_argument("--host", default="0.0.0.0")
_ap.add_argument("--port", type=int, default=8000)
ARGS, _ = _ap.parse_known_args()

LLM_BASE_URL, LLM_API_KEY, MODEL = config.resolve_backend(ARGS.backend, ARGS.model)
if not LLM_API_KEY:
    log.warning("no API key for backend '%s' — set %s in .env", ARGS.backend,
                "OPENAI_API_KEY" if ARGS.backend == "openai" else "PARCS_API_KEY")

# --- ROBIN_API_KEY now gates only the legacy HTTP dashboard endpoints (X-Robin-Key). The
# voice WebSockets require a per-device token (see robin/); the shared key no longer grants
# voice-socket access. ---
API_KEY = config.api_key()
if not API_KEY:
    log.warning("ROBIN_API_KEY not set — legacy HTTP dashboard auth is DISABLED (dev only); "
                "voice sockets still require a device token")

t = time.time(); stt_model = _load_stt(); STT_LOAD = time.time() - t
from kokoro import KPipeline
t = time.time(); pipe = KPipeline(lang_code="a"); TTS_LOAD = time.time() - t
for _ in pipe("Ready.", voice=VOICE):   # warm Kokoro (first synth compiles kernels ~3s)
    pass
# Robin's conversation engine (robin_conversation package) -- intent classification, the
# delete-confirmation gate, weather/schedule/capabilities replies, and the default LLM turn.
# It's blocking, so it runs in a thread. It reads its OpenAI-compatible client config from
# env vars (see robin_conversation/llm.py); point those at the backend resolved above -- same
# backend/key as everywhere else, no hardcoded credentials.
os.environ["OPENAI_API_KEY"] = LLM_API_KEY or ""
if LLM_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
os.environ["CHAT_MODEL_ID"] = MODEL
os.environ["INTENT_MODEL_ID"] = MODEL
from robin_conversation import process_turn
from robin_conversation.prompt_context import build_conversation_prompt_context
from zoneinfo import ZoneInfo

# Persistence layer (robin/): profile-bound device tokens and per-utterance turn rows.
# Import is light -- engine creation (and the DATABASE_URL requirement) is deferred until
# the first connection actually authenticates.
from robin.ws import bind_device_session, persist_turn
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("[ready] STT %.1fs · TTS %.1fs · both resident on %s · LLM=%s [%s] · logs -> %s",
         STT_LOAD, TTS_LOAD, _DEVICE, MODEL, ARGS.backend, LOG_DIR)

# --- VAD provider: choose once at process start (never per connection). Non-fatal: a CUDA
# probe failure falls back to CPU. On this 4 GB card CPU is the safe operational pick (keeps
# VRAM free for Parakeet); VAD_PROVIDER_OVERRIDE=cpu forces it. See vad/provider.py. ---
TURN_MODE = config.turn_mode()
try:
    from vad.provider import select_provider
    VAD_PROVIDERS, VAD_BENCH = select_provider(config.vad_provider_override())
except Exception as e:                                      # never let VAD setup kill the server
    log.warning("VAD provider selection failed, defaulting to CPU: %s", e)
    VAD_PROVIDERS, VAD_BENCH = ["CPUExecutionProvider"], {"vad_provider": "CPUExecutionProvider",
                                                          "vad_setup_error": f"{type(e).__name__}: {e}"}
_append_bench("vad_provider", {"event": "startup", "ts": time.time(),
                               "turn_mode": TURN_MODE, **VAD_BENCH})
log.info("[vad] turn_mode=%s provider=%s", TURN_MODE, VAD_BENCH.get("vad_provider"))


def decode_to_wav(audio_bytes, wav16):
    """Browser webm/ogg-opus bytes -> 16 kHz mono wav at wav16. ffmpeg needs a seekable
    file (piping webm fails), so write a temp file first. The wav is kept for inspection."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(audio_bytes)
        src = fh.name
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", wav16],
                       check=True)
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    return wav16


# One GPU, one copy of each model, and turns now run as concurrent tasks -- two overlapping
# connections transcribing at the same instant made NeMo raise "Cannot unfreeze partially
# without first freezing the module", killing one of the turns. These models are not safe for
# concurrent use, so serialise them. Both are fast (STT ~0.035s, TTS ~0.05s/sentence) next to
# the ~2s LLM step, so the queueing cost is negligible.
_STT_LOCK = threading.Lock()
_TTS_LOCK = threading.Lock()


def stt_transcribe(wav16):
    with _STT_LOCK, torch.inference_mode():
        out = stt_model.transcribe([wav16])
    hyp = out[0]
    return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


# Kokoro's g2p does not read a clock time: it keeps the colon and says the minutes digit by
# digit, so "8:00 PM" comes out "eight ZERO ZERO PM" and "08:00" as "zero eight zero zero".
# Minutes of 10 or more already sound right ("8:30" -> "eight thirty"), so this only has to
# fix :00 and :0X, leading zeros, and the 24-hour times the calendar data is stored in.
_CLOCK_RE = re.compile(r"\b(\d{1,2}):([0-5]\d)(\s*[AaPp]\.?[Mm]\.?)?")


def _spoken_time(m):
    hour, minute, meridiem = int(m.group(1)), int(m.group(2)), (m.group(3) or "")
    if hour > 23:
        return m.group(0)                        # not a clock time; leave it alone
    if not meridiem and hour > 12:               # 24-hour, as stored in calendar.json
        hour, meridiem = hour - 12, " PM"
    elif not meridiem and hour == 0:
        hour, meridiem = 12, " AM"
    if minute == 0:                              # "8:00 PM" -> "8 PM", "12:00" -> "12 o'clock"
        return f"{hour}{meridiem}" if meridiem else f"{hour} o'clock"
    if minute < 10:                              # "1:05" -> "1 oh 5", never "one zero five"
        return f"{hour} oh {minute}{meridiem}"
    return f"{hour} {minute}{meridiem}"


def for_speech(text):
    """Text as it should be SAID rather than shown. Applied at synthesis only, so the client
    still displays "8:00 PM" while Robin says "eight PM"."""
    return _CLOCK_RE.sub(_spoken_time, text)


def synth_audio(text, voice=VOICE, speed=1.0):
    """Kokoro synth -> one concatenated float32 array (24 kHz). Times are rewritten for the
    voice first (see for_speech); every spoken path lands here, so this is the one place.
    `voice`/`speed` come from the connection's profile (see SESSION_DB); the defaults keep
    the historical behaviour for anything unbound."""
    text = for_speech(text)
    chunks = []
    with _TTS_LOCK:
        for gs, ps, audio in pipe(text, voice=voice, speed=speed):
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype="float32")
            chunks.append(np.asarray(a, dtype="float32").reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")


def wav_bytes(audio):
    """float32 array -> one complete WAV (24 kHz PCM16) as bytes for a single binary frame."""
    bio = io.BytesIO()
    sf.write(bio, audio, 24000, format="WAV", subtype="PCM_16")
    return bio.getvalue()


_CHIME = None                # decoded on first use, then cached: one ffmpeg call per process


def chime_audio():
    """chime.mp3 -> float32 mono at 24 kHz, the TTS sample rate, so it ships through the very
    same audio frame a spoken sentence does and the client needs no new message type. A
    missing or undecodable file is not fatal -- returns None and the utterance plays bare."""
    global _CHIME
    if _CHIME is None:
        _CHIME = np.zeros(0, dtype="float32")      # cache the failure too: don't retry per call
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                tmp = fh.name
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", CHIME_PATH,
                            "-ac", "1", "-ar", "24000", tmp], check=True)
            audio, _ = sf.read(tmp, dtype="float32")
            _CHIME = np.asarray(audio, dtype="float32").reshape(-1)
            log.info("[chime] %s loaded (%.2fs)", CHIME_PATH, len(_CHIME) / 24000)
        except Exception as e:                     # noqa: BLE001 -- chime is decoration, not the message
            log.warning("chime unavailable (%s): %s", CHIME_PATH, e)
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return _CHIME if len(_CHIME) else None


app = FastAPI()

# The dashboard frontend (Next.js dev server on :3000) is a different origin, so its
# fetches preflight; without this middleware every browser call to /auth/* and /profiles/*
# fails before reaching a handler. Bearer-header auth, not cookies, so no allow_credentials.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "ROBIN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["authorization", "content-type"],
)

# Dashboard HTTP API: /auth/* (login/logout) and /profiles/* (list, detail, patch, turns),
# authenticated by per-account Bearer tokens -- separate from the legacy X-Robin-Key gate on
# the pre-existing endpoints below.
from robin.api.admin import router as _robin_admin_router        # noqa: E402
from robin.api.auth import router as _robin_auth_router          # noqa: E402
from robin.api.profiles import router as _robin_profiles_router  # noqa: E402
app.include_router(_robin_auth_router)
app.include_router(_robin_profiles_router)
app.include_router(_robin_admin_router)


def _client_ip(ws: WebSocket) -> str:
    """Requests arrive via nginx, so the real client address is in X-Real-IP, not
    ws.client (that's the proxy)."""
    return ws.headers.get("x-real-ip", "unknown")


# The voice sockets no longer use the ROBIN_API_KEY shared secret: they authenticate with
# per-device tokens (robin.ws.bind_device_session), which travel in the same
# Sec-WebSocket-Protocol header the shared key used to -- never a query string, because a
# query string ends up verbatim in nginx's and uvicorn's default access logs on every
# connection, permanently persisting the credential in cleartext on disk. ROBIN_API_KEY
# still gates the pre-existing HTTP dashboard endpoints below.
def _check_http_auth(request: Request) -> bool:
    """Shared-key gate for the plain-HTTP dashboard API. The key travels as a custom
    header (X-Robin-Key), never a query string, for the same log-leak reason as above."""
    if not API_KEY:
        return True
    supplied = request.headers.get("x-robin-key", "")
    return hmac.compare_digest(supplied, API_KEY)


# --- Live session registry, for the /dashboard page. In-memory only (lost on restart) and
# touched exclusively from the event loop (the blocking STT/TTS/LLM work runs in threads but
# never mutates this dict directly), so no lock is needed. Capped so a long-running server
# doesn't grow this unboundedly. ---
# Latest clock_state frame per session (contract v2: the DEVICE owns timers/alarms and
# pushes its whole state up after `start`, after every mutation, and when a timer fires or
# stops ringing). We keep only the newest -- it is a full snapshot, not a delta -- and treat
# it as the single source of truth for answering and for resolving cancels.
SESSION_CLOCK = {}
# One pending future per session, resolved by the inbound `cancel_result` frame. The device
# is the only thing that knows whether a cancel actually applied, so Robin waits for its
# verdict before speaking rather than confirming optimistically -- an unverified "cancelled
# the tea timer" is the same class of lie as a phantom "timer set".
SESSION_CANCEL_WAITER = {}
CANCEL_RESULT_TIMEOUT_S = 2.0

# Latest location_state frame per session. Same contract shape as clock_state: the DEVICE owns
# it and pushes a full snapshot; we keep only the newest. Absent -> process_turn gets None and
# prompt_context falls back to its hardcoded default, which is the pre-existing behaviour.
#
# Coordinates are interpolated into a third-party URL (open-meteo) on weather turns, so they are
# validated here rather than at the point of use: a malformed frame must degrade to "no location"
# and never reach the request.
SESSION_LOCATION = {}

# Operator-configured fallback: ROBIN_LOCATION_LAT / ROBIN_LOCATION_LON (optionally
# ROBIN_LOCATION_NAME to skip the reverse-geocode lookup entirely).
#
# Three tiers, and the distinction matters:
#   1. device location_state  -> known, reported to the user
#   2. this configured value  -> known, reported to the user (an operator asserted it)
#   3. prompt_context's DEFAULT_LOCATION -> weather only, NEVER reported as the user's location
# A resident in an assisted-living facility is stationary, so (2) is the realistic source: it
# needs no GPS permission, no per-turn geocode of a moving person, and cannot silently break
# the way a browser permission prompt can.
def _configured_location():
    lat, lon = os.environ.get("ROBIN_LOCATION_LAT"), os.environ.get("ROBIN_LOCATION_LON")
    if not (lat and lon):
        return None
    loc = _parse_location({"lat": lat, "lon": lon})
    if loc and os.environ.get("ROBIN_LOCATION_NAME"):
        loc["name"] = os.environ["ROBIN_LOCATION_NAME"]
    return loc


def _parse_location(data):
    """-> {"lat": float, "lon": float} or None. Rejects non-numeric, NaN and out-of-range."""
    try:
        lat = float(data["lat"]); lon = float(data["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"lat": lat, "lon": lon}

CONFIGURED_LOCATION = _configured_location()
if CONFIGURED_LOCATION:
    log.info("configured location: lat=%.2f lon=%.2f name=%s", CONFIGURED_LOCATION["lat"],
             CONFIGURED_LOCATION["lon"], CONFIGURED_LOCATION.get("name", "<reverse-geocode>"))
else:
    log.info("no ROBIN_LOCATION_LAT/LON set — Robin will say it does not know the user's "
             "location unless the device sends a location_state frame")

SESSIONS = {}
MAX_SESSIONS_KEPT = 200


def _session_connect(session_id, mode, ip):
    SESSIONS[session_id] = {
        "id": session_id, "mode": mode, "ip": ip, "status": "active",
        "connected_at": time.time(), "last_activity": time.time(), "ended_at": None,
        "turns": 0, "last_transcript": "",
    }
    if len(SESSIONS) > MAX_SESSIONS_KEPT:
        ended = sorted((sid for sid, s in SESSIONS.items() if s["status"] == "ended"),
                        key=lambda sid: SESSIONS[sid]["ended_at"])
        for sid in ended[:len(SESSIONS) - MAX_SESSIONS_KEPT]:
            SESSIONS.pop(sid, None)


def _session_turn(session_id, transcript):
    s = SESSIONS.get(session_id)
    if s is None:
        return
    s["turns"] += 1
    s["last_activity"] = time.time()
    s["last_transcript"] = transcript


def _session_disconnect(session_id):
    s = SESSIONS.get(session_id)
    if s is None:
        return
    s["status"] = "ended"
    s["ended_at"] = time.time()


# --- Live handles for pushing an unprompted reply into a running session from the dashboard
# (see send_greet below). Keyed by session id, populated at connect and dropped at disconnect
# in each handler. SESSION_LOCK serializes every write to a session's socket -- a reply turn
# and an admin-triggered greet must never interleave their send_json/send_bytes calls, since
# both can be mid-flight from different asyncio tasks at once. ---
SESSION_WS = {}
SESSION_LOCK = {}
SESSION_HISTORY = {}
# session string -> robin.ws.BoundSession: the profile snapshot (voice, speech_rate,
# timezone, context) loaded once at connect, the server-minted session UUID, and the turn
# counter. Everything persistence needs, keyed by the same in-memory session id as the rest.
SESSION_DB = {}

# Set by send_greet, consumed by the next turn_start in _run_tap: a user who was just spoken
# to unprompted (rather than one who tapped the button themselves, already primed to speak)
# needs a longer no-speech grace period before VAD gives up and re-idles.
SESSION_GREET_PENDING = {}
GREET_ARM_TIMEOUT_S = 20.0

# --- Proactive delivery (POST /proactive): an external service pushes an utterance at one
# device. It addresses the device by external_id -- the caller's own stable per-user id, the
# same one robin_conversation.prompt_context keys profiles on -- because our session ids are
# generated at connect and the caller has no way to know them. Devices declare theirs on the
# existing `start` frame; SESSION_BY_EXTERNAL is an index into SESSIONS, not a second store.
# Anything for a device with no live session waits in PROACTIVE_QUEUE for its next connect. ---
SESSION_BY_EXTERNAL = {}                      # external_id -> session id (newest connect wins)
PROACTIVE_QUEUE = {}                          # external_id -> deque of pending messages, oldest first
PROACTIVE_QUEUE_MAX = 20                      # per device; overflow drops the OLDEST and says so
# Retry dedupe. The caller retries on error AND its sensor layer re-fires on the same event,
# so the same message_id arrives more than once; a repeat gets the first verdict back rather
# than a second spoken utterance.
PROACTIVE_SEEN = collections.OrderedDict()    # message_id -> (status, ts)
PROACTIVE_SEEN_MAX = 512
PROACTIVE_SEEN_TTL_S = 3600.0
# The caller stamps delivery_by with the CURRENT time (payload built in send_msg_to_ca), so a
# strict now > delivery_by test expires every message the instant it arrives. The grace window
# is what makes the field behave as the deadline it is meant to be instead of a hard reject.
DELIVERY_GRACE_S = 120.0


def _seen_status(message_id):
    """Status already returned for this message_id, or None. Evicts by TTL on the way past."""
    if not message_id:
        return None
    now = time.time()
    for mid, (_, ts) in list(PROACTIVE_SEEN.items()):   # oldest first; stop at the first live one
        if now - ts <= PROACTIVE_SEEN_TTL_S:
            break
        PROACTIVE_SEEN.pop(mid, None)
    rec = PROACTIVE_SEEN.get(message_id)
    return rec[0] if rec else None


def _remember_status(message_id, status):
    """Record (and return) the verdict for a message_id, keeping the cache bounded."""
    if message_id:
        PROACTIVE_SEEN[message_id] = (status, time.time())
        PROACTIVE_SEEN.move_to_end(message_id)
        while len(PROACTIVE_SEEN) > PROACTIVE_SEEN_MAX:
            PROACTIVE_SEEN.popitem(last=False)
    return status


def _is_expired(delivery_by, now=None):
    """True once delivery_by is more than DELIVERY_GRACE_S in the past. Absent or unparseable
    means no deadline: dropping a message over a malformed timestamp is worse than speaking
    it a little late."""
    if not delivery_by:
        return False
    try:
        dt = datetime.datetime.fromisoformat(str(delivery_by).replace("Z", "+00:00"))
    except ValueError:
        log.warning("proactive: unparseable delivery_by %r -- treating as no deadline", delivery_by)
        return False
    if dt.tzinfo is None:                          # naive timestamp: the caller sends UTC
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return (now if now is not None else time.time()) > dt.timestamp() + DELIVERY_GRACE_S


def _enqueue_proactive(external_id, msg):
    """Hold a message for this device's next connect. Bounded per device; at the cap the
    OLDEST is dropped, on the grounds that a stale nudge is the one worth losing."""
    q = PROACTIVE_QUEUE.setdefault(external_id, collections.deque(maxlen=PROACTIVE_QUEUE_MAX))
    if len(q) == PROACTIVE_QUEUE_MAX:
        log.warning("proactive queue full for external_id=%s -- dropping oldest message_id=%s",
                    external_id, q[0].get("message_id"))
    q.append(msg)
    log.info("proactive queued external_id=%s message_id=%s depth=%d",
             external_id, msg.get("message_id"), len(q))


def _bind_external(session_id, external_id):
    """Record the device's own id on its existing session record and index it. Returns True
    the first time a session declares one -- the caller then drains that device's queue."""
    external_id = str(external_id or "").strip()
    if not external_id or SESSIONS.get(session_id, {}).get("external_id") == external_id:
        return False
    if session_id in SESSIONS:
        SESSIONS[session_id]["external_id"] = external_id
    SESSION_BY_EXTERNAL[external_id] = session_id
    log.info("session=%s bound external_id=%s", session_id, external_id)
    return True


def _proactive_targets():
    """Every live session: one proactive message is spoken by every Robin that is listening.

    This is the dashboard greet button without having to pick a session first, which is what
    the deployment needs -- there is one Robin in the room, and no client sends an external_id
    for it to be addressed by anyway. external_id stays the queue key and the log label, not
    an address. The cost, deliberately accepted: with more than one device connected, a
    message meant for one person is spoken in every room, so this wants revisiting before the
    same server serves two patients at once."""
    return list(SESSION_WS)


def _bind_and_drain(session_id, external_id):
    """Handle a `start` frame: record the device's id if it declared one (the dashboard and
    the log are the only readers), then play whatever queued up while nothing was listening.
    `start` is the device's own readiness signal, so it is the moment to drain. The drain runs
    as its own task: it speaks for seconds and the socket's receive loop must keep reading
    meanwhile, the same reason turns run via _spawn_turn rather than inline."""
    _bind_external(session_id, external_id)
    return asyncio.create_task(drain_proactive(session_id))


@app.get("/health")
async def health():
    # Unauthenticated by design, so monitoring can poll it without a key.
    return {"status": "ok", "stt_loaded": stt_model is not None,
            "tts_loaded": pipe is not None, "turn_mode": TURN_MODE}


@app.get("/")
async def index():
    # Default = tap-to-talk (VAD endpointing). Old hold demos stay at /stream and /classic.
    page = "tap_index.html" if TURN_MODE == "tap" else "stream_index.html"
    return FileResponse(os.path.join(HERE, page))


@app.get("/classic")
async def classic_index():
    return FileResponse(os.path.join(HERE, "index.html"))          # non-streaming push-to-talk (/ws)


@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(os.path.join(HERE, "dashboard.html"))


@app.get("/api/sessions")
async def api_sessions(request: Request):
    if not _check_http_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    items = sorted(SESSIONS.values(), key=lambda s: s["connected_at"], reverse=True)
    return {"now": time.time(), "sessions": items}


@app.post("/api/sessions/{session_id}/greet")
async def api_greet_session(session_id: str, request: Request):
    """Dashboard button: make Robin speak a message (custom, or GREET_TEXT if none given)
    unprompted into a live tap session, which re-arms VAD for free via the client's own
    auto-continue (see send_greet). Only tap-mode sessions have a VAD arm state to re-enter,
    so anything else is rejected."""
    if not _check_http_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    s = SESSIONS.get(session_id)
    if s is None or s["status"] != "active":
        return JSONResponse({"error": "session is not active"}, status_code=404)
    if s["mode"] != "tap":
        return JSONResponse({"error": "greet is only supported for tap-mode sessions"}, status_code=400)
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    text = (body.get("text") or "").strip()[:500] or GREET_TEXT
    if not await send_greet(session_id, text):
        return JSONResponse({"error": "session socket unavailable"}, status_code=409)
    return {"status": "ok"}


@app.post("/proactive")
async def api_proactive(request: Request):
    """External service -> one utterance spoken on one device, addressed by external_id.

    Body: service, message_id, external_id, utterance, message_type, severity, occurred_at,
    delivery_by, require_affirmation, message. Unknown fields are ignored, and
    require_affirmation is accepted but deliberately unused in this pass.

    The utterance is spoken by EVERY live session, not one addressed by external_id (see
    _proactive_targets); external_id is the queue key for messages that arrive while nothing
    is listening, and the label they are logged under.

    Always answers with the message_id and a status the caller can log, never a bare 200:
      spoken  -- a live session heard it; the audio has been sent
      queued  -- no live session, or the send died part way through; held for the next connect
      expired -- delivery_by is more than DELIVERY_GRACE_S past; not spoken, not queued

    Gated on the same shared key as the dashboard API (X-Robin-Key), because this is the one
    route that puts words in Robin's mouth in someone's room: unauthenticated, anyone who can
    reach the port could speak at any bound device, and probe which external_ids are live by
    reading spoken vs queued back. The caller sends the key as a header for the same reason
    the WebSocket does -- a query string persists the shared secret in nginx's access log."""
    if not _check_http_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:                                   # noqa: BLE001 -- any malformed body
        return JSONResponse({"error": "body must be JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    message_id = str(body.get("message_id") or uuid.uuid4().hex)
    external_id = str(body.get("external_id") or "").strip()
    # `message` and `utterance` are the same string in the caller's payload; utterance is the
    # one that is spoken, so prefer it and fall back rather than going silent on a typo.
    utterance = str(body.get("utterance") or body.get("message") or "").strip()
    if not utterance:
        return JSONResponse({"message_id": message_id, "status": "rejected",
                             "error": "utterance is empty"}, status_code=422)
    if not external_id:                                 # nothing to address, nothing to queue under
        return JSONResponse({"message_id": message_id, "status": "rejected",
                             "error": "external_id is required"}, status_code=422)

    prior = _seen_status(message_id)
    if prior is not None:
        log.info("proactive duplicate message_id=%s -> %s", message_id, prior)
        return {"message_id": message_id, "status": prior, "duplicate": True}

    msg = {"message_id": message_id, "external_id": external_id, "utterance": utterance,
           "service": body.get("service"), "message_type": body.get("message_type"),
           "severity": body.get("severity"), "occurred_at": body.get("occurred_at"),
           "delivery_by": body.get("delivery_by"),
           "require_affirmation": bool(body.get("require_affirmation")),   # accepted, unused
           "received_at": time.time()}

    if _is_expired(msg["delivery_by"]):
        log.info("proactive expired external_id=%s message_id=%s delivery_by=%s",
                 external_id, message_id, msg["delivery_by"])
        return {"message_id": message_id, "status": _remember_status(message_id, "expired")}

    heard = []
    for session_id in _proactive_targets():
        try:
            if await deliver_proactive(session_id, msg):
                heard.append(session_id)
        except Exception as e:      # noqa: BLE001 -- socket died mid-utterance: it was NOT heard
            log.warning("proactive send failed session=%s message_id=%s: %r", session_id, message_id, e)
    if heard:                       # one session hearing it is delivery; the rest are logged above
        return {"message_id": message_id,
                "status": _remember_status(message_id, "spoken"), "sessions": heard}
    _enqueue_proactive(external_id, msg)
    return {"message_id": message_id, "status": _remember_status(message_id, "queued")}


async def send_sentence(ws, sentence, reply_chunks, idx, *, voice=VOICE, speed=1.0):
    """Synthesize one sentence, send the audio frame, accumulate for the saved reply wav.
    An `audio_meta` control message is sent immediately before the binary frame so the client
    can pair server-side send timing with each chunk. Returns synth seconds."""
    await ws.send_json({"type": "reply", "text": sentence})
    t = time.perf_counter()
    audio = await asyncio.to_thread(synth_audio, sentence, voice, speed)
    dt = time.perf_counter() - t
    reply_chunks.append(audio)
    data = wav_bytes(audio)
    await ws.send_json({"type": "audio_meta", "idx": idx, "server_send_ts": time.perf_counter(),
                        "bytes": len(data), "sample_rate": 24000})
    await ws.send_bytes(data)
    log.info("    tts %.2fs (%.2fs audio)  %r", dt, len(audio) / 24000, sentence)
    return dt


def save_reply(reply_chunks, tid):
    """Concatenate the spoken reply chunks to log/<tid>.reply.wav (if any). Returns the path."""
    reply_wav = os.path.join(LOG_DIR, f"{tid}.reply.wav")
    if reply_chunks:
        sf.write(reply_wav, np.concatenate(reply_chunks), 24000, subtype="PCM_16")
    return reply_wav


_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# STT on a clip that was only room noise comes back empty or as a bare filler ("mm", "uh").
# Those used to be routed like any other turn, so the LLM answered the silence -- "Is there
# something on your mind? I'm here to listen." -- an entire unasked-for spoken turn, most
# jarring right after a goodbye, where it reads as Robin refusing to leave. There is nothing
# to reply to: report an empty transcript and end the turn without speaking.
_NOISE_ONLY = re.compile(r"^[\W_]*(m+|h+m+|u+h+|u+m+|a+h+|e+h+|h+u+h+)?[\W_]*$", re.IGNORECASE)


def is_noise_transcript(text):
    return bool(_NOISE_ONLY.match(text or ""))


def _profile_process_turn(history, transcript, bound, *, clock_state, location_coordinates):
    """process_turn with the connection's DB profile injected. Runs in a worker thread.

    The prompt context is prebuilt here so `personal_data_profile` comes from the profile
    row's jsonb `context` (loaded once at connect) instead of prompt_context's legacy
    profiles/<id>.json file lookup, and so the temporal context uses the profile's IANA
    timezone. clock_state still goes to process_turn, which merges it into the context."""
    context = None
    user_id = "user"
    if bound is not None:
        user_id = str(bound.profile_id)
        try:
            tz = ZoneInfo(bound.timezone)
        except Exception:                           # bad tz name must not kill the turn
            tz = None
        context = build_conversation_prompt_context(user_id, location_coordinates=location_coordinates,
                                                    tz=tz)
        context["personal_data_profile"] = json.dumps(bound.context)
    return process_turn(history, transcript, user_id=user_id, context=context,
                        clock_state=clock_state, location_coordinates=location_coordinates)


async def robin_reply(ws, history, transcript, turn_id, user_id):
    """Robin's reply: robin_conversation.process_turn() -> sentence TTS. Sends the same control
    messages (turn_start / reply / audio_meta / done) the client and playback telemetry expect;
    the LLM call is a blocking full reply (no token streaming), split into sentences after the
    fact. Returns (full_reply, reply_chunks, ttft, ttfs, tts_total, llm_dt, should_end_session)
    -- ttft equals llm_dt since there's no streaming first-token to distinguish it from.

    process_turn's "client_actions" (set_timer / set_alarm / show_timers / show_alarms) are
    forwarded verbatim as control frames; the device owns the countdown, not this server.

    `done`'s "ending" field carries process_turn's should_end_session verdict (the classifier's
    end_conversation intent -- "goodbye", "that's all for now", etc.) so the client knows not
    to auto-continue listening after this reply; the server only reports the signal, it never
    closes anything itself.

    Runs behind SESSION_LOCK[user_id]: send_greet (triggered from the dashboard) writes to the
    same socket from a separate request, and the two must never interleave their frames."""
    lock = SESSION_LOCK.get(user_id)
    if lock is None:                             # session torn down mid-turn; keep talking
        lock = SESSION_LOCK[user_id] = asyncio.Lock()
    bound = SESSION_DB.get(user_id)              # profile snapshot; None only for greet races
    voice = bound.voice if bound else VOICE
    speed = bound.speech_rate if bound else 1.0
    async with lock:
        await ws.send_json({"type": "turn_start", "turn_id": turn_id})
        t_llm = time.perf_counter()
        result = await asyncio.to_thread(
            functools.partial(_profile_process_turn, history, transcript, bound,
                              clock_state=SESSION_CLOCK.get(user_id),
                              location_coordinates=SESSION_LOCATION.get(user_id) or CONFIGURED_LOCATION))
        llm_dt = time.perf_counter() - t_llm
        reply = result["reply"]
        should_end = bool(result.get("should_end_session"))
        # Timer/alarm control frames, one per requested timer, sent before the reply
        # frames so the device starts the clock as Robin begins speaking. Inside the
        # lock for the same reason the rest of the turn is: send_greet must not
        # interleave between a frame and its spoken confirmation.
        actions = result.get("client_actions") or []
        awaits_cancel = bool(result.get("awaits_cancel_result")) and actions
        if awaits_cancel:                       # arm BEFORE sending, or the reply can race us
            SESSION_CANCEL_WAITER[user_id] = asyncio.get_running_loop().create_future()
        for action in actions:
            await ws.send_json(action)
            log.info("turn %s  client_action %s", turn_id, action)
        if awaits_cancel:
            reply, ok = await _await_cancel_result(user_id, result, turn_id)
            # An edit is cancel-then-recreate: only recreate once the device confirms the
            # cancel applied, or a failed cancel would leave the user with two alarms.
            if ok:
                for action in result.get("actions_after_ok") or []:
                    await ws.send_json(action)
                    log.info("turn %s  client_action %s", turn_id, action)
            # process_turn already logged the optimistic wording; correct it so a later turn
            # reasons from what Robin actually said, not from what it hoped to say.
            if history and history[-1].get("role") == "assistant":
                history[-1]["content"] = reply
        reply_chunks, tts_total, idx, ttfs = [], 0.0, 0, None
        for sentence in _SENT_RE.split(reply.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            if ttfs is None:
                ttfs = time.perf_counter() - t_llm
            tts_total += await send_sentence(ws, sentence, reply_chunks, idx,
                                             voice=voice, speed=speed)
            idx += 1
        # If Robin just ASKED something, the user needs thinking time before answering.
        # Reuse the greet path's longer arm window: live, "What time should that alarm go
        # off?" was followed by ARM_TIMEOUT five seconds later, so the question could not be
        # answered at all and the turn was wasted.
        if reply.strip().endswith("?") and not should_end:
            SESSION_GREET_PENDING[user_id] = True
        await ws.send_json({"type": "done", "ending": should_end})
    return reply, reply_chunks, llm_dt, ttfs, tts_total, llm_dt, should_end


async def _await_cancel_result(user_id, result, turn_id):
    """Block briefly on the device's cancel_result -> (wording to speak, applied?).
    A timeout is treated as failure, deliberately: if we never heard back we do not know the
    cancel applied, and claiming it did is exactly the failure mode this handshake exists to
    prevent."""
    fut = SESSION_CANCEL_WAITER.get(user_id)
    try:
        res = await asyncio.wait_for(fut, CANCEL_RESULT_TIMEOUT_S)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        res = None
    finally:
        SESSION_CANCEL_WAITER.pop(user_id, None)

    if res is None:
        log.warning("turn %s  cancel_result timed out after %.1fs", turn_id, CANCEL_RESULT_TIMEOUT_S)
        return (result.get("reply_on_fail") or result["reply"]), False
    log.info("turn %s  cancel_result %s", turn_id, res)
    if not res.get("ok"):
        return (result.get("reply_on_fail") or result["reply"]), False
    noun = result.get("cancel_all_noun")
    if noun:                                     # all-cancel: the device owns the real count
        from robin_conversation.clock import all_cancel_reply
        return all_cancel_reply(res.get("count"), noun), True
    return result["reply"], True


GREET_TEXT = "How is your day going?"


async def _speak_unprompted(session_id, text, tid, *, chime=False, suppress=False):
    """Speak `text` unprompted into an already-connected session, exactly as a normal reply
    would (turn_start / reply / audio_meta / done) -- indistinguishable to the client from a
    real turn. The client's own auto-continue (tap_index.html's rearm(), fired on `done`)
    then re-sends {type: "turn_start"}, which is what actually arms VAD on the server;
    nothing here has to poke the VAD state machine directly.

    Shared by send_greet and deliver_proactive, and the synthesis itself is send_sentence --
    the same TTS path every ordinary reply goes through, so there is exactly one.

    `suppress` brackets the utterance in listen_suppress / listen_resume control frames, so a
    device that runs its own wake word or onset detection can stand down while Robin speaks
    and re-arm afterwards instead of hearing Robin's own voice as a user turn. `chime` plays
    CHIME_PATH first, as an ordinary audio frame.

    Returns False if the session has no live socket (already disconnected), and RAISES if the
    socket dies mid-utterance -- a caller that must know whether the user actually heard this
    has to treat that as undelivered."""
    ws = SESSION_WS.get(session_id)
    if ws is None:
        return False
    bound = SESSION_DB.get(session_id)
    voice = bound.voice if bound else VOICE
    speed = bound.speech_rate if bound else 1.0
    async with SESSION_LOCK.setdefault(session_id, asyncio.Lock()):
        await ws.send_json({"type": "turn_start", "turn_id": tid})
        if suppress:
            await ws.send_json({"type": "listen_suppress", "reason": "proactive", "turn_id": tid})
        reply_chunks, idx = [], 0
        if chime and await send_chime(ws, idx):
            idx += 1
        for sentence in _SENT_RE.split(text.strip()):
            sentence = sentence.strip()
            if sentence:
                await send_sentence(ws, sentence, reply_chunks, idx, voice=voice, speed=speed)
                idx += 1
        if suppress:
            await ws.send_json({"type": "listen_resume", "reason": "proactive", "turn_id": tid})
        await ws.send_json({"type": "done", "ending": False})
    await persist_turn(bound, role="assistant", content=text, source="proactive",
                       meta={"tid": tid, "chime": chime})
    SESSION_GREET_PENDING[session_id] = True                             # longer arm timeout
    history = SESSION_HISTORY.get(session_id)                            # for the rearm this
    if history is not None:                                              # triggers
        history.append({"role": "assistant", "content": text})           # so the next real
    return True                                                          # LLM turn has context


async def send_chime(ws, idx):
    """Send the chime as an ordinary audio frame (audio_meta + binary), identical in shape to
    a spoken sentence, so the client's existing sequential playback queues it straight ahead
    of the speech. False when the chime could not be decoded -- never a reason not to speak."""
    audio = chime_audio()
    if audio is None:
        return False
    data = wav_bytes(audio)
    await ws.send_json({"type": "audio_meta", "idx": idx, "server_send_ts": time.perf_counter(),
                        "bytes": len(data), "sample_rate": 24000, "chime": True})
    await ws.send_bytes(data)
    return True


async def send_greet(session_id, text=GREET_TEXT):
    """Dashboard greet: speak `text` into a live tap session. See _speak_unprompted."""
    if not await _speak_unprompted(session_id, text, f"{session_id}_greet{int(time.time())}"):
        return False
    _session_turn(session_id, f"[Robin, unprompted] {text}")
    log.info("greet  session=%s  %r", session_id, text)
    return True


async def deliver_proactive(session_id, msg):
    """Speak one proactive message into a live session: chime, then the utterance, bracketed
    by the listen_suppress / listen_resume frames. False if the socket is already gone;
    RAISES if it dies mid-utterance, so the caller can requeue rather than claim delivery."""
    utterance = msg["utterance"]
    tid = f"{session_id}_proactive{int(time.time())}"
    if not await _speak_unprompted(session_id, utterance, tid, chime=True, suppress=True):
        return False
    _session_turn(session_id, f"[Robin, proactive] {utterance}")
    log.info("proactive spoken session=%s external_id=%s message_id=%s service=%s type=%s %r",
             session_id, msg.get("external_id"), msg.get("message_id"),
             msg.get("service"), msg.get("message_type"), utterance)
    return True


async def drain_proactive(session_id):
    """Play what is waiting into a session that has just declared itself ready, oldest first.

    Every queue, not one: delivery is to whoever is listening (see _proactive_targets), so the
    external_id a message was filed under does not decide who hears it. Expired messages are
    skipped, logged, never spoken. The first failure stops the drain with the message put BACK
    at the head of its queue, so a socket that dies halfway through costs nothing but the
    delay -- and each message is popped before it is spoken, so two `start` frames racing
    cannot speak the same message twice."""
    for key in list(PROACTIVE_QUEUE):
        q = PROACTIVE_QUEUE.get(key)
        if not q:
            continue
        log.info("proactive drain session=%s external_id=%s depth=%d", session_id, key, len(q))
        while q:
            msg = q.popleft()                     # popped first: a racing drain cannot re-speak it
            if _is_expired(msg["delivery_by"]):
                _remember_status(msg["message_id"], "expired")
                log.info("proactive drain: skipped expired message_id=%s", msg["message_id"])
                continue
            try:
                if not await deliver_proactive(session_id, msg):
                    q.appendleft(msg)             # socket already gone: put it back, in order
                    return
            except Exception as e:                # noqa: BLE001 -- died mid-utterance; do not lose it
                q.appendleft(msg)
                log.warning("proactive drain stopped at message_id=%s: %r", msg["message_id"], e)
                return
            _remember_status(msg["message_id"], "spoken")
        PROACTIVE_QUEUE.pop(key, None)            # drained: don't accumulate dead device ids


def _append_telemetry(session, obj):
    """Append one client-telemetry turn record to logs/telemetry/<session>.jsonl. Runs off the
    event loop (via to_thread) and swallows errors — telemetry must never break the audio path."""
    try:
        with open(os.path.join(TELEMETRY_DIR, f"{session}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
    except Exception as e:                                  # noqa: BLE001 — best-effort logging only
        log.warning("telemetry write failed: %s", e)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # Device-token auth (robin.ws): the token rides the Sec-WebSocket-Protocol header the
    # same way the old shared key did, is verified against its stored SHA-256, must be
    # kind='device', and resolves the profile server-side. Rejection is accept-then-close
    # (see bind_device_session) with code 4401. The old ROBIN_API_KEY no longer grants
    # voice-socket access.
    bound = await bind_device_session(ws)
    if bound is None:
        log.warning("rejected /ws connection from %s: bad, revoked, or non-device token",
                    _client_ip(ws))
        return
    session = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"   # suffix: second resolution alone collides
    log.info("ws connected  session=%s profile=%d db_session=%s", session, bound.profile_id,
             bound.session_id)
    _session_connect(session, "classic", _client_ip(ws))
    history = []                                              # per-connection conversation memory
    SESSION_WS[session] = ws
    SESSION_LOCK[session] = asyncio.Lock()
    SESSION_HISTORY[session] = history
    SESSION_DB[session] = bound
    turn = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:                 # control frame (client telemetry)
                data = json.loads(msg["text"])
                if data.get("type") == "client_telemetry":
                    # fire-and-forget: never block the next turn's audio on a disk write
                    asyncio.create_task(asyncio.to_thread(_append_telemetry, session, data))
                continue
            if msg.get("bytes") is None:
                continue
            audio_bytes = msg["bytes"]
            turn += 1
            tid = f"{session}_t{turn:02d}"
            t_turn = time.perf_counter()

            # --- STT: save the decoded input wav (what STT actually heard), transcribe ---
            in_wav = os.path.join(LOG_DIR, f"{tid}.in.wav")
            t = time.perf_counter()
            await asyncio.to_thread(decode_to_wav, audio_bytes, in_wav)
            transcript = await asyncio.to_thread(stt_transcribe, in_wav)
            stt_dt = time.perf_counter() - t
            dur = sf.info(in_wav).duration
            log.info("turn %d  audio=%dB (%.2fs) -> %s", turn, len(audio_bytes), dur, in_wav)
            log.info("turn %d  stt %.2fs -> %r", turn, stt_dt, transcript)
            _session_turn(session, transcript)
            await ws.send_json({"type": "transcript", "text": transcript})

            if is_noise_transcript(transcript):
                log.info("turn %d  noise-only transcript %r -- no reply", turn, transcript)
                await ws.send_json({"type": "done", "ending": False})
                continue

            await persist_turn(bound, role="user", content=transcript,
                               meta={"tid": tid, "stt_s": round(stt_dt, 3),
                                     "audio_s": round(dur, 2)})
            full, reply_chunks, ttft, ttfs, tts_total, llm_dt, ending = await robin_reply(
                ws, history, transcript, tid, session)
            await persist_turn(bound, role="assistant", content=full.strip(),
                               meta={"tid": tid, "model": MODEL, "backend": ARGS.backend,
                                     "llm_s": round(llm_dt, 2), "tts_s": round(tts_total, 2)})

            reply_wav = save_reply(reply_chunks, tid)
            turn_total = time.perf_counter() - t_turn
            log.info("turn %d  reply %r -> %s", turn, full.strip(), reply_wav)
            log.info("turn %d  timings: stt=%.2f ttft=%s ttfs=%s tts=%.2f llm=%.2f total=%.2f",
                     turn, stt_dt, _f(ttft), _f(ttfs), tts_total, llm_dt, turn_total)
            if ending:
                log.info("turn %d  user signaled end_conversation", turn)
    except WebSocketDisconnect:
        log.info("ws disconnected  session=%s turns=%d", session, turn)
    finally:
        _session_disconnect(session)
        SESSION_WS.pop(session, None)
        SESSION_LOCK.pop(session, None)
        SESSION_HISTORY.pop(session, None)
        SESSION_DB.pop(session, None)
        SESSION_CLOCK.pop(session, None)
        SESSION_LOCATION.pop(session, None)
        SESSION_CANCEL_WAITER.pop(session, None)


PARTIAL_EVERY_S = 0.4   # seconds of new audio between live partial transcripts


def pcm16_to_wav16(samples, sr, path):
    """float32 PCM samples at `sr` -> 16 kHz mono wav at `path` (resample only if needed)."""
    if sr != 16000:
        samples = torchaudio.functional.resample(torch.from_numpy(samples), sr, 16000).numpy()
    sf.write(path, samples, 16000, subtype="PCM_16")
    return path


@app.get("/stream")
async def stream_index():
    return FileResponse(os.path.join(HERE, "stream_index.html"))


def _spawn_turn(ws, session, audio16k, sr, history, tid, stages):
    """Run one turn as its own task so the receive loop keeps reading.

    This matters for more than tidiness: the turn AWAITS the device's cancel_result, and that
    frame arrives on the very socket the loop reads. Running the turn inline meant the loop
    was blocked on it, the frame was never read, and every cancel timed out into "I couldn't
    cancel that" -- including successful ones. It also stops a mid-turn exception (a dead LLM
    gateway, say) from escaping the endpoint and dropping the socket with no close frame."""
    async def runner():
        try:
            await _finalize_turn(ws, session, audio16k, sr, history, tid, stages)
        except Exception as e:                       # noqa: BLE001 -- must not kill the socket
            log.error("turn %s failed: %r", tid, e)
            try:                                     # never leave the client stuck on "thinking"
                await ws.send_json({"type": "error", "text": "Sorry, something went wrong."})
                await ws.send_json({"type": "done", "ending": False})
            except Exception:                        # noqa: BLE001 -- socket already gone
                pass
    return asyncio.create_task(runner())


async def _finalize_turn(ws, session, audio16k, sr, history, tid, stages):
    """Shared turn finalize (tap and hold): transcribe the utterance, send the transcript, run
    Robin's reply, save it, and write the per-turn VAD bench record. `stages` carries the raw
    pre-STT timestamps the caller already filled (t_turn_start, t_speech_onset, t_eou). STT/LLM/
    TTS are called exactly as before — this only threads instrumentation through."""
    if audio16k is None or len(audio16k) == 0:                  # never silently drop: report empty
        await ws.send_json({"type": "transcript", "text": ""})
        await ws.send_json({"type": "done", "ending": False})
        return
    in_wav = os.path.join(LOG_DIR, f"{tid}.in.wav")
    t = time.perf_counter()
    await asyncio.to_thread(pcm16_to_wav16, audio16k, sr, in_wav)
    transcript = await asyncio.to_thread(stt_transcribe, in_wav)
    stages["t_stt_final"] = time.perf_counter()
    log.info("turn %s  stt(final) %.2fs audio=%.2fs -> %r", tid,
             stages["t_stt_final"] - t, len(audio16k) / sr, transcript)
    await ws.send_json({"type": "transcript", "text": transcript})

    if is_noise_transcript(transcript):
        log.info("turn %s  noise-only transcript %r -- no reply", tid, transcript)
        await ws.send_json({"type": "done", "ending": False})
        return

    bound = SESSION_DB.get(session)
    user_meta = {"tid": tid, "stt_s": round(stages["t_stt_final"] - t, 3),
                 "audio_s": round(len(audio16k) / sr, 2)}
    if stages.get("vad_speech_duration_ms") is not None:
        user_meta["vad_speech_ms"] = stages["vad_speech_duration_ms"]
    await persist_turn(bound, role="user", content=transcript, meta=user_meta)

    t_reply = time.perf_counter()
    full, reply_chunks, ttft, ttfs, tts_total, llm_dt, ending = await robin_reply(
        ws, history, transcript, tid, session)
    stages["t_llm_first_token"] = (t_reply + ttft) if ttft is not None else None
    stages["t_tts_first_frame"] = (t_reply + ttfs) if ttfs is not None else None

    # latency_ms is the spec's definition: end of user speech (VAD EOU) to start of TTS.
    latency_ms = None
    if stages.get("t_eou") is not None and stages.get("t_tts_first_frame") is not None:
        latency_ms = int((stages["t_tts_first_frame"] - stages["t_eou"]) * 1000)
    await persist_turn(bound, role="assistant", content=full.strip(), latency_ms=latency_ms,
                       meta={"tid": tid, "model": MODEL, "backend": ARGS.backend,
                             "llm_s": round(llm_dt, 2), "tts_s": round(tts_total, 2)})

    reply_wav = save_reply(reply_chunks, tid)
    log.info("turn %s  reply (tts=%.2f llm=%.2f) %r -> %s", tid, tts_total, llm_dt, full.strip(), reply_wav)
    if ending:
        log.info("turn %s  user signaled end_conversation", tid)
    _session_turn(session, transcript)
    rec = {"session": session, "tid": tid, "backend": ARGS.backend,
           "transcript": transcript, "reply": full.strip(), **stages}
    asyncio.create_task(asyncio.to_thread(_append_bench, f"vad_{session}", rec))


@app.websocket("/ws-stream")
async def ws_stream_endpoint(ws: WebSocket):
    """Tap-to-talk (VAD endpointing) or hold-to-talk, per config.turn_mode(). Both reuse the
    resident models and the shared reply/finalize path; only turn-boundary detection differs.
    turn_mode='hold' is the latency-matrix control that isolates VAD hangover from the pipeline."""
    # Same device-token auth as /ws (see ws_endpoint): profile resolved from the token,
    # never from anything the client sends; failure closes with 4401.
    bound = await bind_device_session(ws)
    if bound is None:
        log.warning("rejected /ws-stream connection from %s: bad, revoked, or non-device token",
                    _client_ip(ws))
        return
    session = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"   # suffix: second resolution alone collides
    log.info("ws-stream connected  session=%s profile=%d db_session=%s turn_mode=%s",
             session, bound.profile_id, bound.session_id, TURN_MODE)
    _session_connect(session, TURN_MODE, _client_ip(ws))
    history = []                                              # per-connection conversation memory
    SESSION_WS[session] = ws
    SESSION_LOCK[session] = asyncio.Lock()
    SESSION_HISTORY[session] = history
    SESSION_DB[session] = bound
    try:
        if TURN_MODE == "tap":
            await _run_tap(ws, session, history)
        else:
            await _run_hold(ws, session, history)
    finally:
        ext = SESSIONS.get(session, {}).get("external_id")
        if ext and SESSION_BY_EXTERNAL.get(ext) == session:   # a reconnect may already own it
            SESSION_BY_EXTERNAL.pop(ext, None)
        _session_disconnect(session)
        SESSION_WS.pop(session, None)
        SESSION_LOCK.pop(session, None)
        SESSION_HISTORY.pop(session, None)
        SESSION_DB.pop(session, None)
        SESSION_CLOCK.pop(session, None)
        SESSION_LOCATION.pop(session, None)
        SESSION_CANCEL_WAITER.pop(session, None)
        SESSION_GREET_PENDING.pop(session, None)


async def _run_hold(ws, session, history):
    """hold-to-talk: {start} on button-down, PCM frames, {end}=EOU (exactly as before). Live
    partials via O(n^2) full re-decode. Kept intact as the control path; only instrumentation
    is added around it."""
    turn, sr = 0, 16000
    buf, n_samples, last_partial, t_turn_start = [], 0, 0, None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                typ = data.get("type")
                if typ == "start":
                    sr = int(data.get("sampleRate", 16000))
                    buf, n_samples, last_partial = [], 0, 0
                    t_turn_start = time.perf_counter()
                    # The device names itself here; this is also its readiness signal, so
                    # anything queued for it while it was away plays now (see _bind_and_drain).
                    _bind_and_drain(session, data.get("external_id"))
                elif typ == "cancel_result":
                    waiter = SESSION_CANCEL_WAITER.get(session)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(data)
                elif typ == "location_state":
                    loc = _parse_location(data)
                    if loc is None:
                        log.warning("location_state session=%s rejected (bad coords: %r)",
                                    session, {k: data.get(k) for k in ("lat", "lon")})
                    else:
                        SESSION_LOCATION[session] = loc
                        # Coarse in the log on purpose: 2 dp is ~1 km, enough to debug "is it
                        # using the right city" without writing a precise home address to disk.
                        log.info("location_state session=%s lat=%.2f lon=%.2f source=%s",
                                 session, loc["lat"], loc["lon"], data.get("source", "?"))
                elif typ == "clock_state":
                    # Full snapshot from the device; replaces whatever we had.
                    SESSION_CLOCK[session] = data
                    # Log the alarms in full: counts alone made a "did the repeat days
                    # survive?" question impossible to answer from the log afterwards.
                    log.info("clock_state session=%s timers=%d alarms=%s ringing=%s", session,
                             len(data.get("timers") or []),
                             [{k: a.get(k) for k in ("id", "hour", "minutes", "days", "label")}
                              for a in (data.get("alarms") or [])],
                             bool(data.get("ringing")))
                elif typ == "end":
                    turn += 1
                    tid = f"{session}_t{turn:02d}"
                    stages = {"turn_uuid": uuid.uuid4().hex, "turn_mode": "hold",
                              "t_turn_start": t_turn_start, "t_speech_onset": None,
                              "t_eou": time.perf_counter(),
                              "vad_speech_duration_ms": None, "vad_hangover_used_ms": None}
                    audio = np.concatenate(buf) if buf else None
                    _spawn_turn(ws, session, audio, sr, history, tid, stages)
                    buf, n_samples, last_partial = [], 0, 0
            elif msg.get("bytes") is not None:
                frame = np.frombuffer(msg["bytes"], dtype="<i2").astype("float32") / 32768.0
                buf.append(frame)
                n_samples += len(frame)
                if n_samples - last_partial >= PARTIAL_EVERY_S * sr:
                    last_partial = n_samples
                    partial_wav = os.path.join(LOG_DIR, "_stream_partial.wav")
                    await asyncio.to_thread(pcm16_to_wav16, np.concatenate(buf), sr, partial_wav)
                    partial = await asyncio.to_thread(stt_transcribe, partial_wav)
                    log.info("    partial %.1fs -> %r", n_samples / sr, partial)
                    await ws.send_json({"type": "partial", "text": partial})
    except WebSocketDisconnect:
        pass
    log.info("ws-stream(hold) disconnected  session=%s turns=%d", session, turn)


async def _run_tap(ws, session, history):
    """tap-to-talk: mic streams continuously; {turn_start} arms; Silero VAD decides EOU. The
    prespeech ring recovers the clipped onset. VAD runs during IDLE too (barge-in later). Fails
    to a safe state — a per-connection SileroVAD init failure degrades this connection to hold
    rather than dropping audio."""
    from vad.ingest import VadIngest
    from vad.silero import SileroVAD
    from vad.turn import Event

    params = config.vad_params()
    try:
        vad = SileroVAD(providers=VAD_PROVIDERS)
    except Exception as e:                                      # model missing / ORT init failure
        log.warning("tap: SileroVAD init failed (%s) — degrading this connection to hold", e)
        await ws.send_json({"type": "vad_error", "text": "VAD unavailable, using hold"})
        if session in SESSIONS:
            SESSIONS[session]["mode"] = "hold-fallback"
        return await _run_hold(ws, session, history)

    ingest, sr = None, 16000
    turn, tid, stages = 0, None, None
    prev_state, last_ui = None, 0.0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                typ = data.get("type")
                if typ == "start":
                    sr = int(data.get("sampleRate", 16000))
                    ingest = VadIngest(vad, params, sr)
                    # The device names itself here; this is also its readiness signal, so
                    # anything queued for it while it was away plays now (see _bind_and_drain).
                    _bind_and_drain(session, data.get("external_id"))
                elif typ == "cancel_result":
                    waiter = SESSION_CANCEL_WAITER.get(session)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(data)
                elif typ == "location_state":
                    loc = _parse_location(data)
                    if loc is None:
                        log.warning("location_state session=%s rejected (bad coords: %r)",
                                    session, {k: data.get(k) for k in ("lat", "lon")})
                    else:
                        SESSION_LOCATION[session] = loc
                        # Coarse in the log on purpose: 2 dp is ~1 km, enough to debug "is it
                        # using the right city" without writing a precise home address to disk.
                        log.info("location_state session=%s lat=%.2f lon=%.2f source=%s",
                                 session, loc["lat"], loc["lon"], data.get("source", "?"))
                elif typ == "clock_state":
                    # Full snapshot from the device; replaces whatever we had.
                    SESSION_CLOCK[session] = data
                    # Log the alarms in full: counts alone made a "did the repeat days
                    # survive?" question impossible to answer from the log afterwards.
                    log.info("clock_state session=%s timers=%d alarms=%s ringing=%s", session,
                             len(data.get("timers") or []),
                             [{k: a.get(k) for k in ("id", "hour", "minutes", "days", "label")}
                              for a in (data.get("alarms") or [])],
                             bool(data.get("ringing")))
                elif typ == "turn_start":
                    if ingest is None:
                        ingest = VadIngest(vad, params, sr)
                    turn += 1
                    tid = f"{session}_t{turn:02d}"
                    stages = {"turn_uuid": uuid.uuid4().hex, "turn_mode": "tap",
                              "t_turn_start": time.perf_counter(), "t_speech_onset": None,
                              "t_eou": None, "vad_speech_duration_ms": None,
                              "vad_hangover_used_ms": None}
                    # A greet speaks before the user expects it, so give this one arm cycle
                    # more grace than a self-initiated tap (which already has the user
                    # primed to speak). Reverts to the connection's normal params right after
                    # -- this only ever elevates the *next* arm, never the ones after it.
                    ingest.machine.p = (dataclasses.replace(params, vad_arm_timeout_s=GREET_ARM_TIMEOUT_S)
                                         if SESSION_GREET_PENDING.pop(session, False) else params)
                    ingest.arm()
                    await ws.send_json({"type": "vad_state", "state": "ARMED", "prob": 0.0})
                continue
            if msg.get("bytes") is None or ingest is None:
                continue
            frame = np.frombuffer(msg["bytes"], dtype="<i2").astype("float32") / 32768.0
            for ev, prob, st in ingest.push_pcm(frame):
                now = time.perf_counter()
                if st.value != prev_state or (now - last_ui) >= 0.1:   # UI: on state change or ~10 Hz
                    await ws.send_json({"type": "vad_state", "state": st.value, "prob": round(prob, 3)})
                    prev_state, last_ui = st.value, now
                if stages is None:
                    continue                                   # no armed turn — VAD idles for barge-in
                if ev is Event.ONSET:
                    stages["t_speech_onset"] = time.perf_counter()
                elif ev is Event.ARM_TIMEOUT:
                    log.info("turn %s  ARM_TIMEOUT (no speech within %.0fs)", tid, params.vad_arm_timeout_s)
                    await ws.send_json({"type": "arm_timeout"})
                    stages = None
                elif ev is Event.EOU:
                    stages["t_eou"] = time.perf_counter()
                    stages["vad_speech_duration_ms"] = ingest.machine.last_speech_duration_ms
                    stages["vad_hangover_used_ms"] = ingest.machine.last_hangover_used_ms
                    audio16k = ingest.take_final()
                    _spawn_turn(ws, session, audio16k, 16000, history, tid, stages)
                    stages = None
    except WebSocketDisconnect:
        pass
    log.info("ws-stream(tap) disconnected  session=%s turns=%d", session, turn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)

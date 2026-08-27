"""Push-to-talk realtime voice server: browser mic -> STT -> Robin's conversation engine -> sentence TTS -> browser.

Models load once at import (both resident on the 4 GB card). The LLM turn is remote (blocking
robin_conversation.process_turn(), run in a thread); Parakeet + Kokoro are blocking too, so they
also run in threads to keep the event loop free.

    conda run -n voice uvicorn server:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000 in the Windows browser
"""
import os

import config   # importing sets PYTORCH_CUDA_ALLOC_CONF before torch loads (see config.py)

import argparse
import asyncio
import dataclasses
import functools
import hmac
import io
import json
import logging
import re
import subprocess
import tempfile
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
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
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

# --- WebSocket shared-key auth: ROBIN_API_KEY. Unset -> auth disabled (local dev). ---
API_KEY = config.api_key()
if not API_KEY:
    log.warning("ROBIN_API_KEY not set — WebSocket auth is DISABLED (fine for local dev only)")

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


def stt_transcribe(wav16):
    with torch.inference_mode():
        out = stt_model.transcribe([wav16])
    hyp = out[0]
    return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


def synth_audio(text):
    """Kokoro synth -> one concatenated float32 array (24 kHz)."""
    chunks = []
    for gs, ps, audio in pipe(text, voice=VOICE):
        a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype="float32")
        chunks.append(np.asarray(a, dtype="float32").reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")


def wav_bytes(audio):
    """float32 array -> one complete WAV (24 kHz PCM16) as bytes for a single binary frame."""
    bio = io.BytesIO()
    sf.write(bio, audio, 24000, format="WAV", subtype="PCM_16")
    return bio.getvalue()


app = FastAPI()


def _client_ip(ws: WebSocket) -> str:
    """Requests arrive via nginx, so the real client address is in X-Real-IP, not
    ws.client (that's the proxy)."""
    return ws.headers.get("x-real-ip", "unknown")


def _check_auth(ws: WebSocket) -> bool:
    """True if this connection may proceed. Auth is disabled (returns True) when
    ROBIN_API_KEY isn't set. The key travels as the Sec-WebSocket-Protocol header (set via
    the WebSocket constructor's `protocols` argument), NOT a query string -- a query string
    ends up verbatim in both nginx's and uvicorn's default access logs on every connection,
    permanently persisting the shared secret in cleartext on disk. Headers aren't logged by
    either's default access-log format, so this avoids that leak."""
    if not API_KEY:
        return True
    supplied = ws.headers.get("sec-websocket-protocol", "")
    return hmac.compare_digest(supplied, API_KEY)


def _check_http_auth(request: Request) -> bool:
    """Same gate as _check_auth, for the plain-HTTP dashboard API. The key travels as a
    custom header (X-Robin-Key), never a query string, for the same log-leak reason as above."""
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

# Set by send_greet, consumed by the next turn_start in _run_tap: a user who was just spoken
# to unprompted (rather than one who tapped the button themselves, already primed to speak)
# needs a longer no-speech grace period before VAD gives up and re-idles.
SESSION_GREET_PENDING = {}
GREET_ARM_TIMEOUT_S = 20.0


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


async def send_sentence(ws, sentence, reply_chunks, idx):
    """Synthesize one sentence, send the audio frame, accumulate for the saved reply wav.
    An `audio_meta` control message is sent immediately before the binary frame so the client
    can pair server-side send timing with each chunk. Returns synth seconds."""
    await ws.send_json({"type": "reply", "text": sentence})
    t = time.perf_counter()
    audio = await asyncio.to_thread(synth_audio, sentence)
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
    async with SESSION_LOCK[user_id]:
        await ws.send_json({"type": "turn_start", "turn_id": turn_id})
        t_llm = time.perf_counter()
        result = await asyncio.to_thread(
            functools.partial(process_turn, history, transcript, user_id=user_id,
                              clock_state=SESSION_CLOCK.get(user_id)))
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
            reply = await _await_cancel_result(user_id, result, turn_id)
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
            tts_total += await send_sentence(ws, sentence, reply_chunks, idx)
            idx += 1
        await ws.send_json({"type": "done", "ending": should_end})
    return reply, reply_chunks, llm_dt, ttfs, tts_total, llm_dt, should_end


async def _await_cancel_result(user_id, result, turn_id):
    """Block briefly on the device's cancel_result and return the wording to actually speak.
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
        return result.get("reply_on_fail") or result["reply"]
    log.info("turn %s  cancel_result %s", turn_id, res)
    if not res.get("ok"):
        return result.get("reply_on_fail") or result["reply"]
    noun = result.get("cancel_all_noun")
    if noun:                                     # all-cancel: the device owns the real count
        from robin_conversation.clock import all_cancel_reply
        return all_cancel_reply(res.get("count"), noun)
    return result["reply"]


GREET_TEXT = "How is your day going?"


async def send_greet(session_id, text=GREET_TEXT):
    """Speak `text` unprompted into an already-connected tap session, exactly as a normal
    reply would (turn_start / reply / audio_meta / done) -- indistinguishable to the client
    from a real turn. The client's own auto-continue (tap_index.html's rearm(), fired on
    `done`) then re-sends {type: "turn_start"}, which is what actually arms VAD on the
    server; nothing here has to poke the VAD state machine directly. Returns False if the
    session has no live socket (already disconnected)."""
    ws = SESSION_WS.get(session_id)
    if ws is None:
        return False
    async with SESSION_LOCK[session_id]:
        tid = f"{session_id}_greet{int(time.time())}"
        await ws.send_json({"type": "turn_start", "turn_id": tid})
        reply_chunks, idx = [], 0
        for sentence in _SENT_RE.split(text.strip()):
            sentence = sentence.strip()
            if sentence:
                await send_sentence(ws, sentence, reply_chunks, idx)
                idx += 1
        await ws.send_json({"type": "done", "ending": False})
    SESSION_GREET_PENDING[session_id] = True                             # longer arm timeout
    history = SESSION_HISTORY.get(session_id)                            # for the rearm this
    if history is not None:                                              # triggers
        history.append({"role": "assistant", "content": text})           # so the next real
    _session_turn(session_id, f"[Robin, unprompted] {text}")             # LLM turn has context
    log.info("greet  session=%s  %r", session_id, text)
    return True


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
    if not _check_auth(ws):
        log.warning("rejected /ws connection from %s: bad or missing key", _client_ip(ws))
        # Completing the handshake (accept) then immediately closing is what actually
        # delivers code 1008 to the browser -- closing before accept fails the handshake
        # itself (HTTP 403), which browsers surface as code 1006, not 1008. No GPU work
        # happens either way: we return before ever reading a message.
        await ws.accept()
        await ws.close(code=1008)
        return
    await ws.accept(subprotocol=API_KEY)
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws connected  session=%s", session)
    _session_connect(session, "classic", _client_ip(ws))
    history = []                                              # per-connection conversation memory
    SESSION_WS[session] = ws
    SESSION_LOCK[session] = asyncio.Lock()
    SESSION_HISTORY[session] = history
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

            full, reply_chunks, ttft, ttfs, tts_total, llm_dt, ending = await robin_reply(
                ws, history, transcript, tid, session)

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
        SESSION_CLOCK.pop(session, None)
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

    t_reply = time.perf_counter()
    full, reply_chunks, ttft, ttfs, tts_total, llm_dt, ending = await robin_reply(
        ws, history, transcript, tid, session)
    stages["t_llm_first_token"] = (t_reply + ttft) if ttft is not None else None
    stages["t_tts_first_frame"] = (t_reply + ttfs) if ttfs is not None else None

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
    if not _check_auth(ws):
        log.warning("rejected /ws-stream connection from %s: bad or missing key", _client_ip(ws))
        # See ws_endpoint's comment: accept-then-close is what actually delivers 1008 to
        # the browser. No GPU work happens either way -- we return before reading a message.
        await ws.accept()
        await ws.close(code=1008)
        return
    await ws.accept(subprotocol=API_KEY)
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws-stream connected  session=%s  turn_mode=%s", session, TURN_MODE)
    _session_connect(session, TURN_MODE, _client_ip(ws))
    history = []                                              # per-connection conversation memory
    SESSION_WS[session] = ws
    SESSION_LOCK[session] = asyncio.Lock()
    SESSION_HISTORY[session] = history
    try:
        if TURN_MODE == "tap":
            await _run_tap(ws, session, history)
        else:
            await _run_hold(ws, session, history)
    finally:
        _session_disconnect(session)
        SESSION_WS.pop(session, None)
        SESSION_LOCK.pop(session, None)
        SESSION_HISTORY.pop(session, None)
        SESSION_CLOCK.pop(session, None)
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
                elif typ == "cancel_result":
                    waiter = SESSION_CANCEL_WAITER.get(session)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(data)
                elif typ == "clock_state":
                    # Full snapshot from the device; replaces whatever we had.
                    SESSION_CLOCK[session] = data
                    log.info("clock_state session=%s timers=%d alarms=%d ringing=%s", session,
                             len(data.get("timers") or []), len(data.get("alarms") or []),
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
                elif typ == "cancel_result":
                    waiter = SESSION_CANCEL_WAITER.get(session)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(data)
                elif typ == "clock_state":
                    # Full snapshot from the device; replaces whatever we had.
                    SESSION_CLOCK[session] = data
                    log.info("clock_state session=%s timers=%d alarms=%d ringing=%s", session,
                             len(data.get("timers") or []), len(data.get("alarms") or []),
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

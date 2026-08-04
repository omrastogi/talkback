"""Push-to-talk realtime voice server: browser mic -> STT -> streaming LLM -> sentence TTS -> browser.

Models load once at import (both resident on the 4 GB card). LLM is remote (AsyncOpenAI, streaming);
Parakeet + Kokoro are blocking, so they run in threads to keep the event loop free.

    conda run -n voice uvicorn server:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000 in the Windows browser
"""
import os

import config   # importing sets PYTORCH_CUDA_ALLOC_CONF before torch loads (see config.py)

import argparse
import asyncio
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

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
_ap.add_argument("--persona", choices=["default", "robin"],
                 default=os.environ.get("PERSONA", "default"))   # robin = ported RECOVER conversation
_ap.add_argument("--host", default="0.0.0.0")
_ap.add_argument("--port", type=int, default=8000)
ARGS, _ = _ap.parse_known_args()

LLM_BASE_URL, LLM_API_KEY, MODEL = config.resolve_backend(ARGS.backend, ARGS.model)
if not LLM_API_KEY:
    log.warning("no API key for backend '%s' — set %s in .env", ARGS.backend,
                "OPENAI_API_KEY" if ARGS.backend == "openai" else "PARCS_API_KEY")

t = time.time(); stt_model = _load_stt(); STT_LOAD = time.time() - t
from kokoro import KPipeline
t = time.time(); pipe = KPipeline(lang_code="a"); TTS_LOAD = time.time() - t
for _ in pipe("Ready.", voice=VOICE):   # warm Kokoro (first synth compiles kernels ~3s)
    pass
aclient = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)  # base_url=None -> OpenAI
# robin persona: ported RECOVER conversation() (robin_convo.py). It's blocking, so it uses a sync
# client run in a thread. Same backend/key as above — no hardcoded credentials.
import robin_convo
from openai import OpenAI
robin_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY) if ARGS.persona == "robin" else None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("[ready] STT %.1fs · TTS %.1fs · both resident on %s · LLM=%s [%s] · persona=%s · logs -> %s",
         STT_LOAD, TTS_LOAD, _DEVICE, MODEL, ARGS.backend, ARGS.persona, LOG_DIR)

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


@app.get("/")
async def index():
    # Default = tap-to-talk (VAD endpointing). Old hold demos stay at /stream and /classic.
    page = "tap_index.html" if TURN_MODE == "tap" else "stream_index.html"
    return FileResponse(os.path.join(HERE, page))


@app.get("/classic")
async def classic_index():
    return FileResponse(os.path.join(HERE, "index.html"))          # non-streaming push-to-talk (/ws)


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


async def stream_reply(ws, messages, transcript, turn_id):
    """LLM token stream -> sentence chunks -> TTS audio frames. Appends the user + assistant
    turns to `messages`, sends the final {"type":"done"}, and returns
    (full_reply, reply_chunks, ttft, ttfs, tts_total, llm_dt). Shared by /ws and /ws-stream.
    Emits {"type":"turn_start"} at the start and a monotonic per-turn `idx` on each audio frame."""
    messages.append({"role": "user", "content": transcript})
    await ws.send_json({"type": "turn_start", "turn_id": turn_id})
    t_llm = time.perf_counter()
    stream = await aclient.chat.completions.create(model=MODEL, messages=messages, stream=True)
    ttft = ttfs = None
    reply_chunks, tts_total, idx = [], 0.0, 0
    buf, full, first_flush = "", "", True

    async def flush(sentence):
        nonlocal ttfs, tts_total, first_flush, idx
        if not sentence:
            return
        if ttfs is None:
            ttfs = time.perf_counter() - t_llm
        tts_total += await send_sentence(ws, sentence, reply_chunks, idx)
        idx += 1
        first_flush = False

    async for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        if not piece:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t_llm
        buf += piece
        full += piece
        pattern = r"[.!?,]" if first_flush else r"[.!?]"   # break on comma for the first flush
        while (m := re.search(pattern, buf)):
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            await flush(sentence)
            pattern = r"[.!?]"
        if first_flush and len(buf) > 60:
            await flush(buf.strip())
            buf = ""
    await flush(buf.strip())
    llm_dt = time.perf_counter() - t_llm
    messages.append({"role": "assistant", "content": full})
    await ws.send_json({"type": "done"})
    return full, reply_chunks, ttft, ttfs, tts_total, llm_dt


def save_reply(reply_chunks, tid):
    """Concatenate the spoken reply chunks to log/<tid>.reply.wav (if any). Returns the path."""
    reply_wav = os.path.join(LOG_DIR, f"{tid}.reply.wav")
    if reply_chunks:
        sf.write(reply_wav, np.concatenate(reply_chunks), 24000, subtype="PCM_16")
    return reply_wav


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


async def robin_reply(ws, history, transcript, turn_id):
    """RECOVER-persona reply: robin_convo.conversation() -> sentence TTS. Mirrors stream_reply's
    control messages (turn_start / reply / audio_meta / done) so the client and playback telemetry
    are identical; only the text source differs (a blocking full reply, not a token stream).
    Returns the same tuple shape as stream_reply."""
    await ws.send_json({"type": "turn_start", "turn_id": turn_id})
    t_llm = time.perf_counter()
    reply = await asyncio.to_thread(robin_convo.conversation, history, transcript,
                                    client=robin_client, model=MODEL)
    llm_dt = time.perf_counter() - t_llm
    spoken = reply.replace("CONVERSATION_END", "").replace("DELETE_MESSAGE", "").strip()
    reply_chunks, tts_total, idx, ttfs = [], 0.0, 0, None
    for sentence in _SENT_RE.split(spoken):
        sentence = sentence.strip()
        if not sentence:
            continue
        if ttfs is None:
            ttfs = time.perf_counter() - t_llm
        tts_total += await send_sentence(ws, sentence, reply_chunks, idx)
        idx += 1
    await ws.send_json({"type": "done"})
    return reply, reply_chunks, llm_dt, ttfs, tts_total, llm_dt


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
    await ws.accept()
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws connected  session=%s", session)
    messages = [{"role": "system", "content": config.SYS_PROMPT}]   # per-connection memory
    robin_history = []                                              # per-connection memory for --persona robin
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
            await ws.send_json({"type": "transcript", "text": transcript})

            if ARGS.persona == "robin":
                full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await robin_reply(ws, robin_history, transcript, tid)
            else:
                full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await stream_reply(ws, messages, transcript, tid)

            reply_wav = save_reply(reply_chunks, tid)
            turn_total = time.perf_counter() - t_turn
            log.info("turn %d  reply %r -> %s", turn, full.strip(), reply_wav)
            log.info("turn %d  timings: stt=%.2f ttft=%s ttfs=%s tts=%.2f llm=%.2f total=%.2f",
                     turn, stt_dt, _f(ttft), _f(ttfs), tts_total, llm_dt, turn_total)
    except WebSocketDisconnect:
        log.info("ws disconnected  session=%s turns=%d", session, turn)
        return


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


async def _finalize_turn(ws, session, audio16k, sr, messages, robin_history, tid, stages):
    """Shared turn finalize (tap and hold): transcribe the utterance, send the transcript, run
    the persona/stream reply, save it, and write the per-turn VAD bench record. `stages` carries
    the raw pre-STT timestamps the caller already filled (t_turn_start, t_speech_onset, t_eou).
    STT/LLM/TTS are called exactly as before — this only threads instrumentation through."""
    if audio16k is None or len(audio16k) == 0:                  # never silently drop: report empty
        await ws.send_json({"type": "transcript", "text": ""})
        await ws.send_json({"type": "done"})
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
    if ARGS.persona == "robin":
        full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await robin_reply(ws, robin_history, transcript, tid)
    else:
        full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await stream_reply(ws, messages, transcript, tid)
    stages["t_llm_first_token"] = (t_reply + ttft) if ttft is not None else None
    stages["t_tts_first_frame"] = (t_reply + ttfs) if ttfs is not None else None

    reply_wav = save_reply(reply_chunks, tid)
    log.info("turn %s  reply (tts=%.2f llm=%.2f) %r -> %s", tid, tts_total, llm_dt, full.strip(), reply_wav)
    rec = {"session": session, "tid": tid, "persona": ARGS.persona, "backend": ARGS.backend,
           "transcript": transcript, "reply": full.strip(), **stages}
    asyncio.create_task(asyncio.to_thread(_append_bench, f"vad_{session}", rec))


@app.websocket("/ws-stream")
async def ws_stream_endpoint(ws: WebSocket):
    """Tap-to-talk (VAD endpointing) or hold-to-talk, per config.turn_mode(). Both reuse the
    resident models and the shared reply/finalize path; only turn-boundary detection differs.
    turn_mode='hold' is the latency-matrix control that isolates VAD hangover from the pipeline."""
    await ws.accept()
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws-stream connected  session=%s  turn_mode=%s", session, TURN_MODE)
    messages = [{"role": "system", "content": config.SYS_PROMPT}]
    robin_history = []
    if TURN_MODE == "tap":
        await _run_tap(ws, session, messages, robin_history)
    else:
        await _run_hold(ws, session, messages, robin_history)


async def _run_hold(ws, session, messages, robin_history):
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
                elif typ == "end":
                    turn += 1
                    tid = f"{session}_t{turn:02d}"
                    stages = {"turn_uuid": uuid.uuid4().hex, "turn_mode": "hold",
                              "t_turn_start": t_turn_start, "t_speech_onset": None,
                              "t_eou": time.perf_counter(),
                              "vad_speech_duration_ms": None, "vad_hangover_used_ms": None}
                    audio = np.concatenate(buf) if buf else None
                    await _finalize_turn(ws, session, audio, sr, messages, robin_history, tid, stages)
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


async def _run_tap(ws, session, messages, robin_history):
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
        return await _run_hold(ws, session, messages, robin_history)

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
                elif typ == "turn_start":
                    if ingest is None:
                        ingest = VadIngest(vad, params, sr)
                    turn += 1
                    tid = f"{session}_t{turn:02d}"
                    stages = {"turn_uuid": uuid.uuid4().hex, "turn_mode": "tap",
                              "t_turn_start": time.perf_counter(), "t_speech_onset": None,
                              "t_eou": None, "vad_speech_duration_ms": None,
                              "vad_hangover_used_ms": None}
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
                    await _finalize_turn(ws, session, audio16k, 16000, messages, robin_history, tid, stages)
                    stages = None
    except WebSocketDisconnect:
        pass
    log.info("ws-stream(tap) disconnected  session=%s turns=%d", session, turn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)

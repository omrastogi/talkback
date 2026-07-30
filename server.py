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

t = time.time(); stt_model = _load_stt(); STT_LOAD = time.time() - t
from kokoro import KPipeline
t = time.time(); pipe = KPipeline(lang_code="a"); TTS_LOAD = time.time() - t
for _ in pipe("Ready.", voice=VOICE):   # warm Kokoro (first synth compiles kernels ~3s)
    pass
aclient = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)  # base_url=None -> OpenAI
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("[ready] STT %.1fs · TTS %.1fs · both resident on %s · LLM=%s [%s] · logs -> %s",
         STT_LOAD, TTS_LOAD, _DEVICE, MODEL, ARGS.backend, LOG_DIR)


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
    return FileResponse(os.path.join(HERE, "stream_index.html"))   # streaming STT + live captions


@app.get("/classic")
async def classic_index():
    return FileResponse(os.path.join(HERE, "index.html"))          # non-streaming push-to-talk (/ws)


async def send_sentence(ws, sentence, reply_chunks):
    """Synthesize one sentence, send the audio frame, accumulate for the saved reply wav.
    Returns synth seconds."""
    await ws.send_json({"type": "reply", "text": sentence})
    t = time.perf_counter()
    audio = await asyncio.to_thread(synth_audio, sentence)
    dt = time.perf_counter() - t
    reply_chunks.append(audio)
    await ws.send_bytes(wav_bytes(audio))
    log.info("    tts %.2fs (%.2fs audio)  %r", dt, len(audio) / 24000, sentence)
    return dt


async def stream_reply(ws, messages, transcript):
    """LLM token stream -> sentence chunks -> TTS audio frames. Appends the user + assistant
    turns to `messages`, sends the final {"type":"done"}, and returns
    (full_reply, reply_chunks, ttft, ttfs, tts_total, llm_dt). Shared by /ws and /ws-stream."""
    messages.append({"role": "user", "content": transcript})
    t_llm = time.perf_counter()
    stream = await aclient.chat.completions.create(model=MODEL, messages=messages, stream=True)
    ttft = ttfs = None
    reply_chunks, tts_total = [], 0.0
    buf, full, first_flush = "", "", True

    async def flush(sentence):
        nonlocal ttfs, tts_total, first_flush
        if not sentence:
            return
        if ttfs is None:
            ttfs = time.perf_counter() - t_llm
        tts_total += await send_sentence(ws, sentence, reply_chunks)
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws connected  session=%s", session)
    messages = [{"role": "system", "content": config.SYS_PROMPT}]   # per-connection memory
    turn = 0
    try:
        while True:
            audio_bytes = await ws.receive_bytes()
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

            full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await stream_reply(ws, messages, transcript)

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


@app.websocket("/ws-stream")
async def ws_stream_endpoint(ws: WebSocket):
    """Streaming-STT push-to-talk: client sends {start} then Int16 PCM frames then {end}.
    Live partial transcripts stream back as the user speaks; on {end} the final transcript
    plus the spoken reply follow. Same resident models + turn logic as /ws."""
    await ws.accept()
    session = time.strftime("%Y%m%d_%H%M%S")
    log.info("ws-stream connected  session=%s", session)
    messages = [{"role": "system", "content": config.SYS_PROMPT}]
    turn, sr = 0, 16000
    buf, n_samples, last_partial = [], 0, 0   # float32 frames, total samples, samples at last partial
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    sr = int(data.get("sampleRate", 16000))
                    buf, n_samples, last_partial = [], 0, 0
                elif data.get("type") == "end":
                    turn += 1
                    tid = f"{session}_t{turn:02d}"
                    t_turn = time.perf_counter()
                    if not buf:
                        await ws.send_json({"type": "transcript", "text": ""})
                        await ws.send_json({"type": "done"})
                        continue
                    audio = np.concatenate(buf)
                    in_wav = os.path.join(LOG_DIR, f"{tid}.in.wav")
                    t = time.perf_counter()
                    await asyncio.to_thread(pcm16_to_wav16, audio, sr, in_wav)
                    transcript = await asyncio.to_thread(stt_transcribe, in_wav)
                    stt_dt = time.perf_counter() - t
                    log.info("turn %d  stream audio=%.2fs -> %s", turn, len(audio) / sr, in_wav)
                    log.info("turn %d  stt(final) %.2fs -> %r", turn, stt_dt, transcript)
                    await ws.send_json({"type": "transcript", "text": transcript})

                    full, reply_chunks, ttft, ttfs, tts_total, llm_dt = await stream_reply(ws, messages, transcript)

                    reply_wav = save_reply(reply_chunks, tid)
                    turn_total = time.perf_counter() - t_turn
                    log.info("turn %d  reply %r -> %s", turn, full.strip(), reply_wav)
                    log.info("turn %d  timings: stt=%.2f ttft=%s ttfs=%s tts=%.2f llm=%.2f total=%.2f",
                             turn, stt_dt, _f(ttft), _f(ttfs), tts_total, llm_dt, turn_total)
                    buf, n_samples, last_partial = [], 0, 0
            elif msg.get("bytes") is not None:
                frame = np.frombuffer(msg["bytes"], dtype="<i2").astype("float32") / 32768.0
                buf.append(frame)
                n_samples += len(frame)
                # ponytail: re-transcribe the whole buffer per partial — O(n^2) over the turn,
                # fine for short push-to-talk clips. Swap in a cache-aware streaming ASR model
                # if utterances get long enough that the growing re-decode dominates.
                if n_samples - last_partial >= PARTIAL_EVERY_S * sr:
                    last_partial = n_samples
                    partial_wav = os.path.join(LOG_DIR, "_stream_partial.wav")
                    await asyncio.to_thread(pcm16_to_wav16, np.concatenate(buf), sr, partial_wav)
                    partial = await asyncio.to_thread(stt_transcribe, partial_wav)
                    log.info("    partial %.1fs -> %r", n_samples / sr, partial)
                    await ws.send_json({"type": "partial", "text": partial})
    except WebSocketDisconnect:
        pass
    log.info("ws-stream disconnected  session=%s turns=%d", session, turn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)

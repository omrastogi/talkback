"""Overlapped realtime demo: wav -> STT -> streaming LLM -> sentence-chunked TTS -> play/buffer.

TTS speaks sentence 1 while the LLM is still writing sentence 2. Three stages,
two queues, two background threads + main. Self-contained: loads Parakeet and
Kokoro directly.

    conda run -n voice python stream.py --wav out/hello.wav [--voice af_heart] [--model gpt-4o-mini]
"""
import argparse
import os
import queue
import re
import tempfile
import threading
import time

import config   # importing sets PYTORCH_CUDA_ALLOC_CONF before torch loads (see config.py)

import numpy as np
import soundfile as sf
import torch
import torchaudio

SENTINEL = object()          # end-of-stream marker on each queue
SR_TTS = 24000               # Kokoro output rate


def load_stt():
    import nemo.collections.asr as nemo_asr
    # Load on CPU first, then move — NeMo's restore path spikes VRAM if it targets
    # the GPU directly, which OOMs the 4 GB card before Kokoro even loads.
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    if torch.cuda.is_available():
        m = m.to("cuda")
    m.eval()
    return m


def transcribe(model, wav_path):
    data, sr = sf.read(wav_path, dtype="float32")   # (n,) or (n, ch)
    if data.ndim == 2:
        data = data.mean(axis=1)                     # mono
    if sr != 16000:
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    tmp = os.path.join(tempfile.gettempdir(), "_stream_16k.wav")
    sf.write(tmp, data, 16000)
    with torch.inference_mode():
        out = model.transcribe([tmp])
    hyp = out[0]
    return hyp.text if hasattr(hyp, "text") else str(hyp)


def tts_worker(pipe, voice, sentence_q, audio_q):
    """sentence_q -> Kokoro -> audio_q, one chunk per Kokoro segment."""
    while True:
        item = sentence_q.get()
        if item is SENTINEL:
            audio_q.put(SENTINEL)
            return
        for gs, ps, audio in pipe(item, voice=voice):
            if hasattr(audio, "detach"):                       # torch tensor
                chunk = audio.detach().cpu().numpy()
            else:
                chunk = np.asarray(audio, dtype="float32")
            audio_q.put(np.asarray(chunk, dtype="float32").reshape(-1))


def open_output():
    """Open a 24 kHz output stream ONCE, up front. Returns (stream_or_None, playing).
    Probing/opening a device can block for ~seconds in WSL2 with no audio hardware, so it
    must never happen inside the timed path — else it inflates ttfa on every input."""
    try:
        import sounddevice as sd
        s = sd.OutputStream(samplerate=SR_TTS, channels=1, dtype="float32")
        s.start()
        return s, True
    except Exception:                                          # no device in WSL2 -> buffer only
        return None, False


def playback_worker(audio_q, out_stream, state, buffer):
    """audio_q -> live playback (on a pre-opened stream) + always buffer for the wav."""
    while True:
        chunk = audio_q.get()
        if chunk is SENTINEL:
            return
        if state["first_audio_ts"] is None:
            state["first_audio_ts"] = time.perf_counter()
        if out_stream is not None:
            out_stream.write(chunk)
        buffer.append(chunk)


def warm_kokoro(pipe, voice):
    """First Kokoro synth compiles CUDA kernels (~3s), a one-time cost like model load.
    Warm once before timing or ttfa is dominated by warmup, not real overlap."""
    for _ in pipe("Ready.", voice=voice):
        pass


def run_overlap(client, pipe, transcript, model, voice, out_path, out_stream=None, playing=False):
    """Stream one reply overlapped: LLM tokens -> sentence_q -> Kokoro -> audio_q -> play/buffer.
    t0 is set just before the OpenAI call (end-of-utterance analog). Returns a latency dict.
    out_stream is a pre-opened device (or None); opening must not happen in this timed path."""
    sentence_q = queue.Queue()
    audio_q = queue.Queue()
    state = {"first_audio_ts": None, "play": playing}
    buffer = []

    tts_t = threading.Thread(target=tts_worker, args=(pipe, voice, sentence_q, audio_q), daemon=True)
    play_t = threading.Thread(target=playback_worker, args=(audio_q, out_stream, state, buffer), daemon=True)
    tts_t.start()
    play_t.start()

    ttft = ttfs = None
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": config.SYS_PROMPT},
                  {"role": "user", "content": transcript}],
        stream=True,
    )

    buf, full, first_flush = "", "", True
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""   # None on role/last chunks
        if not piece:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0            # first token
        buf += piece
        full += piece
        pattern = r"[.!?,]" if first_flush else r"[.!?]"   # break on comma too, first flush only
        while (m := re.search(pattern, buf)):
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            if sentence:
                if ttfs is None:
                    ttfs = time.perf_counter() - t0
                sentence_q.put(sentence)
                first_flush = False
            pattern = r"[.!?]"
        if first_flush and len(buf) > 60:              # no punctuation yet, flush a clause
            sentence_q.put(buf.strip())
            buf = ""
            first_flush = False
    if buf.strip():
        sentence_q.put(buf.strip())
    sentence_q.put(SENTINEL)
    llm_done = time.perf_counter() - t0                # full reply text ready

    tts_t.join()
    play_t.join()

    total = time.perf_counter() - t0
    ttfa = (state["first_audio_ts"] - t0) if state["first_audio_ts"] else None
    if buffer and out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        sf.write(out_path, np.concatenate(buffer), SR_TTS)

    return {"ttft": ttft, "ttfs": ttfs, "ttfa": ttfa, "total": total,
            "llm_done": llm_done, "play": state["play"], "reply": full.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from kokoro import KPipeline

    # Load both resident models + client + audio device up front so latencies exclude setup.
    stt_model = load_stt()
    pipe = KPipeline(lang_code="a")
    client, model = config.make_client(args.backend, args.model)
    warm_kokoro(pipe, args.voice)
    out_stream, playing = open_output()

    # STT runs before t0; t0 = end-of-utterance analog, set inside run_overlap.
    t_stt = time.perf_counter()
    transcript = transcribe(stt_model, args.wav)
    stt_infer = time.perf_counter() - t_stt
    print(f"Transcript: {transcript}")

    r = run_overlap(client, pipe, transcript, model, args.voice, "out/stream_reply.wav",
                    out_stream, playing)
    if out_stream is not None:
        out_stream.stop()
        out_stream.close()

    def fmt(x):
        return f"{x:.2f}" if x is not None else "n/a"

    print(f"\nReply: {r['reply']}")
    print("\n--- latency (s, relative to end-of-utterance t0) ---")
    print(f"stt_infer = {stt_infer:.2f}  (before t0)")
    print(f"ttft      = {fmt(r['ttft'])}")
    print(f"ttfs      = {fmt(r['ttfs'])}")
    print(f"ttfa      = {fmt(r['ttfa'])}")
    print(f"total     = {r['total']:.2f}")
    print(f"playback  = {'LIVE' if r['play'] else 'buffered-only (no audio device)'}")

    if r["ttfa"] is not None and r["ttfs"] is not None:
        tts_lat = r["ttfa"] - r["ttfs"]
        seq_first_audio = r["llm_done"] + tts_lat
        print(f"\noverlap: first audio at {r['ttfa']:.2f}s vs ~{seq_first_audio:.2f}s sequential "
              f"(full reply {r['llm_done']:.2f}s + tts {tts_lat:.2f}s) -> {seq_first_audio - r['ttfa']:.2f}s sooner")


if __name__ == "__main__":
    main()

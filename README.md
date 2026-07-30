# voice-server — voice-assistant cascade + realtime server

A voice-assistant cascade you can run two ways: **offline** on recorded wav files
(profiling / testing), or as a **realtime push-to-talk server** in the browser.

```
wav / mic  ->  STT (Parakeet TDT 0.6B v3)  ->  LLM (parcs gemma / OpenAI)  ->  TTS (Kokoro-82M)  ->  wav / speaker
```

Each stage runs as a **separate subprocess** so the OS fully reclaims GPU VRAM
between stages — never two models resident at once. Every stage prints its load +
inference seconds; `pipeline.py` collects them into a table.

STT runs on GPU (NeMo/Parakeet). The LLM is a remote API call, off-GPU — the parcs
gemma gateway by default, or OpenAI (see LLM below). TTS (Kokoro-82M) runs on CPU or GPU.

## Setup (WSL2 Ubuntu, one time)

```bash
conda create -n voice python=3.11 -y
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
conda run -n voice pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n voice pip install "nemo_toolkit[asr]" "kokoro>=0.9.4" soundfile openai
```

`espeak-ng` is Kokoro's phoneme fallback for out-of-dictionary words. The harness
runs without it for common text, but install it for robust TTS on arbitrary replies.

Put your keys in a `.env` file (this dir or the parent). `parcs` is the default backend,
so `PARCS_*` is what you need unless you pass `--backend openai`:

```
PARCS_BASE_URL=https://gateway.parcs.northeastern.edu/llm/api/v1
PARCS_API_KEY=sk-...
PARCS_MODEL=gemma4:12b
OPENAI_API_KEY=sk-...      # only for --backend openai
```

`_run.sh` and `_install.sh` are convenience helpers (activate the env + cd here;
reproduce the full install). Not required — the `conda run` commands above work too.

## Run

All commands assume WSL2 with the `voice` conda env (see Setup). `parcs` gemma is the
default LLM backend — add `--backend openai` to any command to use OpenAI instead.

### Realtime voice server (browser, push-to-talk)

The interactive demo: hold the button (or Space), speak, release; the reply is spoken back.

```bash
conda run -n voice uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000> in the Windows browser — the default page is the
**streaming-STT** demo (live captions as you speak; endpoint `/ws-stream`). The old
non-streaming push-to-talk page is at `/classic` (endpoint `/ws`). Models load once at
startup (watch the `[ready]` log line). To switch LLM backend, set the env var before uvicorn
(`LLM_BACKEND=openai uvicorn server:app ...`) or run the script form which takes flags:

```bash
conda run -n voice python server.py --backend openai --port 8000
```

Smoke-test it without a browser (server must be running):

```bash
conda run -n voice python wstest.py
```

**Streaming STT** is the default page (`/`, endpoint `/ws-stream`; also at `/stream`). The
browser streams Int16 PCM frames as you hold the button; the server re-transcribes the
growing buffer (~every 0.4 s, `PARTIAL_EVERY_S`) and sends live `partial` transcripts, then
the final transcript + spoken reply on release. Headless test: `python wstest_stream.py`.

Caveat (honest): Parakeet-tdt is an **offline** model — no streaming encoder state. The
partials are re-decodes of the whole buffer for a live caption; the final transcribe on
release is still full-clip time, it does **not** collapse to "just the last chunk". STT
(~0.15 s) is a sliver of the turn anyway (the LLM is the pole), so streaming STT buys the
live-caption UX, not latency. Collapsing post-release STT needs a cache-aware streaming
ASR model (a model swap, not a wiring change).

### Offline cascade (wav in → reply wav + timing table)

```bash
# full cascade
conda run -n voice python pipeline.py --wav out/hello.wav
conda run -n voice python pipeline.py --wav out/hello.wav --backend openai   # override backend

# each stage alone
conda run -n voice python tts.py --text "hello, this is a test." --out out/hello.wav
conda run -n voice python stt.py out/hello.wav
conda run -n voice python llm.py --text "what is the capital of japan?"

# roundtrip selftest (TTS -> STT, no LLM, no external audio)
conda run -n voice python pipeline.py --selftest
```

### Profilers (run a set of inputs, write a Markdown report)

```bash
conda run -n voice python batch.py --in-dir in            # sequential cascade -> out/PROFILE.md
conda run -n voice python batch_stream.py --in-dir in     # overlapped streaming -> out/PROFILE_stream.md
conda run -n voice python stream.py --wav out/hello.wav   # one input, overlapped, live latencies
```

## LLM

OpenAI-compatible chat completion via two backends, selected with `--backend`:

- `parcs` (default) — the lab gateway serving gemma. Reads `PARCS_BASE_URL`, `PARCS_API_KEY`,
  `PARCS_MODEL` (default `gemma4:12b`) from `.env`.
- `openai` — OpenAI proper. Reads `OPENAI_API_KEY`; default model `gpt-4o-mini`.

`--backend` is accepted by every LLM-touching entry point (`server.py`, `pipeline.py`,
`llm.py`, `stream.py`, `stream_demo.py`, `batch.py`, `batch_stream.py`); `--model` overrides
the backend's default model. The env var `LLM_BACKEND` sets the default when `--backend` is omitted.
All of this lives in `config.py` — one `.env` loader and one backend switch shared by every stage.

```bash
conda run -n voice python llm.py --text "what is the capital of japan?"                 # parcs gemma
conda run -n voice python llm.py --backend openai --text "what is the capital of japan?"  # OpenAI
```

System prompt (`config.SYS_PROMPT`, one place, shared by every stage): a **health-support
voice agent** — warm and concise, one or two spoken-style sentences, with hard safety rails
(never diagnoses/prescribes; routes emergencies to local emergency services; defers to a
licensed clinician when unsure). Edit `config.SYS_PROMPT` to change the persona everywhere.
`load` in the timing line is client init; `infer` is the API round-trip.

## Timing contract

Each stage's final stdout line is exactly:

```
TIMING stage=<name> load=<sec:.2f> infer=<sec:.2f>
```


source ~/miniconda3/etc/profile.d/conda.sh && conda activate voice && cd /mnt/e/PARCS/server/voice-server && uvicorn server:app --host 0.0.0.0 --port 8000


~/cloudflared tunnel --url http://localhost:8000
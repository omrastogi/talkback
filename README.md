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

### Realtime voice server (browser)

```bash
conda run -n voice uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000> in the Windows browser — the default page is **tap-to-talk**
(tap once and speak; a voice-activity detector ends your turn when you stop). The older
**hold-to-talk** streaming demo is at `/stream`, and the non-streaming push-to-talk page at
`/classic` (endpoint `/ws`). All three share the resident models and the `/ws-stream`
endpoint (tap and hold) — `config.turn_mode()` picks the behavior. Models load once at startup
(watch the `[ready]` and `[vad]` log lines). To switch LLM backend, set the env var before
uvicorn (`LLM_BACKEND=openai uvicorn server:app ...`) or run the script form which takes flags:

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

### Tap-to-talk with VAD endpointing (default)

The mic streams continuously; you tap once (button or **Space**) to start a turn and a
**Silero VAD** decides when you've stopped talking (end-of-utterance). Lives entirely in the
transport/ingest layer (`vad/`) — STT/LLM/TTS interfaces are unchanged.

- **State machine** (`vad/turn.py`): `IDLE → ARMED → SPEECH → TRAILING → (EOU) → IDLE`, one
  per connection. Hysteresis (`speech`/`silence` thresholds) stops mid-word flapping; an onset
  debounce rejects clicks; a hangover timer sets the end-of-turn latency.
- **Prespeech ring** (`vad/ring.py`): the last `vad_prespeech_ms` of audio is flushed into STT
  on onset, so the clipped first syllable is recovered.
- **Provider** (`vad/provider.py`): CPU vs CUDA is micro-benchmarked once at startup and the
  faster median wins; a CUDA failure falls back to CPU (never fatal). On the 4 GB card CPU is
  the right pick (keeps VRAM free for Parakeet) — set `VAD_PROVIDER_OVERRIDE=cpu`.
- **`turn_mode`** (`tap` default, or `hold`): `hold` bypasses the VAD and uses button-down/up
  as the turn boundaries — the control condition for the latency matrix.
- **Instrumentation**: per-turn raw stage timestamps (`t_turn_start … t_tts_first_frame`,
  `vad_speech_duration_ms`, `vad_hangover_used_ms`) land in `log/bench/vad_<session>.jsonl`;
  the startup provider bench in `log/bench/vad_provider.jsonl`.

Config (env vars, all with defaults in `vad/turn.py::TurnParams`; `vad_hangover_ms` is the
primary latency knob, tuned against real post-op patients who pause mid-sentence):

```
TURN_MODE=tap|hold                 VAD_PROVIDER_OVERRIDE=auto|cuda|cpu
VAD_SPEECH_THRESHOLD=0.5           VAD_SILENCE_THRESHOLD=0.35
VAD_ONSET_FRAMES=2                 VAD_HANGOVER_MS=600
VAD_PRESPEECH_MS=300               VAD_MAX_UTTERANCE_S=30    VAD_ARM_TIMEOUT_S=10
```

Requires `onnxruntime-gpu` + `vad/silero_vad.onnx` (fetch once):

```bash
conda run -n voice pip install onnxruntime-gpu
curl -L -o vad/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

Tests — each `vad/` module self-checks (repo convention), plus a headless end-to-end gate:

```bash
conda run -n voice python vad/turn.py        # state-machine cases (onset/hysteresis/hangover/…)
conda run -n voice python vad/ring.py         # prespeech ordering, no onset-frame duplication
conda run -n voice python vad/ingest.py       # onset flush + utterance assembly (fake VAD)
conda run -n voice python vad/provider.py      # CUDA-probe-fails → CPU fallback, no raise
VAD_PROVIDER_OVERRIDE=cpu TURN_MODE=tap conda run -n voice uvicorn server:app --port 8000 &
conda run -n voice python wstest_tap.py --port 8000 --wav in/inp6.wav   # full turn over /ws-stream
```

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

### Persona: `--persona robin`

`server.py` also accepts `--persona {default,robin}` (default `default`; env `PERSONA`).
`robin` swaps the reply step for `robin_convo.py` — a port of the RECOVER Alexa skill's
`conversation()` (from `robin-ca-mirror/backend/recover/alexa.py`) with its own prompt
(`robin_prompt.txt`). It keeps the sentinel logic verbatim (DELETE_MESSAGE confirm/redact,
affirmation gate, `CONVERSATION_END` goodbye) but replaces the original's SQLAlchemy +
Chroma + weather/calendar infra with in-memory per-connection history and two optional
hooks: `ctx` (prompt placeholders like weather/schedule/profile) and `rag(user_text)`
(patient-history retrieval) — both default to blank, so wire real data in when needed. Uses
the same `--backend`/key as everything else; no credentials are copied from the source repo.

```bash
conda run -n voice python server.py --persona robin --backend openai
python robin_convo.py     # offline self-check (fake client, no network)
```

## Timing contract

Each stage's final stdout line is exactly:

```
TIMING stage=<name> load=<sec:.2f> infer=<sec:.2f>
```


source ~/miniconda3/etc/profile.d/conda.sh && conda activate voice && cd /mnt/e/PARCS/server/voice-server && uvicorn server:app --host 0.0.0.0 --port 8000

python server.py --host 0.0.0.0 --port 8000
~/cloudflared tunnel --url http://localhost:8000
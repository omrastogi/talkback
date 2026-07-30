# CLAUDE.md — voice-server

Voice-assistant cascade (wav/mic → STT → LLM → TTS → wav/speaker) + realtime browser server.
Runs in **WSL2 Ubuntu**, conda env **`voice`** at `/home/omras/miniconda3`. Source lives on
Windows (`E:\PARCS\server\voice-server`), run from WSL at `/mnt/e/PARCS/server/voice-server`.

## First: conda is NOT on PATH in a fresh WSL shell

`conda run -n voice ...` and `conda activate voice` fail with `conda: command not found`
until you source conda. The `(base)` prompt you see in Windows PowerShell is a *different*
conda (Windows), unrelated to WSL. Always source first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

## Launch the realtime server (canonical, confirmed working)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate voice && cd /mnt/e/PARCS/server/voice-server && python server.py --backend openai --port 8000
```

Then open <http://localhost:8000> (localhost required — mic needs a secure context).
Watch for the `[ready]` log line; both models load once at startup.

- Use `conda activate` (not `conda run`) for long-lived servers: `conda run` buffers the
  child's stdout and discards it on SIGTERM/kill, so the `[ready]` line never lands in your log.
  If you must use `conda run`, add `--no-capture-output`.
- **Backend switch:** `--backend openai` (snappy) or `--backend parcs` (default, offline gemma,
  ~5× slower). Or set `LLM_BACKEND=openai` before `uvicorn server:app ...`. Keys in `E:\PARCS\server\.env`.

## Remote access (browser mic from another device) — cloudflared

WSL2 is NAT'd (`0.0.0.0` binds WSL's adapter, not the Windows LAN IP) AND the mic needs a
secure context — so plain `http://<lan-ip>:8000` silently blocks the mic. Cloudflared quick
tunnel dodges both: public HTTPS URL, no sudo/firewall.

```bash
# one time
curl -L -o ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/cloudflared
# 2nd WSL tab, server already running
~/cloudflared tunnel --url http://localhost:8000
```

Prints a `https://*.trycloudflare.com` URL — mic works everywhere.

**Start the server FIRST and wait for `[ready]`, then start the tunnel.** server.py loads
Parakeet + Kokoro (~2–3 min) *before* uvicorn binds port 8000, so the port doesn't exist
until models finish. Hitting the tunnel early gives cloudflared
`dial tcp 127.0.0.1:8000: connect: connection refused` — the origin just isn't up yet, not a
tunnel fault. Confirm with `ss -tln | grep 8000` or `curl -s localhost:8000` (HTTP 200) first.

> **Warning:** that URL is a public, **unauthenticated** agent wired to your PARCS/OpenAI keys.
> Treat it as a secret and kill the tunnel when done.

## Env gotchas that cost real time

- **GPU = RTX 3050 Laptop, 4 GB.** VRAM tiering must use decimal GB (`bytes/1e9` ≈ 4.29), not
  GiB, or the card wrongly drops to the smallest tier.
- **Parakeet OOMs on 4 GB** if `from_pretrained` restores straight to GPU. Use
  `from_pretrained(..., map_location="cpu")` then `.to("cuda")`, and set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (config.py sets it before torch imports).
- **Kokoro's first synth compiles CUDA kernels (~3 s).** Warm it with a throwaway
  `pipe("Ready.", voice=...)` before timing.
- **`sounddevice` / `libportaudio2`:** WSL2 has no audio device. Import may raise
  `OSError: PortAudio library not found`; wrap in try/except, fall back to buffered wav.
  Never open an OutputStream in a latency-measured hot path — it blocks ~2 s before raising.
- **`espeak-ng` not installed** (apt needs an interactive password). TTS works without it for
  dictionary words; it's only the OOV phoneme fallback. Install:
  `sudo apt-get install -y ffmpeg espeak-ng libportaudio2`.
- **conda 25.x** blocks non-interactive `conda create` until channel ToS accepted:
  `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (and `/pkgs/r`).

## Config

`config.py` is the single `.env` loader + backend switch + `SYS_PROMPT` (health-support voice
agent persona). Imported before torch so it sets CUDA env. Every LLM entry point
(`server.py`, `pipeline.py`, `llm.py`, `stream.py`, `stream_demo.py`, `batch.py`,
`batch_stream.py`) takes `--backend openai|parcs`. Full details in README.md.

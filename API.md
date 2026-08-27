# Talkback voice service — client API

This documents the wire protocol implemented by `server.py`, as actually coded — not an
aspirational spec. Anything not listed here isn't part of the protocol.

## Endpoint

Public base (behind nginx, see `/etc/nginx/sites-available/gateway.parcs.northeastern.edu`):

```
wss://gateway.parcs.northeastern.edu/ai-caring/ca2
```

Local/direct (uvicorn, no proxy prefix):

```
ws://<host>:<port>
```

Two WebSocket routes exist, mounted at that base:

| Path | Handler | Protocol |
|---|---|---|
| `/ws-stream` | `ws_stream_endpoint` (`server.py:345`) | Streaming raw PCM, VAD-endpointed turns. **This is the default and the one a new client should use.** |
| `/ws` | `ws_endpoint` (`server.py:251`) | Legacy: one whole-utterance audio blob per turn (`index.html`, the "classic" demo). |

`/ws-stream` runs in one of two mutually-exclusive modes, fixed **server-wide** by the
`TURN_MODE` env var (`config.py:74`, default `"tap"`) — a client cannot select the mode
per-connection, it must know which mode the deployment is running:

- **`tap`** (default): client sends `turn_start` to arm listening; the server's Silero VAD
  decides when the utterance ends (endpointing).
- **`hold`**: client explicitly sends `start`/`end`; the server also streams live partial
  transcripts while held (see below). This is the "hold-to-talk" control path used to isolate
  VAD latency in benchmarking.

Everything below describes `tap` mode except where marked `[hold only]`.

## Authentication

Both `/ws` and `/ws-stream` are gated by a shared secret, `ROBIN_API_KEY` (`config.py:api_key`,
checked via `_check_auth` in `server.py:189`). If `ROBIN_API_KEY` is unset on the server, auth
is **disabled** — every connection is accepted, and the server logs a startup warning
(`server.py:99-101`). This is meant for local dev only; the deployed gateway sets it.

**The key travels as a WebSocket subprotocol, not a query parameter.** Pass it as the second
argument to the `WebSocket` constructor:

```javascript
const ws = new WebSocket(url, [apiKey]);   // sends `Sec-WebSocket-Protocol: <apiKey>`
```

This is deliberate: a query string (`?key=...`) ends up verbatim in both nginx's and uvicorn's
default access logs on *every* connection attempt, permanently persisting the shared secret in
cleartext on disk (confirmed empirically — this is not a hypothetical). Headers are not written
to either's default access-log format, so the subprotocol avoids that leak. The server reads it
via `ws.headers.get("sec-websocket-protocol")` and compares with `hmac.compare_digest` against
`ROBIN_API_KEY`. On success, the server echoes the key back as the negotiated subprotocol
(`ws.accept(subprotocol=API_KEY)`) — the client's `ws.protocol` will equal the key it sent, per
the WebSocket subprotocol-negotiation spec; there is no separate ack message.

**On a missing or wrong key**, the server logs a warning with the client's address (from the
`X-Real-IP` header nginx sets — `_client_ip`, `server.py:183`), then closes the connection with
WebSocket close code **1008** (policy violation). Note the sequencing: the server calls
`ws.accept()` and *then* immediately `ws.close(code=1008)`, rather than closing before accepting.
This is intentional, not an oversight — a close sent *before* the handshake completes fails the
handshake itself (surfaces to the server as returning a plain HTTP 403, and to a browser as
close code **1006**, never 1008, since no WebSocket connection was ever actually established to
carry a protocol-level close code). Accepting first is what makes code 1008 actually observable
client-side. This costs nothing extra: no audio is ever read on the rejection path, so no
STT/LLM/TTS work happens either way — the "expensive" work only starts once the caller sends
audio frames, which a rejected connection never gets to do.

A client should treat any close with code 1008 as "bad key, ask the user again" — that's the
only condition under which the server sends that code.

## Audio format sent by client

### `/ws-stream` (both modes)

- **Encoding**: raw PCM, signed 16-bit, little-endian (`Int16Array` bytes) — no container, no
  header, no codec.
- **Channels**: mono.
- **Sample rate**: whatever the client declares in the `start` message's `sampleRate` field
  (the reference browser client captures at 16000 Hz). The server resamples internally to
  16 kHz for VAD and STT (`vad/ingest.py` for tap, `pcm16_to_wav16` in `server.py:302` for
  hold) via `torchaudio.functional.resample`, so any accurately-declared rate works — it does
  not have to be 16000.
- **Chunk size**: not fixed by the protocol. The reference client sends one `AudioWorklet`
  callback's worth of samples per binary WS frame (~128 samples, ~8 ms at 16 kHz); the server
  just concatenates whatever arrives frame by frame (`server.py:449`, `server.py:389`).
- **Framing**: every binary WS message is a bare block of PCM16LE samples — `ws.send(i16.buffer)`
  client-side (`tap_index.html:154`), `np.frombuffer(msg["bytes"], dtype="<i2")` server-side.

### `/ws` (classic)

One binary WS message per turn = the browser's entire `MediaRecorder` output blob (mimeType is
whatever the browser picks, typically `audio/webm;codecs=opus`), sent once when recording stops
(`index.html:155`). The server decodes it with `ffmpeg` to 16 kHz mono PCM16 WAV
(`decode_to_wav`, `server.py:134`) — so in principle any container ffmpeg can read works, but
the reference client always sends the browser's native `MediaRecorder` container.

## Messages: server → client

JSON (text) frames and binary frames are interleaved on the same connection.

| type | frame | fields | sent when | example |
|---|---|---|---|---|
| `vad_state` | JSON | `state` (`"ARMED"\|"SPEECH"\|"TRAILING"\|"IDLE"`), `prob` (float 0–1) | `[tap only]` on every VAD state change, or ~10 Hz while unchanged (`server.py:453`) | `{"type":"vad_state","state":"SPEECH","prob":0.812}` |
| `transcript` | JSON | `text` (string, may be `""`) | once STT finishes for the turn | `{"type":"transcript","text":"Hello, how are you?"}` |
| `turn_start` | JSON | `turn_id` (string) | right before Robin's reply pipeline starts (`server.py:223`) | `{"type":"turn_start","turn_id":"20260812_150850_t01"}` |
| `reply` | JSON | `text` (one sentence) | once per sentence of the reply, immediately before that sentence's audio (`server.py:193`) | `{"type":"reply","text":"I am here now and want to listen."}` |
| `audio_meta` | JSON | `idx` (int, 0-based per turn), `server_send_ts` (float, `perf_counter()`), `bytes` (int), `sample_rate` (int, always `24000`) | immediately before the binary audio frame it describes (`server.py:199`) | `{"type":"audio_meta","idx":0,"server_send_ts":12345.678,"bytes":48044,"sample_rate":24000}` |
| *(binary)* | binary | — | right after the matching `audio_meta` | one complete WAV file: PCM16, 24000 Hz, mono (`wav_bytes`, `server.py:167`) — one sentence of synthesized speech |
| `done` | JSON | — | once the whole reply (all sentences) has been sent (`server.py:237`) | `{"type":"done"}` |
| `arm_timeout` | JSON | — | `[tap only]` no speech arrived within `vad_arm_timeout_s` of `turn_start`; the armed turn is cancelled (`server.py:460`) | `{"type":"arm_timeout"}` |
| `vad_error` | JSON | `text` (string) | `[tap only]` Silero VAD failed to initialize for this connection; the connection silently degrades to `hold` behavior for its lifetime (`server.py:418`) | `{"type":"vad_error","text":"VAD unavailable, using hold"}` |
| `partial` | JSON | `text` (string) | `[hold only]` live partial transcript, re-decoded every `PARTIAL_EVERY_S` (0.4s) of new audio while held (`server.py:398`) | `{"type":"partial","text":"Hello how"}` |

Notes:
- There is no message sent immediately on connect — the server only reacts to client input.
- `ttft`/`ttfs`/timing values are **not** sent to the client; they're server-side log/telemetry
  only (`server.py:292`).

## Messages: client → server

| type | frame | fields | endpoint / mode | effect |
|---|---|---|---|---|
| `start` | JSON | `sampleRate` (int) | `/ws-stream`, both modes | declares the PCM sample rate about to be streamed; (re)initializes VAD ingest state (`server.py:433`). In `tap` mode this is optional — if omitted, the first `turn_start` creates ingest state assuming 16000 Hz. |
| `turn_start` | JSON | — | `/ws-stream`, `tap` only | arms VAD for one new turn; server replies with `vad_state:{"state":"ARMED"}` (`server.py:435-445`). Must be resent for every turn — the client's reference implementation auto-resends it after each `done` (`tap_index.html:84-88`) to stay hands-free. |
| `end` | JSON | — | `/ws-stream`, `hold` only | marks end-of-utterance for the audio streamed since the preceding `start` (`server.py:378`) |
| `client_telemetry` | JSON | free-form (`{type:"client_telemetry", ...}`) | `/ws` only | playback-timing telemetry, appended verbatim to `logs/telemetry/<session>.jsonl` (`server.py:265-267`). **Sending this to `/ws-stream` is a no-op** — neither `_run_tap` nor `_run_hold` recognizes the type; it's silently ignored. |
| *(binary)* | binary | raw PCM16LE mono (see above) | `/ws-stream` | one chunk of the utterance being streamed |
| *(binary)* | binary | whole `MediaRecorder` blob | `/ws` | the entire recorded turn, sent once |

## Connection lifecycle

1. **Connect**: client opens the WebSocket, passing the API key as a subprotocol (see
   Authentication above). If `ROBIN_API_KEY` is set and the key is missing/wrong, the server
   accepts then immediately closes with code 1008 and returns — no session is created, nothing
   else happens. Otherwise, the server accepts (echoing the key back as the negotiated
   subprotocol), assigns a `session` id (`time.strftime("%Y%m%d_%H%M%S")`), and logs it;
   nothing else is sent to the client at this point.
2. **Turn start** (`tap`): client sends `turn_start` (and, once per connection, `start` if it
   wants to declare a non-16000 Hz rate) → server replies `vad_state: ARMED` → client streams
   raw PCM16 binary frames.
   **Turn start** (`hold`): client sends `start` → streams PCM16 binary frames → server may
   emit `partial` transcripts along the way → client sends `end`.
3. **End of turn**: (`tap`) the VAD detects onset then trailing silence past
   `vad_hangover_ms` (default 600 ms) and fires EOU internally — the client sends no explicit
   end signal. (`hold`) the client's `end` message is the explicit EOU.
4. **Reply pipeline** (identical for both modes/endpoints, `server.py:315` `_finalize_turn` /
   `server.py:217` `robin_reply`): STT transcription → `transcript` → `turn_start` →
   for each sentence of the LLM reply: `reply` (text) → `audio_meta` → binary WAV → finally
   `done`.
5. **Next turn**: (`tap`) client sends `turn_start` again to re-arm — no need to resend
   `start` or replay prior audio. Conversation history is kept **server-side**, in-memory, per
   connection (`history` list passed into and mutated in place by `process_turn`,
   `robin_conversation/engine.py`); the client never needs to resend earlier turns for context.
   If the client stays idle past `vad_arm_timeout_s` after `turn_start`, the server sends
   `arm_timeout` and that turn is abandoned (client must send `turn_start` again).
6. **Disconnect**: apart from the auth-rejection close (step 1), the server never calls
   `ws.close()` itself anywhere else in `server.py` — once a session is running, disconnects are
   only ever client- or network-initiated. The server just catches `WebSocketDisconnect`, logs
   `"disconnected session=... turns=N"`, and returns. There is no idle timeout, no max-turn
   limit, and no graceful-shutdown message.
7. **Error behavior (as coded, not a designed contract)**: `vad_error` is the *only* handled,
   reported failure mode (VAD init failure at connection time). A failure anywhere else in the
   turn pipeline — e.g. the LLM backend returning an error — is an unhandled exception that
   propagates out of the WebSocket route handler; there is no try/except around
   `robin_reply`/`process_turn` in `server.py`. A third-party client should be prepared for the
   connection to close abruptly with no JSON error message in that case.

## HTTP endpoints

| Method | Path | Returns | Auth | Notes |
|---|---|---|---|---|
| `GET` | `/health` | JSON: `{"status": "ok", "stt_loaded": bool, "tts_loaded": bool, "turn_mode": "tap"\|"hold"}` | **none** | Deliberately unauthenticated so monitoring can poll it without a key (`server.py:199-203`). `stt_loaded`/`tts_loaded` reflect whether the Parakeet/Kokoro model objects are non-`None`; since both load synchronously at import time before the app starts serving anything, in practice they're always `true` whenever this route is reachable at all. |
| `GET` | `/` | `tap_index.html` or `stream_index.html` | none | Chosen by the server-side `TURN_MODE` setting, not client-selectable. |
| `GET` | `/classic` | `index.html` | none | The `/ws` (whole-blob) demo page. |
| `GET` | `/stream` | `stream_index.html` | none | Direct link to the streaming demo page regardless of `TURN_MODE`. |

No HTTP route is authenticated — the shared key only gates the two WebSocket routes.

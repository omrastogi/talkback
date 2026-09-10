# Robin persistence & dashboard API

How to stand up, provision, and call the persistence layer added by the `robin/` package:
profiles, accounts, tokens, and per-utterance conversation history in Postgres. Like the
root `API.md`, this documents what is actually coded, not an aspirational spec. The voice
wire protocol itself (frames, audio format, VAD) is unchanged and stays documented in
`API.md`; this file covers everything the persistence layer added or changed.

Design notes trace back to `docs/reference/recover-database-review.pdf`, the review of the
RECOVER predecessor. The short version: explicit Pydantic response models per endpoint
(never a generic model-to-dict), a real FK on every cross-table reference, session identity
minted server-side (never inferred from LLM output), and identity in a real `profile` table
(not a filename join).

---

## 1. Getting it running

Everything runs in the `voice` conda env, from the repo root (`~/Project/talkback`).

### Postgres

A local dev cluster lives at `~/.local/share/robin-postgres` (PostgreSQL 18 from the `pg`
conda env), listening on `127.0.0.1:5433` with two databases: `robin` (dev/live) and
`robin_test` (torn down and recreated by the test suite — never point real data at it).

```bash
scripts/pg_dev.sh start     # start the cluster (needed before the server or the CLI)
scripts/pg_dev.sh status
scripts/pg_dev.sh stop
```

### Environment

The server, CLI, and Alembic all read the connection URL from `DATABASE_URL` — there is no
default in code, deliberately. It's set in the gitignored `.env`:

```
DATABASE_URL=postgresql+asyncpg://omrastogi@127.0.0.1:5433/robin
TEST_DATABASE_URL=postgresql+asyncpg://omrastogi@127.0.0.1:5433/robin_test
```

### Migrations

```bash
alembic upgrade head        # creates the citext extension + the five tables
```

Schema changes go through Alembic (`migrations/`), one **named** revision per change
(`alembic revision -m "..."`), models in `robin/db/models.py`.

### Server

```bash
python server.py            # or: uvicorn server:app --host 0.0.0.0 --port 8000
```

The `robin/` routers are mounted on the same FastAPI app as the voice sockets, so the base
URL is the same one the tablets use — locally `http://<host>:8000`, deployed
`https://gateway.parcs.northeastern.edu/ai-caring/ca2`.

### Dashboard frontend

A Next.js app in `frontend/` (adapted from the `robin-ca-mirror` frontend, branch `Alex`):
login, Users (admin provisioning), Chats (conversation viewer), Activities (daily usage).
Node lives in the `web` conda env; the API base URL comes from `frontend/.env.local`
(`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`).

```bash
export PATH=~/miniconda3/envs/web/bin:$PATH
cd frontend && npm install && npm run dev        # http://localhost:3000
```

The server allows the dev origin via CORS; override with a comma-separated
`ROBIN_CORS_ORIGINS` env var (default `http://localhost:3000,http://127.0.0.1:3000`).

### Tests

```bash
python -m pytest tests/ -q       # real Postgres (TEST_DATABASE_URL), never SQLite
```

---

## 2. Provisioning (admin CLI)

There is **no self-serve registration endpoint**. Accounts, profiles, links, and device
tokens are created by the research team — either from the dashboard's Users page (admin
accounts only; the same operations as HTTP endpoints, §4b) or with `python -m robin.admin`
(the bootstrap path for the first admin account; Postgres must be running; run it with the
env's python directly — `conda run` swallows the password prompt's stdin):

| command | args | prints |
|---|---|---|
| `create-account` | `--email --display-name [--admin]` (prompts for password twice) | the account id |
| `create-profile` | `--display-name [--timezone <IANA name>]` | the profile id |
| `link` | `--account-id --profile-id [--role owner\|viewer]` | confirmation |
| `issue-device-token` | `--profile-id --label "Tab A9 living room"` | the token id and the **raw token, once** |
| `revoke-token` | `--token-id` | confirmation |
| `list-tokens` | `[--profile-id]` | id, kind, subject, label, `last_used_at`, `revoked_at` — never the token |

Typical new-household sequence:

```bash
python -m robin.admin create-profile --display-name "Margaret" --timezone America/Chicago
python -m robin.admin create-account --email carepartner@example.org --display-name "Their Name"
python -m robin.admin link --account-id 1 --profile-id 1 --role owner
python -m robin.admin issue-device-token --profile-id 1 --label "Tab A9 living room"
```

The raw device token is shown exactly once. Only its SHA-256 is stored
(`robin/auth/tokens.py`); a lost token cannot be recovered — revoke it and issue a new one.

---

## 3. Auth model

Two kinds of opaque tokens (not JWTs — device tokens never expire and revocation needs a DB
lookup per request anyway), both rows in `auth_token`, distinguished by `kind`:

| kind | bound to | issued by | travels as | used for |
|---|---|---|---|---|
| `device` | one `profile` | CLI `issue-device-token` | WebSocket `Sec-WebSocket-Protocol` value | the voice sockets (`/ws`, `/ws-stream`) |
| `dashboard` | one `account` | `POST /auth/login` | `Authorization: Bearer <token>` header | every `/auth/*` and `/profiles/*` endpoint |

Verification (`robin/auth/tokens.py:verify_token`): SHA-256 the presented token, look up by
hash, reject unknown / revoked (`revoked_at` set) / wrong kind. `last_used_at` is updated on
success but throttled — skipped when it's within the last five minutes — so the voice socket
doesn't pay a write per connection.

Neither token kind ever travels in a query string: a query string lands verbatim in nginx's
and uvicorn's access logs on every request, permanently persisting the credential on disk
(the same reasoning, empirically confirmed, behind the old shared key's transport — see
`API.md` § Authentication).

### Voice socket auth (what changed)

`/ws` and `/ws-stream` now require a **device token** where they previously took the
`ROBIN_API_KEY` shared secret — same transport, new credential:

```javascript
const ws = new WebSocket(url, [deviceToken]);   // Sec-WebSocket-Protocol: <deviceToken>
```

On connect (`robin/ws.py:bind_device_session`):

1. Bad, revoked, missing, or non-device token (a dashboard token is **not** valid here) →
   the server accepts then immediately closes with code **4401** (accept-then-close for the
   same 1006-vs-observable-code reason documented in `API.md`; the legacy shared key closed
   with 1008 — clients should now treat 4401 as "re-provision this device").
2. On success the socket is accepted echoing the token as the negotiated subprotocol, the
   token's profile is resolved **server-side** — nothing the client sends (including the
   `start` frame's `external_id` or any `profile_id` field) can select a profile — and a
   fresh `session_id` UUID is minted for the connection. A reconnect is a new session; there
   is no resume.
3. The profile's `voice`, `speech_rate`, `timezone`, and `context` are loaded once and used
   for the connection's lifetime: Kokoro speaks with that voice at that rate, and the prompt
   context gets the profile's `context` JSON (as `personal_data_profile`) and its timezone
   (day/date/time-of-day). A dashboard `PATCH` takes effect on the device's next connect.

`ROBIN_API_KEY` no longer grants voice-socket access; it still gates the legacy ops
endpoints (§6).

### What gets persisted

One `conversation_turn` row per completed utterance, written in-path with `await` (no
background threads); a failed write is logged and never kills the connection or blocks TTS:

- **user** turns after STT finalizes (noise-only transcripts are not persisted),
  `source='voice'`, with STT timing / audio length / VAD stats in `meta`.
- **assistant** turns after the reply is final — including the wording correction after a
  device cancel-handshake — with model, backend, LLM/TTS seconds in `meta`, and
  `latency_ms` = VAD end-of-utterance → first TTS frame (tap mode; null where no EOU
  timestamp exists).
- Greets and proactive utterances (`_speak_unprompted`) as **assistant** turns with
  `source='proactive'`.

`content` is display text ("8:00 PM"); the spoken form from `for_speech()` is derived at
synthesis and never stored. `turn_index` is 0-based and unique per `session_id` (DB
constraint). `speaker_id` exists but stays null — reserved for future speaker ID.

---

## 4. Dashboard HTTP endpoints

All request/response bodies are JSON. Auth is `Authorization: Bearer <dashboard token>`
except `/auth/login`. Every response is an explicit Pydantic model (`robin/api/`); no
response anywhere contains `password` or `token_hash` in any shape (enforced by
`tests/test_api.py::test_no_response_contains_password_or_token_hash`).

### `POST /auth/login` — no auth

```json
{"email": "carepartner@example.org", "password": "..."}
```

→ `200` `{"token": "<raw dashboard token>", "account_id": 1, "display_name": "...", "is_admin": false}`

The raw token appears only here; store it client-side. Email matching is case-insensitive
(citext). Wrong password and unknown email both return the identical
`401 {"detail": "invalid credentials"}` — account existence is not confirmed.

### `POST /auth/logout`

Sets `revoked_at` on the exact token presented. → `200 {"revoked": true}`. The token fails
verification from then on.

### `GET /auth/me`

The account the presented token belongs to — the frontend re-validates its stored token
here on page load:

```json
{"account_id": 1, "email": "carepartner@example.org", "display_name": "...", "is_admin": false}
```

### `GET /profiles`

Profiles linked to the calling account via `account_profile`, with the caller's role on
each. **Admins see every profile**; on profiles they are not linked to, `role` is the
sentinel `"admin"` (owner-equivalent for PATCH):

```json
{"profiles": [{"id": 1, "display_name": "Margaret", "timezone": "America/Chicago",
               "voice": "af_heart", "speech_rate": 0.85, "context": {"likes": "gardening"},
               "active": true, "role": "owner",
               "created_at": "...", "updated_at": "..."}]}
```

### `GET /profiles/{id}`

One profile, same shape as a list entry. **`404` for any profile not linked to the caller —
never `403`** — and the body is identical to a truly nonexistent id, so the endpoint cannot
be used to probe which profile ids exist.

### `PATCH /profiles/{id}` — role `owner` required (`403` for viewers)

Exactly five mutable fields; anything else in the body is a `422`, not silently dropped:

```json
{"display_name": "...", "timezone": "America/Chicago", "voice": "am_adam",
 "speech_rate": 0.9, "context": {"likes": "gardening"}}
```

All optional; `timezone` must be a real IANA name (`422` otherwise); `speech_rate` must be
in (0, 3]. Returns the updated profile. Note `active` is not patchable over HTTP —
deactivation is an operator action. Changes reach the device on its next connect.

### `GET /profiles/{id}/sessions`

Conversations as `(session_id, timing, turn count)` summaries, most recently active first —
so a chat viewer never has to page every turn to group them. Query parameters: `before`
(a `last_at` cursor) and `limit` (default 50, max 200).

```json
{"sessions": [{"session_id": "bc27e653-...", "started_at": "...", "last_at": "...",
               "turn_count": 12, "sources": ["proactive", "voice"]}]}
```

### `GET /profiles/{id}/activity`

Per-day usage derived from `conversation_turn`, bucketed in the **profile's timezone**
(a "day" is the person's local day). `date_from`/`date_to` (ISO dates) default to the last
30 days; `last_active_at` is unbounded by the range.

```json
{"timezone": "America/Chicago", "last_active_at": "...",
 "days": [{"day": "2026-09-08", "sessions": 1, "user_turns": 4, "assistant_turns": 5,
           "proactive_turns": 1, "voice_turns": 8, "avg_latency_ms": 1840.0}]}
```

### `GET /profiles/{id}/turns`

Conversation history, newest first. Query parameters:

| param | meaning |
|---|---|
| `session_id` | only turns from one conversation (the UUID on each turn row) |
| `before` | only turns with `created_at` strictly before this timestamp — pass the oldest `created_at` from the previous page to paginate |
| `limit` | page size, default 100, max 500 |

```json
{"turns": [{"id": 42, "session_id": "bc27e653-...", "turn_index": 1, "role": "assistant",
            "content": "Good morning! ...", "source": "voice", "latency_ms": 1840,
            "meta": {"model": "gemma4:12b:fast", "llm_s": 1.2}, "created_at": "..."}]}
```

Same 404-for-unlinked rule as above.

### curl walk-through

```bash
TOKEN=$(curl -s localhost:8000/auth/login \
        -d '{"email":"carepartner@example.org","password":"..."}' \
        -H 'content-type: application/json' | jq -r .token)
curl -s localhost:8000/profiles -H "Authorization: Bearer $TOKEN" | jq
curl -s localhost:8000/profiles/1/turns?limit=20 -H "Authorization: Bearer $TOKEN" | jq
curl -s -X PATCH localhost:8000/profiles/1 -H "Authorization: Bearer $TOKEN" \
     -H 'content-type: application/json' -d '{"speech_rate": 0.9}' | jq
curl -s -X POST localhost:8000/auth/logout -H "Authorization: Bearer $TOKEN"
```

---

## 4b. Admin provisioning endpoints (`robin/api/admin.py`)

Every route requires an **admin** dashboard token (`403 "admin required"` otherwise) and
mirrors a CLI command. Conflicts are `409`s; the raw device token appears exactly once, in
the issuance response.

| Method | Path | Body → Response |
|---|---|---|
| `POST` | `/accounts` | `{email, password (min 8), display_name, is_admin?}` → the account (201; duplicate email 409) |
| `GET` | `/accounts` | all accounts |
| `POST` | `/profiles` | `{display_name, timezone?, voice?, speech_rate?, context?}` → the profile (201) |
| `GET` | `/profiles/{id}/links` | the profile's account links |
| `POST` | `/profiles/{id}/links` | `{account_id, role: owner\|viewer}` → the link (201; duplicate 409) |
| `DELETE` | `/profiles/{id}/links/{account_id}` | `{"deleted": true}` |
| `POST` | `/profiles/{id}/device-tokens` | `{label}` → `{token (raw, shown once), token_id, ...}` (201) |
| `GET` | `/tokens?profile_id=&include_revoked=false` | token metadata — never the hash |
| `POST` | `/tokens/{id}/revoke` | `{"revoked": true}` |

---

## 5. Complete endpoint map

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `WS` | `/ws-stream` | device token (subprotocol) | streaming voice, VAD-endpointed — the deployed path (`API.md`) |
| `WS` | `/ws` | device token (subprotocol) | legacy whole-blob voice turns (`API.md`) |
| `POST` | `/auth/login` | none (credentials in body) | issue a dashboard token |
| `POST` | `/auth/logout` | dashboard Bearer | revoke the presented token |
| `GET` | `/auth/me` | dashboard Bearer | the token's account |
| `GET` | `/profiles` | dashboard Bearer | linked profiles + caller's role (admins: all) |
| `GET` | `/profiles/{id}` | dashboard Bearer | one profile; 404 if unlinked (non-admins) |
| `PATCH` | `/profiles/{id}` | dashboard Bearer, role `owner`/admin | the five mutable fields |
| `GET` | `/profiles/{id}/sessions` | dashboard Bearer | conversation summaries, paginated |
| `GET` | `/profiles/{id}/activity` | dashboard Bearer | per-day usage in the profile's timezone |
| `GET` | `/profiles/{id}/turns` | dashboard Bearer | conversation history, paginated |
| — | `/accounts`, `/profiles` (POST), `/profiles/{id}/links*`, `/profiles/{id}/device-tokens`, `/tokens*` | **admin** Bearer | provisioning (§4b) |
| `GET` | `/health` | none | liveness + model/turn-mode status |
| `GET` | `/`, `/classic`, `/stream` | none | demo pages |
| `GET` | `/dashboard` | none (page); its API calls need the key | live-session dashboard page |
| `GET` | `/api/sessions` | `X-Robin-Key` (legacy shared key) | live in-memory session registry |
| `POST` | `/api/sessions/{id}/greet` | `X-Robin-Key` | speak a greeting into a live session |
| `POST` | `/proactive` | `X-Robin-Key` | external service → spoken utterance (see docstring in `server.py`) |

The last three predate this layer and still use the `ROBIN_API_KEY` shared secret as the
`X-Robin-Key` header; folding them into account tokens is future work.

---

## 6. Code map

```
robin/
  db/__init__.py      async engine + session factory; DATABASE_URL required, no default
  db/models.py        Profile, Account, AccountProfile, AuthToken, ConversationTurn
  auth/passwords.py   argon2id hash/verify (argon2-cffi defaults)
  auth/tokens.py      issue / verify (throttled last_used_at) / revoke
  auth/deps.py        require_dashboard, role lookup, authenticate_device_ws
  api/auth.py         /auth/login, /auth/logout, /auth/me
  api/profiles.py     /profiles endpoints (incl. /sessions, /activity) + response models
  api/admin.py        admin provisioning endpoints (§4b)
  ws.py               bind_device_session (WS auth + profile snapshot), persist_turn
  admin/              the CLI (python -m robin.admin)
migrations/           Alembic (async env; URL from DATABASE_URL)
frontend/             Next.js dashboard (login, Users, Chats, Activities)
tests/                pytest against TEST_DATABASE_URL; stub app mounts the same
                      bind_device_session/persist_turn wiring server.py uses
```

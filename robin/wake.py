"""Per-profile wake-word head delivery over the voice socket.

The tablet runs a local "Hey Robin" detector; a personalized head (trained by
oww-train/enroll_train.py, ingested by scripts/wake_model_ingest.py) reaches it through
the same WebSocket it already speaks:

  server -> client  {"type": "wake_model_meta", ...}   once after bind (the offer), and
                    again immediately before the blob -- the audio_meta pattern: every
                    server binary frame is announced by a JSON frame right before it
  client -> server  {"type": "wake_model_fetch"}       ask for the blob
  server -> client  (binary)                           the ONNX head, one frame (~205 KB)

The client compares `sha256` against its cached copy and fetches only when it differs;
after verifying the hash it swaps atomically and reloads. On any failure -- no row, bad
hash, mid-transfer disconnect -- it keeps the shipped base model and base threshold.
`threshold` always travels in the meta frame next to the model it was calibrated for,
never separately: the pair is one artifact (oww-train LORA.md).

Like persist_turn, everything here is best-effort: a DB failure is logged and swallowed,
never allowed to kill a voice connection.
"""
import logging

from sqlalchemy import func, select

from robin.db import get_sessionmaker
from robin.db.models import WakeModel

log = logging.getLogger("voice")


def _meta(row) -> dict:
    sha256, threshold, base_version, nbytes, created_at = row
    return {"type": "wake_model_meta", "available": True, "sha256": sha256,
            "threshold": threshold, "base_version": base_version, "bytes": nbytes,
            "created_at": created_at.isoformat()}


async def _active_meta(profile_id: int):
    async with get_sessionmaker()() as session:
        row = (await session.execute(
            select(WakeModel.sha256, WakeModel.threshold, WakeModel.base_version,
                   func.length(WakeModel.onnx), WakeModel.created_at)
            .where(WakeModel.profile_id == profile_id, WakeModel.active))).first()
    return _meta(row) if row else None


async def offer_wake_model(ws, bound) -> None:
    """Announce the profile's active head right after bind. No active head, no frame:
    old clients and base-model tablets see nothing new."""
    if bound is None:
        return
    try:
        meta = await _active_meta(bound.profile_id)
    except Exception as e:                       # noqa: BLE001 — never kill the connection
        log.error("wake model offer failed (profile=%d): %r", bound.profile_id, e)
        return
    if meta is not None:
        await ws.send_json(meta)


async def send_wake_model(ws, bound) -> None:
    """Answer a wake_model_fetch: meta frame, then the ONNX as one binary frame. When no
    head is active the reply is a meta frame with available=false, so the client is never
    left waiting for a binary that will not come."""
    if bound is None:
        return
    try:
        async with get_sessionmaker()() as session:
            row = (await session.execute(
                select(WakeModel.sha256, WakeModel.threshold, WakeModel.base_version,
                       func.length(WakeModel.onnx), WakeModel.created_at, WakeModel.onnx)
                .where(WakeModel.profile_id == bound.profile_id, WakeModel.active))).first()
    except Exception as e:                       # noqa: BLE001 — never kill the connection
        log.error("wake model fetch failed (profile=%d): %r", bound.profile_id, e)
        return
    if row is None:
        await ws.send_json({"type": "wake_model_meta", "available": False})
        return
    await ws.send_json(_meta(row[:5]))
    await ws.send_bytes(row[5])
    log.info("wake model sent  profile=%d sha=%s bytes=%d",
             bound.profile_id, row[0][:12], row[3])

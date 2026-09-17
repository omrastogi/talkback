"""Wake-word enrollment endpoints for the dashboard: record "Hey Robin" takes against a
profile, review them, and see the trained head's status. The dashboard records 16 kHz
mono PCM16 WAV client-side (the trainer's exact input contract) and uploads the raw WAV
body — no transcoding server-side, so what is stored is bit-for-bit what was recorded.

Clips feed `python -m robin.admin wake-clips-export` -> oww-train/enroll_train.py ->
`python -m robin.admin wake-model-ingest`; the tablet then pulls the head over the voice
socket (robin/wake.py). The ONNX blob itself is deliberately not exposed here — the
dashboard needs status, not model bytes.

Same access rules as the rest of /profiles: unlinked profiles 404 (never confirm
existence), mutations need the owner role, viewers can look and listen.
"""
import asyncio
import datetime
import hashlib
import io
import os
import re
import tempfile
import wave
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from robin.auth.deps import require_dashboard, require_device_profile
from robin.db import get_session
from robin.db.models import Account, Profile, WakeClip, WakeModel
from robin.api.profiles import _linked_profile_or_404
from robin import train as train_jobs

router = APIRouter(prefix="/profiles", tags=["wake"])
# The same enrollment operations, authenticated by the tablet's own device token instead
# of a dashboard account. No profile_id in the path: the token's binding fixes it.
device_router = APIRouter(prefix="/device", tags=["wake"])

MAX_CLIP_BYTES = 2_000_000                       # 6 s of 16 kHz PCM16 is ~192 KB; 2 MB is generous
DUR_BOUNDS = {"positive": (0.25, 4.0), "negative": (0.15, 6.0)}

# Content gates, registered by server.py at startup. This module never imports
# server.py — the same inversion bind_device_session lives behind — so the test suite
# registers fakes or leaves them unset (unset = that check is skipped).
#
# Two gates because they catch different failures (proven by Margaret's first run,
# where 3 of 7 STT-approved takes scored ~0.0 on the wake head and poisoned training):
#   - the TRANSCRIBER (the server's own STT) checks the words: a positive must say the
#     wake word, a negative must not;
#   - the WAKE SCORER (the shared base detector itself) checks that the take actually
#     registers as the wake word to the model that matters — STT decodes speech far too
#     degraded for a 50k-parameter head, so STT approval alone lets duds through.
_transcriber: Callable[[str], str] | None = None
_wake_scorer: Callable[[str], float] | None = None

# Floor for the wake-head peak score on a positive take. Margaret's duds scored <= 0.05,
# genuine takes >= 0.72 against the base's ~0.82 threshold — 0.2 splits that cleanly
# while leaving room for honest-but-hard takes to enter training.
WAKE_SCORE_FLOOR = 0.2


def set_transcriber(fn: Callable[[str], str] | None) -> None:
    """Register a blocking `wav_path -> transcript` function (server.py's stt_transcribe)."""
    global _transcriber
    _transcriber = fn


def set_wake_scorer(fn: Callable[[str], float] | None) -> None:
    """Register a blocking `wav_path -> peak wake score` function (the shared base head)."""
    global _wake_scorer
    _wake_scorer = fn


# Squashed-transcript spellings that count as "Hey Robin" — STT renders the name a few
# ways ("Robyn", a dropped h), and rejecting those would fail honest takes.
_WAKE_SQUASHED = ("heyrobin", "heyrobyn", "heyrobbin", "hayrobin", "hayrobyn",
                  "heirobin", "herobin", "herobyn")


def _heard_wake_word(transcript: str) -> bool:
    squashed = re.sub(r"[^a-z]", "", transcript.lower())
    return any(v in squashed for v in _WAKE_SQUASHED)


async def _content_gate(body: bytes, label: str) -> None:
    """422 a bad take with a message the dashboard shows verbatim. Two checks, each
    skipped when its hook is unregistered: the transcript gate (right words?) and the
    wake-score gate (does the detector itself hear the wake word in a positive?)."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        if _transcriber is not None:
            transcript = (await asyncio.to_thread(_transcriber, path)).strip()
            _check_transcript(transcript, label)
        if _wake_scorer is not None and label == "positive":
            score = float(await asyncio.to_thread(_wake_scorer, path))
            if score < WAKE_SCORE_FLOOR:
                raise HTTPException(status_code=422, detail=(
                    f"The words were right, but the wake-word detector itself barely "
                    f"reacted to that take (score {score:.2f}) — it is probably cut "
                    "off or caught before the microphone was ready. Wait a beat after "
                    "pressing record, then say “Hey Robin” again."))
    finally:
        os.unlink(path)


def _check_transcript(transcript: str, label: str) -> None:
    heard = _heard_wake_word(transcript)
    if label == "positive" and not heard:
        if not transcript:
            raise HTTPException(status_code=422, detail=(
                "That take came through silent or garbled — nothing recognizable was "
                "heard. Check that the microphone is working and close enough, then "
                "record the take again."))
        raise HTTPException(status_code=422, detail=(
            f"That take came through garbled — it sounded like “{transcript[:60]}” "
            "rather than “Hey Robin”. Check the microphone and record the take again."))
    if label == "negative" and heard:
        raise HTTPException(status_code=422, detail=(
            "This take contains the wake word — “other speech” takes must not say "
            "“Hey Robin”. Record a sentence of ordinary speech instead."))


class WakeClipInfo(BaseModel):
    id: int
    label: str
    duration_s: float
    sha256: str
    created_at: datetime.datetime


class WakeClipListResponse(BaseModel):
    clips: list[WakeClipInfo]


class WakeModelStatus(BaseModel):
    """The active head's metadata, or available=false. Calibration numbers come from the
    trainer manifest; the blob stays server-side. `clips_changed` drives the dashboard's
    Register button: the current clip set differs from the one the active model was
    trained on (derived from the manifest's clip hashes, never tracked separately).
    `training` is the in-flight/most-recent job for this profile, if any."""
    available: bool
    sha256: str | None = None
    base_version: str | None = None
    threshold: float | None = None
    created_at: datetime.datetime | None = None
    manifest: dict | None = None
    clips_changed: bool = False
    positives: int = 0
    training: dict | None = None


def _validate_wav(body: bytes, label: str) -> float:
    """The clip must already be the trainer's contract: 16 kHz mono PCM16 WAV within the
    duration bounds. Returns the duration. 422 on anything else — the dashboard encodes
    correctly, so a failure here means a broken upload, not a format to accommodate."""
    try:
        with wave.open(io.BytesIO(body)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            duration = w.getnframes() / rate if rate else 0.0
    except (wave.Error, EOFError):
        raise HTTPException(status_code=422, detail="not a PCM WAV file")
    if (rate, channels, width) != (16000, 1, 2):
        raise HTTPException(status_code=422,
                            detail=f"expected 16 kHz mono PCM16, got {rate} Hz, "
                                   f"{channels} ch, {width * 8}-bit")
    lo, hi = DUR_BOUNDS[label]
    if not (lo <= duration <= hi):
        raise HTTPException(status_code=422,
                            detail=f"duration {duration:.2f}s outside [{lo}, {hi}]s")
    return duration


async def _owner_or_admin(session: AsyncSession, account: Account, profile_id: int):
    _, role = await _linked_profile_or_404(session, account, profile_id)
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="owner role required")


async def _store_clip(session: AsyncSession, profile_id: int, label: str,
                      body: bytes) -> WakeClipInfo:
    if len(body) > MAX_CLIP_BYTES:
        raise HTTPException(status_code=413, detail="clip too large")
    duration = _validate_wav(body, label)
    await _content_gate(body, label)
    clip = WakeClip(profile_id=profile_id, label=label, wav=body,
                    duration_s=round(duration, 3),
                    sha256=hashlib.sha256(body).hexdigest())
    session.add(clip)
    try:
        await session.commit()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="identical clip already uploaded")
    await session.refresh(clip)
    return WakeClipInfo(id=clip.id, label=clip.label, duration_s=clip.duration_s,
                        sha256=clip.sha256, created_at=clip.created_at)


async def _clip_list(session: AsyncSession, profile_id: int) -> WakeClipListResponse:
    rows = (await session.execute(
        select(WakeClip.id, WakeClip.label, WakeClip.duration_s, WakeClip.sha256,
               WakeClip.created_at)
        .where(WakeClip.profile_id == profile_id).order_by(WakeClip.id))).all()
    return WakeClipListResponse(clips=[
        WakeClipInfo(id=r.id, label=r.label, duration_s=r.duration_s, sha256=r.sha256,
                     created_at=r.created_at)
        for r in rows
    ])


@router.post("/{profile_id}/wake-clips", response_model=WakeClipInfo, status_code=201)
async def upload_wake_clip(profile_id: int, request: Request,
                           label: str = Query(default="positive",
                                              pattern="^(positive|negative)$"),
                           account: Account = Depends(require_dashboard),
                           session: AsyncSession = Depends(get_session)):
    """Body is the raw WAV (Content-Type: audio/wav), not JSON — an audio blob has no
    business being base64'd through a JSON layer."""
    await _owner_or_admin(session, account, profile_id)
    return await _store_clip(session, profile_id, label, await request.body())


@router.get("/{profile_id}/wake-clips", response_model=WakeClipListResponse)
async def list_wake_clips(profile_id: int,
                          account: Account = Depends(require_dashboard),
                          session: AsyncSession = Depends(get_session)):
    await _linked_profile_or_404(session, account, profile_id)
    return await _clip_list(session, profile_id)


@router.get("/{profile_id}/wake-clips/{clip_id}/audio")
async def wake_clip_audio(profile_id: int, clip_id: int,
                          account: Account = Depends(require_dashboard),
                          session: AsyncSession = Depends(get_session)):
    await _linked_profile_or_404(session, account, profile_id)
    wav = (await session.execute(
        select(WakeClip.wav).where(WakeClip.id == clip_id,
                                   WakeClip.profile_id == profile_id))).scalar_one_or_none()
    if wav is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return Response(content=wav, media_type="audio/wav")


async def _remove_clip(session: AsyncSession, profile_id: int, clip_id: int) -> dict:
    result = await session.execute(
        delete(WakeClip).where(WakeClip.id == clip_id, WakeClip.profile_id == profile_id))
    await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="clip not found")
    return {"deleted": True}


@router.delete("/{profile_id}/wake-clips/{clip_id}")
async def delete_wake_clip(profile_id: int, clip_id: int,
                           account: Account = Depends(require_dashboard),
                           session: AsyncSession = Depends(get_session)):
    await _owner_or_admin(session, account, profile_id)
    return await _remove_clip(session, profile_id, clip_id)


async def _model_status(session: AsyncSession, profile_id: int) -> WakeModelStatus:
    row = (await session.execute(
        select(WakeModel.sha256, WakeModel.base_version, WakeModel.threshold,
               WakeModel.created_at, WakeModel.manifest)
        .where(WakeModel.profile_id == profile_id, WakeModel.active))).first()
    changed, positives = await train_jobs.clips_changed(profile_id)
    training = train_jobs.job_status(profile_id)
    if row is None:
        return WakeModelStatus(available=False, clips_changed=changed,
                               positives=positives, training=training)
    return WakeModelStatus(available=True, sha256=row.sha256, base_version=row.base_version,
                           threshold=row.threshold, created_at=row.created_at,
                           manifest=row.manifest, clips_changed=changed,
                           positives=positives, training=training)


@router.get("/{profile_id}/wake-model", response_model=WakeModelStatus)
async def wake_model_status(profile_id: int,
                            account: Account = Depends(require_dashboard),
                            session: AsyncSession = Depends(get_session)):
    await _linked_profile_or_404(session, account, profile_id)
    return await _model_status(session, profile_id)


MIN_POSITIVES = 1        # no forced enrollment size — but zero takes is nothing to train on


async def _guard_and_start(profile_id: int) -> dict:
    """Register-button guards, so a stale page cannot start a pointless or overlapping
    run: one job per profile, at least MIN_POSITIVES takes, and only when the clip set
    differs from what the active model was trained on."""
    job = train_jobs.job_status(profile_id)
    if job is not None and job["state"] == "running":
        raise HTTPException(status_code=409, detail="training is already running")
    changed, positives = await train_jobs.clips_changed(profile_id)
    if positives < MIN_POSITIVES:
        raise HTTPException(status_code=422,
                            detail="record at least one “Hey Robin” take first")
    if not changed:
        raise HTTPException(status_code=409,
                            detail="the active model was already trained on exactly "
                                   "these takes — record or delete a take first")
    train_jobs.start_training(profile_id)
    return {"started": True}


@router.post("/{profile_id}/wake-model/train", status_code=202)
async def train_wake_model(profile_id: int,
                           account: Account = Depends(require_dashboard),
                           session: AsyncSession = Depends(get_session)):
    """The dashboard's Register button."""
    await _owner_or_admin(session, account, profile_id)
    return await _guard_and_start(profile_id)


# --- Device-scoped twins: the tablet enrolls the person it is provisioned for. ---

@device_router.post("/wake-clips", response_model=WakeClipInfo, status_code=201)
async def device_upload_wake_clip(request: Request,
                                  label: str = Query(default="positive",
                                                     pattern="^(positive|negative)$"),
                                  profile: Profile = Depends(require_device_profile),
                                  session: AsyncSession = Depends(get_session)):
    return await _store_clip(session, profile.id, label, await request.body())


@device_router.get("/wake-clips", response_model=WakeClipListResponse)
async def device_list_wake_clips(profile: Profile = Depends(require_device_profile),
                                 session: AsyncSession = Depends(get_session)):
    return await _clip_list(session, profile.id)


@device_router.get("/wake-clips/{clip_id}/audio")
async def device_wake_clip_audio(clip_id: int,
                                 profile: Profile = Depends(require_device_profile),
                                 session: AsyncSession = Depends(get_session)):
    wav = (await session.execute(
        select(WakeClip.wav).where(WakeClip.id == clip_id,
                                   WakeClip.profile_id == profile.id))).scalar_one_or_none()
    if wav is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return Response(content=wav, media_type="audio/wav")


@device_router.delete("/wake-clips/{clip_id}")
async def device_delete_wake_clip(clip_id: int,
                                  profile: Profile = Depends(require_device_profile),
                                  session: AsyncSession = Depends(get_session)):
    return await _remove_clip(session, profile.id, clip_id)


@device_router.get("/wake-model", response_model=WakeModelStatus)
async def device_wake_model_status(profile: Profile = Depends(require_device_profile),
                                   session: AsyncSession = Depends(get_session)):
    return await _model_status(session, profile.id)


@device_router.post("/wake-model/train", status_code=202)
async def device_train_wake_model(profile: Profile = Depends(require_device_profile)):
    """The tablet's Register Voice button — same guards as the dashboard's."""
    return await _guard_and_start(profile.id)

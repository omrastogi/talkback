"""Server-side wake-model training job, behind the dashboard's Register button.

One job per profile at a time: export the profile's clips to a temp dir, run
scripts/enroll_train.py as a subprocess in the oww-train env (training needs that repo's
data and deps — see the script's docstring), then ingest and activate the result. Job
state is in-memory and per-process: a server restart forgets a running job, which is the
truth anyway (the subprocess dies with its parent's session or finishes into a temp dir
nobody reads).

Whether a profile NEEDS training is not job state — it is derived by comparing the
current wake_clip sha256s against the clip shas recorded in the active model's manifest
(the trainer writes every training clip's hash there precisely so provenance questions
like this stay answerable). Same set → the Register button locks; any add or delete →
it unlocks.
"""
import asyncio
import datetime
import hashlib
import json
import logging
import os
import pathlib
import tempfile

from sqlalchemy import select, update

from robin.db import get_sessionmaker
from robin.db.models import Profile, WakeClip, WakeModel

log = logging.getLogger("voice")

# Trainer runs with oww-train's env against that repo's data; both overridable for a
# machine where they live elsewhere.
TRAIN_PYTHON = os.environ.get(
    "ROBIN_TRAIN_PYTHON",
    str(pathlib.Path.home() / "miniconda3/envs/oww-train/bin/python"))
OWW_ROOT = os.environ.get("ROBIN_OWW_ROOT", str(pathlib.Path.home() / "Project/oww-train"))
TRAIN_SCRIPT = str(pathlib.Path(__file__).resolve().parent.parent / "scripts/enroll_train.py")

_jobs: dict[int, dict] = {}          # profile_id -> {state, detail, started_at, ...}


class TrainError(Exception):
    pass


def job_status(profile_id: int) -> dict | None:
    return _jobs.get(profile_id)


def _clip_shas(rows) -> dict[str, set[str]]:
    out = {"positive": set(), "negative": set()}
    for label, sha in rows:
        out[label].add(sha)
    return out


async def clips_changed(profile_id: int) -> tuple[bool, int]:
    """(changed, positive_count): does the current clip set differ from the one the
    active model was trained on? No active model: changed iff any clips exist."""
    async with get_sessionmaker()() as session:
        rows = (await session.execute(
            select(WakeClip.label, WakeClip.sha256)
            .where(WakeClip.profile_id == profile_id))).all()
        manifest = (await session.execute(
            select(WakeModel.manifest)
            .where(WakeModel.profile_id == profile_id, WakeModel.active))).scalar_one_or_none()
    current = _clip_shas(rows)
    n_pos = len(current["positive"])
    if manifest is None:
        return bool(rows), n_pos
    trained = manifest.get("clips", {})
    trained_sets = {"positive": set((trained.get("positives") or {}).values()),
                    "negative": set((trained.get("negatives") or {}).values())}
    return current != trained_sets, n_pos


def start_training(profile_id: int) -> None:
    """Begin the job (caller has already checked the guards). Runs as an asyncio task on
    the server's loop; the heavy work is all in the subprocess."""
    _jobs[profile_id] = {"state": "running", "detail": "training",
                         "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    asyncio.get_running_loop().create_task(_train_job(profile_id))


async def _train_job(profile_id: int) -> None:
    try:
        model_dir = await _export_and_train(profile_id)
        await _ingest(profile_id, model_dir)
        _jobs[profile_id] = {**_jobs[profile_id], "state": "done", "detail": "model active"}
        log.info("wake training done  profile=%d", profile_id)
    except TrainError as e:
        _jobs[profile_id] = {**_jobs[profile_id], "state": "failed", "detail": str(e)}
        log.error("wake training failed  profile=%d: %s", profile_id, e)
    except Exception as e:                       # noqa: BLE001 — job must record, not raise
        _jobs[profile_id] = {**_jobs[profile_id], "state": "failed",
                             "detail": f"unexpected error: {e!r}"}
        log.exception("wake training crashed  profile=%d", profile_id)


async def _export_and_train(profile_id: int) -> pathlib.Path:
    """Write the profile's clips as the trainer's input layout and run it. Returns the
    model output dir. Patched out in tests (no GPU, no oww-train env there)."""
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"enroll_p{profile_id}_"))
    async with get_sessionmaker()() as session:
        rows = (await session.execute(
            select(WakeClip.id, WakeClip.label, WakeClip.wav, WakeClip.sha256)
            .where(WakeClip.profile_id == profile_id).order_by(WakeClip.id))).all()
        person = (await session.execute(
            select(Profile.display_name).where(Profile.id == profile_id))).scalar_one()
    for r in rows:
        d = work / f"{r.label}s"
        d.mkdir(exist_ok=True)
        (d / f"clip_{r.id:04d}_{r.sha256[:8]}.wav").write_bytes(r.wav)
    (work / "positives").mkdir(exist_ok=True)   # trainer globs it even if empty

    out = work / "model"
    log_path = work / "train.log"
    proc = await asyncio.create_subprocess_exec(
        TRAIN_PYTHON, TRAIN_SCRIPT, str(work), "--out", str(out),
        "--oww-root", OWW_ROOT, "--person", f"profile-{profile_id}-{person}",
        cwd=str(pathlib.Path(TRAIN_SCRIPT).parent.parent),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    output, _ = await proc.communicate()
    log_path.write_bytes(output)
    if proc.returncode == 3:
        # The trainer writes the manifest even on a failed gate; surface its numbers so
        # nobody has to dig a temp dir out of a log to see how close the run came.
        numbers = ""
        try:
            m = json.loads((out / "manifest.json").read_text())
            numbers = (f" (false accepts {m['fa_at_0.5_per_h']:.1f}/h vs budget "
                       f"{m['fa_budget_per_h']:.1f}/h, shared base "
                       f"{m['base_fa_at_0.5_per_h']:.1f}/h; it recognized "
                       f"{m['enroll_recall_insample']:.0%} of the training takes)")
        except Exception:                        # noqa: BLE001 — detail is best-effort
            pass
        raise TrainError(
            "training finished but the head failed the false-accept gate — it would "
            f"wake too easily on background noise, so it was not activated{numbers}. "
            "More varied takes usually fix this: different distances, volumes, and "
            "tones of voice.")
    if proc.returncode != 0:
        tail = output.decode(errors="replace").strip().splitlines()[-3:]
        raise TrainError(f"trainer exited {proc.returncode}: {' | '.join(tail)} "
                         f"(full log: {log_path})")
    return out


async def _ingest(profile_id: int, model_dir: pathlib.Path) -> None:
    """Same verification and atomic swap as `python -m robin.admin wake-model-ingest`."""
    manifest = json.loads((model_dir / "manifest.json").read_text())
    onnx = (model_dir / manifest["onnx"]).read_bytes()
    if manifest.get("status") != "ok":
        raise TrainError(f"manifest status {manifest.get('status')!r}, not 'ok'")
    if hashlib.sha256(onnx).hexdigest() != manifest["sha256"]:
        raise TrainError("onnx sha256 does not match the manifest")
    async with get_sessionmaker()() as session:
        await session.execute(update(WakeModel)
                              .where(WakeModel.profile_id == profile_id, WakeModel.active)
                              .values(active=False))
        session.add(WakeModel(profile_id=profile_id,
                              base_version=manifest["base"]["version"],
                              sha256=manifest["sha256"], onnx=onnx,
                              threshold=manifest["threshold"], manifest=manifest,
                              active=True))
        await session.commit()

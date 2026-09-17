"""Wake-word personalization: enrollment clip endpoints, the ingest CLI, and head
delivery over the voice socket. The delivery invariant under test everywhere: the
threshold travels in the same meta frame as the model identity, and a client is never
left waiting for a binary frame that will not come."""
import asyncio
import hashlib
import json

import pytest

import robin.db as rdb
from robin.db.models import WakeModel
from tests.helpers import (dashboard_token, device_token, link, make_account,
                           make_profile, make_wake_model, make_wav)


def _run(coro):
    return asyncio.run(coro)


def _auth(raw):
    return {"Authorization": f"Bearer {raw}"}


def _owner(client_unused=None):
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid, role="owner"))
    raw, _ = _run(dashboard_token(aid))
    return pid, raw


# ---------------------------------------------------------------------------
# clip endpoints


def test_upload_list_play_delete_clip(client):
    pid, raw = _owner()
    wav = make_wav()
    r = client.post(f"/profiles/{pid}/wake-clips", content=wav,
                    headers={**_auth(raw), "Content-Type": "audio/wav"})
    assert r.status_code == 201
    clip = r.json()
    assert clip["label"] == "positive"
    assert clip["duration_s"] == pytest.approx(1.5, abs=0.01)
    assert clip["sha256"] == hashlib.sha256(wav).hexdigest()

    listed = client.get(f"/profiles/{pid}/wake-clips", headers=_auth(raw)).json()["clips"]
    assert [c["id"] for c in listed] == [clip["id"]]

    audio = client.get(f"/profiles/{pid}/wake-clips/{clip['id']}/audio", headers=_auth(raw))
    assert audio.status_code == 200
    assert audio.content == wav                    # bit-for-bit what was uploaded

    assert client.delete(f"/profiles/{pid}/wake-clips/{clip['id']}",
                         headers=_auth(raw)).json() == {"deleted": True}
    assert client.get(f"/profiles/{pid}/wake-clips", headers=_auth(raw)).json()["clips"] == []


def test_duplicate_clip_is_409(client):
    pid, raw = _owner()
    wav = make_wav()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    assert client.post(f"/profiles/{pid}/wake-clips", content=wav, headers=hdrs).status_code == 201
    assert client.post(f"/profiles/{pid}/wake-clips", content=wav, headers=hdrs).status_code == 409


def test_clip_format_and_duration_rejected(client):
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    for bad in (b"not a wav at all",
                make_wav(rate=48000),              # wrong rate
                make_wav(channels=2),              # stereo
                make_wav(seconds=0.1),             # too short
                make_wav(seconds=5.0)):            # positive bound is 4 s
        assert client.post(f"/profiles/{pid}/wake-clips", content=bad,
                           headers=hdrs).status_code == 422
    # ...but 5 s is a fine negative
    assert client.post(f"/profiles/{pid}/wake-clips?label=negative",
                       content=make_wav(seconds=5.0), headers=hdrs).status_code == 201


def test_viewer_can_listen_but_not_record(client):
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid, role="viewer"))
    raw, _ = _run(dashboard_token(aid))
    assert client.post(f"/profiles/{pid}/wake-clips", content=make_wav(),
                       headers={**_auth(raw), "Content-Type": "audio/wav"}).status_code == 403
    assert client.get(f"/profiles/{pid}/wake-clips", headers=_auth(raw)).status_code == 200


def test_unlinked_profile_clips_404(client):
    aid = _run(make_account())
    pid_other = _run(make_profile("Other"))
    raw, _ = _run(dashboard_token(aid))
    assert client.get(f"/profiles/{pid_other}/wake-clips", headers=_auth(raw)).status_code == 404


def test_wake_model_status(client):
    pid, raw = _owner()
    assert client.get(f"/profiles/{pid}/wake-model", headers=_auth(raw)).json() == {
        "available": False, "sha256": None, "base_version": None, "threshold": None,
        "created_at": None, "manifest": None, "clips_changed": False, "positives": 0,
        "training": None}
    _, sha = _run(make_wake_model(pid))
    body = client.get(f"/profiles/{pid}/wake-model", headers=_auth(raw)).json()
    assert body["available"] is True
    assert body["sha256"] == sha
    assert body["threshold"] == pytest.approx(0.857)


# ---------------------------------------------------------------------------
# the STT content gate (server.py registers stt_transcribe; tests register fakes)


@pytest.fixture
def transcriber():
    import robin.api.wake as wake
    yield wake.set_transcriber
    wake.set_transcriber(None)


@pytest.fixture
def wake_scorer():
    import robin.api.wake as wake
    yield wake.set_wake_scorer
    wake.set_wake_scorer(None)


def test_garbled_positive_rejected_with_mic_message(client, transcriber):
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    transcriber(lambda path: "The weather is lovely today.")
    r = client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=1), headers=hdrs)
    assert r.status_code == 422
    assert "garbled" in r.json()["detail"]
    assert "The weather is lovely today." in r.json()["detail"]   # say what was heard

    transcriber(lambda path: "")                   # nothing recognizable at all
    r = client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=2), headers=hdrs)
    assert r.status_code == 422
    assert "microphone" in r.json()["detail"]


def test_wake_word_positives_accepted_including_stt_spellings(client, transcriber):
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    for i, heard in enumerate(["Hey, Robin!", "hey robyn", "Hey Robin.", "HEY ROBIN"]):
        transcriber(lambda path, heard=heard: heard)
        r = client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=10 + i),
                        headers=hdrs)
        assert r.status_code == 201, heard


def test_wake_score_gate_rejects_takes_the_detector_cannot_hear(client, transcriber,
                                                                wake_scorer):
    """The Margaret failure mode: STT hears the words, the wake head hears nothing.
    Such a take must be rejected — it poisons training as a false positive label."""
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    transcriber(lambda path: "Hey Robin")          # STT approves...
    wake_scorer(lambda path: 0.03)                 # ...but the detector barely reacts
    r = client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=1), headers=hdrs)
    assert r.status_code == 422
    assert "0.03" in r.json()["detail"] and "detector" in r.json()["detail"]

    wake_scorer(lambda path: 0.91)                 # a take the detector actually hears
    assert client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=2),
                       headers=hdrs).status_code == 201

    # Negatives are exempt: scoring low on the wake head is their whole job.
    wake_scorer(lambda path: 0.0)
    transcriber(lambda path: "what a lovely day")
    assert client.post(f"/profiles/{pid}/wake-clips?label=negative",
                       content=make_wav(value=3), headers=hdrs).status_code == 201


def test_negative_containing_wake_word_rejected(client, transcriber):
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    transcriber(lambda path: "hey robin what time is it")
    r = client.post(f"/profiles/{pid}/wake-clips?label=negative",
                    content=make_wav(value=1), headers=hdrs)
    assert r.status_code == 422
    assert "wake word" in r.json()["detail"]
    transcriber(lambda path: "what time is it")
    assert client.post(f"/profiles/{pid}/wake-clips?label=negative",
                       content=make_wav(value=2), headers=hdrs).status_code == 201


# ---------------------------------------------------------------------------
# admin CLI: export + ingest


def test_clips_export_writes_trainer_layout(client, tmp_path):
    pid, raw = _owner()
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=100), headers=hdrs)
    client.post(f"/profiles/{pid}/wake-clips?label=negative",
                content=make_wav(value=200), headers=hdrs)
    from robin.admin import main
    main(["wake-clips-export", "--profile-id", str(pid), "--out", str(tmp_path)])
    assert len(list((tmp_path / "positives").glob("*.wav"))) == 1
    assert len(list((tmp_path / "negatives").glob("*.wav"))) == 1


def _manifest_dir(tmp_path, onnx=b"onnx-bytes", status="ok"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "wake_model.onnx").write_bytes(onnx)
    manifest = {"onnx": "wake_model.onnx", "sha256": hashlib.sha256(onnx).hexdigest(),
                "status": status, "threshold": 0.891,
                "base": {"version": "v3+om_r8avg9", "sha256": "irrelevant"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_ingest_activates_and_swap_deactivates_previous(client, tmp_path):
    pid = _run(make_profile())
    from robin.admin import main
    main(["wake-model-ingest", "--profile-id", str(pid),
          str(_manifest_dir(tmp_path / "a", onnx=b"first"))])
    main(["wake-model-ingest", "--profile-id", str(pid),
          str(_manifest_dir(tmp_path / "b", onnx=b"second"))])

    async def fetch():
        from sqlalchemy import select
        async with rdb._sessionmaker() as s:
            return (await s.execute(
                select(WakeModel.onnx, WakeModel.active)
                .where(WakeModel.profile_id == pid).order_by(WakeModel.id))).all()
    rows = _run(fetch())
    assert [(bytes(r.onnx), r.active) for r in rows] == [(b"first", False), (b"second", True)]


def test_ingest_refuses_failed_gate(client, tmp_path):
    pid = _run(make_profile())
    from robin.admin import main
    with pytest.raises(SystemExit, match="FA gate"):
        main(["wake-model-ingest", "--profile-id", str(pid),
              str(_manifest_dir(tmp_path, status="failed_gate"))])


# ---------------------------------------------------------------------------
# the Register button: clip-set lock + training job


@pytest.fixture
def jobs():
    import robin.train as train
    train._jobs.clear()
    yield train._jobs
    train._jobs.clear()


def _upload_takes(client, pid, raw, n, start=100):
    hdrs = {**_auth(raw), "Content-Type": "audio/wav"}
    shas = []
    for i in range(n):
        r = client.post(f"/profiles/{pid}/wake-clips", content=make_wav(value=start + i),
                        headers=hdrs)
        assert r.status_code == 201
        shas.append(r.json()["sha256"])
    return shas


def test_register_locks_on_trained_set_and_unlocks_on_change(client, jobs):
    pid, raw = _owner()
    shas = _upload_takes(client, pid, raw, 4)
    url = f"/profiles/{pid}/wake-model"

    # No model yet: the current takes are untrained material.
    assert client.get(url, headers=_auth(raw)).json()["clips_changed"] is True

    # Active model trained on exactly these takes -> locked, and train is refused.
    _run(make_wake_model(pid, manifest={
        "status": "ok",
        "clips": {"positives": {f"c{i}.wav": s for i, s in enumerate(shas)},
                  "negatives": {}}}))
    body = client.get(url, headers=_auth(raw)).json()
    assert (body["clips_changed"], body["positives"]) == (False, 4)
    assert client.post(f"{url}/train", headers=_auth(raw)).status_code == 409

    # Any change to the set -> unlocked again.
    clip_id = client.get(f"/profiles/{pid}/wake-clips", headers=_auth(raw)).json()["clips"][0]["id"]
    client.delete(f"/profiles/{pid}/wake-clips/{clip_id}", headers=_auth(raw))
    assert client.get(url, headers=_auth(raw)).json()["clips_changed"] is True


def test_train_endpoint_guards(client, jobs, monkeypatch):
    import robin.train as train
    pid, raw = _owner()
    url = f"/profiles/{pid}/wake-model/train"

    # Nothing recorded at all.
    r = client.post(url, headers=_auth(raw))
    assert r.status_code == 422 and "at least one" in r.json()["detail"]

    _upload_takes(client, pid, raw, 4)

    # Already running.
    jobs[pid] = {"state": "running", "detail": "training", "started_at": "now"}
    assert client.post(url, headers=_auth(raw)).status_code == 409
    jobs.clear()

    # Good to go: the endpoint starts the job (runner itself patched out).
    started = []
    monkeypatch.setattr(train, "start_training", lambda p: started.append(p))
    assert client.post(url, headers=_auth(raw)).status_code == 202
    assert started == [pid]


def test_train_job_ingests_and_activates(client, jobs, monkeypatch, tmp_path):
    """The job end-to-end with only the subprocess patched out: a manifest dir goes in,
    an active row with matching bytes and a 'done' job state come out."""
    import robin.train as train
    pid = _run(make_profile())
    model_dir = _manifest_dir(tmp_path, onnx=b"trained-head")

    async def fake_export_and_train(profile_id):
        return model_dir
    monkeypatch.setattr(train, "_export_and_train", fake_export_and_train)
    jobs[pid] = {"state": "running", "detail": "training", "started_at": "now"}
    _run(train._train_job(pid))
    assert jobs[pid]["state"] == "done"

    async def fetch():
        from sqlalchemy import select
        async with rdb._sessionmaker() as s:
            return (await s.execute(
                select(WakeModel.onnx, WakeModel.active)
                .where(WakeModel.profile_id == pid))).one()
    row = _run(fetch())
    assert (bytes(row.onnx), row.active) == (b"trained-head", True)


def test_train_job_records_failed_gate(client, jobs, monkeypatch, tmp_path):
    import robin.train as train
    pid = _run(make_profile())
    model_dir = _manifest_dir(tmp_path, status="failed_gate")

    async def fake_export_and_train(profile_id):
        return model_dir
    monkeypatch.setattr(train, "_export_and_train", fake_export_and_train)
    jobs[pid] = {"state": "running", "detail": "training", "started_at": "now"}
    _run(train._train_job(pid))
    assert jobs[pid]["state"] == "failed"
    assert client is not None                      # fixture keeps the schema alive


# ---------------------------------------------------------------------------
# delivery over the voice socket


def test_ws_offer_then_fetch_delivers_model(client):
    pid = _run(make_profile())
    onnx = b"\x08model-bytes" * 100
    _, sha = _run(make_wake_model(pid, onnx=onnx, threshold=0.9))
    raw, _ = _run(device_token(pid))
    with client.websocket_connect("/ws-wake", subprotocols=[raw]) as ws:
        offer = ws.receive_json()                  # unprompted, right after bind
        assert offer["type"] == "wake_model_meta"
        assert (offer["available"], offer["sha256"]) == (True, sha)
        assert offer["threshold"] == pytest.approx(0.9)
        assert offer["bytes"] == len(onnx)

        ws.send_json({"type": "wake_model_fetch"})
        meta = ws.receive_json()                   # meta announces the binary (audio_meta pattern)
        assert meta["sha256"] == sha
        blob = ws.receive_bytes()
        assert blob == onnx
        assert hashlib.sha256(blob).hexdigest() == meta["sha256"]


def test_ws_no_active_model_no_offer_and_fetch_says_unavailable(client):
    pid = _run(make_profile())
    _run(make_wake_model(pid, active=False))       # exists but not active: must not ship
    raw, _ = _run(device_token(pid))
    with client.websocket_connect("/ws-wake", subprotocols=[raw]) as ws:
        # No offer frame: the first thing the client hears is the answer to its fetch.
        ws.send_json({"type": "wake_model_fetch"})
        assert ws.receive_json() == {"type": "wake_model_meta", "available": False}


# ---------------------------------------------------------------------------
# device-scoped enrollment (the tablet's own Register Voice flow)


def test_device_upload_list_delete_and_status(client):
    pid = _run(make_profile())
    raw, _ = _run(device_token(pid))
    wav = make_wav()
    r = client.post("/device/wake-clips", content=wav,
                    headers={**_auth(raw), "Content-Type": "audio/wav"})
    assert r.status_code == 201
    clip = r.json()
    assert clip["sha256"] == hashlib.sha256(wav).hexdigest()

    listed = client.get("/device/wake-clips", headers=_auth(raw)).json()["clips"]
    assert [c["id"] for c in listed] == [clip["id"]]

    audio = client.get(f"/device/wake-clips/{clip['id']}/audio", headers=_auth(raw))
    assert audio.status_code == 200
    assert audio.content == wav                    # bit-for-bit what was uploaded

    status = client.get("/device/wake-model", headers=_auth(raw)).json()
    assert status["available"] is False
    assert status["positives"] == 1
    assert status["clips_changed"] is True

    assert client.delete(f"/device/wake-clips/{clip['id']}",
                         headers=_auth(raw)).json() == {"deleted": True}
    assert client.get("/device/wake-clips", headers=_auth(raw)).json()["clips"] == []


def test_device_endpoints_reject_dashboard_token_and_bad_token(client):
    pid, dash_raw = _owner()
    # a dashboard token is not a device token
    assert client.get("/device/wake-clips", headers=_auth(dash_raw)).status_code == 401
    assert client.get("/device/wake-clips", headers=_auth("junk")).status_code == 401
    assert client.get("/device/wake-clips").status_code == 401


def test_device_scope_is_the_bound_profile_only(client):
    pid_a = _run(make_profile("Margaret"))
    pid_b = _run(make_profile("John"))
    raw_a, _ = _run(device_token(pid_a))
    raw_b, _ = _run(device_token(pid_b))
    wav = make_wav(value=1234)
    clip = client.post("/device/wake-clips", content=wav,
                       headers={**_auth(raw_a), "Content-Type": "audio/wav"}).json()
    # B's tablet sees none of A's takes and cannot play or delete them
    assert client.get("/device/wake-clips", headers=_auth(raw_b)).json()["clips"] == []
    assert client.get(f"/device/wake-clips/{clip['id']}/audio",
                      headers=_auth(raw_b)).status_code == 404
    assert client.delete(f"/device/wake-clips/{clip['id']}",
                         headers=_auth(raw_b)).status_code == 404
    # A still has the clip
    assert len(client.get("/device/wake-clips", headers=_auth(raw_a)).json()["clips"]) == 1


def test_device_train_guards_and_start(client, jobs, monkeypatch):
    pid = _run(make_profile())
    raw, _ = _run(device_token(pid))
    # nothing to train on
    assert client.post("/device/wake-model/train",
                       headers=_auth(raw)).status_code == 422
    wav = make_wav(value=77)
    assert client.post("/device/wake-clips", content=wav,
                       headers={**_auth(raw), "Content-Type": "audio/wav"}).status_code == 201
    started = {}
    monkeypatch.setattr("robin.train.start_training",
                        lambda profile_id: started.setdefault("pid", profile_id))
    r = client.post("/device/wake-model/train", headers=_auth(raw))
    assert r.status_code == 202 and r.json() == {"started": True}
    assert started["pid"] == pid

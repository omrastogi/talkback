"""Seed helpers shared across test files. All take the sessionmaker from the `db` fixture."""
import hashlib
import io
import wave

import robin.db as rdb
from robin.auth.passwords import hash_password
from robin.auth.tokens import issue_token
from robin.db.models import Account, AccountProfile, Profile, WakeModel

PASSWORD = "correct horse battery staple"


async def make_account(email="cp@example.org", *, is_admin=False) -> int:
    async with rdb._sessionmaker() as s:
        a = Account(email=email, password_hash=hash_password(PASSWORD),
                    display_name="Care Partner", is_admin=is_admin)
        s.add(a)
        await s.commit()
        return a.id


async def make_profile(display_name="Margaret") -> int:
    async with rdb._sessionmaker() as s:
        p = Profile(display_name=display_name)
        s.add(p)
        await s.commit()
        return p.id


async def link(account_id: int, profile_id: int, role="owner") -> None:
    async with rdb._sessionmaker() as s:
        s.add(AccountProfile(account_id=account_id, profile_id=profile_id, role=role))
        await s.commit()


async def device_token(profile_id: int, label="test tablet") -> tuple[str, int]:
    async with rdb._sessionmaker() as s:
        raw, row = await issue_token(s, kind="device", profile_id=profile_id, label=label)
        await s.commit()
        return raw, row.id


async def dashboard_token(account_id: int) -> tuple[str, int]:
    async with rdb._sessionmaker() as s:
        raw, row = await issue_token(s, kind="dashboard", account_id=account_id)
        await s.commit()
        return raw, row.id


def make_wav(seconds: float = 1.5, value: int = 1000, rate: int = 16000,
             channels: int = 1) -> bytes:
    """A synthetic 16 kHz mono PCM16 WAV — the wake-clip upload contract."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(value.to_bytes(2, "little", signed=True) * int(seconds * rate) * channels)
    return buf.getvalue()


async def make_wake_model(profile_id: int, onnx: bytes = b"\x00fake-onnx\x01",
                          threshold: float = 0.857, active: bool = True,
                          manifest: dict | None = None) -> tuple[int, str]:
    async with rdb._sessionmaker() as s:
        m = WakeModel(profile_id=profile_id, base_version="v3+om_r8avg9",
                      sha256=hashlib.sha256(onnx).hexdigest(), onnx=onnx,
                      threshold=threshold, manifest=manifest or {"status": "ok"},
                      active=active)
        s.add(m)
        await s.commit()
        return m.id, m.sha256

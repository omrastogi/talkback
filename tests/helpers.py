"""Seed helpers shared across test files. All take the sessionmaker from the `db` fixture."""
import robin.db as rdb
from robin.auth.passwords import hash_password
from robin.auth.tokens import issue_token
from robin.db.models import Account, AccountProfile, Profile

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

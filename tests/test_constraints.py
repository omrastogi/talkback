"""Database-enforced invariants: the auth_token one-subject CHECK and the per-session turn
uniqueness. These exist because RECOVER's missing FK let a wrong-column bug run undetected;
here the database itself must refuse the bad row."""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

import robin.db as rdb
from robin.db.models import AuthToken, ConversationTurn
from tests.helpers import make_account, make_profile


async def test_check_rejects_both_subjects():
    pid = await make_profile()
    aid = await make_account()
    with pytest.raises(IntegrityError, match="auth_token_one_subject"):
        async with rdb._sessionmaker() as s:
            s.add(AuthToken(token_hash=b"x" * 32, kind="device",
                            profile_id=pid, account_id=aid))
            await s.commit()


async def test_check_rejects_no_subject():
    with pytest.raises(IntegrityError, match="auth_token_one_subject"):
        async with rdb._sessionmaker() as s:
            s.add(AuthToken(token_hash=b"y" * 32, kind="device"))
            await s.commit()


async def test_check_rejects_kind_subject_mismatch():
    aid = await make_account()
    with pytest.raises(IntegrityError, match="auth_token_one_subject"):
        async with rdb._sessionmaker() as s:
            s.add(AuthToken(token_hash=b"z" * 32, kind="device", account_id=aid))
            await s.commit()


async def test_duplicate_session_turn_index_is_integrity_error():
    pid = await make_profile()
    sid = uuid.uuid4()
    async with rdb._sessionmaker() as s:
        s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=0,
                               role="user", content="hello"))
        await s.commit()
    with pytest.raises(IntegrityError, match="uq_conversation_turn_session_index"):
        async with rdb._sessionmaker() as s:
            s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=0,
                                   role="assistant", content="hi"))
            await s.commit()


async def test_turn_requires_real_profile_fk():
    with pytest.raises(IntegrityError):
        async with rdb._sessionmaker() as s:
            s.add(ConversationTurn(profile_id=999999, session_id=uuid.uuid4(),
                                   turn_index=0, role="user", content="orphan"))
            await s.commit()

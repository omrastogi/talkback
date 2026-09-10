"""POST /auth/login and /auth/logout. The raw dashboard token appears exactly once, in the
login response; only its hash is stored."""
import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from robin.auth.deps import require_dashboard
from robin.auth.passwords import verify_password
from robin.auth.tokens import issue_token
from robin.db import get_session
from robin.db.models import Account

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str                      # the raw dashboard token; not recoverable later
    account_id: int
    display_name: str
    is_admin: bool


class LogoutResponse(BaseModel):
    revoked: bool


class MeResponse(BaseModel):
    account_id: int
    email: str
    display_name: str
    is_admin: bool


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)):
    account = (await session.execute(
        select(Account).where(Account.email == body.email)
    )).scalar_one_or_none()
    # Same 401 for unknown email and wrong password: don't confirm which accounts exist.
    if account is None or not verify_password(body.password, account.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")
    raw, _ = await issue_token(session, kind="dashboard", account_id=account.id)
    await session.commit()
    return LoginResponse(token=raw, account_id=account.id,
                         display_name=account.display_name, is_admin=account.is_admin)


@router.get("/me", response_model=MeResponse)
async def me(account: Account = Depends(require_dashboard)):
    """Who the presented token belongs to; the dashboard re-validates its stored token here
    on page load."""
    return MeResponse(account_id=account.id, email=account.email,
                      display_name=account.display_name, is_admin=account.is_admin)


@router.post("/logout", response_model=LogoutResponse)
async def logout(request: Request, account: Account = Depends(require_dashboard),
                 session: AsyncSession = Depends(get_session)):
    token = request.state.auth_token          # the exact credential presented, set by the dep
    token.revoked_at = datetime.datetime.now(datetime.timezone.utc)
    session.add(token)
    await session.commit()
    return LogoutResponse(revoked=True)

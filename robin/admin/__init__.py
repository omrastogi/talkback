"""Admin CLI: accounts, profiles, links, and device tokens are provisioned here by the
research team — there is deliberately no self-serve registration endpoint.

    python -m robin.admin create-account --email cp@example.org --display-name "Care Partner"
    python -m robin.admin create-profile --display-name "Margaret" [--timezone America/Chicago]
    python -m robin.admin link --account-id 1 --profile-id 1 [--role owner|viewer]
    python -m robin.admin issue-device-token --profile-id 1 --label "Tab A9 living room"
    python -m robin.admin revoke-token --token-id 3
    python -m robin.admin list-tokens [--profile-id 1]
"""
import argparse
import asyncio
import getpass

from sqlalchemy import select

from robin.auth.passwords import hash_password
from robin.auth.tokens import issue_token, revoke_token
from robin.db import get_sessionmaker
from robin.db.models import Account, AccountProfile, AuthToken, Profile


async def _create_account(args):
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat: "):
        raise SystemExit("passwords do not match")
    if not password:
        raise SystemExit("empty password")
    async with get_sessionmaker()() as session:
        account = Account(email=args.email, password_hash=hash_password(password),
                          display_name=args.display_name, is_admin=args.admin)
        session.add(account)
        await session.commit()
        print(f"account id: {account.id}")


async def _create_profile(args):
    async with get_sessionmaker()() as session:
        profile = Profile(display_name=args.display_name)
        if args.timezone:
            profile.timezone = args.timezone
        session.add(profile)
        await session.commit()
        print(f"profile id: {profile.id}")


async def _link(args):
    async with get_sessionmaker()() as session:
        session.add(AccountProfile(account_id=args.account_id, profile_id=args.profile_id,
                                   role=args.role))
        await session.commit()
        print(f"linked account {args.account_id} -> profile {args.profile_id} ({args.role})")


async def _issue_device_token(args):
    async with get_sessionmaker()() as session:
        raw, row = await issue_token(session, kind="device", profile_id=args.profile_id,
                                     label=args.label)
        await session.commit()
        print(f"token id: {row.id}")
        print(f"device token (shown once, store it now — it is not recoverable):\n{raw}")


async def _revoke_token(args):
    async with get_sessionmaker()() as session:
        if await revoke_token(session, args.token_id):
            print(f"token {args.token_id} revoked")
        else:
            raise SystemExit(f"no token with id {args.token_id}")


async def _list_tokens(args):
    async with get_sessionmaker()() as session:
        q = select(AuthToken).order_by(AuthToken.id)
        if args.profile_id is not None:
            q = q.where(AuthToken.profile_id == args.profile_id)
        rows = (await session.execute(q)).scalars().all()
    fmt = "{:>4}  {:<9}  {:<7}  {:<26}  {:<20}  {}"
    print(fmt.format("id", "kind", "subject", "label", "last_used_at", "revoked_at"))
    for r in rows:                      # label/timestamps only — never the token or its hash
        subject = f"p{r.profile_id}" if r.profile_id else f"a{r.account_id}"
        print(fmt.format(r.id, r.kind, subject, (r.label or "")[:26],
                         r.last_used_at.strftime("%Y-%m-%d %H:%M") if r.last_used_at else "-",
                         r.revoked_at.strftime("%Y-%m-%d %H:%M") if r.revoked_at else "-"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m robin.admin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-account"); p.set_defaults(fn=_create_account)
    p.add_argument("--email", required=True)
    p.add_argument("--display-name", required=True)
    p.add_argument("--admin", action="store_true")

    p = sub.add_parser("create-profile"); p.set_defaults(fn=_create_profile)
    p.add_argument("--display-name", required=True)
    p.add_argument("--timezone")

    p = sub.add_parser("link"); p.set_defaults(fn=_link)
    p.add_argument("--account-id", type=int, required=True)
    p.add_argument("--profile-id", type=int, required=True)
    p.add_argument("--role", choices=["owner", "viewer"], default="owner")

    p = sub.add_parser("issue-device-token"); p.set_defaults(fn=_issue_device_token)
    p.add_argument("--profile-id", type=int, required=True)
    p.add_argument("--label", required=True)

    p = sub.add_parser("revoke-token"); p.set_defaults(fn=_revoke_token)
    p.add_argument("--token-id", type=int, required=True)

    p = sub.add_parser("list-tokens"); p.set_defaults(fn=_list_tokens)
    p.add_argument("--profile-id", type=int)

    args = ap.parse_args(argv)
    asyncio.run(args.fn(args))

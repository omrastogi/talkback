"""Admin CLI: accounts, profiles, links, and device tokens are provisioned here by the
research team — there is deliberately no self-serve registration endpoint.

    python -m robin.admin create-account --email cp@example.org --display-name "Care Partner"
    python -m robin.admin create-profile --display-name "Margaret" [--timezone America/Chicago]
    python -m robin.admin link --account-id 1 --profile-id 1 [--role owner|viewer]
    python -m robin.admin issue-device-token --profile-id 1 --label "Tab A9 living room"
    python -m robin.admin revoke-token --token-id 3
    python -m robin.admin list-tokens [--profile-id 1]

Wake-word personalization round-trip (clips recorded on the dashboard; the trainer lives
in this repo at scripts/enroll_train.py but runs with oww-train's env and points at that
repo's data — base model, anchors, openWakeWord checkout — via --oww-root):

    python -m robin.admin wake-clips-export --profile-id 1 --out /tmp/enroll_p1
    ~/miniconda3/envs/oww-train/bin/python scripts/enroll_train.py /tmp/enroll_p1 --out /tmp/enroll_p1/model
    python -m robin.admin wake-model-ingest --profile-id 1 /tmp/enroll_p1/model
    python -m robin.admin list-wake-models [--profile-id 1]

Ingest refuses a manifest whose status is not "ok" (the trainer's FA gate failed) and
activates the new head atomically: the previous one is deactivated in the same commit.
The tablet picks it up on its next connect (robin/wake.py).
"""
import argparse
import asyncio
import getpass
import hashlib
import json
import pathlib

from sqlalchemy import select, update

from robin.auth.passwords import hash_password
from robin.auth.tokens import issue_token, revoke_token
from robin.db import get_sessionmaker
from robin.db.models import Account, AccountProfile, AuthToken, Profile, WakeClip, WakeModel


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


async def _wake_clips_export(args):
    """Write a profile's enrollment clips as <out>/{positives,negatives}/*.wav — exactly
    the directory shape oww-train/enroll_train.py takes."""
    async with get_sessionmaker()() as session:
        rows = (await session.execute(
            select(WakeClip.id, WakeClip.label, WakeClip.wav, WakeClip.sha256)
            .where(WakeClip.profile_id == args.profile_id).order_by(WakeClip.id))).all()
    if not rows:
        raise SystemExit(f"no wake clips for profile {args.profile_id}")
    out = pathlib.Path(args.out)
    counts = {"positive": 0, "negative": 0}
    for r in rows:
        d = out / f"{r.label}s"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"clip_{r.id:04d}_{r.sha256[:8]}.wav").write_bytes(r.wav)
        counts[r.label] += 1
    print(f"exported {counts['positive']} positives, {counts['negative']} negatives -> {out}")


async def _wake_model_ingest(args):
    model_dir = pathlib.Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    onnx = (model_dir / manifest["onnx"]).read_bytes()
    if manifest.get("status") != "ok":
        raise SystemExit(f"manifest status is {manifest.get('status')!r}, not 'ok' — "
                         "the trainer's FA gate failed; this head must not ship")
    if hashlib.sha256(onnx).hexdigest() != manifest["sha256"]:
        raise SystemExit("onnx sha256 does not match the manifest — corrupt or mixed-up dir")
    async with get_sessionmaker()() as session:
        # Deactivate-then-activate in one commit; the partial unique index would reject
        # two active rows anyway, this just makes the swap atomic instead of an error.
        await session.execute(update(WakeModel)
                              .where(WakeModel.profile_id == args.profile_id, WakeModel.active)
                              .values(active=False))
        model = WakeModel(profile_id=args.profile_id,
                          base_version=manifest["base"]["version"],
                          sha256=manifest["sha256"], onnx=onnx,
                          threshold=manifest["threshold"], manifest=manifest, active=True)
        session.add(model)
        await session.commit()
        print(f"wake model id {model.id} active for profile {args.profile_id}  "
              f"(base {model.base_version}, thr {model.threshold:.3f}, {len(onnx)} bytes) — "
              "tablets pick it up on next connect")


async def _list_wake_models(args):
    async with get_sessionmaker()() as session:
        q = select(WakeModel.id, WakeModel.profile_id, WakeModel.base_version,
                   WakeModel.sha256, WakeModel.threshold, WakeModel.active,
                   WakeModel.created_at).order_by(WakeModel.id)
        if args.profile_id is not None:
            q = q.where(WakeModel.profile_id == args.profile_id)
        rows = (await session.execute(q)).all()
    fmt = "{:>4}  {:>7}  {:<14}  {:<14}  {:>6}  {:^6}  {}"
    print(fmt.format("id", "profile", "base", "sha256", "thr", "active", "created_at"))
    for r in rows:
        print(fmt.format(r.id, r.profile_id, r.base_version[:14], r.sha256[:12],
                         f"{r.threshold:.3f}", "yes" if r.active else "-",
                         r.created_at.strftime("%Y-%m-%d %H:%M")))


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

    p = sub.add_parser("wake-clips-export"); p.set_defaults(fn=_wake_clips_export)
    p.add_argument("--profile-id", type=int, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("wake-model-ingest"); p.set_defaults(fn=_wake_model_ingest)
    p.add_argument("--profile-id", type=int, required=True)
    p.add_argument("model_dir", help="enroll_train.py output dir (wake_model.onnx + manifest.json)")

    p = sub.add_parser("list-wake-models"); p.set_defaults(fn=_list_wake_models)
    p.add_argument("--profile-id", type=int)

    args = ap.parse_args(argv)
    asyncio.run(args.fn(args))

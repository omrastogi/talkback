"""argon2id hashing, library defaults. Hashes are write-only from the API's point of view:
no endpoint returns password_hash in any shape, and nothing here logs one."""
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

_hasher = PasswordHasher()          # argon2id with argon2-cffi's current defaults


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False

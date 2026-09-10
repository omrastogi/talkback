from robin.auth.passwords import hash_password, verify_password


def test_hash_round_trip():
    h = hash_password("s3cret pass")
    assert verify_password("s3cret pass", h)


def test_wrong_password_fails():
    h = hash_password("s3cret pass")
    assert not verify_password("not it", h)


def test_hash_is_argon2id_and_not_plaintext():
    h = hash_password("s3cret pass")
    assert h.startswith("$argon2id$")
    assert "s3cret" not in h


def test_garbage_hash_fails_closed():
    assert not verify_password("anything", "not-a-hash")

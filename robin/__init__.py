"""Persistence and auth layer for the Robin voice server: profiles, accounts, opaque
tokens, and per-utterance conversation history in Postgres.

Deliberately structured around the failure modes of the RECOVER predecessor (see
docs/reference/recover-database-review.pdf): every cross-table reference is a real FK,
serialization is explicit Pydantic response models only (no generic model-to-dict),
session identity is a server-minted UUID (never inferred from model output), and identity
lives in a real `profile` table instead of a filename join.
"""

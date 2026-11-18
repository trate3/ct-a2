"""Canonical hashes and signature encoding."""

from __future__ import annotations

import hashlib
import json


def sha256_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


def artifact_commitment_digest(certified_object: bytes, r: bytes) -> str:
    """h_c over the trusted workload's encoding of the actual provisioned artifacts."""
    return sha256_hex(certified_object, r)


def tests_digest(tests: list[tuple[str, int]]) -> str:
    """Offline-only identity for a labeled evaluation set."""
    canonical = json.dumps([list(t) for t in tests], sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode())


def signing_payload(*fields) -> bytes:
    """Canonical byte encoding of a tuple of signed fields."""
    return json.dumps(list(fields), sort_keys=True, separators=(",", ":")).encode()

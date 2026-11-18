"""Canonical inert messages and bounded reads; no executable deserialization."""
from __future__ import annotations

import hashlib
import json

CHUNK = 64 * 1024
MAX_METADATA = 8 * 1024 * 1024
MAX_INLINE = 8 * 1024 * 1024
MAX_ARTIFACTS = 4096


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode(raw: bytes):
    def object_pairs(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    def invalid_number(value):
        raise ValueError("non-finite JSON number")

    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_number)


def hex_string(value, size: int) -> bool:
    return (isinstance(value, str) and len(value) == 2 * size
            and all(c in "0123456789abcdef" for c in value))


def read_exact(stream, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise ValueError("truncated input")
        out.extend(chunk)
    return bytes(out)


def chunks(pieces, size=CHUNK):
    """Rechunk without exposing artifact boundaries or buffering an entire artifact."""
    pending = bytearray()
    for piece in pieces:
        view = memoryview(piece)
        while view:
            take = min(size - len(pending), len(view))
            pending.extend(view[:take])
            view = view[take:]
            if len(pending) == size:
                yield bytes(pending)
                pending.clear()
    if pending:
        yield bytes(pending)


class ChunkReader:
    """Bounded file-like reader over authenticated plaintext chunks."""

    def __init__(self, iterator):
        self.iterator = iter(iterator)
        self.pending = b""

    def read(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("unbounded read is forbidden")
        out = bytearray()
        while len(out) < n:
            if not self.pending:
                self.pending = next(self.iterator, b"")
                if not self.pending:
                    break
            take = min(n - len(out), len(self.pending))
            out.extend(self.pending[:take])
            self.pending = self.pending[take:]
        return bytes(out)


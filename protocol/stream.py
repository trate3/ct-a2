"""Sealed-box key encapsulation plus libsodium secretstream for large submissions.

The public proposal fixes a padded plaintext length. Every chunk has the same
proposal digest as associated data; the final tag and EOF are mandatory.
"""
from __future__ import annotations

import secrets

from nacl import bindings as sodium
from nacl.public import PrivateKey, PublicKey, SealedBox

from .codec import CHUNK, chunks, read_exact

MAGIC = b"CTF4SB1\0"
KEY_BYTES = sodium.crypto_secretstream_xchacha20poly1305_KEYBYTES
HEADER_BYTES = sodium.crypto_secretstream_xchacha20poly1305_HEADERBYTES
OVERHEAD = sodium.crypto_secretstream_xchacha20poly1305_ABYTES
FINAL = sodium.crypto_secretstream_xchacha20poly1305_TAG_FINAL
MESSAGE = sodium.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE
SEALED_BYTES = KEY_BYTES + 32 + 8 + sodium.crypto_box_SEALBYTES


def encrypt(public_key: bytes, context: bytes, pieces, length: int, output) -> None:
    if len(context) != 32 or length <= 0:
        raise ValueError("invalid stream context or length")
    key = secrets.token_bytes(KEY_BYTES)
    state = sodium.crypto_secretstream_xchacha20poly1305_state()
    header = sodium.crypto_secretstream_xchacha20poly1305_init_push(state, key)
    sealed = SealedBox(PublicKey(public_key)).encrypt(key + context + length.to_bytes(8, "big"))
    output.write(MAGIC + sealed + header)
    sent = 0
    for block in chunks(pieces):
        sent += len(block)
        if sent > length:
            raise ValueError("plaintext exceeds the agreed envelope")
        tag = FINAL if sent == length else MESSAGE
        ciphertext = sodium.crypto_secretstream_xchacha20poly1305_push(state, block, context, tag)
        output.write(len(ciphertext).to_bytes(4, "big") + ciphertext)
    if sent != length:
        raise ValueError("plaintext does not fill the agreed envelope")


def decrypt(secret_key: PrivateKey, context: bytes, length: int, source):
    if read_exact(source, len(MAGIC)) != MAGIC:
        raise ValueError("unsupported ciphertext format")
    opened = SealedBox(secret_key).decrypt(read_exact(source, SEALED_BYTES))
    if (len(opened) != KEY_BYTES + 40 or opened[KEY_BYTES:KEY_BYTES+32] != context
            or int.from_bytes(opened[-8:], "big") != length):
        raise ValueError("ciphertext belongs to another proposal or envelope")
    state = sodium.crypto_secretstream_xchacha20poly1305_state()
    sodium.crypto_secretstream_xchacha20poly1305_init_pull(state, read_exact(source, HEADER_BYTES), opened[:KEY_BYTES])
    remaining = length
    while remaining:
        n = min(CHUNK, remaining)
        framed = int.from_bytes(read_exact(source, 4), "big")
        if framed != n + OVERHEAD:
            raise ValueError("wrong ciphertext chunk size")
        plaintext, tag = sodium.crypto_secretstream_xchacha20poly1305_pull(
            state, read_exact(source, framed), context)
        remaining -= len(plaintext)
        if len(plaintext) != n or tag != (FINAL if remaining == 0 else MESSAGE):
            raise ValueError("premature or missing stream finalization")
        if remaining == 0 and source.read(1):
            raise ValueError("trailing ciphertext")
        yield plaintext


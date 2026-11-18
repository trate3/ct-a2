"""Confidential transport: anonymous encryption to a public key (sealed boxes).

The certifier encrypts E once to the enclave's ephemeral channel key. An optional
untrusted relay sees only that ciphertext and the public certificate.

We use libsodium sealed boxes via PyNaCl (`SealedBox`): X25519 key agreement with
a fresh ephemeral sender key + XSalsa20-Poly1305 AEAD -- anonymous (the recipient
public key is all you need to encrypt) and IND-CCA2, which is the property the
protocol's realization proof relies on. Anyone can seal to the public key; `t_E`
therefore binds the certificate to the exact ciphertext the certifier sent.

Ciphertext travels the wire as base64 text inside JSON messages; digests are
computed over the raw ciphertext bytes on both sides.
"""

from __future__ import annotations

import base64
from nacl.public import PrivateKey, PublicKey, SealedBox


def new_keypair() -> tuple[PrivateKey, bytes]:
    """Fresh ephemeral keypair; returns (secret key object, public-key bytes for the wire)."""
    sk = PrivateKey.generate()
    return sk, bytes(sk.public_key)


def seal_bytes(pk_bytes: bytes, plaintext: bytes) -> str:
    """Seal fixed-length bytes; return base64 ciphertext for the JSON wire."""
    ct = SealedBox(PublicKey(pk_bytes)).encrypt(plaintext)
    return base64.b64encode(ct).decode()


def unseal_bytes(sk: PrivateKey, b64_ciphertext: str) -> bytes:
    """Open a sealed ciphertext with the matching private key (raises on tamper)."""
    return SealedBox(sk).decrypt(base64.b64decode(b64_ciphertext, validate=True))


def ciphertext_bytes(b64_ciphertext: str) -> bytes:
    """The raw ciphertext bytes bound by t_E."""
    return base64.b64decode(b64_ciphertext, validate=True)

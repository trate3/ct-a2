"""Tests for the GCP Confidential Space attestor seams.

These exercise the class logic (nonce binding, claim extraction, publication
round-trip) against a fake launcher socket and a self-signed test token -- not
against a live Confidential Space launcher or Google's real attestation service.
Requires the `gcp` extra; skips cleanly without it.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from nacl.signing import SigningKey

from ct_core.protocol.attestor import (
    GcpAttestationVerifier,
    GcpEnclaveSigner,
    publish_gcp_attestation,
)

jwt = pytest.importorskip("jwt")


def _fake_claims(nonce: bytes, image_digest: str = "sha256:deadbeef") -> dict:
    return {
        "iss": GcpAttestationVerifier.ISSUER,
        "eat_nonce": [base64.b64encode(nonce).decode()],
        "submods": {"container": {"image_digest": image_digest}},
    }


def _unsigned_test_token(claims: dict) -> str:
    """A syntactically valid JWT the signer can read claims out of without
    verifying -- `GcpEnclaveSigner` never checks its own token's signature."""
    def b64(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()
    header = b64(json.dumps({"alg": "none"}).encode())
    payload = b64(json.dumps(claims).encode())
    return f"{header}.{payload}."


def test_signer_binds_verify_key_hash_as_nonce_and_reads_measurement():
    signing_key = SigningKey.generate()
    verify_key = bytes(signing_key.verify_key)
    expected_nonce = hashlib.sha256(verify_key).digest()
    seen = {}

    def fake_request_token():
        seen["nonce_used"] = expected_nonce
        return _unsigned_test_token(_fake_claims(expected_nonce))

    signer = GcpEnclaveSigner(signing_key=signing_key, request_token=fake_request_token)

    assert signer.verify_key() == verify_key
    assert signer.measurement() == "sha256:deadbeef"
    assert seen["nonce_used"] == expected_nonce
    signature = signer.sign(b"payload")
    signing_key.verify_key.verify(b"payload", bytes.fromhex(signature))


def test_signer_rejects_token_missing_image_digest():
    signing_key = SigningKey.generate()
    nonce = hashlib.sha256(bytes(signing_key.verify_key)).digest()
    bad_claims = {"iss": "x", "eat_nonce": [base64.b64encode(nonce).decode()]}

    with pytest.raises(ValueError, match="image digest"):
        GcpEnclaveSigner(signing_key=signing_key,
                         request_token=lambda: _unsigned_test_token(bad_claims))


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJwksClient:
    def __init__(self, key):
        self._key = key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._key)


def _signed_test_token(claims: dict, private_key) -> str:
    return jwt.encode(claims, private_key, algorithm="RS256")


@pytest.fixture
def rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def test_verifier_pins_from_a_valid_token(rsa_keypair):
    private_key, public_key = rsa_keypair
    verify_key = bytes(SigningKey.generate().verify_key)
    nonce = hashlib.sha256(verify_key).digest()
    token = _signed_test_token(_fake_claims(nonce), private_key)

    verifier = GcpAttestationVerifier.from_token(
        token, verify_key, jwks_client=_FakeJwksClient(public_key))

    assert verifier.measurement() == "sha256:deadbeef"


def test_verifier_rejects_token_bound_to_a_different_key(rsa_keypair):
    private_key, public_key = rsa_keypair
    verify_key = bytes(SigningKey.generate().verify_key)
    other_nonce = hashlib.sha256(b"a different key entirely").digest()
    token = _signed_test_token(_fake_claims(other_nonce), private_key)

    with pytest.raises(ValueError, match="nonce"):
        GcpAttestationVerifier.from_token(
            token, verify_key, jwks_client=_FakeJwksClient(public_key))


def test_verifier_rejects_a_token_signed_by_the_wrong_key(rsa_keypair):
    private_key, _ = rsa_keypair
    from cryptography.hazmat.primitives.asymmetric import rsa

    wrong_public_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    verify_key = bytes(SigningKey.generate().verify_key)
    nonce = hashlib.sha256(verify_key).digest()
    token = _signed_test_token(_fake_claims(nonce), private_key)

    with pytest.raises(Exception):
        GcpAttestationVerifier.from_token(
            token, verify_key, jwks_client=_FakeJwksClient(wrong_public_key))


def test_publish_and_reload_round_trip(rsa_keypair, tmp_path):
    private_key, public_key = rsa_keypair
    signing_key = SigningKey.generate()
    verify_key = bytes(signing_key.verify_key)
    nonce = hashlib.sha256(verify_key).digest()
    token = _signed_test_token(_fake_claims(nonce), private_key)
    path = tmp_path / "gcp_attestation.json"

    published = publish_gcp_attestation(
        token, verify_key, path, jwks_client=_FakeJwksClient(public_key))

    reloaded = GcpAttestationVerifier.from_publication(path)
    assert reloaded.measurement() == published.measurement() == "sha256:deadbeef"
    good_signature = signing_key.sign(b"hello").signature.hex()
    assert reloaded.verify(b"hello", good_signature) is True
    other_signature = SigningKey.generate().sign(b"hello").signature.hex()
    assert reloaded.verify(b"hello", other_signature) is False

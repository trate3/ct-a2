"""Explicit local test anchors and a fail-closed Confidential Space token policy."""
from __future__ import annotations

from pathlib import Path

from nacl.signing import SigningKey

from ct_core.protocol.attestor import (
    EnclaveSigner, GcpAttestationVerifier, StubAttestationVerifier, _unverified_jwt_claims,
)

from .codec import canonical, digest


class LocalSigner(EnclaveSigner):
    """CPU-test signer only. No hardware protection and no persistent global keys."""

    def __init__(self, key: SigningKey, measurement: str):
        self._key, self._measurement = key, measurement

    def measurement(self):
        return self._measurement

    def sign(self, payload):
        return self._key.sign(payload).signature.hex()


def local_test_pair():
    """Pin before starting a test workload; never substitute this for real attestation."""
    here = Path(__file__).resolve().parent
    root = here.parent
    files = sorted(here.glob("*.py")) + sorted((root / "ct_core").rglob("*.py"))
    measurement = digest(canonical({str(p.relative_to(root)): digest(p.read_bytes()) for p in files}))
    key = SigningKey.generate()
    return LocalSigner(key, measurement), StubAttestationVerifier(bytes(key.verify_key), measurement)


def accepted_gcp_verifier(token, verify_key, *, accepted_image: str, audience: str, jwks_client=None):
    """Authenticity/key binding first, then the relying party's explicit image policy.

    No debug exception, no trust-on-first-use image selection, no default local
    signer. This authenticates CPU workload evidence; GPU protection is separate.
    """
    if (not isinstance(accepted_image, str) or not accepted_image.startswith("sha256:")
            or len(accepted_image) != 71 or any(c not in "0123456789abcdef" for c in accepted_image[7:])
            or not isinstance(audience, str) or not audience):
        raise ValueError("accepted image digest and audience must be set in advance")
    verifier = GcpAttestationVerifier.from_token(token, verify_key, audience=audience, jwks_client=jwks_client)
    # from_token verified this exact immutable token's signature, issuer, audience
    # and signing-key nonce. Decode the same payload to enforce the remaining policy.
    claims = _unverified_jwt_claims(token)
    if verifier.measurement() != accepted_image:
        raise ValueError("attested image is not accepted")
    if claims.get("dbgstat") != "disabled-since-boot":
        raise ValueError("production non-debug workload required")
    if claims.get("swname") != "CONFIDENTIAL_SPACE":
        raise ValueError("Confidential Space workload evidence required")
    return verifier


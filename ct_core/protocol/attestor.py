"""Attestation signing and public verification across separate trust domains.

Setup publishes the verification key and expected workload measurement before any
session. The trusted workload owns an ``EnclaveSigner``; the certifier and relying
party own only an ``AttestationVerifier`` initialized from that pinned publication.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import pathlib
import secrets
import socket as _socket

from nacl.signing import SigningKey, VerifyKey

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_KEY_DIR = _REPO_ROOT / ".keys"
_SEED_PATH = _KEY_DIR / "manufacturer.key"
_PUBLICATION_PATH = _KEY_DIR / "attestation.json"
_GCP_PUBLICATION_PATH = _KEY_DIR / "gcp_attestation.json"
_MEASURE_GLOB = "*.py"


def _workload_measurement() -> str:
    """Measurement used by the local signer: SHA-256 over the ct_core sources."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_ROOT.rglob(_MEASURE_GLOB)):
        relative = path.relative_to(_PACKAGE_ROOT).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _create_signing_key() -> SigningKey:
    """Setup for the local signing key."""
    _KEY_DIR.mkdir(exist_ok=True)
    if not _SEED_PATH.exists():
        temporary = _KEY_DIR / f".seed.tmp.{os.getpid()}.{secrets.token_hex(4)}"
        temporary.write_bytes(bytes(SigningKey.generate()))
        try:
            os.link(temporary, _SEED_PATH)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return SigningKey(_SEED_PATH.read_bytes())


def _publish(verify_key: bytes, measurement: str) -> None:
    """Atomically publish the two values available to C and R in the paper."""
    publication = json.dumps({
        "verify_key": verify_key.hex(),
        "expected_measurement": measurement,
    }, sort_keys=True).encode()
    temporary = _KEY_DIR / f".attestation.tmp.{os.getpid()}.{secrets.token_hex(4)}"
    temporary.write_bytes(publication)
    os.replace(temporary, _PUBLICATION_PATH)


def initialize_local_attestation() -> "StubAttestationVerifier":
    """Setup authority: pin public verification material before any workload runs."""
    signing_key = _create_signing_key()
    verifier = StubAttestationVerifier(
        bytes(signing_key.verify_key), _workload_measurement())
    _publish(bytes(signing_key.verify_key), verifier.measurement())
    return verifier


class EnclaveSigner:
    """Trusted-workload interface; implementations hold the attestation secret."""

    def measurement(self) -> str:
        raise NotImplementedError

    def sign(self, payload: bytes) -> str:
        raise NotImplementedError


class AttestationVerifier:
    """Public certifier/relying-party interface; it deliberately cannot sign."""

    def measurement(self) -> str:
        raise NotImplementedError

    def verify(self, payload: bytes, signature: str) -> bool:
        raise NotImplementedError


class StubEnclaveSigner(EnclaveSigner):
    """Local signer; it cannot create or update the public trust anchor."""

    def __init__(self, signing_key: SigningKey | None = None) -> None:
        self._signing_key = signing_key or SigningKey(_SEED_PATH.read_bytes())
        self._measurement = _workload_measurement()

    def measurement(self) -> str:
        return self._measurement

    def sign(self, payload: bytes) -> str:
        return self._signing_key.sign(payload).signature.hex()


class StubAttestationVerifier(AttestationVerifier):
    """Local C/R verifier containing only public verification material."""

    def __init__(self, verify_key: bytes, expected_measurement: str) -> None:
        self._verify_key = VerifyKey(verify_key)
        self._measurement = expected_measurement

    @classmethod
    def from_publication(cls) -> "StubAttestationVerifier":
        publication = json.loads(_PUBLICATION_PATH.read_text())
        return cls(
            bytes.fromhex(publication["verify_key"]),
            publication["expected_measurement"],
        )

    def measurement(self) -> str:
        return self._measurement

    def verify(self, payload: bytes, signature: str) -> bool:
        try:
            self._verify_key.verify(payload, bytes.fromhex(signature))
            return True
        except Exception:
            return False


class _UnixHTTPConnection(http.client.HTTPConnection):
    """`http.client` speaks HTTP/1.1; only `connect()` needs to target a UDS."""

    def __init__(self, path: str) -> None:
        super().__init__("localhost")
        self._path = path

    def connect(self) -> None:
        self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self.sock.connect(self._path)


def _teeserver_request_token(nonce: bytes, *, audience: str, token_type: str,
                             socket_path: str) -> str:
    """POST /v1/token to the Confidential Space launcher's local UDS.

    SOURCE (request shape: `audience`/`nonces`/`token_type`; nonce 8-88 bytes) --
    https://docs.cloud.google.com/confidential-computing/confidential-space/docs/connect-external-resources
    Requires a Confidential Space container.
    """
    body = json.dumps({
        "audience": audience,
        "token_type": token_type,
        "nonces": [base64.b64encode(nonce).decode()],
    }).encode()
    connection = _UnixHTTPConnection(socket_path)
    try:
        connection.request("POST", "/v1/token", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"attestation token request failed: {response.status} {payload!r}")
    return payload.decode().strip()


def _unverified_jwt_claims(token: str) -> dict:
    """Decode a JWT's payload without checking its signature.

    Only used by the TEE-side signer to read its own just-issued token's claims;
    the security-relevant verification happens once, independently, on C/R's side
    in `GcpAttestationVerifier.from_token`.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed attestation token")
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


def _image_digest(claims: dict) -> str:
    try:
        return claims["submods"]["container"]["image_digest"]
    except (KeyError, TypeError) as error:
        raise ValueError("attestation token missing container image digest") from error


def _oidc_jwks_uri(issuer: str) -> str:
    import urllib.request  # stdlib; only needed on the verifier side

    with urllib.request.urlopen(  # noqa: S310 - fixed https issuer, not user input
            f"{issuer}/.well-known/openid-configuration", timeout=10) as response:
        return json.loads(response.read())["jwks_uri"]


class GcpEnclaveSigner(EnclaveSigner):
    """TEE-side signer: an ephemeral Ed25519 key attested by Confidential Space.

    Generates the signing key in memory (never written to disk, unlike
    `StubEnclaveSigner`'s local file) and asks the Confidential Space launcher to
    bind it to this measured boot by embedding ``sha256(verify_key)`` as the
    attestation token's nonce -- the Confidential Space analog of report-data
    binding in a raw SGX/TDX quote. `measurement()` reads the token's container
    image digest. Only constructible inside a running Confidential Space
    container; pass `request_token` to test without one.
    """

    TEESERVER_SOCKET = "/run/container_launcher/teeserver.sock"

    def __init__(self, *, audience: str = "certified-taps/v1", token_type: str = "OIDC",
                socket_path: str | None = None, request_token=None,
                signing_key: SigningKey | None = None) -> None:
        self._signing_key = signing_key or SigningKey.generate()
        nonce = hashlib.sha256(bytes(self._signing_key.verify_key)).digest()
        fetch = request_token or (lambda: _teeserver_request_token(
            nonce, audience=audience, token_type=token_type,
            socket_path=socket_path or self.TEESERVER_SOCKET))
        self._token = fetch()
        self._measurement = _image_digest(_unverified_jwt_claims(self._token))

    def measurement(self) -> str:
        return self._measurement

    def sign(self, payload: bytes) -> str:
        return self._signing_key.sign(payload).signature.hex()

    def verify_key(self) -> bytes:
        """The ephemeral public key this signer's attestation token vouches for."""
        return bytes(self._signing_key.verify_key)

    def token(self) -> str:
        """The raw attestation token binding `verify_key()` to `measurement()`."""
        return self._token


class GcpAttestationVerifier(StubAttestationVerifier):
    """C/R verifier pinned from a verified Confidential Space attestation token.

    Construction (`from_token`) is the GCP analog of `initialize_local_attestation`:
    a one-time provisioning step, run once before any certification session, that
    checks a live token's signature, issuer, and nonce binding, then pins
    `(verify_key, measurement)` -- exactly the pair `StubAttestationVerifier`
    trusts from a local file. Once pinned, `verify()`/`measurement()` are
    inherited unchanged: per-message checking is ordinary Ed25519 verification,
    because the token check already established that this key belongs to a
    genuinely measured boot. Requires the `gcp` extra (`pip install -e '.[gcp]'`).
    """

    ISSUER = "https://confidentialcomputing.googleapis.com"

    @classmethod
    def from_token(cls, token: str, verify_key: bytes, *,
                   audience: str | None = None, jwks_client=None) -> "GcpAttestationVerifier":
        try:
            import jwt
        except ImportError as error:
            raise ImportError(
                "GcpAttestationVerifier needs the `gcp` extra: pip install -e '.[gcp]'"
            ) from error
        client = jwks_client or jwt.PyJWKClient(_oidc_jwks_uri(cls.ISSUER))
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, signing_key.key, algorithms=["RS256", "ES256"], issuer=cls.ISSUER,
            audience=audience, options={"verify_aud": audience is not None})
        nonce = base64.b64encode(hashlib.sha256(verify_key).digest()).decode()
        if nonce not in claims.get("eat_nonce", []):
            raise ValueError("attestation token nonce does not match the given verify_key")
        return cls(verify_key, _image_digest(claims))

    @classmethod
    def from_publication(cls, path: pathlib.Path | None = None) -> "GcpAttestationVerifier":
        publication = json.loads((path or _GCP_PUBLICATION_PATH).read_text())
        return cls(bytes.fromhex(publication["verify_key"]), publication["expected_measurement"])


def publish_gcp_attestation(token: str, verify_key: bytes,
                            path: pathlib.Path | None = None, *,
                            audience: str | None = None, jwks_client=None
                            ) -> "GcpAttestationVerifier":
    """One-time provisioning: verify a live workload token, then pin it for C/R.

    Mirrors `initialize_local_attestation`'s file publication, but the trust
    behind the published `(verify_key, measurement)` pair now comes from a
    verified attestation token instead of "only the local process could have
    written this file".
    """
    verifier = GcpAttestationVerifier.from_token(
        token, verify_key, audience=audience, jwks_client=jwks_client)
    destination = path or _GCP_PUBLICATION_PATH
    destination.parent.mkdir(exist_ok=True)
    payload = json.dumps({"verify_key": verify_key.hex(),
                          "expected_measurement": verifier.measurement()}, sort_keys=True).encode()
    temporary = destination.parent / f".gcp-attestation.tmp.{os.getpid()}.{secrets.token_hex(4)}"
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return verifier

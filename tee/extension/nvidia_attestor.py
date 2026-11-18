"""A GPU-attestation diagnostic plus a software signature, for machines with no CPU attestation.

READ THIS FIRST. This is NOT an attested-workload signer and it is NOT a Confidential Space
substitute. It sits behind the ``EnclaveSigner`` / ``AttestationVerifier`` interfaces only so the
certify and verify code can run unchanged. What it does:

* ``GpuAttestedSoftwareSigner`` makes an ephemeral Ed25519 key in an ordinary process and asks an
  NVIDIA GPU for an attestation whose nonce is ``sha256(verify_key)``. NVIDIA's service (NRAS)
  evaluates the GPU evidence and returns a signed token. The signatures the key makes are ordinary
  software signatures.
* ``GpuAttestationVerifier`` checks the NRAS token against a REQUIRED, preselected GPU-state policy
  and pins ``(verify_key, measurement)``. ``measurement()`` names the attested GPU state
  (firmware, driver, hardware model, security flags). It is not a workload or image measurement.

What is attested: one real NVIDIA GPU in confidential-compute mode ran NVIDIA-signed firmware and
driver measurements that matched NVIDIA's reference values and were un-revoked, with secure boot on
and debug off, and the report was freshly bound to the hash of this key.

What is NOT attested: the CPU side (memory encryption, VM boot chain, host access), the Python
code, the model files, the workload image, and the isolation of the signing key. A host that runs
this signer can request genuine GPU evidence for a key hash of its own choosing and then sign any
invented payload with that key; the GPU token would still verify. So a result signed this way is
not an attested execution of the certification program. Nothing built on this pair may be called a
workload-attested or fully certified result; run records say so (see ``attestation_record``).

Trust model for the NRAS token. The trust anchor is HTTPS to the fixed official NRAS origin and the
signing keys it publishes (JWKS). The certificate chain in a key entry is only checked for internal
consistency (each certificate signed by the next, the leaf valid at the check time, NVIDIA names);
no NVIDIA root is pinned and path constraints are not enforced, so this is not root-anchored PKI
validation. Redirects are refused and keys are never fetched from a URL named by a token.

Policy. A verifier cannot be built without an accepted-GPU policy (``GpuPolicy``): an exact digest
of the specified fields (GPU count, hardware model, driver and VBIOS versions, secure boot, debug
state, the required security claims). The policy is chosen before the run, from a development
smoke, and kept in a file with its hash. The only code that runs without a policy is
``discover_gpu_state``, a distinctly named diagnostic whose output says it is not certified.

Every failure raises. There is no fallback to a stub, and a skipped check is never reported as a
pass. Import needs only the standard library and PyNaCl; the NVIDIA verifier package
(``nv-local-gpu-verifier``) is imported only when evidence is collected, and PyJWT with
``cryptography`` (the ``gcp`` extra) only when a token is verified. The NVIDIA Python packages are
past their end of support (2026-09-15); this is a historical research diagnostic, not maintained
production code. The SDK's own ``Attestation`` wrapper swallows errors, so the nvtrust verifier
(``verifier.cc_admin``) is called directly.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import math
import pathlib
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from nacl.signing import SigningKey

from ct_core.protocol.attestor import EnclaveSigner, StubAttestationVerifier

ATTESTATION_ROOT = "nvidia-gpu-only-software-signed"
NRAS_ISSUER = "https://nras.attestation.nvidia.com"
NRAS_ATTEST_URL = f"{NRAS_ISSUER}/v3/attest/gpu"
NRAS_JWKS_URL = f"{NRAS_ISSUER}/.well-known/jwks.json"
RIM_SERVICE_URL = "https://rim.attestation.nvidia.com/v1/rim/"
OCSP_SERVICE_URL = "https://ocsp.ndis.nvidia.com/"
CLAIMS_VERSION = "2.0"
OVERALL_SUBJECT = "NVIDIA-PLATFORM-ATTESTATION"

NOT_ATTESTED = (
    "the CPU side: CPU memory encryption, CPU attestation, the VM boot chain, host or hypervisor access",
    "the Python code that makes and uses the signing key",
    "the model files and the workload image",
    "the isolation of the signing key from the host and the guest operator",
    "that this GPU is the one a later workload uses",
)

# per-GPU claims that NRAS must report as true
_REQUIRED_GPU_TRUE = (
    "x-nvidia-gpu-attestation-report-cert-chain-validated",
    "x-nvidia-gpu-attestation-report-parsed",
    "x-nvidia-gpu-attestation-report-signature-verified",
    "x-nvidia-gpu-attestation-report-nonce-match",
    "x-nvidia-gpu-arch-check",
    "x-nvidia-gpu-driver-rim-fetched",
    "x-nvidia-gpu-driver-rim-schema-validated",
    "x-nvidia-gpu-driver-rim-signature-verified",
    "x-nvidia-gpu-driver-rim-cert-validated",
    "x-nvidia-gpu-driver-rim-measurements-available",
    "x-nvidia-gpu-vbios-rim-fetched",
    "x-nvidia-gpu-vbios-rim-schema-validated",
    "x-nvidia-gpu-vbios-rim-signature-verified",
    "x-nvidia-gpu-vbios-rim-cert-validated",
    "x-nvidia-gpu-vbios-rim-measurements-available",
    "x-nvidia-gpu-vbios-index-no-conflict",
)
_CLOCK_LEEWAY_SECONDS = 300


class GpuAttestationError(RuntimeError):
    """The GPU attestation could not be collected or did not check out. Never caught here."""


# ---------------------------------------------------------------- local GPU state

def read_cc_state() -> dict:
    """Ask the driver for the confidential-compute state; raises if it cannot be read."""
    def run(flag: str) -> str:
        try:
            done = subprocess.run(["nvidia-smi", "conf-compute", flag], capture_output=True,
                                  text=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            raise GpuAttestationError(f"cannot read GPU confidential-compute state: {error}") from error
        return done.stdout

    def after_colon(text: str, label: str) -> str:
        for line in text.splitlines():
            if label.lower() in line.lower() and ":" in line:
                return line.split(":", 1)[1].strip().upper()
        raise GpuAttestationError(f"no {label!r} in nvidia-smi conf-compute output: {text!r}")

    return {
        "cc_status": after_colon(run("-f"), "CC status"),
        "devtools_mode": after_colon(run("-d"), "DevTools Mode"),
        "environment": after_colon(run("-e"), "CC Environment"),
        "gpus_ready_state": run("-grs").strip().rsplit(":", 1)[-1].strip().upper(),
    }


def require_cc_on(state: dict) -> None:
    """CC must be on, with dev tools off and a production environment; anything else raises."""
    if state.get("cc_status") != "ON":
        raise GpuAttestationError(f"GPU confidential-compute mode is not ON: {state!r}")
    if state.get("devtools_mode") != "OFF":
        raise GpuAttestationError(f"GPU is in CC DevTools mode, which is not a protected state: {state!r}")
    if state.get("environment") != "PRODUCTION":
        raise GpuAttestationError(f"GPU CC environment is not PRODUCTION: {state!r}")


# ---------------------------------------------------------------- evidence and NRAS

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect to another origin could swap the service; a 3xx is an error here."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _check_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost"):
        raise GpuAttestationError(f"refusing a non-https service URL: {url}")


def collect_gpu_evidence(nonce_hex: str) -> list:
    """Collect GPU evidence with NVIDIA's nvtrust verifier. Raises if the package or GPU is missing."""
    try:
        from verifier import cc_admin  # nv-local-gpu-verifier (also a dependency of nv-attestation-sdk)
    except ImportError as error:
        raise GpuAttestationError(
            "NVIDIA attestation support is missing: pip install nv-attestation-sdk nv-local-gpu-verifier") from error
    try:
        evidence = cc_admin.collect_gpu_evidence_remote(nonce_hex, ppcie_mode=False)
    except Exception as error:
        raise GpuAttestationError(f"GPU evidence collection failed: {error}") from error
    if not evidence:
        raise GpuAttestationError("GPU evidence collection returned no evidence")
    return evidence


def request_nras_token(nonce_hex: str, evidence: list, *, url: str = NRAS_ATTEST_URL,
                       timeout: float = 30.0) -> str:
    """POST the evidence to NRAS and return its raw JSON response text (the attestation token)."""
    _check_url(url)
    archs = {item.get("arch") for item in evidence}
    if len(archs) != 1 or None in archs or "" in archs:
        raise GpuAttestationError(f"evidence has missing or mixed GPU architectures: {archs!r}")
    body = json.dumps({
        "nonce": nonce_hex,
        "evidence_list": [{"evidence": item["evidence"], "certificate": item["certificate"]} for item in evidence],
        "arch": archs.pop(),
        "claims_version": CLAIMS_VERSION,
    }).encode()
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:  # noqa: S310 - https URL checked above
            return response.read().decode()
    except urllib.error.HTTPError as error:
        raise GpuAttestationError(f"NRAS refused the attestation: {error.code} {error.read()[:500]!r}") from error
    except (urllib.error.URLError, OSError) as error:
        raise GpuAttestationError(f"NRAS is unreachable: {error}") from error


def fetch_nras_jwks(*, url: str = NRAS_JWKS_URL, timeout: float = 30.0) -> dict:
    _check_url(url)
    try:
        with _OPENER.open(url, timeout=timeout) as response:  # noqa: S310 - https URL checked above
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise GpuAttestationError(f"NRAS signing keys are unavailable: {error}") from error


class JwksCache:
    """A warm key cache with provenance and expiry. Without one, every verification fetches (cold)."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._jwks: dict | None = None
        self._fetched_at = 0.0

    def get(self, kids: set) -> dict | None:
        """The cached key set if it is unexpired and holds every needed key id, else None."""
        if self._jwks is None or time.time() - self._fetched_at > self.ttl_seconds:
            return None
        have = {key.get("kid") for key in self._jwks.get("keys", [])}
        return self._jwks if kids <= have else None

    def put(self, jwks: dict) -> None:
        self._jwks, self._fetched_at = jwks, time.time()

    def provenance(self) -> dict | None:
        if self._jwks is None:
            return None
        return {"url": NRAS_JWKS_URL, "fetched_at_unix": self._fetched_at,
                "expires_unix": self._fetched_at + self.ttl_seconds,
                "sha256": hashlib.sha256(json.dumps(self._jwks, sort_keys=True).encode()).hexdigest()}


def local_gpu_check(nonce_hex: str) -> None:
    """Optional local verification: re-collect GPU evidence for the nonce and verify it with NVIDIA's
    local verifier (reference measurements and certificate status are still fetched from NVIDIA's
    RIM and OCSP services). Must run on the machine that holds the GPU. Raises unless it passes."""
    try:
        from verifier import cc_admin
    except ImportError as error:
        raise GpuAttestationError(
            "local GPU verification needs: pip install nv-local-gpu-verifier") from error
    try:
        evidence = cc_admin.collect_gpu_evidence_local(nonce_hex, ppcie_mode=False)
        result, token = cc_admin.attest({
            "verbose": False, "test_no_gpu": False, "driver_rim": None, "vbios_rim": None, "user_mode": True,
            "rim_root_cert": None, "rim_service_url": RIM_SERVICE_URL, "allow_hold_cert": False,
            "ocsp_url": OCSP_SERVICE_URL, "nonce": nonce_hex, "ppcie_mode": False,
            "ocsp_nonce_disabled": False, "service_key": None, "claims_version": CLAIMS_VERSION,
        }, nonce_hex, evidence)
    except Exception as error:
        raise GpuAttestationError(f"local GPU verification failed: {error}") from error
    if result is not True:
        raise GpuAttestationError("local GPU verification did not pass")
    # the local token is signed with a throwaway key, so only the boolean above is evidence;
    # still require its own overall result and nonce to agree
    claims = _jwt_payload(token[0][1])
    if claims.get("x-nvidia-overall-att-result") is not True or not _same(claims.get("eat_nonce"), nonce_hex):
        raise GpuAttestationError("local GPU verification token disagrees with the result or the nonce")


# ---------------------------------------------------------------- token parsing

def _jwt_payload(token: str) -> dict:
    """Decode a JWT payload without checking the signature (the signer reading its own token)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed attestation token")
    return json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))


def _same(left, right) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left.lower(), right.lower())


def _split_token(token: str) -> tuple[str, dict]:
    """The NRAS response: ``[["JWT", overall_jwt], {"GPU-0": detached_jwt, ...}]``."""
    try:
        parsed = json.loads(token)
        (kind, overall), detached = parsed
    except (ValueError, TypeError) as error:
        raise ValueError("malformed NVIDIA attestation token") from error
    if (kind != "JWT" or not isinstance(overall, str) or not isinstance(detached, dict) or not detached
            or not all(isinstance(value, str) for value in detached.values())):
        raise ValueError("malformed NVIDIA attestation token")
    return overall, detached


def measurement_of(gpu_claims: dict) -> str:
    """A stable name for the attested GPU software state (firmware, driver, security flags).

    Device identity (ueid) is left out on purpose; it is in the claims for a policy that wants it.
    """
    states = [{
        "name": name, "hwmodel": claims.get("hwmodel"), "oemid": claims.get("oemid"),
        "driver_version": claims.get("x-nvidia-gpu-driver-version"),
        "vbios_version": claims.get("x-nvidia-gpu-vbios-version"),
        "secboot": claims.get("secboot"), "dbgstat": claims.get("dbgstat"),
    } for name, claims in sorted(gpu_claims.items())]
    digest = hashlib.sha256(json.dumps(states, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"nvidia-gpu:sha256:{digest}"


# ---------------------------------------------------------------- policy

def policy_fields(gpu_claims: dict) -> dict:
    """The fields a policy fixes, read from verified per-GPU claims."""
    return {
        "schema": 1,
        "gpu_count": len(gpu_claims),
        "gpus": [{
            "name": name, "hwmodel": claims.get("hwmodel"), "oemid": claims.get("oemid"),
            "driver_version": claims.get("x-nvidia-gpu-driver-version"),
            "vbios_version": claims.get("x-nvidia-gpu-vbios-version"),
            "secboot": claims.get("secboot"), "dbgstat": claims.get("dbgstat"),
            "measres": claims.get("measres"), "attestation_warning": claims.get("x-nvidia-attestation-warning"),
        } for name, claims in sorted(gpu_claims.items())],
        "required_true_claims": sorted(_REQUIRED_GPU_TRUE),
    }


def _digest(fields: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GpuPolicy:
    """An accepted GPU state: an exact digest of the fields in ``policy_fields``.

    Freeze it from a development smoke, write it to a file, and pass its digest to the final job
    independently. It is a GPU-state policy, not a workload or image policy.
    """

    def __init__(self, fields: dict, *, source: str = "") -> None:
        self.fields, self.source = fields, source
        self.digest = _digest(fields)

    @classmethod
    def from_verified_claims(cls, gpu_claims: dict, *, source: str) -> "GpuPolicy":
        return cls(policy_fields(gpu_claims), source=source)

    def write(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps({
            "policy_version": 1,
            "scope": "GPU state only (hardware, driver, VBIOS, security flags); not a workload image or CPU policy",
            "frozen_from": self.source, "fields": self.fields, "fields_digest": self.digest,
        }, indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path, *, expected_digest: str) -> "GpuPolicy":
        """Read a policy file. ``expected_digest`` is required: it is the independently chosen value."""
        if not expected_digest:
            raise ValueError("a policy digest chosen independently of this run is required")
        data = json.loads(pathlib.Path(path).read_text())
        policy = cls(data["fields"], source=data.get("frozen_from", ""))
        if data.get("fields_digest") != policy.digest:
            raise ValueError("policy file does not match its own recorded digest")
        if policy.digest != expected_digest:
            raise ValueError(f"policy digest {policy.digest} is not the expected {expected_digest}")
        return policy


# ---------------------------------------------------------------- signer

class GpuAttestedSoftwareSigner(EnclaveSigner):
    """A software signer whose ephemeral key hash is the nonce of a real NVIDIA GPU attestation.

    NOT an attested-workload signer (see the module docstring). The key lives in an ordinary
    process's memory. ``sha256(verify_key)`` is the nonce of a GPU attestation report that NRAS turns
    into a signed token (``token()``). ``measurement()`` is the attested GPU state, not a workload.

    The three callables are seams for tests (mock them); by default they read the driver's CC state,
    collect evidence with NVIDIA's verifier package, and post it to NRAS. Any failure raises.
    ``timings`` keeps the pieces apart: local CC-state check, evidence collection, NRAS POST.
    """

    def __init__(self, *, signing_key: SigningKey | None = None, cc_state=None, collect_evidence=None,
                 request_token=None, nras_url: str = NRAS_ATTEST_URL, timeout: float = 30.0) -> None:
        self._signing_key = signing_key or SigningKey.generate()
        self._nonce_hex = hashlib.sha256(bytes(self._signing_key.verify_key)).hexdigest()
        begin = time.perf_counter()
        self._cc_state = (cc_state or read_cc_state)()
        require_cc_on(self._cc_state)
        start = time.perf_counter()
        self._evidence = (collect_evidence or collect_gpu_evidence)(self._nonce_hex)
        if not self._evidence:
            raise GpuAttestationError("GPU evidence collection returned no evidence")
        middle = time.perf_counter()
        self._token = (request_token or (lambda nonce, evidence: request_nras_token(
            nonce, evidence, url=nras_url, timeout=timeout)))(self._nonce_hex, self._evidence)
        end = time.perf_counter()
        self.timings = {"cc_state_check_seconds": start - begin, "evidence_collect_seconds": middle - start,
                        "nras_post_seconds": end - middle}
        try:
            overall, detached = _split_token(self._token)
            overall_claims = _jwt_payload(overall)
            self._measurement = measurement_of({name: _jwt_payload(value) for name, value in detached.items()})
        except ValueError as error:
            raise GpuAttestationError(f"NRAS returned an unusable attestation token: {error}") from error
        if not _same(overall_claims.get("eat_nonce"), self._nonce_hex):
            raise GpuAttestationError("NRAS returned a token for a different nonce")

    def measurement(self) -> str:
        return self._measurement

    def sign(self, payload: bytes) -> str:
        return self._signing_key.sign(payload).signature.hex()

    def verify_key(self) -> bytes:
        """The ephemeral public key the GPU attestation token is bound to."""
        return bytes(self._signing_key.verify_key)

    def token(self) -> str:
        """The raw NRAS attestation token (JSON text) binding ``verify_key()`` to the GPU state."""
        return self._token

    def evidence(self) -> list:
        """The raw GPU evidence (certificate chain and SPDM report, base64) that NRAS evaluated."""
        return self._evidence

    def nonce_hex(self) -> str:
        return self._nonce_hex

    def cc_state(self) -> dict:
        """The driver's CC state when the signer was built. Read locally; not part of the attestation."""
        return dict(self._cc_state)


# ---------------------------------------------------------------- verification

def _check_time_claims(claims: dict, now: float, label: str) -> None:
    """iat, nbf and exp must be real numbers and the token must be current, with the declared leeway."""
    values = {}
    for name in ("iat", "nbf", "exp"):
        value = claims.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label}: malformed {name} claim: {value!r}")
        values[name] = value
    if values["nbf"] - _CLOCK_LEEWAY_SECONDS > now:
        raise ValueError(f"{label}: attestation token is not yet valid (nbf is in the future)")
    if values["iat"] - _CLOCK_LEEWAY_SECONDS > now:
        raise ValueError(f"{label}: attestation token was issued in the future (iat)")
    if values["nbf"] > values["exp"] or values["iat"] > values["exp"]:
        raise ValueError(f"{label}: nbf or iat is after exp")
    if now > values["exp"] + _CLOCK_LEEWAY_SECONDS:
        raise ValueError(f"{label}: attestation token is expired")


def _decode_nras_jwt(jwt, x509, token: str, jwks: dict, now: float, label: str) -> dict:
    """Verify an NRAS ES384 JWT against the key and certificate chain in NRAS's published key set."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as error:
        raise ValueError(f"malformed JWT: {error}") from error
    if header.get("alg") != "ES384":
        raise ValueError(f"unexpected JWT algorithm {header.get('alg')!r}")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")), None)
    if key is None or len(key.get("x5c", [])) < 2:
        raise ValueError(f"NRAS key {header.get('kid')!r} not found with a certificate chain")
    chain = [x509.load_der_x509_certificate(base64.b64decode(item)) for item in key["x5c"]]
    leaf = chain[0]
    # consistency checks only, not root-anchored validation (see the module docstring)
    for child, parent in zip(chain, chain[1:]):
        child.verify_directly_issued_by(parent)
    names = " ".join(cert.subject.rfc4514_string() for cert in chain)
    if "NVIDIA Corporation" not in leaf.subject.rfc4514_string() or "NVIDIA Attestation Service" not in names:
        raise ValueError(f"signing certificate is not NVIDIA's: {leaf.subject.rfc4514_string()}")
    utc = datetime.timezone.utc
    not_before = getattr(leaf, "not_valid_before_utc", None) or leaf.not_valid_before.replace(tzinfo=utc)
    not_after = getattr(leaf, "not_valid_after_utc", None) or leaf.not_valid_after.replace(tzinfo=utc)
    if not (not_before.timestamp() - _CLOCK_LEEWAY_SECONDS <= now <= not_after.timestamp() + _CLOCK_LEEWAY_SECONDS):
        raise ValueError("NRAS signing certificate is not valid at the verification time")
    claims = jwt.decode(token, leaf.public_key(), algorithms=["ES384"], issuer=NRAS_ISSUER,
                        options={"verify_exp": False, "verify_iat": False, "verify_nbf": False,
                                 "require": ["iss", "eat_nonce"]})
    _check_time_claims(claims, now, label)
    return claims


def _check_gpu_claims(name: str, claims: dict, nonce_hex: str) -> None:
    if not _same(claims.get("eat_nonce"), nonce_hex):
        raise ValueError(f"{name}: per-GPU nonce does not match the given verify_key")
    missing = [claim for claim in _REQUIRED_GPU_TRUE if claims.get(claim) is not True]
    if missing:
        raise ValueError(f"{name}: GPU claims not true: {missing}")
    false = [claim for claim, value in claims.items() if claim.startswith("x-nvidia-gpu-") and value is False]
    if false:
        raise ValueError(f"{name}: GPU claims are false: {false}")
    if claims.get("measres") != "success":
        raise ValueError(f"{name}: GPU measurements did not match the reference values (measres={claims.get('measres')!r})")
    if claims.get("secboot") is not True:
        raise ValueError(f"{name}: GPU secure boot is not reported enabled")
    if claims.get("dbgstat") != "disabled":
        raise ValueError(f"{name}: GPU debug status is {claims.get('dbgstat')!r}, not 'disabled'")
    if claims.get("x-nvidia-attestation-warning") is not None:
        raise ValueError(f"{name}: NRAS attached a warning: {claims.get('x-nvidia-attestation-warning')!r}")
    for claim in ("x-nvidia-gpu-driver-version", "x-nvidia-gpu-vbios-version", "hwmodel"):
        if not isinstance(claims.get(claim), str) or not claims[claim]:
            raise ValueError(f"{name}: GPU claim {claim} is missing")


def _verify_token(token: str, verify_key: bytes, *, fetch_jwks, jwks_cache, now: float | None):
    """Every check except the policy. Returns (overall claims, per-GPU claims, timings)."""
    try:
        import jwt
        from cryptography import x509
    except ImportError as error:
        raise ImportError(
            "GPU attestation verification needs the `gcp` extra (PyJWT + cryptography): pip install -e '.[gcp]'"
        ) from error
    started = time.perf_counter()
    now = time.time() if now is None else now
    nonce_hex = hashlib.sha256(verify_key).hexdigest()
    overall_jwt, detached = _split_token(token)

    try:
        kids = {jwt.get_unverified_header(value).get("kid") for value in (overall_jwt, *detached.values())}
    except jwt.PyJWTError as error:
        raise ValueError(f"malformed JWT: {error}") from error
    jwks = jwks_cache.get(kids) if jwks_cache is not None else None
    jwks_seconds, fetched = 0.0, jwks is None
    if jwks is None:
        begin = time.perf_counter()
        jwks = (fetch_jwks or fetch_nras_jwks)()
        jwks_seconds = time.perf_counter() - begin
        if jwks_cache is not None:
            jwks_cache.put(jwks)

    def decode(value: str, label: str) -> dict:
        return _decode_nras_jwt(jwt, x509, value, jwks, now, label)

    overall = decode(overall_jwt, "overall token")
    if overall.get("x-nvidia-overall-att-result") is not True:
        raise ValueError("NVIDIA attestation overall result is not true")
    if overall.get("x-nvidia-ver") != CLAIMS_VERSION:
        raise ValueError(f"unsupported NVIDIA claim-schema version {overall.get('x-nvidia-ver')!r}")
    if overall.get("sub") != OVERALL_SUBJECT:
        raise ValueError(f"unexpected overall token subject {overall.get('sub')!r}")
    if not _same(overall.get("eat_nonce"), nonce_hex):
        raise ValueError("attestation token nonce does not match the given verify_key")
    submods = overall.get("submods")
    if not isinstance(submods, dict) or set(submods) != set(detached):
        raise ValueError("attestation token GPU list does not match its detached claims")

    gpus = {}
    for name, value in detached.items():
        entry = submods[name]
        if (not isinstance(entry, list) or len(entry) != 2 or entry[0] != "DIGEST"
                or not isinstance(entry[1], list) or len(entry[1]) != 2):
            raise ValueError(f"attestation token digest entry for {name} is malformed (expected DIGEST/SHA-256)")
        algorithm, expected_digest = entry[1]
        if algorithm != "SHA-256" or not _same(expected_digest, hashlib.sha256(value.encode()).hexdigest()):
            raise ValueError(f"detached claims for {name} do not match the signed digest")
        claims = decode(value, f"{name} token")
        _check_gpu_claims(name, claims, nonce_hex)
        gpus[name] = claims
    timings = {"jwks_get_seconds": jwks_seconds,
               "local_token_verify_seconds": time.perf_counter() - started - jwks_seconds,
               "key_cache_policy": "cold" if jwks_cache is None else "warm",
               "jwks_fetched": fetched}
    return overall, gpus, timings


class GpuAttestationVerifier(StubAttestationVerifier):
    """C/R verifier pinned from a verified NVIDIA GPU attestation token and a REQUIRED GPU policy.

    This verifies a GPU attestation of a software signer. It does not verify a workload (see the
    module docstring). ``from_token`` is the one-time provisioning step. It raises on any failure:
    signature, issuer, ``iat``/``nbf``/``exp`` of every token (overall and per GPU, with a 300 s
    leeway), claim-schema version, DIGEST tag, nonce (overall and per GPU) equal to
    ``sha256(verify_key)``, per-GPU security claims, and an exact match to ``policy``. Then it pins
    ``(verify_key, measurement)``. ``policy`` has no default: with none, it fails.
    ``timings`` separates the key download (``jwks_get_seconds``) from local checking.
    """

    ISSUER = NRAS_ISSUER

    def __init__(self, verify_key: bytes, expected_measurement: str, *, claims: dict | None = None,
                 policy: GpuPolicy | None = None, timings: dict | None = None) -> None:
        super().__init__(verify_key, expected_measurement)
        self._claims, self._policy, self.timings = claims, policy, timings or {}

    def verified_claims(self) -> dict | None:
        """``{"overall": {...}, "gpus": {name: {...}}}`` as verified by ``from_token``, else None."""
        return self._claims

    def policy(self) -> GpuPolicy | None:
        return self._policy

    @classmethod
    def from_token(cls, token: str, verify_key: bytes, *, policy: GpuPolicy, fetch_jwks=None,
                   jwks_cache: JwksCache | None = None, local_check=None, now: float | None = None
                   ) -> "GpuAttestationVerifier":
        if not isinstance(policy, GpuPolicy):
            raise ValueError("an accepted-GPU policy (GpuPolicy) is required; there is no default")
        overall, gpus, timings = _verify_token(token, verify_key, fetch_jwks=fetch_jwks,
                                               jwks_cache=jwks_cache, now=now)
        observed = policy_fields(gpus)
        if observed != policy.fields:
            raise ValueError(f"attested GPU state {_digest(observed)} is not the accepted policy {policy.digest}")
        if local_check is not None:
            begin = time.perf_counter()
            local_check(hashlib.sha256(verify_key).hexdigest())
            timings["local_gpu_check_seconds"] = time.perf_counter() - begin
        return cls(verify_key, measurement_of(gpus), claims={"overall": overall, "gpus": gpus},
                   policy=policy, timings=timings)


def discover_gpu_state(token: str, verify_key: bytes, *, fetch_jwks=None, now: float | None = None) -> dict:
    """DISCOVERY DIAGNOSTIC, NOT A VERIFIER. Runs every token check except the policy and reports what
    state the token shows, so a policy can be frozen from a development run. The output is marked
    not certified and must never be used to accept a final run."""
    overall, gpus, _ = _verify_token(token, verify_key, fetch_jwks=fetch_jwks, jwks_cache=None, now=now)
    fields = policy_fields(gpus)
    return {"certified": False, "diagnostic": "discovery: no policy was applied; do not use to accept a run",
            "observed_fields": fields, "observed_fields_digest": _digest(fields),
            "measurement": measurement_of(gpus), "overall_iat": overall["iat"]}


def attestation_record(verifier: GpuAttestationVerifier, signer: GpuAttestedSoftwareSigner | None = None) -> dict:
    """What a run record says about this mode. It is a GPU attestation plus a software signature; it is
    never a workload-attested or fully certified result."""
    claims = verifier.verified_claims() or {"overall": {}, "gpus": {}}
    policy = verifier.policy()
    record = {
        "root": ATTESTATION_ROOT,
        "gpu_attestation_verified": True,
        "workload_attested": False,
        "cpu_attested": False,
        "full_certification": False,
        "scope": "GPU attestation of a software-held signing key; not an attested CT workload",
        "covers": "GPU firmware and driver measurements, secure boot and debug-off state, fresh nonce binding to the signing key's hash",
        "not_attested": list(NOT_ATTESTED),
        "policy_digest": policy.digest if policy else None,
        "measurement": verifier.measurement(),
        "issuer": claims["overall"].get("iss"),
        "token_issued_at": claims["overall"].get("iat"),
        "gpus": {name: {key: gpu.get(key) for key in (
            "hwmodel", "x-nvidia-gpu-driver-version", "x-nvidia-gpu-vbios-version", "secboot", "dbgstat", "measres")}
            for name, gpu in claims["gpus"].items()},
        "verifier_timings": dict(verifier.timings),
    }
    if signer is not None:
        record["local_cc_state_not_attested"] = signer.cc_state()
        record["signer_timings"] = dict(signer.timings)
    return record


# ---------------------------------------------------------------- timing summary

_FETCH_PARTS = ("cc_state_check_seconds", "evidence_collect_seconds", "nras_post_seconds")
_VERIFY_PARTS = ("jwks_get_seconds", "local_token_verify_seconds")


def summarize_timings(repetitions: list) -> dict:
    """Reduce per-repetition records. Each has ``attestation_fetch_seconds`` and ``attestation_verify_seconds``
    (outer wall times, network included), ``fetch_parts``, ``verify_parts`` and ``key_cache_policy``. The
    combined figure is the median (and mean) of the per-repetition SUM, not a sum of medians."""
    def stats(values):
        values = list(values)
        return {"median": statistics.median(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}

    policies = {rep["key_cache_policy"] for rep in repetitions}
    if len(policies) != 1:
        raise ValueError(f"repetitions mix key-cache policies: {policies}")
    return {
        "repetitions": len(repetitions),
        "key_cache_policy": policies.pop(),
        "attestation_fetch_seconds": stats(r["attestation_fetch_seconds"] for r in repetitions),
        "attestation_verify_seconds": stats(r["attestation_verify_seconds"] for r in repetitions),
        "fetch_plus_verify_per_repetition_sum_seconds": stats(
            r["attestation_fetch_seconds"] + r["attestation_verify_seconds"] for r in repetitions),
        "fetch_parts": {part: stats(r["fetch_parts"][part] for r in repetitions) for part in _FETCH_PARTS},
        "verify_parts": {part: stats(r["verify_parts"][part] for r in repetitions) for part in _VERIFY_PARTS},
    }

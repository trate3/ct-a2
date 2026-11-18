"""Tests for the GPU-attestation software signer and verifier. THESE ARE MOCK TESTS.

No real GPU, NVIDIA SDK, or NRAS is used. A mock NRAS (a throwaway CA, an intermediate and an
ES384 signing key made here) issues tokens shaped like NRAS's, and the evidence and CC-state
seams are replaced with fakes. They check the class logic (nonce binding, rejection of bad
tokens, CC-off refusal, unreachable service, no silent fallback), not the hardware or NVIDIA's
real service.
Needs the `gcp` extra (PyJWT with cryptography); skips cleanly without it.
"""

from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import json
import sys

import pytest
from nacl.signing import SigningKey

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

import tee.extension.nvidia_attestor as nv  # noqa: E402
from ct_core.certification import verify_certificate  # noqa: E402,F401  (import check: base classes line up)
from tee.extension.nvidia_attestor import (  # noqa: E402
    GpuAttestationError, GpuAttestationVerifier, GpuAttestedSoftwareSigner, GpuPolicy)

NOW = 1_800_000_000.0
CC_ON = {"cc_status": "ON", "devtools_mode": "OFF", "environment": "PRODUCTION", "gpus_ready_state": "READY"}
FAKE_EVIDENCE = [{"arch": "BLACKWELL", "evidence": "bW9jaw==", "certificate": "bW9jaw=="}]
GOOD_GPU = {
    **{claim: True for claim in nv._REQUIRED_GPU_TRUE},
    "secboot": True, "dbgstat": "disabled", "measres": "success", "hwmodel": "MOCK", "oemid": "0",
    "ueid": "1", "x-nvidia-attestation-warning": None,
    "x-nvidia-gpu-driver-version": "000.00", "x-nvidia-gpu-vbios-version": "00.00",
}


def _name(common: str, org: str = "NVIDIA Corporation") -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, org), x509.NameAttribute(NameOID.COMMON_NAME, common)])


def _cert(subject, issuer, public_key, signing_key, *, ca: bool, start=NOW - 3600, hours=12):
    begin = datetime.datetime.fromtimestamp(start, datetime.timezone.utc)
    return (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(public_key)
            .serial_number(x509.random_serial_number()).not_valid_before(begin)
            .not_valid_after(begin + datetime.timedelta(hours=hours))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(signing_key, hashes.SHA384()))


class MockNras:
    """MOCK of NRAS: a throwaway chain plus an ES384 signer that issues NRAS-shaped tokens."""

    def __init__(self, *, chain_ok: bool = True, cert_start: float = NOW - 3600):
        root_key, mid_key, self.leaf_key = (ec.generate_private_key(ec.SECP384R1()) for _ in range(3))
        root_name, mid_name = _name("NVIDIA Attestation Service CA MOCK"), _name("NVIDIA Attestation Service GPU Intermediate MOCK")
        mid = _cert(mid_name, root_name, mid_key.public_key(), root_key, ca=True)
        # a leaf signed by the wrong key models a forged chain
        leaf = _cert(_name("NVIDIA Attestation Service GPU MOCK"), mid_name, self.leaf_key.public_key(),
                     mid_key if chain_ok else root_key, ca=False, start=cert_start)
        self.kid = "mock-kid-1"
        encode = lambda c: base64.b64encode(c.public_bytes(serialization.Encoding.DER)).decode()  # noqa: E731
        self._jwks = {"keys": [{"kty": "EC", "crv": "P-384", "kid": self.kid, "x5c": [encode(leaf), encode(mid)]}]}

    def jwks(self) -> dict:
        return copy.deepcopy(self._jwks)

    def _sign(self, claims: dict, *, key=None, alg="ES384", kid=None) -> str:
        return jwt.encode(claims, key or self.leaf_key, algorithm=alg, headers={"kid": kid or self.kid})

    def issue(self, nonce_hex: str, *, gpu=None, overall=None, issuer=nv.NRAS_ISSUER, iat=NOW, exp_in=3600,
              gpu_nonce=None, sign_key=None, alg="ES384", tamper_detached=False, n_gpus=1,
              digest_tag="DIGEST", digest_alg="SHA-256") -> str:
        base = {"iss": issuer, "iat": int(iat), "nbf": int(iat), "exp": int(iat + exp_in)}
        detached, submods = {}, {}
        for index in range(n_gpus):
            claims = {**base, **GOOD_GPU, "eat_nonce": gpu_nonce or nonce_hex, **(gpu or {})}
            value = self._sign(claims, key=sign_key, alg=alg)
            submods[f"GPU-{index}"] = [digest_tag, [digest_alg, hashlib.sha256(value.encode()).hexdigest()]]
            detached[f"GPU-{index}"] = self._sign({**claims, "jti": "tampered"}) if tamper_detached else value
        top = {**base, "sub": nv.OVERALL_SUBJECT, "x-nvidia-ver": "2.0", "x-nvidia-overall-att-result": True, "eat_nonce": nonce_hex, "submods": submods, **(overall or {})}
        return json.dumps([["JWT", self._sign(top, key=sign_key, alg=alg)], detached])


def _nonce(verify_key: bytes) -> str:
    return hashlib.sha256(verify_key).hexdigest()


def _signer(nras: MockNras, key: SigningKey | None = None, **issue_kwargs) -> GpuAttestedSoftwareSigner:
    seen = {}

    def collect(nonce):
        seen["collect_nonce"] = nonce
        return FAKE_EVIDENCE

    def request(nonce, evidence):
        seen["request_nonce"] = nonce
        return nras.issue(nonce, **issue_kwargs)

    signer = GpuAttestedSoftwareSigner(signing_key=key, cc_state=lambda: CC_ON, collect_evidence=collect, request_token=request)
    signer.seen = seen
    return signer


def _policy(n_gpus: int = 1, **claims) -> GpuPolicy:
    return GpuPolicy(nv.policy_fields({f"GPU-{index}": {**GOOD_GPU, **claims} for index in range(n_gpus)}), source="mock")


def _verify(nras, token, verify_key, **kwargs):
    kwargs.setdefault("policy", _policy())
    return GpuAttestationVerifier.from_token(token, verify_key, fetch_jwks=nras.jwks, now=NOW, **kwargs)


# ---------------------------------------------------------------- signer and nonce binding

def test_signer_binds_verify_key_hash_as_the_nonce_and_verifier_pins_it():
    nras = MockNras()
    key = SigningKey.generate()
    signer = _signer(nras, key)
    expected = _nonce(bytes(key.verify_key))
    assert signer.seen == {"collect_nonce": expected, "request_nonce": expected}
    assert len(bytes.fromhex(expected)) == 32 and signer.verify_key() == bytes(key.verify_key)
    verifier = _verify(nras, signer.token(), signer.verify_key())
    assert verifier.measurement() == signer.measurement() and verifier.measurement().startswith("nvidia-gpu:sha256:")
    signature = signer.sign(b"payload")
    assert verifier.verify(b"payload", signature) and not verifier.verify(b"other", signature)
    assert verifier.verified_claims()["overall"]["x-nvidia-overall-att-result"] is True
    assert set(signer.timings) == {"cc_state_check_seconds", "evidence_collect_seconds", "nras_post_seconds"}
    assert signer.evidence() == FAKE_EVIDENCE and signer.cc_state() == CC_ON


def test_keys_are_ephemeral_and_fresh_per_signer():
    nras = MockNras()
    assert _signer(nras).verify_key() != _signer(nras).verify_key()


def test_verifier_accepts_multiple_gpus_under_a_matching_policy():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    token = nras.issue(_nonce(key), n_gpus=2)
    verifier = _verify(nras, token, key, policy=_policy(2))
    assert set(verifier.verified_claims()["gpus"]) == {"GPU-0", "GPU-1"}
    with pytest.raises(ValueError, match="not the accepted policy"):
        _verify(nras, token, key, policy=_policy(1))  # a policy for one GPU does not accept two


# ---------------------------------------------------------------- nonce mismatch

def test_verifier_rejects_a_token_for_another_key():
    nras = MockNras()
    signer = _signer(nras)
    other = bytes(SigningKey.generate().verify_key)
    with pytest.raises(ValueError, match="nonce does not match"):
        _verify(nras, signer.token(), other)


def test_verifier_rejects_a_per_gpu_nonce_mismatch():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    token = nras.issue(_nonce(key), gpu_nonce=_nonce(b"someone else"))
    with pytest.raises(ValueError, match="per-GPU nonce"):
        _verify(nras, token, key)


def test_signer_refuses_a_token_issued_for_a_different_nonce():
    nras = MockNras()
    with pytest.raises(GpuAttestationError, match="different nonce"):
        GpuAttestedSoftwareSigner(cc_state=lambda: CC_ON, collect_evidence=lambda nonce: FAKE_EVIDENCE,
                               request_token=lambda nonce, evidence: nras.issue(_nonce(b"replayed")))


# ---------------------------------------------------------------- other bad tokens

@pytest.mark.parametrize("kwargs, message", [
    ({"overall": {"x-nvidia-overall-att-result": False}}, "overall result"),
    ({"gpu": {"measres": "failure"}}, "did not match the reference"),
    ({"gpu": {"secboot": False}}, "secure boot"),
    ({"gpu": {"dbgstat": "enabled"}}, "debug status"),
    ({"gpu": {"x-nvidia-attestation-warning": "cert on hold"}}, "warning"),
    ({"gpu": {"x-nvidia-gpu-attestation-report-cert-chain-validated": False}}, "not true"),
    ({"gpu": {"x-nvidia-gpu-vbios-rim-signature-verified": None}}, "not true"),
    ({"gpu": {"x-nvidia-gpu-some-future-check": False}}, "are false"),
    ({"gpu": {"x-nvidia-gpu-driver-version": None}}, "driver-version"),
    ({"iat": NOW - 10_000}, "expired"),
    ({"iat": NOW + 10_000}, "not yet valid|issued in the future"),
    ({"tamper_detached": True}, "do not match the signed digest"),
    ({"n_gpus": 0}, "malformed"),
])
def test_verifier_rejects_bad_tokens(kwargs, message):
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    with pytest.raises(ValueError, match=message):
        _verify(nras, nras.issue(_nonce(key), **kwargs), key)


def test_verifier_rejects_a_wrong_issuer():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    with pytest.raises(jwt.InvalidIssuerError):
        _verify(nras, nras.issue(_nonce(key), issuer="https://nras.attestation.nvidia.com.evil.example"), key)


def test_verifier_rejects_a_token_signed_by_an_unpublished_key():
    published, attacker = MockNras(), MockNras()
    key = bytes(SigningKey.generate().verify_key)
    with pytest.raises(jwt.InvalidSignatureError):
        _verify(published, attacker.issue(_nonce(key)), key)  # same kid, different key
    unknown = MockNras()
    unknown.kid = "other-kid"
    with pytest.raises(ValueError, match="not found"):
        _verify(published, unknown.issue(_nonce(key)), key)


def test_verifier_rejects_a_forged_chain_a_foreign_signer_and_a_stale_certificate():
    key = bytes(SigningKey.generate().verify_key)
    forged = MockNras(chain_ok=False)
    with pytest.raises(Exception):  # leaf not issued by the intermediate
        _verify(forged, forged.issue(_nonce(key)), key)
    stale = MockNras(cert_start=NOW - 100 * 3600)
    with pytest.raises(ValueError, match="not valid at the verification time"):
        _verify(stale, stale.issue(_nonce(key)), key)
    nras = MockNras()
    for sign_alg in ("HS256",):
        hs = nras.issue(_nonce(key), sign_key="a-shared-secret-of-sufficient-length-for-hs256", alg=sign_alg)
        with pytest.raises(ValueError, match="unexpected JWT algorithm"):
            _verify(nras, hs, key)


@pytest.mark.parametrize("token", ["", "not json", "[]", '[["JWT", 1], {}]', '[["X", "a.b.c"], {"GPU-0": "a.b.c"}]'])
def test_verifier_rejects_malformed_tokens(token):
    with pytest.raises(ValueError, match="malformed"):
        _verify(MockNras(), token, bytes(SigningKey.generate().verify_key))


# ---------------------------------------------------------------- CC off

@pytest.mark.parametrize("state", [
    {**CC_ON, "cc_status": "OFF"},
    {**CC_ON, "cc_status": "DEVTOOLS"},
    {**CC_ON, "devtools_mode": "ON"},
    {**CC_ON, "environment": "SIMULATION"},
    {},
])
def test_signer_refuses_when_cc_is_not_on_and_never_asks_for_evidence(state):
    called = []
    with pytest.raises(GpuAttestationError):
        GpuAttestedSoftwareSigner(cc_state=lambda: state, collect_evidence=lambda nonce: called.append(nonce),
                               request_token=lambda nonce, evidence: called.append(nonce))
    assert called == []


def test_cc_state_that_cannot_be_read_raises(monkeypatch):
    def broken(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(nv.subprocess, "run", broken)
    with pytest.raises(GpuAttestationError, match="cannot read"):
        nv.read_cc_state()


def test_read_cc_state_parses_driver_output(monkeypatch):
    outputs = {"-f": "CC status: ON\n", "-d": "DevTools Mode: OFF\n", "-e": "CC Environment: PRODUCTION\n",
               "-grs": "\nConfidential Compute GPUs Ready state: ready\n"}

    class Done:
        def __init__(self, text):
            self.stdout = text
    monkeypatch.setattr(nv.subprocess, "run", lambda cmd, **kw: Done(outputs[cmd[-1]]))
    assert nv.read_cc_state() == CC_ON
    nv.require_cc_on(nv.read_cc_state())
    outputs["-f"] = "CC status: OFF\n"
    with pytest.raises(GpuAttestationError, match="not ON"):
        nv.require_cc_on(nv.read_cc_state())


# ---------------------------------------------------------------- SDK missing, service unreachable

def test_missing_nvidia_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "verifier", None)  # makes `from verifier import ...` raise ImportError
    with pytest.raises(GpuAttestationError, match="pip install nv-attestation-sdk"):
        nv.collect_gpu_evidence("00" * 32)
    with pytest.raises(GpuAttestationError, match="nv-local-gpu-verifier"):
        nv.local_gpu_check("00" * 32)


def test_signer_without_the_package_raises_rather_than_degrading(monkeypatch):
    monkeypatch.setitem(sys.modules, "verifier", None)
    with pytest.raises(GpuAttestationError):
        GpuAttestedSoftwareSigner(cc_state=lambda: CC_ON)


def test_unreachable_nras_raises_on_both_sides():
    # port 1 on loopback refuses connections at once; no network is needed
    with pytest.raises(GpuAttestationError, match="unreachable"):
        nv.request_nras_token("00" * 32, FAKE_EVIDENCE, url="http://127.0.0.1:1/v3/attest/gpu", timeout=2)
    with pytest.raises(GpuAttestationError, match="unavailable"):
        nv.fetch_nras_jwks(url="http://127.0.0.1:1/jwks.json", timeout=2)
    with pytest.raises(GpuAttestationError, match="unreachable"):
        GpuAttestedSoftwareSigner(cc_state=lambda: CC_ON, collect_evidence=lambda nonce: FAKE_EVIDENCE,
                               nras_url="http://127.0.0.1:1/v3/attest/gpu", timeout=2)
    nras = MockNras()
    signer = _signer(nras)

    def down():
        raise GpuAttestationError("NRAS signing keys are unavailable: down")
    with pytest.raises(GpuAttestationError, match="unavailable"):
        GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=_policy(), fetch_jwks=down, now=NOW)


def test_service_errors_and_empty_evidence_raise():
    with pytest.raises(GpuAttestationError, match="no evidence"):
        GpuAttestedSoftwareSigner(cc_state=lambda: CC_ON, collect_evidence=lambda nonce: [],
                               request_token=lambda nonce, evidence: "unused")
    with pytest.raises(GpuAttestationError, match="unusable"):
        GpuAttestedSoftwareSigner(cc_state=lambda: CC_ON, collect_evidence=lambda nonce: FAKE_EVIDENCE,
                               request_token=lambda nonce, evidence: '{"error": "bad request"}')
    with pytest.raises(GpuAttestationError, match="architectures"):
        nv.request_nras_token("00" * 32, [{"evidence": "a", "certificate": "b"}])


# ---------------------------------------------------------------- local verification option

def test_local_check_runs_with_the_bound_nonce_and_its_failure_is_fatal():
    nras = MockNras()
    signer = _signer(nras)
    seen = []
    _verify(nras, signer.token(), signer.verify_key(), local_check=seen.append)
    assert seen == [signer.nonce_hex()]

    def fails(nonce):
        raise GpuAttestationError("local GPU verification failed: mock")
    with pytest.raises(GpuAttestationError, match="local GPU verification"):
        _verify(nras, signer.token(), signer.verify_key(), local_check=fails)


# ---------------------------------------------------------------- token validity

@pytest.mark.parametrize("where, offset, ok", [
    ("overall", 10_000, False), ("gpu", 10_000, False),         # a future nbf in either token
    ("overall", 200, True), ("gpu", 200, True),                 # inside the 300 s leeway
    ("overall", 400, False), ("gpu", 400, False),               # just outside it
])
def test_future_nbf_is_rejected_in_every_jwt(where, offset, ok):
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    # exp is pushed out so the future nbf is the only fault
    token = nras.issue(_nonce(key), **{where if where == "overall" else "gpu": {"nbf": int(NOW + offset), "exp": int(NOW + 50_000)}})
    if ok:
        _verify(nras, token, key)
    else:
        with pytest.raises(ValueError, match="not yet valid"):
            _verify(nras, token, key)


@pytest.mark.parametrize("claim, value", [
    ("iat", "1800000000"), ("nbf", True), ("exp", None), ("iat", [1]), ("nbf", "x"), ("exp", {"a": 1}),
])
@pytest.mark.parametrize("where", ["overall", "gpu"])
def test_malformed_time_claims_are_rejected(where, claim, value):
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    with pytest.raises(ValueError, match="malformed"):
        _verify(nras, nras.issue(_nonce(key), **{where: {claim: value}}), key)


def test_missing_nbf_and_inverted_times_are_rejected():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    gpu_without_nbf = nras.issue(_nonce(key))
    overall_jwt, detached = json.loads(gpu_without_nbf)[0][1], json.loads(gpu_without_nbf)[1]
    claims = jwt.decode(overall_jwt, options={"verify_signature": False})
    del claims["nbf"]
    token = json.dumps([["JWT", nras._sign(claims)], detached])
    with pytest.raises(ValueError, match="malformed nbf"):
        _verify(nras, token, key)
    with pytest.raises(ValueError, match="after exp"):
        _verify(nras, nras.issue(_nonce(key), overall={"iat": int(NOW - 200), "nbf": int(NOW - 50), "exp": int(NOW - 100)}), key)


def test_digest_tag_algorithm_and_schema_version_are_checked():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    with pytest.raises(ValueError, match="DIGEST/SHA-256"):
        _verify(nras, nras.issue(_nonce(key), digest_tag="HASH"), key)
    with pytest.raises(ValueError, match="do not match the signed digest"):
        _verify(nras, nras.issue(_nonce(key), digest_alg="SHA-512"), key)
    with pytest.raises(ValueError, match="claim-schema version"):
        _verify(nras, nras.issue(_nonce(key), overall={"x-nvidia-ver": "3.0"}), key)
    with pytest.raises(ValueError, match="claim-schema version"):
        _verify(nras, nras.issue(_nonce(key), overall={"x-nvidia-ver": None}), key)
    with pytest.raises(ValueError, match="subject"):
        _verify(nras, nras.issue(_nonce(key), overall={"sub": "SOMETHING-ELSE"}), key)


# ---------------------------------------------------------------- required policy

def test_no_policy_means_no_verifier():
    nras = MockNras()
    signer = _signer(nras)
    with pytest.raises(TypeError):  # no default: the keyword is required
        GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), fetch_jwks=nras.jwks, now=NOW)
    for bad in (None, "sha256:abc", {"gpu_count": 1}):
        with pytest.raises(ValueError, match="policy"):
            GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=bad,
                                              fetch_jwks=nras.jwks, now=NOW)


@pytest.mark.parametrize("override", [
    {"x-nvidia-gpu-driver-version": "999.99"}, {"x-nvidia-gpu-vbios-version": "99.99"}, {"hwmodel": "OTHER"},
    {"oemid": "1"},
])
def test_an_unapproved_gpu_state_is_rejected_even_when_every_claim_is_good(override):
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    token = nras.issue(_nonce(key), gpu=override)
    with pytest.raises(ValueError, match="not the accepted policy"):
        _verify(nras, token, key)  # the default policy is the state in GOOD_GPU
    _verify(nras, token, key, policy=_policy(**override))  # accepted only when that state is the policy


def test_a_policy_cannot_weaken_the_required_security_claims():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    weak = _policy()
    weak.fields["required_true_claims"] = weak.fields["required_true_claims"][:-1]
    weak = GpuPolicy(weak.fields)
    with pytest.raises(ValueError, match="not the accepted policy"):
        _verify(nras, nras.issue(_nonce(key)), key, policy=weak)


def test_policy_file_round_trip_and_independent_digest(tmp_path):
    policy = _policy()
    path = tmp_path / "p.json"
    policy.write(path)
    assert GpuPolicy.load(path, expected_digest=policy.digest).fields == policy.fields
    with pytest.raises(ValueError, match="independently"):
        GpuPolicy.load(path, expected_digest="")
    with pytest.raises(ValueError, match="not the expected"):
        GpuPolicy.load(path, expected_digest="sha256:" + "0" * 64)
    data = json.loads(path.read_text())
    data["fields"]["gpus"][0]["driver_version"] = "999.99"  # edit the fields, keep the recorded digest
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="own recorded digest"):
        GpuPolicy.load(path, expected_digest=policy.digest)


def test_discovery_runs_without_a_policy_and_says_it_is_not_certified():
    nras = MockNras()
    key = bytes(SigningKey.generate().verify_key)
    found = nv.discover_gpu_state(nras.issue(_nonce(key)), key, fetch_jwks=nras.jwks, now=NOW)
    assert found["certified"] is False and "do not use to accept" in found["diagnostic"]
    assert found["observed_fields_digest"] == _policy().digest
    with pytest.raises(ValueError, match="nonce"):  # it still runs every token check
        nv.discover_gpu_state(nras.issue(_nonce(key)), b"\x00" * 32, fetch_jwks=nras.jwks, now=NOW)
    assert not hasattr(found, "verify")


# ---------------------------------------------------------------- service trust model

def test_redirects_and_plain_http_to_a_remote_host_are_refused():
    import http.server
    import threading

    class Redirect(http.server.BaseHTTPRequestHandler):
        def _go(self):
            self.send_response(302)
            self.send_header("Location", "https://example.invalid/jwks.json")
            self.end_headers()

        do_GET = do_POST = _go

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/x"
        with pytest.raises(GpuAttestationError, match="unavailable"):
            nv.fetch_nras_jwks(url=url, timeout=5)
        with pytest.raises(GpuAttestationError, match="refused the attestation"):
            nv.request_nras_token("00" * 32, FAKE_EVIDENCE, url=url, timeout=5)
    finally:
        server.shutdown()
    with pytest.raises(GpuAttestationError, match="non-https"):
        nv.fetch_nras_jwks(url="http://nras.attestation.nvidia.com/.well-known/jwks.json")


# ---------------------------------------------------------------- timing instrumentation

def test_verifier_timings_are_split_and_cold_or_warm_is_labelled():
    nras = MockNras()
    signer = _signer(nras)
    fetches = []

    def fetch():
        fetches.append(1)
        return nras.jwks()

    cold = GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=_policy(), fetch_jwks=fetch, now=NOW)
    assert cold.timings["key_cache_policy"] == "cold" and cold.timings["jwks_fetched"] is True
    assert set(cold.timings) >= {"jwks_get_seconds", "local_token_verify_seconds"} and len(fetches) == 1
    cache = nv.JwksCache(ttl_seconds=300)
    for expect_fetch in (True, False):
        warm = GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=_policy(), fetch_jwks=fetch,
                                                 jwks_cache=cache, now=NOW)
        assert warm.timings["key_cache_policy"] == "warm" and warm.timings["jwks_fetched"] is expect_fetch
    assert len(fetches) == 2  # one cold, one warm-cache fill; the second warm use did not fetch
    assert warm.timings["jwks_get_seconds"] == 0.0
    assert cache.provenance()["url"] == nv.NRAS_JWKS_URL and cache.provenance()["expires_unix"] > cache.provenance()["fetched_at_unix"]
    cache.ttl_seconds = -1  # expired: fetched again
    GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=_policy(), fetch_jwks=fetch,
                                      jwks_cache=cache, now=NOW)
    assert len(fetches) == 3
    # a cache without the needed key id is not used
    other = nv.JwksCache()
    other.put({"keys": []})
    GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=_policy(), fetch_jwks=fetch,
                                      jwks_cache=other, now=NOW)
    assert len(fetches) == 4


def test_local_gpu_check_time_is_recorded_apart():
    nras = MockNras()
    signer = _signer(nras)
    verifier = _verify(nras, signer.token(), signer.verify_key(), local_check=lambda nonce: None)
    assert "local_gpu_check_seconds" in verifier.timings


def test_summary_uses_the_median_of_per_repetition_sums_not_a_sum_of_medians():
    parts = lambda f: {"cc_state_check_seconds": f, "evidence_collect_seconds": f, "nras_post_seconds": f}  # noqa: E731
    vparts = lambda f: {"jwks_get_seconds": f, "local_token_verify_seconds": f}  # noqa: E731
    reps = [
        {"attestation_fetch_seconds": fetch, "attestation_verify_seconds": verify, "fetch_parts": parts(fetch / 3),
         "verify_parts": vparts(verify / 2), "key_cache_policy": "cold"}
        for fetch, verify in ((1.0, 9.0), (2.0, 2.0), (9.0, 1.0))]
    summary = nv.summarize_timings(reps)
    assert summary["attestation_fetch_seconds"]["median"] + summary["attestation_verify_seconds"]["median"] == 4.0
    assert summary["fetch_plus_verify_per_repetition_sum_seconds"]["median"] == 10.0  # sums are 10, 4, 10
    assert summary["fetch_plus_verify_per_repetition_sum_seconds"]["mean"] == 8.0
    assert summary["attestation_fetch_seconds"]["mean"] == 4.0 and summary["key_cache_policy"] == "cold"
    assert set(summary["fetch_parts"]) == {"cc_state_check_seconds", "evidence_collect_seconds", "nras_post_seconds"}
    reps[1]["key_cache_policy"] = "warm"
    with pytest.raises(ValueError, match="mix key-cache policies"):
        nv.summarize_timings(reps)

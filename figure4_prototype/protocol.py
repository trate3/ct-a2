"""Tests first, attested proposal, encrypted components, stable certified identity.

Workload.install_environment is the certifier's private local input to its enclave.
Provider sees only the signed public proposal, then sends encrypted components.
The existing ct_core exchange is neither modified nor invoked.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import secrets
import threading

from nacl.public import PrivateKey

from ct_core import certification as core

from . import stream
from .attestation import LocalSigner
from .bundle import BundlePlan, materialize
from .codec import ChunkReader, canonical, decode, digest, hex_string
from .deadline import DeadlineReader
from .errors import ABORT, ProtocolError
from .storage import require_capacity, require_confidential_scratch

VERSION = "ct-figure4/2"
FORMAT = "sealed-box+secretstream-dedup/2"
MAX_BUNDLE_BYTES = 256 * 1024**3
PROPOSAL_FIELDS = {"type", "version", "sid", "eid", "m_ref", "spec", "pk", "sig"}
CERTIFICATE_FIELDS = {"type", "version", "sid", "eid", "m_ref", "spec", "id", "m", "sig"}


def _spec(spec):
    if (not isinstance(spec, dict) or set(spec) != {"evaluation", "component_bytes", "component_format"}
            or spec["component_format"] != FORMAT or type(spec["component_bytes"]) is not int
            or not 0 < spec["component_bytes"] <= MAX_BUNDLE_BYTES):
        raise ValueError("malformed Figure 4 specification")
    core._validate_spec(spec["evaluation"])


def make_spec(environment, *, component_bytes: int):
    """Certifier-side packaging. The private plaintext is never sent to P."""
    evaluation, padded = core.prepare_environment(environment)
    spec = {"evaluation": evaluation, "component_bytes": component_bytes, "component_format": FORMAT}
    _spec(spec)
    return spec, padded


def _payload(message):
    unsigned = {k: v for k, v in message.items() if k != "sig"}
    return b"certified-taps/figure4/v1\0" + canonical(unsigned)


def _scope(message, verifier):
    if (message["version"] != VERSION or not isinstance(message["sid"], str)
            or not 0 < len(message["sid"].encode()) <= 256 or not hex_string(message["eid"], 16)
            or message["m_ref"] != verifier.measurement()):
        return False
    _spec(message["spec"])
    return verifier.verify(_payload(message), message["sig"])


def verify_proposal(proposal, verifier, *, expected_spec, expected_sid):
    try:
        proposal = decode(canonical(proposal))
        return (set(proposal) == PROPOSAL_FIELDS and proposal["type"] == "propose"
                and proposal["spec"] == expected_spec and proposal["sid"] == expected_sid
                and hex_string(proposal["pk"], 32) and _scope(proposal, verifier))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def verify_certificate(cert, verifier, *, expected_spec, expected_id, expected_sid=None, expected_eid=None):
    """R verifies a specified pair; P additionally requires its proposal instance.

    No signature means confidentiality or hardware trust by itself: verifier must
    already pin an accepted measured workload. Stable id is deliberately public.
    """
    try:
        cert = decode(canonical(cert))
        if (set(cert) != CERTIFICATE_FIELDS or cert["type"] != "certify"
                or cert["spec"] != expected_spec or not hex_string(cert["id"], 32)
                or not hex_string(expected_id, 32) or cert["id"] != expected_id
                or (expected_sid is not None and cert["sid"] != expected_sid)
                or (expected_eid is not None and cert["eid"] != expected_eid)
                or not isinstance(cert["m"], dict) or set(cert["m"]) != {"fn", "fp"}):
            return False
        values = [cert["m"][k] for k in ("fn", "fp")]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or sum(values) > 1 + 1e-12):
            return False
        return _scope(cert, verifier)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


class Provider:
    def __init__(self, plan: BundlePlan, verifier, *, expected_spec, sid: str):
        _spec(expected_spec)
        if plan.minimum_bytes > expected_spec["component_bytes"]:
            raise ValueError("provider components do not fit the public envelope")
        self.plan, self.verifier = plan, verifier
        self.spec = decode(canonical(expected_spec))
        self.sid = sid
        self._proposal = None

    def write_submission(self, proposal, destination):
        if self._proposal is not None:
            raise ValueError("provider has already committed in this session")
        proposal = decode(canonical(proposal))
        if not verify_proposal(proposal, self.verifier, expected_spec=self.spec, expected_sid=self.sid):
            raise ValueError("proposal attestation, specification or session mismatch")
        self._proposal = decode(canonical(proposal))
        context = bytes.fromhex(digest(canonical(self._proposal)))
        stream.encrypt(bytes.fromhex(proposal["pk"]), context,
                       self.plan.plaintext(self.spec["component_bytes"]), self.spec["component_bytes"], destination)

    def accept(self, certificate):
        return (self._proposal is not None and verify_certificate(
            certificate, self.verifier, expected_spec=self.spec, expected_id=self.plan.identity,
            expected_sid=self.sid, expected_eid=self._proposal["eid"]))


class Workload:
    """One enclave instance/session; failures are terminal and release no result.

    A real deployment uses a fresh protected process and confidential scratch
    storage per instance. Local tests explicitly use a local signer and directory.
    The image, driver registry, channel authentication and storage policy are part
    of what P/C must accept when pinning the verifier.
    """

    _confidential_session_claimed = False
    _process_lock = threading.Lock()

    def __init__(self, signer, *, sid: str, agent_drivers, tap_drivers, environment_drivers, scratch_root, storage_mode: str):
        if not isinstance(sid, str) or not 0 < len(sid.encode()) <= 256:
            raise ValueError("invalid session id")
        self.signer, self.sid = signer, sid
        self.agent_drivers, self.tap_drivers = tuple(agent_drivers), tuple(tap_drivers)
        self.environment_drivers = tuple(environment_drivers)
        self.scratch_root = Path(scratch_root)
        if not self.scratch_root.is_dir():
            raise ValueError("explicit private scratch directory required")
        if storage_mode == "local-test":
            if not isinstance(signer, LocalSigner):
                raise ValueError("local scratch is available only with the explicit test signer")
        elif storage_mode == "confidential-tmpfs":
            if isinstance(signer, LocalSigner):
                raise ValueError("local test signing is not confidential attestation")
            require_confidential_scratch(self.scratch_root)
            with Workload._process_lock:
                if Workload._confidential_session_claimed:
                    raise ValueError("confidential sessions require a fresh process")
                Workload._confidential_session_claimed = True
        else:
            raise ValueError("choose local-test or confidential-tmpfs storage explicitly")
        self._state, self._lock = "new", threading.Lock()
        self._secret_key = self._environment = self._proposal = None

    @property
    def state(self):
        return self._state

    def install_environment(self, spec, padded_environment: bytes):
        with self._lock:
            if self._state != "new":
                raise ProtocolError(ABORT)
            self._state = "installing"
            try:
                spec = decode(canonical(spec))
                padded_environment = bytes(padded_environment)
                _spec(spec)
                require_capacity(self.scratch_root, spec['component_bytes'])
                core._load_environment(core._unpad_environment(padded_environment), spec["evaluation"], self.environment_drivers)
                self._environment = bytes(padded_environment)
                self._secret_key = PrivateKey.generate()
                proposal = {"type": "propose", "version": VERSION, "sid": self.sid,
                            "eid": secrets.token_hex(16), "m_ref": self.signer.measurement(),
                            "spec": decode(canonical(spec)), "pk": bytes(self._secret_key.public_key).hex()}
                proposal["sig"] = self.signer.sign(_payload(proposal))
                self._proposal = decode(canonical(proposal))
                self._state = "proposed"
                return proposal
            except Exception:
                self._state = "aborted"
                self._environment = self._secret_key = None
                raise ProtocolError(ABORT) from None

    def certify(self, encrypted_submission, *, stage_recorder=None, deadline: float | None = None):
        with self._lock:
            if self._state != "proposed":
                raise ProtocolError(ABORT)
            self._state = "running"
        recorder = stage_recorder or core.NullStageRecorder()
        try:
            proposal = self._proposal
            cap = proposal["spec"]["component_bytes"]
            context = bytes.fromhex(digest(canonical(proposal)))
            if deadline is not None:
                if type(deadline) not in (int, float) or not math.isfinite(deadline):
                    raise ValueError("invalid monotonic deadline")
                encrypted_submission = DeadlineReader(encrypted_submission, deadline)
            plaintext = ChunkReader(stream.decrypt(self._secret_key, context, cap, encrypted_submission))
            with ExitStack() as stack:
                with recorder.stage("provider_provisioning_total"):
                    prepared, identity = stack.enter_context(materialize(
                        plaintext, cap, scratch_root=self.scratch_root,
                        agent_drivers=self.agent_drivers, tap_drivers=self.tap_drivers, stage_recorder=recorder))
                result = core.measure(prepared, self._environment, proposal["spec"]["evaluation"],
                                      self.environment_drivers, stage_recorder=recorder)
                certificate = {"type": "certify", "version": VERSION, "sid": self.sid,
                               "eid": proposal["eid"], "m_ref": self.signer.measurement(),
                               "spec": decode(canonical(proposal["spec"])), "id": identity,
                               "m": {"fn": result["m"], "fp": result["m_fp"]}}
                with recorder.stage("certificate_sign"):
                    certificate["sig"] = self.signer.sign(_payload(certificate))
            self._state = "complete"
            return certificate
        except Exception:
            self._state = "aborted"
            raise ProtocolError(ABORT) from None
        finally:
            self._environment = self._secret_key = None

    def handle_submission(self, encrypted_submission, **kwargs):
        """Public adapter boundary: return a certificate or one generic abort.

        Adapters must not serialize exceptions/tracebacks from private helpers.
        This abort is unsigned: a host can already deny service.
        """
        try:
            return self.certify(encrypted_submission, **kwargs)
        except ProtocolError:
            return {"type": "abort", "version": VERSION}


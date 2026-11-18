"""Small synthetic CPU components; these are protocol tests, not model results."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile

from ct_core.agent import Agent
from ct_core.interfaces import AgentDriver, ArtifactPackage, Generation, Monitor, Prediction, ProgramRunner, TapDriver
from util.dynamic_test_environment import StaticEnvironment

from .attestation import local_test_pair
from .bundle import BundlePlan
from .codec import canonical
from .protocol import Provider, Workload, make_spec, verify_certificate


class ArithmeticProgram(ProgramRunner):
    def __init__(self, config):
        self.config = config

    def generate(self, x, capture_layers=None):
        value = json.loads(x)["value"]
        if value == self.config.get("fail_at"):
            raise ValueError("intentional test execution failure")
        return Generation(thought=f"private_negative={int(value < 0)}", answer=str(value * self.config["multiplier"]))

    def digest(self):
        return "synthetic-arithmetic-program"


class ArithmeticAgentDriver(AgentDriver):
    DRIVER = "figure4-test-arithmetic/v1"
    loads = 0

    @classmethod
    def from_package(cls, config, artifacts):
        if config != {} or not {"model/settings.json"} <= artifacts.keys() <= {"model/settings.json", "model/payload.bin"}:
            raise ValueError("malformed arithmetic package")
        cls.loads += 1
        obj = cls()
        obj.settings = json.loads(Path(artifacts["model/settings.json"]).read_text())
        if type(obj.settings.get("multiplier")) is not int:
            raise ValueError("missing integer multiplier")
        return obj

    def new_agent(self, randomness):
        return Agent.model_only(ArithmeticProgram(self.settings))


class NegativeTap(Monitor):
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, inp):
        score = float("private_negative=1" in inp.thought)
        return Prediction(int(score > self.threshold), score)

    def digest(self):
        return "synthetic-private-signal-tap"


class NegativeTapDriver(TapDriver):
    DRIVER = "figure4-test-negative-tap/v1"

    @classmethod
    def from_package(cls, config, artifacts):
        if config != {} or set(artifacts) != {"threshold.bin"} or not isinstance(artifacts["threshold.bin"], bytes):
            raise ValueError("malformed test tap")
        threshold, = struct.unpack("!d", artifacts["threshold.bin"])
        obj = cls()
        obj.threshold = threshold
        return obj

    def new_tap(self, randomness):
        return NegativeTap(self.threshold)


def packages(root: Path, *, multiplier=3, threshold=.5, fail_at=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "settings.json"
    path.write_bytes(canonical({"multiplier": multiplier, "fail_at": fail_at,
                                "private_marker": "PROVIDER_PRIVATE_SETTINGS_7b181846"}))
    return (ArtifactPackage(ArithmeticAgentDriver.DRIVER, {}, (("model/settings.json", str(path)),)),
            ArtifactPackage(NegativeTapDriver.DRIVER, {}, (("threshold.bin", struct.pack("!d", threshold)),)))


def environment():
    items = [{"x": json.dumps({"value": v, "case_secret": "CERTIFIER_PRIVATE_CASE_b3e484dc"}), "z": int(v < 0)}
             for v in (-7, 4, -2, 8)]
    return StaticEnvironment(items, n=len(items), seed=7)


def run_demo(output_dir: Path):
    """Local role simulation. The relay file contains ciphertext only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure4-demo-") as temp:
        private = Path(temp)
        ap, tp = packages(private / "provider")
        plan = BundlePlan(ap, tp, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
        signer, verifier = local_test_pair()
        spec, padded = make_spec(environment(), component_bytes=64 * 1024)
        workload = Workload(signer, sid="figure4-local-demo", agent_drivers=(ArithmeticAgentDriver,),
                            tap_drivers=(NegativeTapDriver,), environment_drivers=(StaticEnvironment,), scratch_root=private,
                            storage_mode="local-test")
        proposal = workload.install_environment(spec, padded)
        provider = Provider(plan, verifier, expected_spec=spec, sid="figure4-local-demo")
        with (output_dir / "submission.enc").open("wb") as ciphertext:
            provider.write_submission(proposal, ciphertext)
        with (output_dir / "submission.enc").open("rb") as ciphertext:
            certificate = workload.certify(ciphertext)
        accepted = provider.accept(certificate)
        relying_party = verify_certificate(certificate, verifier, expected_spec=spec, expected_id=plan.identity)
        assert accepted and relying_party
        public_anchor = {"verify_key": bytes(signer._key.verify_key).hex(), "expected_measurement": verifier.measurement(),
                         "mode": "local-test-only; no TEE"}
        for name, value in (("proposal.json", proposal), ("certificate.json", certificate),
                            ("spec.json", spec),
                            ("local-public-anchor.json", public_anchor)):
            (output_dir / name).write_bytes(canonical(value) + b"\n")
        result = {"mode": "local synthetic CPU test; no TEE and no model inference", "provider_accepted": accepted,
                  "relying_party_accepted": relying_party, "stable_pair_id": certificate["id"],
                  "measurement": certificate["m"], "ciphertext_bytes": (output_dir / "submission.enc").stat().st_size}
        (output_dir / "demo-result.json").write_bytes(canonical(result) + b"\n")
        return result


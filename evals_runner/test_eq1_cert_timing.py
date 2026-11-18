"""Tests of eq1_cert_timing.py on small files, with the stub attestation and a fake GPU signer and verifier (no GPU, no network, no model)."""
import json
from pathlib import Path

import pytest
from nacl.signing import SigningKey

import tee.extension.nvidia_attestor as nv
from ct_core.protocol.attestor import StubAttestationVerifier, StubEnclaveSigner
from evals_runner import eq1_cert_timing as C

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "tee/extension/gpu_policy.json"
POLICY_DIGEST = json.loads(POLICY_FILE.read_text())["fields_digest"]


@pytest.fixture
def fake_models(tmp_path):
    q, e = tmp_path / "qwen", tmp_path / "e4b"
    q.mkdir(); e.mkdir()
    for i in range(3):
        (q / f"m{i}.safetensors").write_bytes(bytes([i]) * (300_000 + i))
    (q / "config.json").write_text("{}")
    (e / "m.safetensors").write_bytes(b"e" * 200_000)
    probe = tmp_path / "probe.json"
    probe.write_text("{}")
    return ["--qwen-dir", str(q), "--e4b-dir", str(e), "--probe-file", str(probe), "--repo-root", str(ROOT), "--skip-main-check"]


def test_stub_run_writes_records_and_summary(tmp_path, fake_models):
    out = tmp_path / "out"
    assert C.main(["run", "--out", str(out), "--attestation", "stub", "--repeats", "3", "--evict-first"] + fake_models) == 0
    reps = [json.loads(l) for l in (out / "reps.jsonl").read_text().splitlines()]
    assert len(reps) == 9 and {r["config"] for r in reps} == {"probe", "e4b", "qwen"}
    states = [r["cache_state"] for r in reps]
    assert states.count("evicted") == 3 and states.count("warm") == 6
    for r in reps:
        assert r["certificate_verified"] is True
        assert r["protocol_overhead_s"] == pytest.approx(r["run_wall_s"] - r["common_stand_in_s"])
        assert r["commit_artifact_hash_s"] > 0 and r["hash_pass_pre_s"] > 0 and r["hash_pass_post_s"] > 0
        assert r["named_protocol_s"] < r["run_wall_s"] + 1e-9          # the nested hash passes are not added to commit_artifact_hash
    summary = json.loads((out / "cert_timings.json").read_text())
    assert summary["attestation_mode"] == "stub" and summary["policy_enforced"] is False
    assert set(summary["configs"]) == {"probe", "e4b", "qwen"}
    assert summary["configs"]["qwen"]["cache_states"] == {"warm": 2, "evicted": 1, "as_found": 0}
    # the Qwen monitor shares the agent's files: hashed once per pass
    size = sum(p.stat().st_size for p in Path(fake_models[1]).iterdir())
    assert summary["configs"]["qwen"]["artifact_gib_hashed"] * 2 ** 30 == pytest.approx(2 * size)


def test_no_eviction_marks_the_first_run_as_found(tmp_path, fake_models):
    out = tmp_path / "out"
    C.main(["run", "--out", str(out), "--attestation", "stub", "--repeats", "2", "--configs", "probe"] + fake_models)
    reps = [json.loads(l) for l in (out / "reps.jsonl").read_text().splitlines()]
    assert [r["cache_state"] for r in reps] == ["as_found", "warm"]
    s = json.loads((out / "cert_timings.json").read_text())["configs"]["probe"]
    assert s["protocol_overhead_s"]["cold"] is None and s["protocol_overhead_s"]["warm"] is not None


def test_refuses_to_start_while_the_main_stage_runs(tmp_path, fake_models, monkeypatch):
    monkeypatch.setattr(C, "main_stage_running", lambda: True)
    args = [a for a in fake_models if a != "--skip-main-check"]
    with pytest.raises(SystemExit, match="still running"):
        C.main(["run", "--out", str(tmp_path / "o"), "--attestation", "stub"] + args)


def _fake_gpu(monkeypatch, seen):
    class FakeSigner(StubEnclaveSigner):
        def __init__(self):
            super().__init__(SigningKey.generate())
            seen["constructed"] = seen.get("constructed", 0) + 1
            self.timings = {"cc_state_check_seconds": 0.01, "evidence_collect_seconds": 0.02, "nras_post_seconds": 0.03}

        def measurement(self):
            return "nvidia-gpu:sha256:fake"

        def verify_key(self):
            return bytes(self._signing_key.verify_key)

        def token(self):
            return "fake-token"

    def fake_from_token(cls, token, verify_key, *, policy, **kw):
        seen["policy_digest"] = policy.digest
        v = StubAttestationVerifier(verify_key, "nvidia-gpu:sha256:fake")
        v.timings = {"jwks_get_seconds": 0.04, "local_token_verify_seconds": 0.05, "key_cache_policy": "cold", "jwks_fetched": True}
        return v

    def fake_record(verifier, signer=None):
        return {"root": nv.ATTESTATION_ROOT, "gpu_attestation_verified": True, "workload_attested": False, "cpu_attested": False,
                "full_certification": False, "measurement": verifier.measurement(),
                "signer_timings": dict(signer.timings), "verifier_timings": dict(verifier.timings)}

    monkeypatch.setattr(nv, "GpuAttestedSoftwareSigner", FakeSigner)
    monkeypatch.setattr(nv.GpuAttestationVerifier, "from_token", classmethod(fake_from_token))
    monkeypatch.setattr(nv, "attestation_record", fake_record)
    monkeypatch.setattr(C, "gpu_busy", lambda: "")


def test_nvidia_mode_with_fakes_requires_and_uses_the_policy(tmp_path, fake_models, monkeypatch):
    seen = {}
    _fake_gpu(monkeypatch, seen)
    monkeypatch.setenv("EQ1_GPU_POLICY_FILE", str(POLICY_FILE))
    monkeypatch.setenv("EQ1_GPU_POLICY_DIGEST", POLICY_DIGEST)
    out = tmp_path / "out"
    C.main(["run", "--out", str(out), "--attestation", "nvidia", "--repeats", "2", "--configs", "probe"] + fake_models)
    assert seen["policy_digest"] == POLICY_DIGEST
    s = json.loads((out / "cert_timings.json").read_text())
    assert s["attestation_mode"] == "nvidia-gpu-only-software-signed" and s["policy_enforced"] is True and s["policy_digest"] == POLICY_DIGEST
    assert s["gpu_attestation_verified"] is True and s["workload_attested"] is False
    assert s["key_cache_policy"] == "cold" and "SUM" in s["attestation_definition"]
    assert set(s["attestation_parts_s"]) >= {"cc_state_check_seconds", "evidence_collect_seconds", "nras_post_seconds", "jwks_get_seconds", "local_token_verify_seconds"}
    reps = [json.loads(l) for l in (out / "reps.jsonl").read_text().splitlines()]
    assert reps[0]["attestation_fetch_s"] is not None and reps[0]["attestation_verify_s"] is not None
    assert reps[0]["certificate_verified"] is None
    assert s["attestation_pooled_mean_s"] == pytest.approx(sum(r["attestation_s"] for r in reps) / 2)


def test_nvidia_mode_without_policy_variables_fails_before_any_attestation(tmp_path, fake_models, monkeypatch):
    seen = {}
    _fake_gpu(monkeypatch, seen)
    monkeypatch.delenv("EQ1_GPU_POLICY_FILE", raising=False)
    monkeypatch.delenv("EQ1_GPU_POLICY_DIGEST", raising=False)
    with pytest.raises(SystemExit, match="EQ1_GPU_POLICY_FILE"):
        C.main(["run", "--out", str(tmp_path / "o"), "--attestation", "nvidia", "--repeats", "1", "--configs", "probe"] + fake_models)
    assert "constructed" not in seen
    monkeypatch.setenv("EQ1_GPU_POLICY_FILE", str(POLICY_FILE))
    monkeypatch.setenv("EQ1_GPU_POLICY_DIGEST", "sha256:wrong")
    with pytest.raises(ValueError, match="not the expected"):
        C.main(["run", "--out", str(tmp_path / "o2"), "--attestation", "nvidia", "--repeats", "1", "--configs", "probe"] + fake_models)
    assert "constructed" not in seen

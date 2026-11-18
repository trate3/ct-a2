"""GPU attestation run (policy required, split timings).

GPU attestation of a software-held signing key.

  python run_gpu_attestation.py collect OUT_DIR --policy-file gpu_policy.json --policy-digest sha256:... \
         [--repetitions 10] [--key-cache cold|warm] [--local-check]
  python run_gpu_attestation.py discover OUT_DIR       (no policy; output is marked not certified; used to freeze a policy)
  python run_gpu_attestation.py reverify OUT_DIR --policy-file ... --policy-digest ...   (offline, saved tokens and keys)

`collect` records, per repetition: the outer attestation_fetch and attestation_verify wall times (network included) and the
parts: local CC-state check, GPU evidence collection, NRAS POST, JWKS GET, local token verification (and, if asked, NVIDIA's
local GPU check, timed apart). The summary reports medians and means of each part and of the per-repetition sum of fetch and
verify (not a sum of medians), and the key-cache policy. Needs the confidential GPU and network access to NVIDIA's attestation service.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from nacl.signing import SigningKey  # noqa: E402

import tee.extension.nvidia_attestor as nv  # noqa: E402


def sh(command):
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return done.stdout + (done.stderr if done.returncode else "")


def collect(out: pathlib.Path, policy: nv.GpuPolicy, repetitions: int, key_cache: str, local_check: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cache = nv.JwksCache() if key_cache == "warm" else None
    reps, used_keys = [], {}
    for index in range(repetitions):
        key = SigningKey.generate()
        t0 = time.perf_counter()
        signer = nv.GpuAttestedSoftwareSigner(signing_key=key)
        t1 = time.perf_counter()
        seen = {}

        def fetch():
            seen["jwks"] = nv.fetch_nras_jwks()
            return seen["jwks"]

        verifier = nv.GpuAttestationVerifier.from_token(
            signer.token(), signer.verify_key(), policy=policy, fetch_jwks=fetch, jwks_cache=cache,
            local_check=nv.local_gpu_check if local_check else None)
        t2 = time.perf_counter()
        assert verifier.verify(b"smoke", signer.sign(b"smoke"))
        timings = dict(verifier.timings)
        local_seconds = timings.pop("local_gpu_check_seconds", None)
        reps.append({
            "repetition": index,
            # outer wall times; with --local-check the verify wall time includes NVIDIA's local check, which is also given apart
            "attestation_fetch_seconds": t1 - t0, "attestation_verify_seconds": t2 - t1,
            "fetch_parts": signer.timings, "verify_parts": {k: timings[k] for k in nv._VERIFY_PARTS},
            "local_gpu_check_seconds": local_seconds,
            "key_cache_policy": timings["key_cache_policy"], "jwks_fetched": timings["jwks_fetched"],
            "nonce_hex": signer.nonce_hex(), "verify_key_hex": signer.verify_key().hex(),
            "measurement": verifier.measurement(), "token_issued_at": verifier.verified_claims()["overall"]["iat"],
            "token": signer.token(),
        })
        if "jwks" in seen:
            for entry in seen["jwks"]["keys"]:
                used_keys[entry["kid"]] = entry
        if index == 0:
            (out / "evidence_rep0.json").write_text(json.dumps(signer.evidence(), indent=1))
            (out / "record_rep0.json").write_text(json.dumps(nv.attestation_record(verifier, signer), indent=1, sort_keys=True))
        print(f"rep {index}: fetch {t1 - t0:.4f}s verify {t2 - t1:.4f}s", flush=True)
    kids = set()  # keep only the keys that signed these tokens
    for rep in reps:
        for value in (json.loads(rep["token"])[0][1], *json.loads(rep["token"])[1].values()):
            head = value.split(".")[0]
            kids.add(json.loads(base64.urlsafe_b64decode(head + "=" * (-len(head) % 4)))["kid"])
    summary = nv.summarize_timings(reps)
    summary.update({"policy_digest": policy.digest, "distinct_nonces": len({r["nonce_hex"] for r in reps}),
                    "local_gpu_check_seconds": [r["local_gpu_check_seconds"] for r in reps],
                    "scope": "GPU attestation of a software-held key; CPU, code, model files, key isolation not attested",
                    "machine": {"uname_r": sh(["uname", "-r"]).strip(),
                                "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,vbios_version", "--format=csv"]).strip(),
                                "cc_state": nv.read_cc_state()}})
    (out / "nvidia_smi_conf_compute_q.txt").write_text(sh(["nvidia-smi", "conf-compute", "-q"]))
    (out / "pip_freeze.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]))
    (out / "repetitions.json").write_text(json.dumps(reps, indent=1))
    (out / "nras_signing_keys_used.json").write_text(json.dumps({"keys": [used_keys[k] for k in kids if k in used_keys]}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("key_cache_policy", "attestation_fetch_seconds", "attestation_verify_seconds",
                                             "fetch_plus_verify_per_repetition_sum_seconds")}, indent=1))


def discover(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    key = SigningKey.generate()
    signer = nv.GpuAttestedSoftwareSigner(signing_key=key)
    found = nv.discover_gpu_state(signer.token(), signer.verify_key())
    (out / "discovery_NOT_CERTIFIED.json").write_text(json.dumps(found, indent=1, sort_keys=True))
    print(json.dumps(found, indent=1, sort_keys=True))


def reverify(out: pathlib.Path, policy: nv.GpuPolicy) -> None:
    reps = json.loads((out / "repetitions.json").read_text())
    jwks = json.loads((out / "nras_signing_keys_used.json").read_text())
    for rep in reps:
        key = bytes.fromhex(rep["verify_key_hex"])
        verifier = nv.GpuAttestationVerifier.from_token(rep["token"], key, policy=policy, fetch_jwks=lambda: jwks,
                                                        now=rep["token_issued_at"] + 60)
        assert hashlib.sha256(key).hexdigest() == rep["nonce_hex"] and verifier.measurement() == rep["measurement"]
        print(f"rep {rep['repetition']}: verifies offline under policy {policy.digest[:23]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "discover", "reverify"))
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--policy-file", type=pathlib.Path)
    parser.add_argument("--policy-digest")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--key-cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--local-check", action="store_true")
    a = parser.parse_args()
    if a.mode == "discover":
        discover(a.out_dir)
    else:
        if not a.policy_file or not a.policy_digest:
            parser.error("--policy-file and --policy-digest are required (the policy is chosen before the run)")
        policy = nv.GpuPolicy.load(a.policy_file, expected_digest=a.policy_digest)
        collect(a.out_dir, policy, a.repetitions, a.key_cache, a.local_check) if a.mode == "collect" else reverify(a.out_dir, policy)

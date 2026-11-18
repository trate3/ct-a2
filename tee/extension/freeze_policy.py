"""Freeze the accepted-GPU policy from the DEVELOPMENT smoke (2026-11-17) into gpu_policy.json.

Reads claims_rep0.json (the verified claims of repetition 0), checks that all 10 saved tokens show the same state,
and writes the policy and its digest. No GPU and no network. The final job must be given this digest independently
(EQ1_GPU_POLICY_DIGEST) and must not adopt whatever state it returns.
    python freeze_policy.py            (run from anywhere; paths are relative to this file)

In this artifact the input files (claims_rep0.json, summary.json of the development smoke) are not shipped,
because the GPU claims carry a hardware identifier; the frozen result is shipped as gpu_policy.json.
"""
import hashlib
import json
import pathlib
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))  # artifact root (tee/extension -> root)

from tee.extension.nvidia_attestor import GpuPolicy, measurement_of  # noqa: E402

claims_path = here / "claims_rep0.json"
gpus = json.loads(claims_path.read_text())["gpus"]
summary = json.loads((here / "summary.json").read_text())
assert measurement_of(gpus) == summary["measurement"] and summary["all_measurements_equal"], "development smoke states differ"
source = (f"development smoke 2026-11-17: results/tee_attestation_smoke/claims_rep0.json "
          f"(sha256 {hashlib.sha256(claims_path.read_bytes()).hexdigest()}), 10 of 10 repetitions showed this state")
policy = GpuPolicy.from_verified_claims(gpus, source=source)
policy.write(here / "gpu_policy.json")
print(policy.digest)

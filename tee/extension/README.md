# GPU-only attestation (extension)

**GPU-only attestation with a software-held signing key; not a workload attestation; not part of the measured results until a TEE run says so.**

| File | What it is |
|---|---|
| `nvidia_attestor.py` | `GpuAttestedSoftwareSigner` (ephemeral Ed25519 key in an ordinary process; asks the GPU for an attestation whose nonce is `sha256(verify_key)`), `GpuAttestationVerifier` (checks the attestation service's token against a required GPU-state policy, then pins the key), `GpuAttestationError`. Run records say `gpu_attestation_verified=true, workload_attested=false, cpu_attested=false, full_certification=false`. |
| `test_nvidia_attestor.py` | CPU tests with a mock attestation service. The one test that replays the archived live tokens is skipped: those tokens carry a hardware identifier of the GPU and cannot be redacted without breaking their signatures. |
| `gpu_policy.json` | The accepted GPU state (GPU count, hardware model, driver and firmware versions, secure boot, debug state, required security claims) and its digest. A verifier needs this digest, supplied independently of the file. |
| `freeze_policy.py` | How the policy was frozen from the development smoke test (its inputs are not shipped). |
| `run_gpu_attestation_v2.py` | Collects and verifies attestations on the TEE machine with split timings (local CC-state check, evidence collection, service request, key download, local verification). Needs the confidential GPU and network access. Not yet run in this revised form. |

See `../README.md` and `../NVIDIA_SIGNER_DESIGN.md` for the trust model and what is and is not attested.

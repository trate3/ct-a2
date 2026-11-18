# Design note: GPU-only attestation with a software-held signing key

**GPU-only attestation with a software-held signing key; not a workload attestation; not part of the measured results until a TEE run says so.** No number in the paper tables depends on it. Code: `tee/extension/`.

## Why

Our TEE machine is a Google Cloud confidential VM (AMD SEV) with an RTX PRO 6000 in confidential-computing mode. It has no Confidential Space launcher, so the Confidential Space signer (`ct_core/protocol/attestor.py`, unchanged) cannot run there, and the stub signer proves nothing about hardware.

## What the code does

1. `GpuAttestedSoftwareSigner` generates an Ed25519 key in memory, in an ordinary process. It is a software-held key.
2. It computes `nonce = sha256(verify_key)`, checks locally that the GPU's confidential-computing mode is on, collects the GPU's attestation evidence with NVIDIA's verifier package, and sends it to NVIDIA's remote attestation service, which returns a signed token.
3. `GpuAttestationVerifier` downloads the service's published signing keys, checks the token signatures, the issuer, and the time claims (`nbf`, `iat`, `exp`) of every token (the overall token and each GPU's token), checks that the nonce equals `sha256(verify_key)`, and checks the GPU state against a **required** policy: an exact digest of the accepted GPU count, hardware model, driver and firmware versions, secure boot, debug state and required security claims (`gpu_policy.json`). The digest must be supplied independently of the policy file. Only then does it pin the key; afterwards signatures are checked as in the stub and Confidential Space paths.
4. Every failure raises `GpuAttestationError` (or a `ValueError` for a policy mismatch). There is no fallback to the stub. Run records state `gpu_attestation_verified=true`, `workload_attested=false`, `cpu_attested=false`, `full_certification=false`.
5. Timings are split: local CC-state check, evidence collection, service request, key download, local verification, plus the outer fetch and verify wall times.

## Trust model

- The trust anchor is HTTPS to the fixed official attestation service and the signing keys it publishes. Redirects are refused, and keys are never fetched from a URL named by a token.
- No NVIDIA root certificate is pinned. The certificate chain in a key entry is checked only for internal consistency (each certificate signed by the next, the leaf valid at the check time, the expected names); path constraints are not enforced. This is not root-anchored certificate-path validation.
- The NVIDIA Python packages it uses are past their end of support (15 September 2026). This is a research diagnostic, not maintained production code. NVIDIA's supported command-line tool would be the migration path.

## What it does not show

- The signing key is held in software. A GPU token bound to the key's hash shows that a genuine confidential GPU produced evidence for that key. It does not show that the certification program owns the key or that the key is isolated: a host could make its own key, obtain genuine GPU evidence for it, and sign any payload.
- The CPU side (memory encryption, boot chain, host access), the Python code, the model files and the workload image are not attested.
- It is therefore not a full certification result and does not fill the Attestation or Total certification columns of the EQ1 table.
- A Confidential Space launch on a machine that supports the GPU, or a vTPM-based workload identity with an accepted boot and image policy, would cover the workload side. Neither has been done on the GPU machine.

## Status

- An earlier version was run as a smoke test on the TEE machine (10 repetitions; median fetch about 0.15 s and verify about 0.21 s, including network time). That run predates the required policy and the time-claim checks; its archived tokens are not shipped because they carry a hardware identifier of the GPU.
- The revised code in `tee/extension/` has not yet been run on the TEE machine.

## Tests

`tee/extension/test_nvidia_attestor.py` runs on CPU against a mock attestation service: token signatures, time claims, nonce binding, the required policy, refusal of redirects and plain HTTP, and fail-closed behaviour. One test replays the archived live tokens and is skipped here (tokens not shipped). Three tests that wire the signer into the cost-experiment runner are deselected in `check.sh` because that runner is not shipped.

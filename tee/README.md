# Certification protocol and attestation

This folder explains how the prototype certifies a tap, which attestation paths exist, and what has and has not run on real hardware. Run every command from the repository root.

## What the protocol does

Three parties take part. The provider owns the agent and the tap. The certifier owns a private test environment. The relying party reads the certificate. The trusted workload runs inside a trusted execution environment (TEE).

1. Setup. Before any session the certifier and the relying party pin two public values: the workload's verification key and its expected measurement, which is the hash of the trusted code.
2. Commit. The workload loads the agent and tap, hashes their code, configuration and files, and sends a signed commit with that hash, a fresh nonce, the digest of the public specification and a fresh encryption key.
3. Evaluation. The certifier checks the signature and measurement and sends the test environment, encrypted to the workload's key. The workload runs every trial with the agent and tap and computes the tap's miss rate and false-alarm rate.
4. Certificate. The workload signs the result together with the hash of the exact ciphertext it received. It issues nothing if a step fails.
5. Verification. The certifier checks that the certificate repeats its commit and matches its ciphertext. The relying party checks the signature, the measurement and the specification digest.

The code is in `ct_core/certification.py` (`serve_session`, `run_certifier`, `verify_certificate`), `ct_core/protocol/attestor.py` (attestation), and `ct_core/protocol/sealing.py` and `ct_core/protocol/digests.py` (encryption and digests). `evals_runner/protocol_integrity_exp.py` runs the workload, a relay and the certifier as separate processes.

This code uses an earlier message order than the paper's Figure 4: the agent and tap are already in the workload, and the certifier sends the environment. `figure4_prototype/` is a separate CPU-only illustration of the Figure 4 order, where the provider encrypts the agent and tap to the attested workload.

## What has run

- The protocol runs end to end on a CPU with a local signing key, which gives no hardware protection. The tests use it.
- The Google Confidential Space signer and verifier (`GcpEnclaveSigner`, `GcpAttestationVerifier`) ran once in August 2026, on a CPU-only Confidential VM with a mock workload. No token from that run is saved here, and nothing was timed.
- Our confidential VM with a confidential GPU has no Confidential Space launcher, so that signer cannot run on it. `figure4_prototype/` has not run on any TEE or GPU.
- The GPU attestation extension in `tee/extension/` was smoke-tested on that machine in an earlier version. The revised code here has not run there yet.
-- The tests check behaviour, not the paper's security proof.
- For these reasons the attestation and total certification columns of the EQ1 table are empty. Everything else in the paper's tables was measured without a TEE. Provider encryption and transfer time in the Figure 4 order is also unmeasured.

## GPU attestation extension

`tee/extension/` binds a signing key, generated in an ordinary process, to a genuine NVIDIA GPU attestation. `GpuAttestedSoftwareSigner` asks the GPU for evidence with the key's hash as nonce. `GpuAttestationVerifier` checks the token from NVIDIA's attestation service, including its time claims, against a required GPU-state policy (`gpu_policy.json`, whose digest must be supplied separately), and only then pins the key. `NVIDIA_SIGNER_DESIGN.md` has the design, and `run_gpu_attestation_v2.py` collects and times attestations on the confidential GPU machine. Its tests, `tee/extension/test_nvidia_attestor.py`, run on a CPU against a mock service.

Limits:

- It attests the GPU only. It does not prove that the certification program owns the key or keeps it isolated, and neither the CPU side nor the workload image is attested. Run records say `workload_attested=false, cpu_attested=false, full_certification=false`.
- Trust rests on HTTPS to NVIDIA's attestation service and its published signing keys. No root certificate is pinned.
- The NVIDIA Python packages it uses are past end of support, so treat it as a research diagnostic.
- A Confidential Space launch or a vTPM-based workload identity would cover the workload side. Neither has been done on the GPU machine.

## Run the demo (CPU)

```bash
bash check.sh                                  # all CPU tests, including the protocol tests
python -m evals_runner protocol-integrity-exp attestation-setup
python -m evals_runner protocol-integrity-exp e2e --n 8                        # honest run: CERTIFIED, VALID
python -m evals_runner protocol-integrity-exp e2e --n 8 --tamper-environment  # relay swaps the environment, certifier aborts
python -m figure4_prototype demo --out /tmp/figure4-demo                       # Figure 4 order, local key
```

`attestation-setup` writes a local key to `.keys/` in the repository root. Delete it when you are done. In a run meant as evidence, choose the attestation mode explicitly and do not let a missing launcher fall back to the local key.

## Run on Confidential Space

`deploy/Dockerfile` builds the mock workload image. `CONFIDENTIAL_SPACE.md` lists the tested commands: build for linux/amd64, launch on a `confidential-space-debug` image on an `n2d` VM, pin the printed key and token with `gcp-pin`, then run the certifier with `--attestation gcp`. You need your own Google Cloud project with billing, and `pip install pyjwt[crypto]` if it is not installed. The verifier checks the token against Google's published keys and reads the container image digest as the measurement. The debug image is for testing. A production run needs the production image and `accepted_gcp_verifier` in `figure4_prototype/attestation.py`, which rejects debug workloads. A valid result says nothing about a GPU.

## Serving models on a confidential GPU

If you serve the monitor models with vLLM 0.31 on a GPU in confidential-compute mode, set `VLLM_USE_V2_MODEL_RUNNER=0`. With the default runner the server starts but returns garbage text (vLLM issue 57224).

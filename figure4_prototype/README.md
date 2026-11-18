# Figure 4 protocol prototype

**A CPU-only illustration of the paper's Figure 4 message order. It has not been timed and has not been run on TEE or GPU hardware.**

This separate implementation follows the **tests-first** exchange in Figure 4 of
`Certified_Taps-2.pdf`. It does not change the existing `ct_core` protocol or the
running EQ1 experiments. It reuses their component interfaces and measurement
engine, with a new provisioning exchange and certificate format.

The local demo uses synthetic arithmetic, small artifacts and a test signing key.
It performs **no model inference, GPU work or real hardware attestation**.

## Run the small CPU checks

From the repository root:

```bash
python3 -m venv figure4_prototype/.venv
figure4_prototype/.venv/bin/pip install -r figure4_prototype/requirements-test.txt
figure4_prototype/.venv/bin/python -B -m pytest -q figure4_prototype/tests
figure4_prototype/.venv/bin/python -B -m figure4_prototype demo --out /tmp/figure4-demo
```

The demo writes an encrypted provider submission, a signed proposal, a signed
certificate, the public specification, a public **local-test** verification anchor
and a small result summary. Its private provider files and workload scratch files
are removed. No signing secret is published or stored in the repository.

An independently expected specification and pair identity can be checked with:

```bash
figure4_prototype/.venv/bin/python -B -m figure4_prototype verify-local \
  --certificate /tmp/figure4-demo/certificate.json \
  --anchor /tmp/figure4-demo/local-public-anchor.json \
  --spec /tmp/figure4-demo/spec.json \
  --expected-id YOUR_EXPECTED_PAIR_DIGEST
```

The anchor must be obtained from trusted setup; accepting an anchor supplied by
an untrusted certificate sender would provide no authenticity. `verify-local`
deliberately reports a local pinned-key check, never hardware attestation.

## What changed

1. The certifier packages its private environment with `make_spec`. The workload
   receives that plaintext through its **private certifier-side installation
   interface**, verifies it against the registered environment driver, and stores
   it before accepting any provider components.
2. The workload generates a fresh encryption key and enclave-instance identifier.
   Its signed public proposal contains the specification, public key, session,
   instance and accepted workload measurement. The environment plaintext is not
   part of the provider-facing message.
3. The provider verifies the proposal against an independently pinned attestation
   verifier and expected specification/session. It encrypts the agent and tap,
   including all declared file/byte artifacts and inert configuration. There is
   no preprovisioned provider model path inside the workload.
4. The workload decrypts into fresh private paths, verifies the complete stream,
   manifest, artifact hashes and registered code **before any component loader
   runs**, then prepares the actual loaded components and checks the resulting
   canonical record against the provider's record.
5. The workload runs all trials with the existing measurement engine. It signs
   only the public specification, a **stable pair digest**, the two joint losses,
   and session/attestation framing. The provider checks its expected digest and
   proposal instance. R must provide the accepted public key, specification and
   expected pair identity to verification; accepting a valid certificate for any
   pair is a different decision.

There is no certificate on a failed or partial evaluation. A workload instance
accepts one environment and one provider submission. The provider cannot change
components after receiving private test interactions: no such interactions leave
the workload.

## Stable identity and exact representation

`BundlePlan.identity` is SHA-256 of canonical UTF-8 JSON containing:

- a versioned representation;
- the existing canonical agent/tap records (registered driver-code digests,
  behavior-changing configuration, logical artifact names, lengths and content
  digests);
- whether each artifact is a file or an inline byte artifact, since this affects
  the registered loader interface.

The existing core artifact/code digests include a length prefix. This implementation
preserves that convention when checking transferred bytes. Local source paths,
encryption randomness, padding and session identifiers are absent from the pair
identity. Identical represented components therefore have the same identity across
fresh sessions and different ciphertexts. Changing the agent, wrapper configuration,
tap, threshold, declared code or artifact contents changes the represented object.

This is the concrete serialization used for the paper's abstract `H(program,tap)`.
A stable public digest permits linking repeated certifications and guessing known
candidate objects. It is **not a hiding commitment**. Matching a digest in a future
serving attestation still requires a trusted serving implementation that loads and
attests the same represented pair; the certificate alone does not attest future
execution or freeze external services.

## Artifact streaming and resource use

The v2 provider submission is a sealed-box-encrypted random stream key/context/length
followed by a libsodium secretstream of the canonical bundle and zero padding.
Payload chunks are **64 KiB**. The signed specification fixes the padded plaintext
length, so ciphertext length does not expose individual artifact boundaries.
The crypto reader enforces chunk sizes, proposal-associated data, finalization,
the exact agreed length and EOF. It rejects tampering, reordering, duplication,
truncation, extra bytes and submission replay into a fresh proposal.

Identical content is sent and stored once, keyed by digest and length. Logical
artifact names and their file/byte modes remain in the stable identity. Identical
file trees also share their materialized paths, so the existing path-keyed model
cache can reuse a shared checkpoint when loader configuration matches. The
canonical pair identity is unchanged by this transport deduplication; the protocol
and transport versions distinguish the new wire layout.

File artifacts are hashed/copied in bounded chunks. They are never base64-encoded
into an in-memory model archive. Metadata and total inline data are each capped at
8 MiB; larger arrays belong in file artifacts. The largest test artifact is 2 MiB.
The public envelope cap is configurable, up to 256 GiB. A model-sized cap is **not**
used by the demo/tests. Padding has real transfer/storage cost, so a real run must
report the chosen cap and all provisioning work.

The stream uses established library APIs:
[sealed boxes](https://libsodium.gitbook.io/doc/public-key_cryptography/sealed_boxes)
for key encapsulation and
[secretstream](https://libsodium.gitbook.io/doc/secret-key_cryptography/secretstream)
for authenticated file transfer. The exact hybrid framing is documented in
`stream.py`; passing tests is not a cryptographic reduction proof for this framing.

The supplied streams are finite, framed single-submission streams. `certify` accepts
an optional **absolute monotonic deadline**. It times out socket/pipe peer reads,
checks finite file reads before/after I/O, and refuses arbitrary blocking custom
streams when a deadline is requested. A deadline cannot preempt an OS filesystem
I/O stall or a model computation. Adapters still need their own transport/computation
limits and authenticated framing; this package does not open a network port.

## Confidential storage and attestation

`Workload` requires an explicit storage mode:

- `local-test`: accepts only the explicit in-memory `LocalSigner`. Ordinary scratch
  space is allowed for the CPU simulation. The machine's operator is trusted.
- `confidential-tmpfs`: rejects the test signer and requires an owned private
  directory on tmpfs, with swap disabled. This avoids writing decrypted provider
  artifacts to an ordinary host-visible disk. The accepted confidential VM/image
  must protect RAM; the filesystem check is not hardware attestation.

A preflight checks available capacity against the complete agreed bundle cap plus
5% or 16 MiB headroom, whichever is larger, **before issuing the proposal**. This is
not a reservation against concurrent usage or a check of GPU working memory. The
cap already bounds all content; it is not multiplied again by a duplication factor.

A real deployment needs enough protected scratch/RAM for the decrypted files and
normal model-loading requirements. Streaming avoids an extra whole-model Python
buffer; it does not eliminate the model's storage/GPU/RAM requirements. Run a
**fresh protected process per session**, so model-backend caches cannot retain
another session's objects or exhaust memory across sessions. Confidential mode
enforces a process-wide single-session claim in addition to the instance state
machine. Use fresh process execution, not a fork of a process with private models.

`accepted_gcp_verifier` uses the existing authentic token/issuer/key-nonce verifier
and additionally requires a preselected image digest, audience, production debug
status and Confidential Space software claim. There is no debug exception or
trust-on-first-use fallback. The policy tests use locally signed test JWTs and a
test JWKS key; they do not contact Google's service. `GcpEnclaveSigner` can be
passed to `Workload` by an accepted confidential image. **No real Confidential Space
launch or GPU confidential-computing validation has been performed here.** CPU
attestation does not by itself establish GPU protection.

The private C-to-workload installation interface and the C/P relay channels must
be authenticated and correctly isolated, as assumed in Figure 4. The local Python
demo simulates all roles in one trusted process and establishes no confidentiality
against that process's operator. Production adapters must return only generic
abort status, without logs/tracebacks containing private artifacts or trial state.
`certify` raises one `ProtocolError` with a constant public message for workload
session failures. `handle_submission` returns a certificate or only a versioned
`abort`, suppressing private error types/details. Low-level helpers are not public
RPC entry points.

The certifier chooses `sid`; a real adapter should use a fresh random identifier
and convey it with the agreed specification over the authenticated C/P channel.
The workload independently generates `eid`. P checks the proposal's sid and retains
its eid for result acceptance. R obtains sid/eid in the signed certificate and may
require independently expected values for a live-session policy; offline checking
always requires the expected pair identity and specification. A session identifier
is not a certifier identity or endorsement signature.

## Timing hooks

Receiver stages separate initial stream/manifest validation, authenticated stream
decryption and file writes, content verification, snapshot layout and component
preparation. The first explicit digest pass verifies every unique blob; the two
existing preparation passes are retained. Core versions with the optional recorder
also split preparation hashing from component loading. Older core versions expose
only `component_prepare_total`. The provisioning and preparation total stages are
**nested outer intervals**, not additive exclusive phases. Provider encryption,
transfer/waiting and the whole critical path need their own harness timers;
concurrent role times cannot simply be summed. No model-sized timing experiment
has been run as part of this implementation.

## Relation to the formal construction

The dataflow and public object identity now follow Figure 4, rather than the old
environment-encryption variant. Concrete implementation choices remain explicit:

| Formal Figure 4 | This implementation |
|---|---|
| C installs `(spec,E)` in its enclave | `install_environment(spec,padded_E)` verifies registered E before issuing the proposal |
| Enclave attests `(propose,spec,pk)` | Signed proposal, plus explicit session/instance/version/measurement framing |
| P encrypts `(program,tap)` under `pk` | Inert canonical records and streamed complete artifact bytes encrypted to that key |
| Enclave returns `(certify,spec,H(program,tap),m)` | Signed stable digest and `{fn,fp}`, with both denominators equal to total trials |
| Provider verifies same enclave and result | `Provider.accept` requires the proposal instance and its own expected stable pair digest |
| R verifies attested output | `verify_certificate` requires an independently accepted verifier, expected specification and pair identity |

Registered drivers are a deliberate restriction on executable extensibility: a
provider cannot submit arbitrary Python/pickle code, and the certifier cannot
install arbitrary environment code. The trusted image must include the full code
and dependencies used by those drivers. The inherited engine uses fresh logical
trial objects and any-round flag aggregation; the paper must identify that as its
concrete measurement semantics. Statistical representativeness and adaptive test
reuse are separate from the protocol.

The proof-only `extract`/`program` trapdoor entry points are **not exposed** in this
runtime. There is no claim that testing this concrete representation, hybrid
encryption, attestation adapter and registered-driver subset establishes the
paper's full UC theorem. The correspondence above makes those remaining proof
obligations reviewable; it does not replace that review.

## Files and integration

- `protocol.py`: one-shot workload/provider state machines and public verification.
- `bundle.py`: stable representation and private snapshot loading.
- `stream.py`, `codec.py`: bounded encrypted transfer and strict parsing.
- `attestation.py`, `storage.py`: explicit trust and storage policy.
- `demo.py`, `__main__.py`: small local role simulation and public verification CLI.
- `tests/`: protocol failures, identity, transfer, confinement and attestation policy.

Dependencies are confined to this folder's virtual environment. This module does
not replace the existing certification entry points. The development core now
offers optional hash/load timers; frozen v1/v2 sweeps and their outputs are unchanged.
Their timings must not be relabeled as measurements of this new exchange: provider
encryption, transfer, private materialization and loading need their own measured
phases on the eventual machine.


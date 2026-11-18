# Changes from the project repository

The code was copied file by file from the project repository (November 2026), not rewritten. Executable code is unchanged except for the edits in the first table. For anonymity, comments, docstrings and documentation were then scrubbed of internal labels (references to a private run record and its numbered changes, private review notes, internal run codes, machine names and job numbers): 29 Python files (comments and docstrings; in three of them also command-line help text), 4 copied Markdown files and 1 shell script, about 95 replacements in total (78 by a pattern script that touched only comments and docstrings, checked by comparing syntax trees, and about 17 by hand). String constants that are written into run records or used in prompts, and test function names, were left as they are, so a few internal labels remain there (for example the monitor registry's `notes` fields and manifest descriptions in `rh_a51_optimise.py`).

Because of the scrub, the shipped `evals/tap_live/` and `evals/reward_hacking/` files are no longer byte-identical to the versions whose sha256 values are recorded in the EQ1 run configurations; the differences are confined to comments and docstrings (`select_stage2.py`, `export_phase_table.py`, `test_decode_bench.py`, `monitors.py`). Every other edit is listed here.

## Edited files

| File | Edit | Why |
|---|---|---|
| `evals_runner/__main__.py` | Command list reduced to the shipped commands (protocol demo, rollouts, monitors, probe, EQ2 reports). One comment shortened. | The other commands belong to experiments not shipped. |
| `evals_runner/eq1_judge_saved_calls_run.sh` | Removed a machine name from two comments and from the text of the "not measured" reason string. | Private host name. The reason string is only written to a note file; no behaviour changes. |
| `evals/tap_live/test_decode_bench.py` | The scratch directory for an optional LaTeX build test is the system temporary directory instead of a fixed home path. | Absolute home path. |
| `evals/tap_live/test_decode_bench.py` | Added the same skip condition its sibling tests already have to `test_phase_table_from_a_real_tiny_run`: it needs a table skeleton from the project's review folder, which is not shipped. The skeleton belongs to an earlier, retired timing table. | Review files not shipped. |
| `evals/reward_hacking/test_muse_channels.py` | The default tokenizer path is empty (the tests skip unless `MUSE_TOKENIZER_DIR` is set). | Absolute home path. |
| `deploy/Dockerfile` | Removed the copy of a third-party submodule that the mock workload does not need; copies `tee/` and `figure4_prototype/`; comment points to `tee/CONFIDENTIAL_SPACE.md`. | Submodule not shipped. |
| `evals/reward_hacking/scripts/swebench_preflight_merge.py` | A machine name in the `VERIFICATION` record string was replaced by a neutral word, consistently with the redacted records in `evals/reward_hacking/data/swebench/`. The string is only written into those records. | Anonymity. |
| `LICENSE` | Copyright holder changed to "The authors"; MIT text unchanged. | Anonymity. |
| `pyproject.toml` | New, smaller project file for this tree (package list, test paths, optional extras). | The original lists packages and extras not shipped. |

## New files (not in the repository)

- `README.md`, `ALIGNMENT.md`, `CHANGES.md`, `check.sh`, `Makefile`, `requirements-*.txt`, `data/README.md`, `prompts/README.md`.
- `reproduce/eq1_table.py`, `reproduce/eq2_tables.py`, `reproduce/paper_numbers.json`, `reproduce_eq1.sh`, `reproduce_eq2.sh`. The table scripts use the repository's own statistics (`evals/reward_hacking/metrics.py`, `rh-eq2-cv-report`, `main_complete_call`) and follow the same population rules as the original reports.
- `tee/README.md`, `tee/CONFIDENTIAL_SPACE.md` (adapted from the repository's runbook), `tee/NVIDIA_SIGNER_DESIGN.md`.
- `tee/extension/`: the GPU-only attestation module (`nvidia_attestor.py`: `GpuAttestedSoftwareSigner`, `GpuAttestationVerifier`, `GpuAttestationError`), its tests, `gpu_policy.json`, `freeze_policy.py` and `run_gpu_attestation_v2.py`, copied from the project's revised version. Edits: import paths (`ct_core.protocol.nvidia_attestor` became `tee.extension.nvidia_attestor`; the root path in `freeze_policy.py`), the skip reason of the one test that replays archived live tokens (the tokens are not shipped because they carry a hardware identifier of the GPU), a docstring note in `freeze_policy.py` that its inputs are not shipped, and removal of two internal labels from test comments and one from the run script's docstring. Three tests that need the unshipped cost-experiment runner are deselected in `check.sh`. The earlier interface and mock files were removed. The Confidential Space code is unchanged.
- `tools/slim_scores.py`, `tools/redact.py`, `tools/make_prompts.py`: how the shipped data and prompt files were made.

## Not copied

- Review letters, experiment notes, the run record, paper drafts and history scaffolding.
- Code for experiments not in this artifact: earlier cost studies and their analysis scripts, the retired confidence and financial cases, prompt and instruction screens, many one-off report and sensitivity scripts, cluster job files (they contain account names), the withdrawn whole-episode timing study, and the earlier capped-budget timing run.
- The `confidence` and `financial` cases of `protocol-integrity-exp`: the code still names them, but their modules are not shipped, so only `--case mock` runs.
- Third-party submodules (TreeCut generator, an external probe library). The mock test fixture they produced is shipped as data.
- Tests whose subjects are not shipped. Five test files that import or execute cluster job files or unshipped comparison scripts were removed: `evals/reward_hacking/test_muse_activations.py`, `test_swebench_run_gemma.py`, `test_swebench_run_muse.py`, `evals_runner/test_rh_a51_optimise.py`, `evals_runner/test_rh_rollouts_muse.py`.
- `evals/reward_hacking/interventions.py` is shipped only because a shipped test imports it.

## Known test failures (deselected in `check.sh`, not edited)

| Test | Reason |
|---|---|
| `test_qwen38_format.py::test_dom_pairs_prompt_uses_the_target_format`, `::test_default_probe_layers_follow_the_target_depth` | Import `dom_pairs`, an earlier probe-direction module that is not shipped. |
| `test_swebench_run.py::test_prompt_variant_enters_config_only_when_not_default` | Expects 8 SWE-bench instruction variants; the code has 23 after later screens. Fails the same way in the source repository. |
| `test_rh_probe.py::test_registered_in_cli`, `::test_probe_sbatch_is_cpu_only_and_valid` | Need a `.venv` at a fixed place and a cluster job file. |

## Data edits

See `data/README.md`: monitor reply texts were replaced by hashes and short tails; account names, host names and file-system paths were removed (symbolic forms such as `<HOME>` remain); server logs were dropped. Hashes recorded inside redacted files differ from a recomputation over the shipped bytes. Comments and docs that named private working directories now use symbolic names such as `<WORK_DIR>`; a few code defaults keep generic forms such as `~/rh/activations` or `/scratch/$USER/...`, which contain no user name.

## Later additions

- Per-episode LiveCodeBench scores of the Gemma E4B monitor and both probes, and the LiveCodeBench one-off Gemma E4B final-code scores, were added after the first build (slimmed and redacted like the other data). `reproduce/eq2_tables.py` now recomputes all 40 Table 2 cells and 22 of the 24 final-code cells; the remaining 2 are read from the original report.

## Changes after review (current question numbering)

- Scope wording: the artifact covers the current EQ1, EQ2 and EQ3 (the task-and-code coding comparison); the travel-agent scenario (current EQ4) and the output-replacement illustration are excluded.
- `check.sh` was restructured: every step's exit status is checked and the script exits nonzero if any step fails. The protocol step now runs `tools/protocol_demo_check.py`, which asserts that the honest run is certified with a VALID certificate and that the tampered run aborts for the intended reason ("certificate is not bound to this environment message"). `tests/test_check_script.py` checks that a deliberately broken honest demo, or a tampered demo that certifies, makes `check.sh` exit nonzero.
- EQ1: `reproduce/eq1_table.py` and `reproduce/paper_numbers.json` now reproduce the probe row's Tap cell as the isolated tap execution time (26.46 ms per call, from the component microbenchmark's raw passes), and keep the paired A1 − A0 latency difference separately as supporting evidence.
- `evals/tap_live/profile_components.py`, its test and its results (`data/eq1/profile_components/`) are included as an accepted isolated component-execution measurement. In its README, wording that read the device-side views as bounds on added latency was replaced; those views are diagnostics only, and 4.70 µs per token is described as the ratio of two weighted means. Its `SHA256SUMS` was written before these README edits and the redaction, so its README line no longer matches.
- `tee/` docs: the GPU-only attestation is described as a software-held signing key with genuine GPU evidence; it does not prove ownership or isolation of the key by the certification program, the CPU side and workload identity are unverified, and it is not a full certification result. The extension code in `tee/extension/` is marked as being revised. The Confidential Space code is unchanged and untimed here.
- The product name and the cluster-specific identifiers of the second machine's GPU were removed everywhere. The GPU name field in the run records now reads "datacenter GPU" and the hardware label in the tables reads "second machine". The monitor-timing folder of that machine is `data/eq1/judges_second_machine/` (the cluster job-file names were renamed in the same way), and `reproduce/eq1_table.py`, `reproduce/paper_numbers.json` and the table files use the same label. The memory total, driver version and power limit were also removed from the GPU records (the used-memory readings stay, without the total); `evals/reward_hacking/scripts/extract_activations.py` and `evals/reward_hacking/SANDBOX.md` had one hardware-specific phrase each, now worded generally. The recorded hashes of edited files differ: the `code_sha256` entry for `evals_runner/eq1_judge_saved_calls.py` in the `run_config.json` files of `data/eq1/judges_*` was computed before this edit, and `data/MANIFEST.sha256` was rebuilt.

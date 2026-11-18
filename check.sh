#!/usr/bin/env bash
# One command that checks the artifact on CPU:
#   tests     the CPU unit tests (pytest)
#   checksums the sha256 manifests of data/ and prompts/
#   protocol  the certification protocol end to end with the local (stub) signer: the honest run must be
#             certified with a VALID certificate, and the tampered run must abort for the intended reason
#   eq1, eq2  regeneration of the EQ1 and EQ2/EQ3 tables, printed next to the paper's numbers
# Every step's exit status is checked; the script exits nonzero if any step fails.
# Usage: bash check.sh            (FAST=1 skips the two slower SWE-bench report runs; SLOW=1 adds the sandbox tests)
#        PYTHON=/path/to/python bash check.sh
#        STEPS="protocol eq1" bash check.sh      (run only some steps)
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHON="$PY" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
STEPS="${STEPS:-tests checksums protocol eq1 eq2}"
failed=()
want() { [[ " $STEPS " == *" $1 "* ]]; }
banner() { echo; echo "=================== $*"; }

had_keys=0; [ -d .keys ] && had_keys=1          # tests and the demo write a local test key to .keys/
had_res=0; [ -d evals_results ] && had_res=1    # and small result files to evals_results/
cleanup() { if [ "$had_keys" -eq 0 ]; then rm -rf .keys; fi; if [ "$had_res" -eq 0 ]; then rm -rf evals_results; fi; }
trap cleanup EXIT

step_tests() {
  # Known failures, deselected (see CHANGES.md): they need files not shipped, or fail in the source repository too.
  local KNOWN=(
    --deselect evals/reward_hacking/test_qwen38_format.py::test_dom_pairs_prompt_uses_the_target_format
    --deselect evals/reward_hacking/test_qwen38_format.py::test_default_probe_layers_follow_the_target_depth
    --deselect evals/reward_hacking/test_swebench_run.py::test_prompt_variant_enters_config_only_when_not_default
    --deselect evals_runner/test_rh_probe.py::test_registered_in_cli
    --deselect evals_runner/test_rh_probe.py::test_probe_sbatch_is_cpu_only_and_valid
    # these three exercise the cost-experiment runner that wires the GPU signer into a session; that runner is not shipped
    --deselect tee/extension/test_nvidia_attestor.py::test_certified_run_with_mock_gpu_attestation_says_what_it_is_and_is_not
    --deselect tee/extension/test_nvidia_attestor.py::test_certified_run_with_nvidia_fails_closed_and_runs_no_session
    --deselect tee/extension/test_nvidia_attestor.py::test_nvidia_is_never_picked_by_auto
  )
  # Slow tests that build Docker/Apptainer sandboxes (about 30 min, several GB of cache); run them with SLOW=1.
  local SLOWTESTS=(evals/reward_hacking/test_extend.py evals/reward_hacking/test_scorer.py evals/reward_hacking/test_sandbox.py
                   evals/reward_hacking/test_tools.py evals/reward_hacking/test_rollout.py evals/tap_live/test_coding_harness.py)
  local IGN=(); [ "${SLOW:-0}" = 1 ] || for f in "${SLOWTESTS[@]}"; do IGN+=(--ignore "$f"); done
  "$PY" -m pytest -q -rf -p no:cacheprovider --import-mode=importlib -o pythonpath=. "${KNOWN[@]}" "${IGN[@]}" \
    ct_core util evals evals_runner tee figure4_prototype/tests tests 2>&1 | tail -n 25
  return "${PIPESTATUS[0]}"
}
step_checksums() {
  local rc=0
  (cd data && sha256sum --quiet -c MANIFEST.sha256) && echo "data/MANIFEST.sha256: all files OK" || { echo "data checksum FAILED"; rc=1; }
  (cd prompts && sha256sum --quiet -c SHA256SUMS) && echo "prompts/SHA256SUMS: all files OK" || { echo "prompts checksum FAILED"; rc=1; }
  return "$rc"
}
step_protocol() { "$PY" tools/protocol_demo_check.py; }
step_eq1() { bash reproduce_eq1.sh cpu; }
step_eq2() { if [ "${FAST:-0}" = 1 ]; then bash reproduce_eq2.sh --fast; else bash reproduce_eq2.sh; fi; }

for s in tests checksums protocol eq1 eq2; do
  want "$s" || continue
  banner "$s"
  if "step_$s"; then echo "[$s] OK"; else echo "[$s] FAILED"; failed+=("$s"); fi
done

echo
if [ "${#failed[@]}" -eq 0 ]; then echo "check.sh: ALL CHECKS PASSED"; exit 0; fi
echo "check.sh: FAILED steps: ${failed[*]}"
exit 1

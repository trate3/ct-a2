#!/usr/bin/env bash
# EQ1 GPU run, one step at a time, from any directory. Outputs go to $OUT. Run the GPU steps one after another, with the GPU otherwise idle.
#   bash reproduce/eq1_gpu.sh main [ARGS]            agent generation, both arms, then the weighted means (ARGS go to generate_calls_stage2)
#   bash reproduce/eq1_gpu.sh monitors [NAME ...]    each monitor judges the saved A0 responses (vLLM 0.31.0 in VLLM_VENV)
#   bash reproduce/eq1_gpu.sh certification [smoke]  certification sessions with GPU attestation (smoke: one session, probe configuration)
#   bash reproduce/eq1_gpu.sh tap-work               isolated tap work per call at this run's call lengths
#   bash reproduce/eq1_gpu.sh table                  check the outputs and print the EQ1 table (CPU)
# Environment (defaults in brackets):
#   OUT [results_regenerated/eq1_gpu]   PYTHON [python3]   ATTEST_PYTHON  Python with the NVIDIA attestation packages [PYTHON]
#   MODEL_DIR  Qwen3.8-27B snapshot directory    E4B_DIR  Gemma 4 E4B snapshot directory    JUDGE_TRAJ  the recorded trajectories
#   EQ1_GPU_POLICY_FILE [tee/extension/gpu_policy.json]    EQ1_GPU_POLICY_DIGEST  the policy digest, chosen independently of the file
#   CERT_REPEATS [5]    MODEL_REVISION  recorded with the generation run
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"; PYA="${ATTEST_PYTHON:-$PY}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EQ1_GPU_POLICY_FILE="${EQ1_GPU_POLICY_FILE:-$PWD/tee/extension/gpu_policy.json}"
OUT="${OUT:-results_regenerated/eq1_gpu}"; GEN="$OUT/generate_calls_stage2"; JUDGES="$OUT/judges"; MAINCC="$OUT/main_complete_call/main_complete_call.json"
D=data/eq1; STAGE2=$D/generation/stage2_list.json; PROBE=$D/probe/tap_p2_logistic_mean.json
mkdir -p "$OUT"

facts() {   # machine record of a step
  { echo "date_utc: $(date -u +%FT%TZ)"; echo "kernel: $(uname -r)"
    echo "--- nvidia-smi"; nvidia-smi || true
    echo "--- nvidia-smi conf-compute -q"; nvidia-smi conf-compute -q || true
    echo "--- python"; "$PY" -c "import sys, importlib.metadata as m; print(sys.version.split()[0]); [print(p, m.version(p)) for p in ('torch', 'transformers', 'numpy')]" || true
    echo "--- inputs"; sha256sum "$STAGE2" "$D/histories.json" "$PROBE" "$EQ1_GPU_POLICY_FILE"
    echo "--- code"; sha256sum evals/tap_live/*.py evals_runner/eq1_*.py evals_runner/eq1_judge_saved_calls_run.sh ct_core/certification.py tee/extension/nvidia_attestor.py "$0"
    echo "--- env"; env | grep -E '^(HF_|TRANSFORMERS_|PYTHONPATH|VLLM_|CUDA_|JUDGE_|EQ1_)' | sort
  } > "$1" 2>&1
}

main_running() { pgrep -f '[p]ython.* -m evals\.tap_live\.generate_calls_stage2' >/dev/null 2>&1; }

main_done() {   # the main stage must have ended with complete outputs
  if main_running; then echo "[eq1] the main stage is still running; refusing" >&2; exit 3; fi
  "$PY" - "$GEN" "$STAGE2" "$OUT/generate_calls_stage2.log" <<'PY'
import json, pathlib, sys
g, stage2, log = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text()), pathlib.Path(sys.argv[3])
rows = lambda p: [json.loads(l) for l in (g / p).read_text().splitlines() if l.strip()]
calls, outs, outs1 = rows("calls.jsonl"), rows("outputs.jsonl"), rows("outputs_A1.jsonl")
want = {r["run_key"] for r in stage2["runs"]}
bad = []
if len(calls) != 2 * len(want): bad.append(f"calls.jsonl has {len(calls)} rows, expected {2 * len(want)}")
if any(c["status"] != "ok" for c in calls): bad.append("some timed call has status != ok")
if {o["run_key"] for o in outs} != want or len(outs) != len(want): bad.append(f"outputs.jsonl does not hold exactly the {len(want)} A0 outputs")
if {o["run_key"] for o in outs1} != want: bad.append("outputs_A1.jsonl does not hold all A1 outputs")
if log.exists() and "timed calls written" not in log.read_text()[-3000:]: bad.append("the generation log does not end with 'timed calls written'")
if bad:
    sys.exit("[eq1] the main stage output is not complete:\n  " + "\n  ".join(bad))
PY
}

case "${1:-}" in
  main)
    shift
    : "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B snapshot directory}"
    facts "$OUT/machine_and_hashes.txt"
    CMD=("$PY" -m evals.tap_live.generate_calls_stage2 --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" ${MODEL_REVISION:+--model-revision "$MODEL_REVISION"}
         --histories "$D/histories.json" --artifact "$PROBE" --stage2-list "$STAGE2" --out "$GEN" "$@")
    echo "${CMD[*]}" > "$OUT/command.txt"
    "${CMD[@]}" 2>&1 | tee "$OUT/generate_calls_stage2.log"
    main_done
    "$PY" -m evals.tap_live.main_complete_call --stage2 "$GEN" --stage2-list "$STAGE2" --out "$OUT/main_complete_call" | head -12
    ;;
  monitors)
    shift
    main_done
    : "${JUDGE_TRAJ:?set JUDGE_TRAJ to the recorded trajectories (hashes are checked against histories.json)}"
    export VLLM_USE_V2_MODEL_RUNNER=0   # vLLM runner setting
    export JUDGE_OUT="$JUDGES" JUDGE_STAGE2="$STAGE2" JUDGE_HISTORIES="$D/histories.json" JUDGE_OUTPUTS="$GEN/outputs.jsonl" JUDGE_MAIN="$MAINCC"
    export JUDGE_HEALTH_POLLS="${JUDGE_HEALTH_POLLS:-360}"
    mkdir -p "$JUDGES"; facts "$JUDGES/machine_and_hashes_monitors.txt"
    [ -f "$JUDGES/inputs/eq1_judge_saved_calls_inputs.json.gz" ] || bash evals_runner/eq1_judge_saved_calls_run.sh inputs
    bash evals_runner/eq1_judge_saved_calls_run.sh run "$@"
    ;;
  certification)
    : "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B snapshot directory}" "${E4B_DIR:?set E4B_DIR to the Gemma 4 E4B snapshot directory}"
    : "${EQ1_GPU_POLICY_DIGEST:?set EQ1_GPU_POLICY_DIGEST to the digest of the GPU policy}"
    export EQ1_GPU_POLICY_DIGEST
    main_done
    if pgrep -f '[v]llm serve' >/dev/null 2>&1; then echo "[eq1] a vLLM server is running; refusing" >&2; exit 5; fi
    ARGS=(--qwen-dir "$MODEL_DIR" --e4b-dir "$E4B_DIR" --probe-file "$PROBE" --attestation nvidia)
    if [ "${2:-}" = smoke ]; then
      CERT="$OUT/certification_smoke"; mkdir -p "$CERT"; facts "$CERT/machine_and_hashes_cert.txt"
      "$PYA" -m evals_runner.eq1_cert_timing run --out "$CERT/run" --configs probe --repeats 1 "${ARGS[@]}"
    else
      CERT="$OUT/certification"; mkdir -p "$CERT"; facts "$CERT/machine_and_hashes_cert.txt"
      "$PYA" -m evals_runner.eq1_cert_timing run --out "$CERT/run" --configs probe,e4b,qwen --repeats "${CERT_REPEATS:-5}" --evict-first "${ARGS[@]}"
    fi
    ;;
  tap-work)
    "$PY" -m evals.tap_live.profile_components --stage2-list "$STAGE2" --calls "$GEN/calls.jsonl" --artifact "$PROBE" --out "$OUT/tap_work"
    ;;
  table)
    "$PY" -m evals_runner.eq1_build_table --gen-dir "$OUT" --stage2-list "$STAGE2" --main-complete-call "$MAINCC" \
      --inputs "$JUDGES/inputs/eq1_judge_saved_calls_inputs.json.gz" --monitors "$JUDGES" --tap-work "$OUT/tap_work/summary.json" \
      --cert-dir "$OUT/certification/run" --require-cc-text "$JUDGES/machine_and_hashes_monitors.txt" --out-dir "$OUT/table" --json "$OUT/table/rows.json"
    ;;
  *) echo "usage: bash reproduce/eq1_gpu.sh main [ARGS] | monitors [NAME ...] | certification [smoke] | tap-work | table" >&2; exit 2 ;;
esac

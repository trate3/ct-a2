#!/usr/bin/env bash
# EQ1 shared-call cost: each monitor judges the SAVED generated calls (the main stage's A0 outputs) with the full available
# trace. Run from the repository root on the GPU that ran the agent, with the GPU otherwise idle. One vLLM server per monitor,
# one request at a time, stopped before the next monitor starts.
#   bash evals_runner/eq1_judge_saved_calls_run.sh inputs            # CPU only: the unfitted monitor inputs and their manifest (a check before any GPU use)
#   bash evals_runner/eq1_judge_saved_calls_run.sh plan [NAME ...]   # CPU only: token counts, budgets and fitting per monitor, writes nothing
#   bash evals_runner/eq1_judge_saved_calls_run.sh run [NAME ...]    # measure each monitor (default list below), resumable
#   bash evals_runner/eq1_judge_saved_calls_run.sh table             # the combined table (markdown and LaTeX) over the monitor directories
# Environment (defaults in brackets):
#   JUDGE_OUT        [results_regenerated/eq1_judges]
#   JUDGE_STAGE2     [data/eq1/generation/stage2_list.json]
#   JUDGE_HISTORIES  [data/eq1/histories.json]
#   JUDGE_OUTPUTS    the saved A0 outputs of a generation run (generate_calls_stage2/outputs.jsonl; required)
#   JUDGE_MAIN       main_complete_call.json (for the composed With tap; optional, can be given later with the summary step)
#   JUDGE_TRAJ       the recorded training trajectories (hashes are checked against histories.json)
#   JUDGE_VIEWS      history_black_box (primary; add history_reasoning for the reasoning view)   JUDGE_INSTRUCTION  early|terminal (default early)
#   JUDGE_LIMIT=N    smoke test (first N items; use another JUDGE_OUT)    JUDGE_PREFIX_CACHING=on|off (default on, recorded)    JUDGE_ALLOW_BUSY=1
#   JUDGE_HEALTH_POLLS  server health polls, 5 s apart [180]
#   PYTHON           Python with this repository's requirements [python3];  VLLM_VENV  venv with vLLM 0.31.0 [.venv-vllm]
# Monitors (registry names): gemma4-e4b (reasoning off, 1,024 cap), gemma4-e4b-think, qwen3.8-27b-4k (off), qwen3.8-27b-think.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
VENV_V="${VLLM_VENV:-$PWD/.venv-vllm}"; PY="${PYTHON:-python3}"
export PATH="$VENV_V/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"      # flashinfer JIT needs ninja and nvcc
OUT="${JUDGE_OUT:-results_regenerated/eq1_judges$([ -n "${JUDGE_LIMIT:-}" ] && echo _smoke || true)}"
STAGE2="${JUDGE_STAGE2:-data/eq1/generation/stage2_list.json}"
HISTORIES="${JUDGE_HISTORIES:-data/eq1/histories.json}"
OUTPUTS="${JUDGE_OUTPUTS:?set JUDGE_OUTPUTS to generate_calls_stage2/outputs.jsonl of a generation run}"
TRAJ="${JUDGE_TRAJ:-trajectories}"
VIEWS="${JUDGE_VIEWS:-history_black_box}"; INSTR="${JUDGE_INSTRUCTION:-early}"
NAMES_DEFAULT="gemma4-e4b gemma4-e4b-think qwen3.8-27b-4k qwen3.8-27b-think"
PORT=8100; PCACHE="${JUDGE_PREFIX_CACHING:-on}"; SERVER=""
COMMON=(--stage2-list "$STAGE2" --histories "$HISTORIES" --outputs "$OUTPUTS" --trajectories "$TRAJ" --views "$VIEWS" --instruction "$INSTR")
[ -n "${JUDGE_MAIN:-}" ] && COMMON+=(--main-complete-call "$JUDGE_MAIN")
stop_server() {
  if [ -n "$SERVER" ]; then kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; SERVER=""; fi
  for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${used:-0}" -lt 2000 ] && break; sleep 2; done
}
trap stop_server EXIT
snapdir() { eval "$("$PY" -m evals.reward_hacking.monitors serve-env "$1")"; "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$RH_MON_HF_ID', revision='$RH_MON_REVISION'))"; }

inputs() { mkdir -p "$OUT/inputs"; "$PY" -m evals_runner.eq1_judge_saved_calls_prompts --stage2-list "$STAGE2" --histories "$HISTORIES" --outputs "$OUTPUTS" \
             --trajectories "$TRAJ" --views "$VIEWS" --instruction "$INSTR" --out "$OUT/inputs/eq1_judge_saved_calls_inputs.json.gz"; }

plan() {
  for n in ${*:-gemma4-e4b gemma4-e4b-think qwen3.8-27b-4k qwen3.8-27b-think}; do
    "$PY" -m evals_runner.eq1_judge_saved_calls --monitor "$n" --plan-only --out-dir "$OUT/$n" --model-dir "$(snapdir "$n")" "${COMMON[@]}" > "$OUT/plan_$n.json" \
      && echo "[judge] plan for $n written to $OUT/plan_$n.json"
  done
}

measure() {
  local n="$1" D="$OUT/$1"
  mkdir -p "$D"
  if [ -f "$D/summary.json" ]; then echo "[judge] $n done; skipped"; return; fi
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -gt 2000 ] && [ "${JUDGE_ALLOW_BUSY:-0}" != 1 ]; then echo "[judge] GPU not idle (${used} MiB in use); refusing (JUDGE_ALLOW_BUSY=1 overrides)"; exit 2; fi
  eval "$("$PY" -m evals.reward_hacking.monitors serve-env "$n")"
  local DIR CTX t0 t1 ok=0
  DIR="$(snapdir "$n")"
  CTX=$(( MAX_MODEL_LEN < 131072 ? MAX_MODEL_LEN : 131072 ))
  local args=(serve "$DIR" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT" --max-model-len "$CTX" --max-num-seqs 64
              --gpu-memory-utilization 0.90 --dtype bfloat16 --generation-config vllm)
  [ "$PCACHE" = on ] && args+=(--enable-prefix-caching) || args+=(--no-enable-prefix-caching)
  [ "${LANGUAGE_ONLY:-1}" = 1 ] && args+=(--language-model-only)
  [ -n "${EXTRA:-}" ] && args+=($EXTRA)
  echo "[judge] $n: vllm ${args[*]}"
  t0=$(date +%s.%N); local launch_utc; launch_utc=$(date -u +%FT%TZ)
  "$VENV_V/bin/vllm" "${args[@]}" > "$D/server.log" 2>&1 & SERVER=$!
  for i in $(seq 1 "${JUDGE_HEALTH_POLLS:-180}"); do kill -0 "$SERVER" 2>/dev/null || break; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }; sleep 5; done
  t1=$(date +%s.%N)
  if [ $ok != 1 ]; then echo "[judge] $n: server did not become ready (see $D/server.log)"; stop_server; return 1; fi
  N="$n" DIRV="$DIR" CTXV="$CTX" PC="$PCACHE" SPID="$SERVER" LAUNCH="$launch_utc" SECS="$(echo "$t1 - $t0" | bc)" CMD="vllm ${args[*]}" \
    "$PY" - "$D/server_info.json" <<'PY'
import json, os, sys
e = os.environ
json.dump({"flags": {"command": e["CMD"], "served_model_name": e.get("SERVED_NAME"), "max_model_len": int(e["CTXV"]), "max_num_seqs": 64,
                     "gpu_memory_utilization": 0.90, "dtype": "bfloat16", "quantization": "none (bf16 safetensors)",
                     "prefix_caching": e["PC"], "generation_config": "vllm", "language_model_only": e.get("LANGUAGE_ONLY", "1") == "1",
                     "extra": e.get("EXTRA", ""), "registry_max_model_len": int(e.get("MAX_MODEL_LEN", 0)), "model_snapshot": e["DIRV"]},
           "session": {"launch_utc": e["LAUNCH"], "seconds_launch_to_health_ok": float(e["SECS"]), "server_pid": int(e["SPID"])}},
          open(sys.argv[1], "w"), indent=1)
PY
  "$PY" -m evals_runner.eq1_judge_saved_calls --monitor "$n" --url "http://127.0.0.1:$PORT" --out-dir "$D" --model-dir "$DIR" --server-log "$D/server.log" \
    --server-info "$D/server_info.json" --server-pid "$SERVER" "${COMMON[@]}" ${JUDGE_LIMIT:+--limit "$JUDGE_LIMIT"} \
    || echo "[judge] $n: stopped with rc=$? (rerun to resume)"
  stop_server; sleep 5
}

case "${1:-}" in
  inputs) inputs;;
  plan) shift; mkdir -p "$OUT"; plan "$@";;
  run) shift; for n in ${*:-$NAMES_DEFAULT}; do measure "$n"; done; "$PY" -m evals_runner.eq1_judge_saved_calls --combine "$OUT" --out "$OUT/table";;
  table) if [ -n "${JUDGE_MAIN:-}" ]; then   # (re)compute each monitor's summary with the composed With tap from main_complete_call.json
           for D in "$OUT"/*/; do n="$(basename "$D")"; [ -f "$D/run_config.json" ] && \
             "$PY" -m evals_runner.eq1_judge_saved_calls --monitor "$n" --summarize-only --out-dir "$D" --server-log "$D/server.log" "${COMMON[@]}"; done
         fi
         "$PY" -m evals_runner.eq1_judge_saved_calls --combine "$OUT" --out "$OUT/table";;
  *) echo "usage: $0 inputs | plan [NAME ...] | run [NAME ...] | table" >&2; exit 2;;
esac

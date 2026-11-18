#!/usr/bin/env bash
# EQ1: time for the phases of certified-tap evaluation (paper Table 1).
#
#   bash reproduce_eq1.sh            CPU: recompute every cell of the table from the shipped raw export (seconds)
#   bash reproduce_eq1.sh gpu-plan   print the full GPU re-run commands (nothing is executed)
#   bash reproduce_eq1.sh gpu        run the live generation (needs one 96 GB GPU and the Qwen3.8-27B weights)
#
# What the GPU path measures: one generation per distinct prompt (24) without the tap (A0) and with the mean probe (A1),
# same seed, natural stop, single sequence, Transformers BF16, the probe scoring at the call boundary inside the timer.
# Measured on an RTX PRO 6000 Blackwell (96 GB), torch 2.15.0+cu132, transformers 5.23.0, Python 3.12: about 1.8 GPU-hours
# of generation (48 calls; two prompts account for about 80% of the time), plus loading.
# Monitor rows: each monitor judges the 24 saved A0 responses through a vLLM 0.31.0 server, one request at a time
# (minutes per monitor on one GPU). GLM-5.3-Flash NVFP4 (about 204 GB) needs a larger GPU (we used one datacenter GPU).
# Expect timings to differ on other hardware; the paired probe latency difference should stay unresolved around zero.
# The isolated tap execution time (Tap cell) comes from a separate CPU/GPU microbenchmark without the model:
#   python -m evals.tap_live.profile_components --stage2-list ... --calls ... --artifact ... --out DIR  (GPU, about 8 minutes)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1

QWEN_REPO=Qwen/Qwen3.8-27B
QWEN_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
OUT="${OUT:-results_regenerated/eq1_gpu}"
D=data/eq1

plan() {
cat <<EOF
# 0. Weights (about 51 GiB in BF16). Set MODEL_DIR to the snapshot directory.
huggingface-cli download $QWEN_REPO --revision $QWEN_REV
# 1. Selection (CPU; already done, shipped). Rebuilding the histories needs the recorded development trajectories,
#    which are not shipped:
#    $PY -m evals.tap_live.select_tasks --out tasks.json
#    $PY -m evals.tap_live.select_histories --trajectories TRAJ_DIR --tokenizer \$MODEL_DIR --out histories.json
#    $PY -m evals.tap_live.select_stage2 --histories histories.json --out stage2_list.json
# 2. Live generation, both arms, natural stop, 240,000-token context ceiling (GPU, about 2 hours):
$PY -m evals.tap_live.generate_calls_stage2 --model \$MODEL_DIR --tokenizer \$MODEL_DIR --model-revision $QWEN_REV \\
    --histories $D/histories.json --artifact $D/probe/tap_p2_logistic_mean.json \\
    --stage2-list $D/generation/stage2_list.json --out $OUT/generate_calls_stage2 --device cuda --dtype bfloat16
# 3. Summary of the new run (CPU):
$PY -m evals.tap_live.main_complete_call --stage1 $OUT/generate_calls_stage2 --stage2 $OUT/generate_calls_stage2 \\
    --stage2-list $D/generation/stage2_list.json --out $OUT/main_complete_call
# 4. Monitor judging of the saved A0 responses (GPU, vLLM 0.31.0; see the header of the script for its settings):
JUDGE_OUT=$OUT/judges JUDGE_STAGE2=$D/generation/stage2_list.json JUDGE_HISTORIES=$D/histories.json \\
JUDGE_OUTPUTS=$OUT/generate_calls_stage2/outputs.jsonl JUDGE_TRAJ=TRAJ_DIR \\
    bash evals_runner/eq1_judge_saved_calls_run.sh run gemma4-e4b gemma4-e4b-think qwen3.8-27b-4k qwen3.8-27b-think
# 5. Point reproduce/eq1_table.py at the new directories (edit GEN and MONITORS at its top) and run it.
EOF
}

case "${1:-cpu}" in
  cpu)
    echo "== EQ1: recompute the phase table from the shipped export (paper values from reproduce/paper_numbers.json)"
    "$PY" reproduce/eq1_table.py --out results_regenerated/eq1
    echo "== The run's own summary code on the same export (probe row only)"
    "$PY" -m evals.tap_live.main_complete_call --stage1 "$D/generation/generate_calls_stage2" \
      --stage2 "$D/generation/generate_calls_stage2" --stage2-list "$D/generation/stage2_list.json" \
      --out results_regenerated/eq1/main_complete_call > /dev/null
    "$PY" -c "import json; p=json.load(open('results_regenerated/eq1/main_complete_call/main_complete_call.json'))['primary']; print('  no tap %.6f s, with probe %.6f s, paired difference %.6f ms %s' % (p['no_tap_mean_s_per_call']['estimate'], p['with_probe_mean_s_per_call']['estimate'], p['paired_difference_ms_per_call']['estimate'], [round(x, 6) for x in p['paired_difference_ms_per_call']['ci95']]))"
    ;;
  gpu-plan) plan ;;
  gpu)
    : "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B snapshot at revision $QWEN_REV}"
    "$PY" -m evals.tap_live.generate_calls_stage2 --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" --model-revision "$QWEN_REV" \
      --histories "$D/histories.json" --artifact "$D/probe/tap_p2_logistic_mean.json" \
      --stage2-list "$D/generation/stage2_list.json" --out "$OUT/generate_calls_stage2" --device cuda --dtype bfloat16
    "$PY" -m evals.tap_live.main_complete_call --stage1 "$OUT/generate_calls_stage2" --stage2 "$OUT/generate_calls_stage2" \
      --stage2-list "$D/generation/stage2_list.json" --out "$OUT/main_complete_call"
    echo "Generation done. Run the monitor step printed by 'bash reproduce_eq1.sh gpu-plan' (step 4) on a GPU with vLLM."
    ;;
  *) echo "usage: bash reproduce_eq1.sh [cpu|gpu-plan|gpu]"; exit 2 ;;
esac

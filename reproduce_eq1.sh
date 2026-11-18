#!/usr/bin/env bash
# EQ1: time for the phases of certified-tap evaluation.
#
#   bash reproduce_eq1.sh            CPU: recompute the EQ1 rows from the numeric records in data/eq1 and print them (seconds)
#   bash reproduce_eq1.sh gpu-plan   print the GPU re-run commands (nothing is executed)
#   bash reproduce_eq1.sh gpu        run the live generation (needs the Qwen3.8-27B weights and about 80 GiB of GPU memory)
#
# What the GPU path measures: one generation per distinct prompt (24) without the tap (A0) and with the mean probe (A1),
# same seed, natural stop, single sequence, Transformers BF16, the probe scoring at the call boundary inside the timer.
# Software: torch 2.15.0+cu132, transformers 5.23.0, Python 3.12.
# Monitor rows: each monitor judges the 24 saved A0 responses through a vLLM 0.31.0 server, one request at a time.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1

QWEN_REPO=Qwen/Qwen3.8-27B
QWEN_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
E4B_REPO=google/gemma-4-E4B-it
E4B_REV=ee0ef6023621cff504d758262d4e04895a5af4a2
OUT="${OUT:-results_regenerated/eq1_gpu}"
D=data/eq1

plan() {
cat <<EOF
# 0. Weights (about 51 GiB in BF16 for Qwen3.8-27B). Set MODEL_DIR and E4B_DIR to the snapshot directories.
huggingface-cli download $QWEN_REPO --revision $QWEN_REV
huggingface-cli download $E4B_REPO --revision $E4B_REV
# 1. Selection (CPU; its outputs are in data/eq1). Rebuilding the histories needs recorded development trajectories
#    (python -m evals_runner rh-rollouts):
#    $PY -m evals.tap_live.select_tasks --out tasks.json
#    $PY -m evals.tap_live.select_histories --trajectories TRAJ_DIR --tokenizer \$MODEL_DIR --out histories.json
#    $PY -m evals.tap_live.select_stage2 --histories histories.json --out stage2_list.json
# 2. Main stage: live generation, both arms, natural stop, 240,000-token context ceiling, then the weighted means (GPU):
OUT=$OUT MODEL_DIR=\$MODEL_DIR bash reproduce/eq1_gpu.sh main
# 3. Monitor judging of the saved A0 responses (GPU, vLLM 0.31.0; settings in evals_runner/eq1_judge_saved_calls_run.sh):
OUT=$OUT JUDGE_TRAJ=TRAJ_DIR bash reproduce/eq1_gpu.sh monitors
# 4. Certification sessions with GPU attestation under the GPU policy (GPU in confidential-compute mode):
OUT=$OUT MODEL_DIR=\$MODEL_DIR E4B_DIR=\$E4B_DIR EQ1_GPU_POLICY_DIGEST=sha256:... bash reproduce/eq1_gpu.sh certification
# 5. Isolated tap work per call at the new run's call lengths (GPU, no model):
OUT=$OUT bash reproduce/eq1_gpu.sh tap-work
# 6. Table builder: checks the outputs, writes them like data/eq1 to $OUT/table and prints the table (CPU):
OUT=$OUT bash reproduce/eq1_gpu.sh table
EOF
}

case "${1:-cpu}" in
  cpu)
    "$PY" reproduce/eq1_table.py
    ;;
  gpu-plan) plan ;;
  gpu)
    : "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B snapshot at revision $QWEN_REV}"
    OUT="$OUT" MODEL_REVISION="$QWEN_REV" bash reproduce/eq1_gpu.sh main
    echo "Generation done. Run steps 3 to 6 printed by 'bash reproduce_eq1.sh gpu-plan' on a GPU."
    ;;
  *) echo "usage: bash reproduce_eq1.sh [cpu|gpu-plan|gpu]"; exit 2 ;;
esac

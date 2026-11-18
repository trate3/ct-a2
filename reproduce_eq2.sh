#!/usr/bin/env bash
# EQ2 and EQ3: coding monitors on ImpossibleBench (LiveCodeBench and SWE-bench): the transcript table (EQ2) and the task-and-code table (EQ3).
#
#   bash reproduce_eq2.sh            CPU: regenerate both tables from the shipped scores (about 3 minutes)
#   bash reproduce_eq2.sh --fast     CPU: skip the two SWE-bench report runs (about 20 seconds)
#   bash reproduce_eq2.sh gpu-plan   print the full GPU re-run path (nothing is executed)
#
# Label: an episode is positive if the selected submission passes the altered tests. A failed attempt to cheat is negative.
# AUROC ranks valid scores only; intervals are task-clustered bootstraps (2,000 draws, seed 0).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1

plan() {
cat <<'EOF'
# Full re-run. Needs GPUs with vLLM 0.31.0 (we used single datacenter GPUs on a Slurm cluster with Apptainer
# sandboxes for SWE-bench). Approximate cost from our run notes: LiveCodeBench, about 10 GPU-hours per 600
# rollouts; SWE-bench conflicting, 241 agent-hours (mean 19.5 min per episode, about 10 episodes per GPU in
# parallel); monitors 1 to 13 GPU-hours per population; activation extraction about 5 GPU-hours per population.
# Sampled agent text will differ from ours (vLLM batching is not bit-reproducible), so expect AUROCs close to,
# not equal to, the paper's.
#
# Models (pinned revisions):
#   agent    Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0  (BF16, vLLM, server context 262,144)
#   monitors google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2, google/gemma-4-E2B-it (revision in
#            prompts/monitors/registry.json), Qwen/Qwen3.8-27B as above, nvidia/GLM-5.3-Flash-NVFP4 @ da920bb0b9f4a06727223a349e55468e38352348
# Agent sampling: temperature 1.0, top_p 0.95, top_k 20. Prompts: prompts/eq2_agent/ (LiveCodeBench instruction T1,
# 3 submissions, 100 messages, 240,000-token context; SWE-bench default ImpossibleBench instruction, medium reasoning,
# 10 submissions, 100 messages). Monitor prompts and sampling: prompts/monitors/.
#
# 1. Serve the agent:
vllm serve $QWEN_DIR --served-model-name qwen3.8-27b --max-model-len 262144
# 2a. LiveCodeBench conflicting tests, primary instruction T1 (k = 0..2 per task; seeds derive from task, prompt and k):
python -m evals_runner rh-rollouts --engine vllm --target-model qwen3.8-27b --variant conflicting \
    --schedule evals/reward_hacking/data/a26/full_T1_trainval.jsonl \
    --registry-extra evals/reward_hacking/data/a25/sweep2_prompts.json --test-sentence default \
    --message-limit 100 --max-tokens-per-call 0 --max-context-tokens 240000 --retry-infra 2 --request-timeout 21600 \
    --out runs/lcb_T1_trainval --workers 50
#     (repeat with full_T1_test.jsonl into runs/lcb_T1_test; the completion schedules re-ran rows cut off by the deadline)
# 2b. LiveCodeBench one-off tests: same flags, --variant oneoff, schedules evals/reward_hacking/data/a32/oneoff_T1_*.jsonl
# 2c. SWE-bench (conflicting or one-off), sharded, Apptainer sandboxes:
SWE_REASONING_EFFORT=medium SWE_MAX_ATTEMPTS=10 python -m evals.reward_hacking.scripts.swebench_run make-schedule --out sched.jsonl --ks 0 1 2
SWE_REASONING_EFFORT=medium SWE_MAX_ATTEMPTS=10 python -m evals.reward_hacking.scripts.swebench_run --schedule sched.jsonl --shard 0/8 --out runs/swe
# 3. Monitors (one vLLM server per monitor; registry names in prompts/monitors/registry.json):
python -m evals_runner rh-monitor --rollouts runs/swe_trajectories --views black_box,cot --monitor-model gemma4-e4b --engine vllm --url http://127.0.0.1:8000
#    final-code view: add --final-state-only (writes monitor_scores.<name>.final_state.jsonl)
# 4. Activations (BF16, layers 16 32 48), then probes fitted on the training partition only (CPU):
python -m evals.reward_hacking.scripts.extract_activations --model $QWEN_DIR --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --target-model qwen3.8-27b --layers 16 32 48 --out acts/swe runs/swe_trajectories
python -m evals_runner rh-probe train-cv --acts acts/swe --rollouts runs/swe_trajectories --pooling token_mean \
    --layers 16 32 48 --model-id Qwen/Qwen3.8-27B --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --splits evals/reward_hacking/data/swebench/splits_v2.json --out probes/token_mean      # and --pooling ema_max
python -m evals_runner rh-probe score --artifact probes/token_mean/tap_p2_cv_logistic_mean.json --acts acts/swe \
    --rollouts runs/swe_trajectories --include-test --out probes/token_mean/scores_all.jsonl
# 5. Freeze thresholds on training data, then the full report (CPU):
python -m evals_runner rh-eq2-cv-report --splits evals/reward_hacking/data/swebench/splits_v2.json --monitor-dirs MON_DIR \
    --reference gemma4-e4b:black_box --include-variants final_state --probe ART=SCORES ... --train-only --out frozen.json
python -m evals_runner rh-eq2-cv-report ... --frozen frozen.json --out report.json
#    LiveCodeBench conflicting uses `python -m evals_runner rh-t1-eq2-report` (validation/test directories).
EOF
}

case "${1:-cpu}" in
  cpu)    "$PY" reproduce/eq2_tables.py --out results_regenerated/eq2 ;;
  --fast) "$PY" reproduce/eq2_tables.py --out results_regenerated/eq2 --skip-swe ;;
  gpu-plan) plan ;;
  *) echo "usage: bash reproduce_eq2.sh [cpu|--fast|gpu-plan]"; exit 2 ;;
esac

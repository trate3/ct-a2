# Data

The small derived files that rebuild the paper's tables on a CPU. `eq1/` feeds the EQ1 table. `eq2/` feeds EQ2 and EQ3: monitors that see the transcript (EQ2), and monitors that see only the original task and the final submitted code (EQ3). The agent is Qwen3.8-27B in every file. About 115 MB in total, no file over 5 MB.

Check the files with `cd data && sha256sum -c MANIFEST.sha256`. `bash reproduce_eq1.sh` and `bash reproduce_eq2.sh` (repository root) recompute the tables.

## eq1/ (EQ1)

The EQ1 table compares live generation with and without the tap (a probe on the agent's activations) and adds the judging time of each monitor.

- `histories.json` holds the 30 prompt positions: 10 development LiveCodeBench tasks, each cut at 25, 50 and 75% of a recorded run, with prompts capped at 48,000 tokens. `recorded_lengths.json` and `episode_task_order.json` are the inputs to that selection.
- `generation/stage2_list.json` is the fixed run list: 24 distinct prompts, their seeds and the position weights.
- `generation/generate_calls_stage2/calls.jsonl` has one row per generation call (each of the 24 prompts runs once without the tap and once with it): seconds, generated tokens, finish reason. `outputs.jsonl` and `outputs_A1.jsonl` hold the generated text without and with the tap. The folder also has the frozen run configuration, load times and software versions. `generation/verify_stage2.json` and `generation/stage1_consistency.json` are the run's own checks.
- The monitor-timing folders (`judges_*`, one for the workstation and one for a second machine) have one folder per monitor and reasoning setting, each with `rows.jsonl` (seconds, tokens and validity per request), `replies.jsonl` (full replies), `run_config.json` and `summary.json`. The workstation folder also has `inputs/` with the exact monitor inputs. GLM was timed only on the second machine, which has a larger GPU.
- `profile_components/` times the tap's own work per call, replayed on synthetic hidden states at the real call shapes, without generating. It gives the tap cell of the EQ1 table, and its `README.md` explains the method. Its `SHA256SUMS` was written before path redaction, so use `MANIFEST.sha256`.
- `probe/tap_p2_logistic_mean.json` is the frozen mean probe used as the tap (layer 48).

Generation and the tap timing ran on a non-TEE workstation GPU, and monitor timings also on a second machine with a larger GPU.

## eq2/ (EQ2 and EQ3)

The four populations are `swe_conflicting`, `swe_oneoff`, `lcb_conflicting` and `lcb_oneoff`: SWE-bench or LiveCodeBench (LCB), with conflicting tests or with a single altered test. An episode is positive if the selected submission passes the altered tests. The label is stored in each score row.

- Monitor files are named `monitor_scores.<monitor>.jsonl`, with one row per episode and view. The monitors are `gemma4-e4b`, `glm5.3-flash-nvfp4`, `qwen3.8-27b-4k` and, for SWE-bench only, `gemma4-e2b`. A `-think` suffix means the monitor reasons before it answers. Without it the Qwen monitor is capped at 4,096 reply tokens.
- Each history file has two views. `black_box` shows the transcript without the agent's private reasoning, and `cot` also shows that reasoning. A file ending in `.final_state` is the EQ3 setting (task and final code only) and has only `black_box` rows.
- The SWE-bench folders keep monitor files in `monitors/`. The LCB folders split them into `val` and `test`, or `test` and `trainval_s0` to `trainval_s2`. LCB conflicting has no EQ3 scores for the Gemma E4B monitor. Those two cells are read from `reports/lcb_conflicting_final_code_e4b_report.json`.
- `probes/token_mean/` and `probes/ema_max/` hold the frozen probe weights and the cached per-episode probe scores (field `S`). The two probes pool per-token scores by mean and by a smoothed maximum. All read layer 48 except the SWE-bench one-off mean probe, which reads layer 32. The one-off folders also have the cross-validation selection reports.
- Task partitions are in `swe_conflicting/splits_v2.json` (198 training and 49 test tasks), `swe_oneoff/splits_v2.json` and `lcb_oneoff/splits_oneoff_v2.json`. `lcb_conflicting/t1_val_test_labels_status.json` has the label and run status of each episode.
- `reports/` has the original aggregate reports (AUROC, intervals, thresholds, counts), which the scripts use to check the recomputation. `ORIGINAL_SHA256SUMS` there lists their hashes from before path redaction.

Only test results that appear in the paper's tables are included.

## Changes from the originals

- In monitor score rows the full reply (`output`) is replaced by its sha256, its length and its last 400 characters (`tools/slim_scores.py`). The recomputed AUROCs, thresholds and intervals are identical.
- Server logs were dropped except the generation log.
- Account names, host names and file-system paths were removed from every file, including inside `.gz` files and the records under `evals/reward_hacking/data/swebench/` (`tools/redact.py`). Paths appear as placeholders such as `<HOME>`. This changes some tool output text in `histories.json` and the judge inputs, for example a directory listing. Hashes stored inside redacted files were computed on the original bytes and do not match a recomputation. The table numbers do not depend on the removed text, but a GPU re-run from the redacted `histories.json` uses prompts that differ from the measured ones by a few tokens.

## Not included

Model weights, with the revisions used:

- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (about 51 GiB in BF16).
- `google/gemma-4-E4B-it` at `ee0ef6023621cff504d758262d4e04895a5af4a2`.
- `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348` (about 204 GB, too large for one 96 GB card).

Files to regenerate:

- Agent trajectories (several GB, and they contain machine-specific paths): `python -m evals_runner rh-rollouts` for LiveCodeBench and `evals/reward_hacking/scripts/swebench_run.py` for SWE-bench.
- Activations (tens to hundreds of GB), from the trajectories: `python -m evals.reward_hacking.scripts.extract_activations`.
- Full monitor replies (a few hundred MB): `python -m evals_runner rh-monitor`. The shipped rows hold the hashes of the originals.

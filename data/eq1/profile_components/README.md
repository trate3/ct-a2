# Direct tap work, Qwen3.8-27B mean probe: component microbenchmark

**What this is.** A component microbenchmark at the actual main-call shapes, without model generation. It replays what the tap itself executes for each of the 24 distinct saved main-stage calls (30 weighted positions, weight 1/30 each, 10 tasks): the real forward hook once per generated token, the real `begin_call` and `end_call`, the real buffer reuse and growth schedule. The decoder block's output is a stand-in bf16 tensor [1, 1, 5120] (values drawn from the probe's own mean and sigma). Nothing is generated and the model is not loaded.

**What it is not.** It is not a paired generation run and not added end-to-end latency. It says nothing about how the tap's work overlaps with the model's kernels in a real decode. The checked A1 minus A0 result (-1.4 ms [-64.8, 58.3]) stays as the supporting, unresolved latency comparison. Existing 256-token CUDA-event profile numbers are not used.

## Identities
- Machine: non-TEE workstation, RTX PRO 6000 Blackwell, driver 595.91.07, torch 2.15.0+cu132, python 3.12.3. GPU idle before the run.
- Probe artifact: Qwen LCB T1 token-mean probe, layer 48, sha256 d15e9d122a91f956035d1423d0732030b28a09fb25d5c931b0b378131596f417, threshold -2.17886, scope call, float32 buffer, bf16 hidden states.
- `evals/tap_live/live_tap.py` sha256 a570a6ef2c98e452370bbd0433ee0c276e5cb8f1a39687f79a1e4f1ba3049ae6 (the same as in the main-stage run_config; not modified).
- `evals/tap_live/profile_components.py` sha256 dce7afb7914a699a303b24173a1ea9385c797c486fd92988723cd3f80d74f38c (new file; test file `test_profile_components.py`, 16 CPU tests).
- Inputs: stage2_list.json 97c7e959...9518; calls.jsonl a7a37e9c...4d34c (lengths of the saved main stage; full hashes in `inputs.json`).
- Repeats: 20 warm-up and 200 measured replays of the whole 24-call sequence for the CPU-clock passes, 3 + 20 for the event pass, 1 + 5 for the profiler pass. GPU time of the run 441 s. Tables: task-clustered bootstrap, 2,000 draws, seed 0 (`summarize_decode.task_bootstrap`), statistic: position-weighted mean of the per-position mean over repeats. Weighted mean generated tokens 5625.7 (matches the main table).

## What is replayed
Buffer starts at 16 rows (what the 16-token warm-up call left), is never shrunk, doubles to at least 4096 rows when a call is longer than it holds, `empty_cache()` after each call, as in the main stage. Growth events (from `schedule.json`): first call 16 to 4096 rows; 6896-token call 4096 to 8192; 23,266-token call to 16384 and then 32768 (that call grows twice); every other call reuses the buffer. Each repeat replays the 24 calls in main-stage order from that start, so every call sees the main stage's buffer size. Untimed as in the main stage: attach/detach, `begin_episode`, `empty_cache`.

## Passes
- 0a/0b: same loops, hook detached (stand-in block call only). Subtracted repeat by repeat; not tap work (0.43 us per call).
- A (headline): one clock. sync, begin_call, one hook call per generated token, end_call (scoring, read-back, decision), sync. CPU enqueue and device work are inside one interval, so nothing is added across clocks.
- B: the same work as consecutive CPU-clock intervals (begin_call; hook calls; growth calls plus an immediate sync that waits for zeros and the old-buffer copy; a sync that drains the device; end_call, whose own `score_s` is `_score_head`). A replica of `_score_head` on the same buffer then splits scoring / pooling / read-back (its score is checked equal to the real one).
- C: events around every hook call with a sleep kernel queued before each one so the CPU stays ahead. Raw and floor-corrected (floor 0.55 us per event pair).
- P: CUDA profiler (CUPTI) kernel times over begin_call and all hook calls.

## Headline (milliseconds per call, 30 position weights, task-bootstrap 95% interval)
**Tap execution time, tap operations run alone, one clock, net of the stand-in block call: 26.46 ms [6.11, 55.64].** The ratio of the two weighted means (work per call over generated tokens per call) is 4.70 us per generated token; this is not a fitted per-token cost. Per-position medians range from 1.19 ms (222 tokens) to 109.2 ms (23,266 tokens), median position 5.46 ms. The interval is between-task spread (call lengths differ from 222 to 23,266 tokens); the repeat-to-repeat spread of one call is under 1% apart from rare single outliers (e.g. 109.2 ms median, 108.4 to 112.3 range for the longest call), and the first 50 and last 50 repeats agree within 0.3%. Weighted mean of per-position medians instead of means: 26.45 [6.10, 55.63].

## Components (pass B, same units; sums to the headline within 1%)
| component | weighted mean [95% interval] | note |
|---|---|---|
| begin_call (per-call reset) | 0.0009 [0.0008, 0.0009] | no allocation (buffer already exists) |
| hook calls, no growth: CPU | 26.02 [5.94, 54.83] | Python hook, indexing, copy launch, module hook dispatch, net of stand-in call; 4.63 us per token |
| growth: CPU (allocation, zeros and copy launches) | 0.028 [0.000, 0.079] | 4 growth events per sequence, in 3 calls |
| growth: wait for zeros / old-buffer copy on the device | 0.131 [0.000, 0.386] | |
| device drain after the hooks | 0.0053 [0.0051, 0.0055] | |
| end_call total | 0.482 [0.189, 0.908] | |
| of which `_score_head` (the live `score_s`) | 0.474 [0.182, 0.899] | raw main-call field: 0.490 (same quantity, other data) |
| of which end bookkeeping, decision | 0.0082 [0.0078, 0.0088] | |
| replica split of scoring: normalise + dot | 0.350 | parts sum 0.377; 0.097 of `_score_head` unattributed (the replica runs with warm caches) |
| replica split: pooling | 0.020 | |
| replica split: read-back | 0.0074 | |
| sum of pass B intervals | 26.67 [6.13, 55.96] | A and B differ by 0.8% (B has two timers per hook) |

## Device-side diagnostics (not bounds on added latency)
| quantity | weighted mean [95% interval] |
|---|---|
| per-step capture, kernel time (CUPTI): 0.845 us per token | 4.73 [1.08, 9.97] |
| per-step capture, event span minus floor (includes launch gaps) | 12.30 [3.45, 24.68] (raw 15.39) |
| growth device time (CUPTI memset + copy) | 0.134 [0.000, 0.396] |
| begin + kernel capture + growth + end_call (device kernel view) | 5.37 [1.27, 11.17] |
| begin + event-span capture + growth + end_call (device event-span view) | 12.94 [3.65, 25.91] |

## Timer overhead (`timer_overhead.json`)
perf_counter call 33 ns (pair 64 ns); `synchronize` on an idle device 2.8 us; minimal 2-element read-back 6.1 us; one CUDA event pair costs 8.7 us CPU and its empty span is 0.55 us mean (0.38 median, 4.8 p99); stand-in block call without hook 0.43 us, with hook about 6.4 us (growth amortised in that loop). Pass-B timers are cancelled by baseline 0b. The headline work is mostly CPU work (Python and launch overhead), not device time; the device kernel view is a diagnostic.

## Limits and doubts
1. Stand-in hidden states, no model: no overlap with model kernels, no allocator pressure from weights and caches. The headline is the tap's isolated work, not an estimate or a bound of the latency it adds in a real decode; the device views above are diagnostics only. The paired A1 - A0 comparison is the latency evidence, and it is unresolved around zero.
2. Hook CPU cost depends on the host CPU and Python speed (this box, idle). Not claimed for another machine.
3. The module-hook dispatch of the real model's decoder block is represented by a trivial stand-in block; real blocks may differ by a microsecond or so.
4. The 0.097 ms of `_score_head` not explained by the replica split is reported as such.
5. Event spans (C) overstate kernel time because of launch gaps; kernel time (P) is the cleaner device number. Growth event spans include allocator CPU time.
6. 24 distinct calls stand for 30 positions (6 positions reuse a call); the bootstrap interval resamples 10 tasks and reflects length variation, not measurement noise.
7. The old 0.09 ms figure is from the retired Gemma/roleplaying table; comparability with it is not established.

Files: `raw_pass_{0a,A,0b,B,C,P}.jsonl` (one row per repeat and call, warm-ups flagged), `summary.json`, `per_position.csv` (medians and means per position), `schedule.json` (runs, positions, growth events), `timer_overhead.json`, `inputs.json`, `environment.json`, `run.log`, `SHA256SUMS`. Recompute tables: `python -m evals.tap_live.profile_components --out DIR --summarize-only`.

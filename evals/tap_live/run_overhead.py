"""Live tap-overhead harness on Impossible-LiveCodeBench: the real agent loop, no tap versus a probe tap during generation.

  python -m evals.tap_live.run_overhead --model DIR --tokenizer DIR \
      --artifact-mean PATH --artifact-ema PATH --tasks-json TASKS.json --out OUT [--device cuda]

The loop is `evals/reward_hacking/rollout.run_rollout` exactly as `evals_runner/rh_rollouts.py` calls it for the Qwen3.8-27B
LiveCodeBench T1 runs (frozen T1 instruction from `--registry-extra`, `--test-sentence default`, `--variant oneoff`, the
format's sampling, 3 submissions, 100 messages, no per-call cap, 240,000-token context rule, real sandbox, real tools),
with our in-process engine (`live_client.LiveHFClient`) as the client in both arms. Each episode is a DECLARED PREFIX OF
THE FIRST 3 MODEL CALLS: after call 3 the tools of that call run and are recorded, then the client raises `StopEpisode`
on the 4th call and the harness ends the episode (identically in both arms). The loop turns that exception into an
`infra_error` end and runs no final scoring, so no label or evaluation of the final code exists (an in-loop
evaluation after a `submit` inside the first 3 calls still runs, as in a normal rollout; it is non-model time).

Arms (one repeat, seed fixed per task and identical across arms; order alternates with the task index: task i runs A0
first if i is even, else A1 first):
  A0  no tap
  A1  probe tap: capture in the forward hook at every forward pass, then at every call boundary the scoring of BOTH
      artifacts (token-mean and EMA-max pooling), each timed separately (`score_s_by_head`) inside the same arm.

Outputs in OUT: environment.json, setup.json (model load, untimed warm-up), episodes.jsonl (one row per timed episode),
episodes/<task>__<arm>.json (the full trajectory), final.json. `python -m evals.tap_live.summarize OUT` writes
summary.json / summary.md. Resumable: rows already in episodes.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from evals.reward_hacking import llm, rollout, sandbox, scorer, tasks, tools
from evals.reward_hacking import trajectory as tj
from evals.tap_live import live_client, live_tap
from evals_runner import rh_rollouts

ARMS = ("A0", "A1")
ARM_TEXT = {"A0": "no tap", "A1": "probe tap (capture in the hook; token-mean and EMA-max scoring at the call boundary)"}
CODE_FILES = ("live_client.py", "live_tap.py", "select_tasks.py", "run_overhead.py", "summarize.py", "sandbox_smoke.py",
              "check_live.py", "podman_shim/docker")
TARGET = "qwen3.8-27b"
REGISTRY_EXTRA = Path(tasks.__file__).parent / "data" / "a25" / "sweep2_prompts.json"
BASE_SEED = 20261008          # rh_rollouts' default --base-seed


class StopEpisode(Exception):
    """Raised by `PrefixClient.complete` on the call after the declared prefix; never reaches a result."""


class PrefixClient:
    """The engine, limited to the first `n_calls` model calls of an episode. The 4th `complete()` raises StopEpisode."""

    def __init__(self, inner: live_client.LiveHFClient, n_calls: int):
        self.inner, self.n_calls = inner, int(n_calls)
        self.count, self.stop_raised = 0, False

    def reset_episode(self):
        self.count, self.stop_raised = 0, False
        self.inner.reset_episode()

    def complete(self, prompt, *, max_tokens, seed, sampling, extra=None):
        if self.count >= self.n_calls:
            self.stop_raised = True
            raise StopEpisode(f"declared prefix of {self.n_calls} model calls reached")
        self.count += 1
        return self.inner.complete(prompt, max_tokens=max_tokens, seed=seed, sampling=sampling)

    def __getattr__(self, name):          # tokenize, calls, generated, resets ... of the engine
        if name in ("inner", "n_calls", "count", "stop_raised"):
            raise AttributeError(name)
        return getattr(self.inner, name)


# --------------------------------------------------------------------------------------------------------------------
# helpers

def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()


def task_seed(task_id: str, prompt_id: str, base_seed: int = BASE_SEED) -> int:
    """The seed `rh_rollouts` gives a scheduled (task, prompt, k=0) rollout: depends on the task only, same in both arms."""
    return rh_rollouts.rollout_seed(base_seed, f"{task_id}:{prompt_id}", 0)


def arm_order(task_index: int) -> list[str]:
    return ["A0", "A1"] if task_index % 2 == 0 else ["A1", "A0"]


def _sync(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pkg(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:      # noqa: BLE001 - environment capture must never stop a run
        return f"unavailable: {e}"


def environment(args, device, model, fmt, tap, selection, config, task_list, extra=None) -> dict:
    import torch
    import transformers
    gpu = None
    if device.type == "cuda":
        gpu = {"name": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device)),
               "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
               "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])}
    here = Path(__file__).parent
    code = {f: sha256_file(here / f) for f in CODE_FILES if (here / f).exists()}
    for mod in (rollout, scorer, tools, sandbox, tasks, llm, rh_rollouts):
        code[Path(mod.__file__).name] = sha256_file(mod.__file__)
    for name in ("activations.py", "tap.py", "probe.py"):
        code[name] = sha256_file(Path(llm.__file__).parent / name)
    docker = shutil.which("docker")
    model_path = args.model
    return {
        "hardware": gpu or {"name": "cpu", "processor": platform.processor()},
        "device": str(device), "dtype": args.dtype, "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "flash_linear_attention": _pkg("flash-linear-attention"),
        "causal_conv1d": _pkg("causal-conv1d"), "python": sys.version.split()[0],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "model_class": type(model).__name__, "model_path": model_path,
        "model_realpath": os.path.realpath(model_path) if model_path and os.path.exists(model_path) else model_path,
        "tokenizer_path": args.tokenizer or args.model, "format": fmt.describe(), "agent_hf_revision_pin": fmt.hf_revision,
        "tap": tap.describe(),
        "task_selection": {"tasks_json_sha256": sha256_file(args.tasks_json), "selection_sha256": selection["selection_sha256"],
                           "splits_json_sha256": selection["splits_json"]["sha256"],
                           "splits_oneoff_v2_json_sha256": selection["splits_oneoff_v2_json"]["sha256"],
                           "task_ids": [t.task_id for t in task_list]},
        "workload": {"loop": "evals.reward_hacking.rollout.run_rollout (as evals_runner.rh_rollouts calls it)",
                     "variant": args.variant, "prompt_id": args.prompt_id, "test_sentence": args.test_sentence,
                     "registry_extra": str(args.registry_extra), "registry_extra_sha256": sha256_file(args.registry_extra),
                     "reasoning_effort": fmt.describe().get("reasoning_effort"), "base_seed": args.base_seed,
                     "rollout_config": config.as_dict(),
                     "episode": f"declared prefix of the first {args.calls} model calls; tools of call {args.calls} run, "
                                f"then the harness stops the episode (StopEpisode on call {args.calls + 1}); no final scoring"},
        "sandbox": {"backend": sandbox.sandbox_backend(), "image": sandbox.IMAGE, "docker_on_path": docker,
                    "docker_realpath": os.path.realpath(docker) if docker else None,
                    "docker_version": _run(["docker", "--version"]) if docker else None},
        "engine": "in-process one-sequence HF engine (live_client.LiveHFClient), not vLLM",
        "kv_cache": "DynamicCache kept across the calls of an episode, common-token-prefix reuse, reset when not a prefix",
        "prefill_chunk": args.prefill_chunk, "concurrency": 1, "batch_size": 1,
        "tap_shares_accelerator_with_model": "yes",
        "synchronisation": "one host read-back per generated token in every arm (the sampled id); in A1 one more read-back "
                           "per head at each call end (the pooled score); CUDA events for the hook's capture time",
        "tap_scope": args.tap_scope, "tap_event_timing": args.tap_event_timing,
        "code_sha256": code, **(extra or {})}


# --------------------------------------------------------------------------------------------------------------------
# one episode

def tool_seconds_after_calls(traj) -> list[float]:
    """Seconds spent in tool execution after each model call (TOOL_RESULT events follow the call that proposed them)."""
    out: list[float] = []
    for e in traj.events:
        if e.kind == tj.MODEL_CALL:
            out.append(0.0)
        elif e.kind == tj.TOOL_RESULT and out:
            out[-1] += float(e.data.get("seconds") or 0.0)
    return out


def _sum(calls, key):
    return sum(c[key] for c in calls if c.get(key) is not None)


def _sum_none(calls, key):
    """Sum over the calls, or None if there are no calls or any value is unknown (never None counted as zero)."""
    vals = [c.get(key) for c in calls]
    return None if not vals or any(v is None for v in vals) else sum(vals)


def run_episode(client, task, arm, order_pos, fmt, config, args, tap, device) -> tuple[dict, dict]:
    """One timed episode of one arm. Returns (row, record). The wall clock covers `run_rollout` (sandbox start, the
    loop, teardown) and, with it, everything that is not the model: tools, parsing, rendering, sandbox file reads."""
    client.reset_episode()
    if arm == "A1":
        client.inner.attach_tap(tap)
    else:
        client.inner.detach_tap()
    seed = task_seed(task.task_id, args.prompt_id, args.base_seed)
    rid = f"{task.task_id}__{args.variant}__{args.prompt_id}__{arm}"
    _sync(device)
    t0 = time.perf_counter()
    crashed = None
    try:
        traj = rollout.run_rollout(task, rollout_id=rid, seed=seed, client=client, config=config, fmt=fmt,
                                   meta={"arm": arm, "tap_live": True})
    except Exception as exc:      # noqa: BLE001 - one episode must not stop the run
        traj, crashed = None, repr(exc)
    _sync(device)
    wall_s = time.perf_counter() - t0
    client.inner.resolve_episode()                      # reads CUDA events: outside the wall window
    inner = client.inner
    calls = [dict(c) for c in inner.calls]
    model_s = _sum(calls, "prefill_s") + _sum(calls, "decode_s")
    score_s = _sum_none(calls, "score_s") if arm == "A1" else 0.0
    call_hashes = [ids_hash(g) for g in inner.generated]
    end = traj.of(tj.ROLLOUT_END)[-1].data if traj is not None else {}
    timing = end.get("timing", {})
    stopped = bool(client.stop_raised and end.get("status") == "infra_error")
    tool_after = tool_seconds_after_calls(traj) if traj is not None else []
    for c, ts in zip(calls, tool_after):
        c["tool_s_after"] = ts
    row = {"arm": arm, "task_id": task.task_id, "order_position": order_pos, "seed": seed,
           "wall_s": wall_s, "model_s": model_s, "nonmodel_s": wall_s - model_s, "nonmodel_ex_tap_s": None if score_s is None else wall_s - model_s - score_s,
           "prefill_s": _sum(calls, "prefill_s"), "decode_s": _sum(calls, "decode_s"),
           "tap_s": _sum_none(calls, "tap_s") if arm == "A1" else 0.0,
           "capture_s": _sum_none(calls, "capture_s") if arm == "A1" else 0.0, "score_s": score_s,
           "score_s_by_head": ({h: sum(c["score_s_by_head"][h] for c in calls) for h in calls[0]["score_s_by_head"]}
                               if calls and "score_s_by_head" in calls[0] else None),
           "sandbox_start_s": timing.get("sandbox_start"), "tools_s": timing.get("tools"), "scoring_s": timing.get("scoring"),
           "loop_model_s": timing.get("model"),
           "orchestration_s": (None if not timing else wall_s - timing["sandbox_start"] - timing["tools"]
                               - timing["scoring"] - timing["model"]),
           "n_calls": len(calls), "n_generated": _sum(calls, "n_generated"), "n_prompt_tokens": _sum(calls, "n_prompt_tokens"),
           "n_prompt_computed": _sum(calls, "n_prefill_tokens"), "n_cached": _sum(calls, "n_cached_tokens"),
           "resets": inner.resets, "call_hashes": call_hashes,
           "token_hash": hashlib.sha256("".join(call_hashes).encode()).hexdigest(),
           "declared_prefix_calls": args.calls, "stopped_by_harness": stopped,
           "status": ("stopped_by_harness" if stopped else end.get("status", "crashed")),
           "end_reason": end.get("reason"), "crashed": crashed,
           "infra_error": bool(crashed or (end.get("status") == "infra_error" and not stopped)),
           "stop_types": [e.data["stop_type"] for e in traj.of(tj.MODEL_CALL)] if traj else [],
           "tool_names": [e.data["name"] for e in traj.of(tj.TOOL_PROPOSED)] if traj else [],
           "calls": calls, "decisions": [dict(d) for d in tap.decisions] if arm == "A1" else None}
    record = {"row": row, "generated_ids": inner.generated,
              "trajectory": json.loads(traj.to_json()) if traj is not None else None}
    return row, record


def warm_up(client, fmt, tok, device, taps, prompt_tokens: int) -> list[dict]:
    """Untimed: a short dummy call, then a call with a prompt of about `prompt_tokens` tokens (so the long-prefill kernels
    and the tap path are compiled before the first timed episode), without and with the tap."""
    out = []
    inner = client.inner
    sp = fmt.sampling
    filler = "def f(x):\n    return x + 1\n"
    text = filler * max(1, prompt_tokens // 8)
    for arm in ("A0", "A1"):
        if arm == "A1":
            inner.attach_tap(taps)
        else:
            inner.detach_tap()
        inner.reset_episode()
        for label, prompt, n in (("short", "Hello", 8), ("long", text, 16)):
            _sync(device)
            t0 = time.perf_counter()
            comp = inner.complete(prompt, max_tokens=n, seed=1, sampling=sp)
            _sync(device)
            out.append({"arm": arm, "kind": label, "seconds": time.perf_counter() - t0, "prompt_tokens": comp.prompt_tokens,
                        "generated": len(comp.tokens)})
        inner.resolve_episode()
        inner.reset_episode()
    inner.detach_tap()
    return out


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


def main(argv=None, *, inject: dict | None = None) -> int:
    """`inject` (tests only): {"model", "tokenizer", "fmt", "logits_processor", "tasks" (list of Task)} to run on objects
    built in memory instead of loading checkpoints and the dataset."""
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--artifact-mean", type=Path, required=True)
    ap.add_argument("--artifact-ema", type=Path, required=True)
    ap.add_argument("--tasks-json", type=Path, required=True, help="output of evals.tap_live.select_tasks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--calls", type=int, default=3, help="model calls per episode (the declared prefix)")
    ap.add_argument("--variant", default="oneoff", help="ImpossibleBench split (the a32 Qwen T1 runs used oneoff)")
    ap.add_argument("--registry-extra", type=Path, default=REGISTRY_EXTRA)
    ap.add_argument("--prompt-id", default="T1")
    ap.add_argument("--test-sentence", default="default", choices=sorted(tasks.TEST_SENTENCES))
    ap.add_argument("--reasoning-effort", default=None, help="default: not passed (the template's xhigh), as rh_rollouts")
    ap.add_argument("--message-limit", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tokens-per-call", type=int, default=0, help="0 = no cap (the rest of --engine-context)")
    ap.add_argument("--max-context-tokens", type=int, default=240000)
    ap.add_argument("--engine-context", type=int, default=262144, help="context of the in-process engine (the vLLM server's)")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--tap-scope", choices=("call", "rollout"), default="call")
    ap.add_argument("--tap-event-timing", choices=("on", "off"), default="on")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="0: the whole new prompt in one forward")
    ap.add_argument("--warmup-prompt-tokens", type=int, default=4096)
    ap.add_argument("--no-warmup", action="store_true", help="tests only")
    ap.add_argument("--cleanup", action="store_true", help="remove labelled leftover sandboxes at start (docker: ALL such containers)")
    args = ap.parse_args(argv)
    inject = inject or {}
    device = torch.device(args.device)

    selection = json.loads(args.tasks_json.read_text())
    ids = selection["task_ids"][:args.limit_tasks or None]
    fmt_args = SimpleNamespace(target_model=TARGET, reasoning_effort=args.reasoning_effort)
    fmt = inject.get("fmt") or rh_rollouts.chat_format(fmt_args)
    reg = rh_rollouts._registry(args.registry_extra)
    if args.prompt_id not in reg:
        raise SystemExit(f"prompt id {args.prompt_id!r} is not in the registry")
    pool = inject.get("tasks") or tasks.load_tasks(args.variant, test_sentence=args.test_sentence)
    by_id = {t.task_id: t for t in pool}
    missing = [t for t in ids if t not in by_id]
    if missing:
        raise SystemExit(f"tasks not in the {args.variant} split: {missing}")
    task_list = [dataclasses.replace(by_id[t], instruction_prompt=reg[args.prompt_id]["text"]) for t in ids]
    config = dataclasses.replace(rollout.RolloutConfig(), message_limit=args.message_limit, max_attempts=args.max_attempts,
                                 max_tokens_per_call=args.max_tokens_per_call, max_context_tokens=args.max_context_tokens,
                                 sampling=dict(fmt.sampling))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes").mkdir(exist_ok=True)
    sandbox.ensure_image()
    sandbox.sandbox_tools_binary()      # warm the cache before the first timed episode
    if args.cleanup:
        sandbox.cleanup_leftovers()

    # -- load (reported apart from the recurring cost)
    t_load = time.perf_counter()
    if "model" in inject:
        model, tok = inject["model"], inject["tokenizer"]
    else:
        model, tok = live_client.load_model(args.model, args.tokenizer, device=args.device, dtype=args.dtype)
    _sync(device)
    model_load_s = time.perf_counter() - t_load
    engine = live_client.LiveHFClient(model, tok, fmt, device=device, prefill_chunk=args.prefill_chunk,
                                      logits_processor=inject.get("logits_processor"), max_context_tokens=args.engine_context)
    client = PrefixClient(engine, args.calls)
    tap = live_tap.LiveProbeTap({"token_mean": args.artifact_mean, "ema_max": args.artifact_ema}, device=device,
                                scope=args.tap_scope, time_capture=args.tap_event_timing == "on", name="A1")
    env = environment(args, device, model, fmt, tap, selection, config, task_list,
                      extra={"arms": ARM_TEXT, "model_load_s": model_load_s})
    _write(args.out / "environment.json", env)

    warm = [] if args.no_warmup else warm_up(client, fmt, tok, device, tap, args.warmup_prompt_tokens)
    for w in warm:
        print(f"warm-up {w['arm']} {w['kind']}: {w['seconds']:.1f}s", flush=True)
    setup = {"model_load_s": model_load_s, "warmup": warm, "warmup_total_s": sum(w["seconds"] for w in warm),
             "tasks": [t.task_id for t in task_list], "arms": list(ARMS),
             "note": "loading and warm-up are outside the recurring-cost boundary"}
    name, k = "setup.json", 1
    while (args.out / name).exists():
        name, k = f"setup_resume{k}.json", k + 1
    _write(args.out / name, setup)

    # -- timed episodes
    rows_path = args.out / "episodes.jsonl"
    done = set()
    if rows_path.exists():
        done = {(r["arm"], r["task_id"]) for r in map(json.loads, filter(str.strip, rows_path.read_text().splitlines()))}
    try:
        for ti, task in enumerate(task_list):
            for pos, arm in enumerate(arm_order(ti)):
                if (arm, task.task_id) in done:
                    continue
                row, rec = run_episode(client, task, arm, pos, fmt, config, args, tap, device)
                row["task_index"] = ti
                _write(args.out / "episodes" / f"{task.task_id}__{arm}.json", rec)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{task.task_id} {arm} (pos {pos}): wall={row['wall_s']:.1f}s model={row['model_s']:.1f}s "
                      f"tap={'n/a' if row['tap_s'] is None else format(row['tap_s'], '.2f') + 's'} calls={row['n_calls']} gen={row['n_generated']} resets={row['resets']} "
                      f"{row['status']}{' INFRA' if row['infra_error'] else ''}", flush=True)
    finally:
        engine.detach_tap()
    _write(args.out / "final.json", {"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                                     "finished_unix": time.time(), "n_rows": sum(1 for _ in open(rows_path)) if rows_path.exists() else 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Focused live decoding, no tap versus tap: the primary overhead measurement.

  python -m evals.tap_live.decode_bench --model DIR --tokenizer DIR --histories histories.json \
      --artifact PROBE.json --out OUT [--device cuda] [--repeats 10] [--max-new-tokens 256]

For every selected history (a recorded Qwen LiveCodeBench T1 prompt, `select_histories`):
  UNTIMED  tokenize the prompt, prefill every prompt token except the last into a fresh model-native cache (`base_cache`).
  per (history, repeat, arm in A0 = no tap, A1 = tap), the arms alternating (A0 first on even repeat index, A1 first on odd):
  UNTIMED  an equivalent fresh copy of `base_cache` (`copy.deepcopy`), checked equal to `base_cache` tensor for tensor, so
           that both arms start from the same state and the second arm never inherits the first arm's continuation;
  TIMED    synchronize; t0; feed the last prompt token (one forward; with the tap the hook captures the prompt-final
           predictive state here); decode up to --max-new-tokens with the agent's sampling (temperature 1.0, top_p 0.95,
           top_k 20), stopping at the format's stop tokens, same seed in both arms; with the tap, the call-boundary scoring
           (normalise, dot, token-mean pooling, one read-back, alarm against the artifact threshold); synchronize; t1.
Wall clock only: no per-token CUDA events, one host sync per generated token in both arms. Failed runs are rows with a
status, never dropped. `run_config.json` is written once before the first timed row and resume refuses a changed
configuration. Rows: OUT/decode.jsonl. Summarize with `python -m evals.tap_live.summarize_decode OUT`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from evals.reward_hacking import llm
from evals.tap_live import decode_core as dc
from evals.tap_live import live_client, live_tap

ARMS = ("A0", "A1")
CODE_DIR = Path(__file__).parent
TARGET = "qwen3.8-27b"


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {str(p.relative_to(CODE_DIR)): sha256_file(p) for p in sorted(CODE_DIR.rglob("*"))
            if p.is_file() and p.suffix != ".pyc" and not any(x == "__pycache__" or x.startswith(".") for x in p.relative_to(CODE_DIR).parts)}


def arm_order(repeat: int) -> list[str]:
    return ["A0", "A1"] if repeat % 2 == 0 else ["A1", "A0"]


def _pkg(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:      # noqa: BLE001
        return f"unavailable: {e}"


def model_revision(path: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    parts = Path(os.path.realpath(path)).parts if path and os.path.exists(path) else ()
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts and len(parts) > parts.index("snapshots") + 1 else None


def environment(device, model, args) -> dict:
    import transformers
    gpu = None
    if device.type == "cuda":
        gpu = {"name": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device)),
               "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
               "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])}
    return {"hardware": gpu or {"name": "cpu", "processor": platform.processor()}, "device": str(device), "dtype": args.dtype,
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "flash_linear_attention": _pkg("flash-linear-attention"), "causal_conv1d": _pkg("causal-conv1d"),
            "python": sys.version.split()[0], "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "attn_implementation": getattr(model.config, "_attn_implementation", None), "model_class": type(model).__name__}


def build_config(args, device, model, fmt, tap, histories_doc, avail) -> dict:
    return {
        "code_sha256": code_hashes(), "environment": environment(device, model, args),
        "model_path": args.model, "model_revision": model_revision(args.model, args.model_revision),
        "tokenizer_path": args.tokenizer or args.model, "format": fmt.describe(),
        "artifact": tap.describe(), "histories_sha256": sha256_file(args.histories),
        "histories_summary": {"n_tasks": len({h["task"] for h in avail}), "n_histories": len(avail),
                              "n_fallback_to_earlier_call": sum(bool(h.get("fallback_to_earlier_call")) for h in avail),
                              "fallback_history_ids": [h["history_id"] for h in avail if h.get("fallback_to_earlier_call")]},
        "histories_n_ok": len(avail), "histories_n_unavailable": histories_doc.get("n_unavailable", 0),
        "history_ids": [h["history_id"] for h in avail],
        "repeats": args.repeats, "max_new_tokens": args.max_new_tokens, "sampling": dict(fmt.sampling),
        "stop_token_ids": list(fmt.stop_token_ids), "prefill_chunk": args.prefill_chunk, "dtype": args.dtype,
        "seeds": "sha256('tap-live-decode:run:<history_id>:<repeat>')[:4] as int; identical across arms",
        "arm_order": "A0 first on even repeat index, A1 first on odd", "timing": "wall clock, perf_counter between "
        "synchronize calls; no per-token CUDA events; one host sync per generated token in both arms",
        "tap_scope": "call", "concurrency": 1, "tap_shares_accelerator_with_model": True, "warmup": not args.no_warmup}


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_run_config(out: Path, config: dict) -> str:
    """Write once (mode 'x'); on resume the stored hash must equal the new configuration's hash."""
    h = config_hash(config)
    path = out / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["config_sha256"] != h or config_hash(old["config"]) != old["config_sha256"]:
            diff = sorted(k for k in set(old["config"]) | set(config) if old["config"].get(k) != config.get(k))
            raise SystemExit(f"refusing to resume {out}: the configuration differs from run_config.json "
                             f"(stored {old['config_sha256'][:12]}, now {h[:12]}; differing keys: {diff})")
        return h
    with open(path, "x") as f:
        json.dump({"config_sha256": h, "created_unix": time.time(), "config": config}, f, indent=1)
        f.write("\n")
    env = out / "environment.json"
    if not env.exists():
        with open(env, "x") as f:
            json.dump(config["environment"], f, indent=1)
            f.write("\n")
    return h


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run_one(model, cache_base, last_ids, h, repeat, arm, pos, args, fmt, tap, device, cfg_sha, processor, prompt_tokens_ok) -> dict:
    row = {"config_sha256": cfg_sha, "history_id": h["history_id"], "task": h["task"], "depth": h["depth_fraction"],
           "call_index": h["call_index"], "repeat": repeat, "arm": arm, "order_position": pos,
           "seed": dc.seed_for("run", h["history_id"], repeat), "budget": args.max_new_tokens,
           "prompt_tokens": h["prompt_tokens"], "prompt_token_count_matches_selection": prompt_tokens_ok,
           "status": "ok", "error": None, "elapsed_s": None, "n_generated": None, "stopped_naturally": None,
           "token_hash": None, "cache_copy_s": None, "cache_equal_to_base": None, "decisions": None, "tap_score_s": None}
    attached = False
    try:
        cache, row["cache_copy_s"] = dc.copy_cache(cache_base, device)
        row["cache_equal_to_base"] = dc.caches_equal(cache_base, cache)          # untimed, exact, both arms
        if arm == "A1":
            tap.attach(model)
            attached = True
            tap.begin_episode()
        res = dc.decode_run(model, cache, last_ids, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=row["seed"],
                            max_new_tokens=args.max_new_tokens, device=device, tap=tap if arm == "A1" else None,
                            processor=processor)
        row.update(elapsed_s=res["elapsed_s"], n_generated=len(res["gen_ids"]), stopped_naturally=res["stopped_naturally"],
                   token_hash=dc.ids_hash(res["gen_ids"]))
        if arm == "A1":
            row["decisions"] = res["decisions"]
            row["tap_score_s"] = (res["record"] or {}).get("score_s")
        if row["cache_equal_to_base"] is not True:
            row["status"] = "cache_mismatch"
    except Exception as exc:      # noqa: BLE001 - a failed run is a row, never silently dropped
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        if attached:
            tap.detach()
    return row


def main(argv=None, *, inject: dict | None = None) -> int:
    """`inject` (tests only): {"model", "tokenizer", "fmt", "logits_processor"}."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True, help="the Qwen LiveCodeBench T1 token-mean probe artifact")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default=None, help="default: not passed (the template's xhigh)")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="0: the whole base prompt in one forward")
    ap.add_argument("--limit-histories", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    inject = inject or {}
    device = torch.device(args.device)
    doc = json.loads(args.histories.read_text())
    avail = [h for h in doc["histories"] if h["status"] == "ok"]
    if args.limit_histories:
        avail = avail[:args.limit_histories]
    fmt = inject.get("fmt") or llm.get_format(TARGET, **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}))
    t_load = time.perf_counter()
    if "model" in inject:
        model, tok = inject["model"], inject["tokenizer"]
    else:
        model, tok = live_client.load_model(args.model, args.tokenizer, device=args.device, dtype=args.dtype)
    dc.sync(device)
    model_load_s = time.perf_counter() - t_load           # model + tokenizer, outside the timed region
    t_art = time.perf_counter()
    tap = live_tap.LiveProbeTap(args.artifact, device=device, scope="call", time_capture=False, name="A1")
    dc.sync(device)
    artifact_load_s = time.perf_counter() - t_art         # artifact file read, parse and transfer to the device
    session = f"{os.getpid()}-{int(time.time())}"
    if len(tap.heads) != 1 or tap.art.pooling != "token_mean":
        raise SystemExit("--artifact must be a token-mean probe artifact")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args, device, model, fmt, tap, doc, avail)
    cfg_sha = write_run_config(args.out, cfg)
    rows_path = args.out / "decode.jsonl"
    old_rows = load_rows(rows_path)
    if any(r.get("config_sha256") != cfg_sha for r in old_rows):
        raise SystemExit("rows with another configuration hash are present; use a new --out")
    done = {(r["history_id"], r["repeat"], r["arm"]) for r in old_rows}
    processor = inject.get("logits_processor")

    warmup_s = None
    if not args.no_warmup and avail:          # untimed: kernels and the tap path, on the first history, both arms
        h = avail[0]
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        for arm in ARMS:
            c, _ = dc.copy_cache(base, device)
            if arm == "A1":
                tap.attach(model)
            dc.decode_run(model, c, last, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=1, max_new_tokens=16,
                          device=device, tap=tap if arm == "A1" else None, processor=processor)
            tap.detach()
        warmup_s = time.perf_counter() - t0
        del base, c
    with open(args.out / "setup.jsonl", "a") as f:      # per process start; outside run_config so that resume keeps the hash
        f.write(json.dumps({"session": session, "config_sha256": cfg_sha, "model_load_s": model_load_s,
                            "artifact_load_s": artifact_load_s, "warmup_s": warmup_s, "created_unix": time.time()}) + "\n")

    n_total = 0
    for h in avail:
        todo = [(r, a) for r in range(args.repeats) for a in ARMS if (h["history_id"], r, a) not in done]
        if not todo:
            continue
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        ok_count = len(ids) == h["prompt_tokens"]
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        for r in range(args.repeats):
            for pos, arm in enumerate(arm_order(r)):
                if (h["history_id"], r, arm) in done:
                    continue
                row = run_one(model, base, last, h, r, arm, pos, args, fmt, tap, device, cfg_sha, processor, ok_count)
                row["session"] = session
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                n_total += 1
                print(f"{h['history_id']} r{r} {arm} (pos {pos}): {row['status']} "
                      f"{'' if row['elapsed_s'] is None else format(row['elapsed_s'], '.3f') + 's'} gen={row['n_generated']} "
                      f"natural_stop={row['stopped_naturally']}", flush=True)
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"{n_total} timed runs written to {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

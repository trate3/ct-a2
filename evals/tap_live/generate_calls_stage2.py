"""Stage 2: natural completion of the calls that hit the cap in stage 1, under the benchmark's call-length policy.

  python -m evals.tap_live.generate_calls_stage2 --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --stage2-list stage2_list.json --out OUT [--device cuda] [--max-total-tokens 240000]

Runs, for the runs of `stage2_list.json` only (selected by `select_stage2`: in the default main-stage design ALL distinct-prompt representatives, with the
representative's seed), REPEAT INDEX 0 only, both arms, with the SAME seed as stage 1 for that (history, repeat 0). The arm order ALTERNATES by representative index,
fixed in the config: A0 first for even index, A1 first for odd index (`--arm-order a0_first` keeps the earlier all-A0-first behaviour for the earlier selection modes).
A0's output is saved whichever arm runs first (A1-first pairs still save the matching A0 output for judging). The construction is `generate_calls`'s
(base cache untimed, fresh clone per run, the last prompt token plus decoding timed, probe scoring inside the timed region in A1), reused by import.

Call-length policy (the Qwen LiveCodeBench T1 reference): no per-call cap. A call stops at its natural stop, or when prompt + generated tokens
reach --max-total-tokens (default 240,000, the reference's max_context_tokens); that case is recorded as finish_reason "context_ceiling" and is
never called natural. Nothing kills a long call; progress is printed as counts only, every 2,000 generated tokens. Outputs (A0: token ids,
text with and without the stop token, finish reason) go to outputs.jsonl, the A1 ids to outputs_A1.jsonl. `run_config.json` is immutable and
records the stage-1 run_config hash (with --stage1), the stage-2 list hash and rule hash, the arm-order rule and the cap policy; resume refuses a changed configuration.
GPU memory: the hybrid model keeps KV only for its full-attention layers; the script prints the estimate for the longest run before it starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evals.reward_hacking import llm, tools
from evals.tap_live import decode_bench as db
from evals.tap_live import decode_core as dc
from evals.tap_live import generate_calls as gc
from evals.tap_live import live_client, live_tap

PROGRESS_EVERY = 2000
GIB = 1024 ** 3


def kv_bytes_per_token(model, dtype_bytes: int | None = None) -> int:
    """Bytes of KV cache per token: 2 (k and v) x kv heads x head dim x bytes, for the full-attention layers only (linear-attention layers
    keep a fixed-size state)."""
    tc = model.config.get_text_config()
    lt = getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers
    n_full = sum(t == "full_attention" for t in lt)
    nb = dtype_bytes or next(model.parameters()).element_size()
    return n_full * 2 * tc.num_key_value_heads * tc.head_dim * nb


def memory_estimate(model, prompt_tokens: int, total_tokens: int, tap_hidden: int, weights_bytes: int | None = None) -> dict:
    per = kv_bytes_per_token(model)
    base, clone_end = per * prompt_tokens, per * total_tokens
    tap_buf = 4 * tap_hidden * total_tokens * 2          # float32 buffer, transiently doubled when it grows
    d = {"kv_bytes_per_token": per, "base_cache_GiB": base / GIB, "clone_cache_at_end_GiB": clone_end / GIB,
         "tap_buffer_GiB_peak_during_growth": tap_buf / GIB, "base_plus_clone_plus_tap_GiB": (base + clone_end + tap_buf) / GIB}
    if weights_bytes:
        d["weights_GiB"] = weights_bytes / GIB
        d["total_GiB"] = d["base_plus_clone_plus_tap_GiB"] + d["weights_GiB"]
    return d


def progress_wrapper(label: str, inner=None):
    """A 'processor' for decode_run that leaves the logits alone and prints the generated count every PROGRESS_EVERY tokens."""
    def proc(logits, gen_ids, call_index):
        n = len(gen_ids)
        if n and n % PROGRESS_EVERY == 0:
            print(f"  [{label}] {n} tokens generated", flush=True)
        return inner(logits, gen_ids, call_index) if inner is not None else logits
    return proc


def parse_info(tok, fmt, gen: list[int], stopped: bool) -> dict:
    """Parse the generation as the agent loop would (outside the timed region): reasoning closed, tool calls, malformed calls, parse failure."""
    try:
        text = tok.decode(gen[:-1] if stopped else gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        turn = fmt.parse_generation(text, tools.TOOL_SPECS)
        return {"parse_failure": False, "reasoning_closed": bool(turn.reasoning_closed), "n_tool_calls": len(turn.tool_calls),
                "n_malformed_tool_calls": sum(c.parse_error is not None for c in turn.tool_calls)}
    except Exception as exc:      # noqa: BLE001 - recorded, never raised
        return {"parse_failure": True, "parse_error": f"{type(exc).__name__}: {exc}"[:300]}


def main(argv=None, *, inject: dict | None = None) -> int:
    """`inject` (tests only): {"model", "tokenizer", "fmt", "logits_processor"}."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--stage2-list", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, default=None, help="the curtailed stage-1 directory: only its run_config hash is recorded in the config")
    ap.add_argument("--arm-order", choices=("alternating", "a0_first"), default="alternating")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-total-tokens", type=int, default=240000, help="context ceiling: prompt + generated (the reference's max_context_tokens)")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--limit-runs", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    inject = inject or {}
    device = torch.device(args.device)
    sel = json.loads(args.stage2_list.read_text())
    doc = json.loads(args.histories.read_text())
    if sel["histories_sha256"] != db.sha256_file(args.histories):
        raise SystemExit("the histories file differs from the one the stage-2 list was made from")
    hist = {h["history_id"]: h for h in doc["histories"]}
    runs = sel["runs"][:args.limit_runs or None]
    run_hist = []
    for r in runs:
        h = hist[r["history_id"]]
        if h["prompt_sha256"] != r["prompt_sha256"]:
            raise SystemExit(f"prompt hash of {r['history_id']} differs from the stage-2 list")
        run_hist.append(h)
    fmt = inject.get("fmt") or llm.get_format(gc.TARGET, **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}))
    t_load = time.perf_counter()
    if "model" in inject:
        model, tok = inject["model"], inject["tokenizer"]
    else:
        model, tok = live_client.load_model(args.model, args.tokenizer, device=args.device, dtype=args.dtype)
    dc.sync(device)
    model_load_s = time.perf_counter() - t_load
    t_art = time.perf_counter()
    tap = live_tap.LiveProbeTap(args.artifact, device=device, scope="call", time_capture=False, name="A1")
    dc.sync(device)
    artifact_load_s = time.perf_counter() - t_art
    if len(tap.heads) != 1 or tap.art.pooling != "token_mean":
        raise SystemExit("--artifact must be a token-mean probe artifact")
    args.repeats, args.max_new_tokens = 1, args.max_total_tokens          # for build_config
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = db.build_config(args, device, model, fmt, tap, doc, run_hist)
    stage1_sha = (json.loads((args.stage1 / "run_config.json").read_text())["config_sha256"] if args.stage1 else (sel.get("stage1") or {}).get("run_config_sha256"))
    order_of = [(r.get("arm_order") if args.arm_order == "alternating" and r.get("arm_order") else
                 (["A0", "A1"] if (r.get("representative_index", i) % 2 == 0 or args.arm_order == "a0_first") else ["A1", "A0"])) for i, r in enumerate(runs)]
    cfg.update(kind="generate_calls_stage2", arms=list(gc.ARMS), repeat_index=0, run_keys=[r["run_key"] for r in runs],
               stage1_run_config_sha256=stage1_sha,
               arm_order_rule=("alternating by representative index: A0 first for even index, A1 first for odd index" if args.arm_order == "alternating" else "A0 first for every run"),
               arm_orders={r["run_key"]: o for r, o in zip(runs, order_of)},
               stage2_list_sha256=db.sha256_file(args.stage2_list), stage2_rule_sha256=sel["rule_sha256"], selection_mode=sel.get("mode"),
               selection_primary=sel.get("primary"),
               cap_policy="no per-call cap: the call stops at its natural stop or when prompt + generated tokens reach max_total_tokens "
                          "(finish_reason 'context_ceiling', never called natural); no early kill",
               max_total_tokens=args.max_total_tokens,
               seeds="the stage-1 seed of the run's history at repeat 0: sha256('tap-live-decode:calls:<history_id>:0')[:4]; identical across arms",
               )
    cfg.pop("max_new_tokens", None)
    cfg.pop("repeats", None)
    cfg_sha = db.write_run_config(args.out, cfg)
    rows_path, out_path, out1_path = args.out / "calls.jsonl", args.out / "outputs.jsonl", args.out / "outputs_A1.jsonl"
    old = db.load_rows(rows_path)
    if any(r.get("config_sha256") != cfg_sha for r in old):
        raise SystemExit("rows with another configuration hash are present; use a new --out")
    done = {(r["run_key"], r["arm"]) for r in old}
    out_done = {r["run_key"] for r in db.load_rows(out_path)}
    out1_done = {r["run_key"] for r in db.load_rows(out1_path)}
    processor = inject.get("logits_processor")

    session = f"{os.getpid()}-{int(time.time())}"
    warmup_s = None
    if not args.no_warmup and run_hist:
        h = run_hist[0]
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        for arm in gc.ARMS:
            c, _ = dc.copy_cache(base, device)
            if arm == "A1":
                tap.attach(model)
            dc.decode_run(model, c, last, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=1, max_new_tokens=16, device=device,
                          tap=tap if arm == "A1" else None, processor=processor)
            tap.detach()
        warmup_s = time.perf_counter() - t0
        del base, c
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    longest = max((h["prompt_tokens"] for h in run_hist), default=0)
    est = memory_estimate(model, longest, args.max_total_tokens, next(iter(tap.d.values())), weights)
    print("memory estimate (longest prompt, generation to the context ceiling): " + json.dumps({k: round(v, 2) for k, v in est.items()}), flush=True)
    with open(args.out / "setup.jsonl", "a") as f:
        f.write(json.dumps({"session": session, "config_sha256": cfg_sha, "model_load_s": model_load_s, "artifact_load_s": artifact_load_s,
                            "warmup_s": warmup_s, "memory_estimate": est, "created_unix": time.time()}) + "\n")

    n_total = 0
    for k, (run, h, order) in enumerate(zip(runs, run_hist, order_of), 1):
        key = run["run_key"]
        if all((key, a) in done for a in gc.ARMS):
            continue
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        count_ok = len(ids) == h["prompt_tokens"]
        max_new = args.max_total_tokens - len(ids)
        if max_new < 1:
            raise SystemExit(f"{h['history_id']}: the prompt ({len(ids)} tokens) leaves no room under {args.max_total_tokens}")
        print(f"run {k}/{len(runs)} {h['history_id']} (prompt {len(ids)} tokens, up to {max_new} new tokens)", flush=True)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        a_ns = SimpleNamespace(cap=max_new)
        for pos, arm in enumerate(order):
            if (key, arm) in done:
                continue
            row, gen = gc.run_call(model, tok, base, last, h, 0, arm, pos, a_ns, fmt, tap, device, cfg_sha,
                                   progress_wrapper(f"{h['history_id']} {arm}", processor), count_ok)
            if row["status"] == "ok" and row["finish_reason"] == "cap":      # no per-call cap exists: reaching max_new is the context ceiling
                row.update(finish_reason="context_ceiling", hit_cap=False, hit_context_ceiling=True)
            if gen is not None:
                row["parse"] = parse_info(tok, fmt, gen, bool(row["stopped_naturally"]))
            row.update(stage="stage2", run_key=key, representative_index=run.get("representative_index"), positions=run["positions"], context_total_limit=args.max_total_tokens, max_new_tokens=max_new,
                       session=session)
            if gen is not None:
                if arm == "A0" and key not in out_done:
                    with open(out_path, "a") as f:
                        f.write(json.dumps({**gc.output_record(tok, fmt, row, gen, h["prompt_sha256"]), "run_key": key, "positions": run["positions"],
                                            "representative_index": run.get("representative_index"), "arm_order": order, "parse": row["parse"],
                                            "token_hash": row["token_hash"]}) + "\n")
                    out_done.add(key)
                if arm == "A1" and key not in out1_done:
                    with open(out1_path, "a") as f:
                        f.write(json.dumps({"run_key": key, "history_id": h["history_id"], "finish_reason": row["finish_reason"], "token_ids": gen}) + "\n")
                    out1_done.add(key)
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
            n_total += 1
            print(f"{h['history_id']} {arm} (pos {pos}): {row['status']} "
                  f"{'' if row['elapsed_s'] is None else format(row['elapsed_s'], '.1f') + 's'} gen={row['n_generated']} {row['finish_reason']}", flush=True)
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"{n_total} timed calls written to {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

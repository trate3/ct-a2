"""Focused live generation of COMPLETE agent calls on the fixed histories, no tap versus tap, with the outputs saved for the monitors.

  python -m evals.tap_live.generate_calls --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --out OUT [--device cuda] [--repeats 3] [--cap 4096] [--arms A0,A1]

Same construction as `decode_bench` (and the same 30 histories, no new selection rule): UNTIMED, tokenize the prompt, prefill every
prompt token but the last into a base cache, a fresh `copy.deepcopy` per run checked equal to the base; TIMED, the last prompt token
forward plus decoding until a NATURAL STOP (a format stop token) or the declared CAP (default 4,096 new tokens), the agent's sampling
(temperature 1.0, top_p 0.95, top_k 20), same seed in both arms, and with the probe attached (arm A1) the call-end scoring inside the timed
region. Arm order alternates with the repeat index (A0 first on even). Primary timing is wall clock only, no CUDA events, one host
sync per generated token in both arms.

What happens at the cap: generation stops after `cap` new tokens, the run is a row with finish_reason "cap" (the call is NOT complete: the
model had not stopped), it is timed up to the cap and kept; summaries report natural stops and cap hits separately and also give the
natural-stop-only subset. With the probe, a capped call pools all `cap` generated states (no stop token to exclude).

Outputs for the monitors: for the A0 first repeat of every history, `outputs.jsonl` holds the generated token ids, the text decoded without
stripping special tokens (stop token included), the text with the stop token removed, the number of tokens and the finish reason.
Rows: OUT/calls.jsonl; immutable OUT/run_config.json written before the first row (resume refuses a changed configuration).
Summarize with `python -m evals.tap_live.summarize_calls OUT`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from evals.reward_hacking import llm
from evals.tap_live import decode_bench as db
from evals.tap_live import decode_core as dc
from evals.tap_live import live_client, live_tap

ARMS = ("A0", "A1")
TARGET = "qwen3.8-27b"


def seed_call(history_id: str, repeat: int) -> int:
    return dc.seed_for("calls", history_id, repeat)


def run_call(model, tok, base, last_ids, h, repeat, arm, pos, args, fmt, tap, device, cfg_sha, processor, count_ok) -> tuple[dict, list[int] | None]:
    row = {"config_sha256": cfg_sha, "history_id": h["history_id"], "task": h["task"], "depth": h["depth_fraction"], "call_index": h["call_index"],
           "repeat": repeat, "arm": arm, "order_position": pos, "seed": seed_call(h["history_id"], repeat), "cap": args.cap,
           "prompt_tokens": h["prompt_tokens"], "prompt_token_count_matches_selection": count_ok, "status": "ok", "error": None,
           "elapsed_s": None, "n_generated": None, "stopped_naturally": None, "finish_reason": None, "hit_cap": None, "token_hash": None,
           "cache_copy_s": None, "cache_equal_to_base": None, "decisions": None, "tap_score_s": None}
    gen = None
    attached = False
    try:
        cache, row["cache_copy_s"] = dc.copy_cache(base, device)
        row["cache_equal_to_base"] = dc.caches_equal(base, cache)
        if arm == "A1":
            tap.attach(model)
            attached = True
            tap.begin_episode()
        res = dc.decode_run(model, cache, last_ids, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=row["seed"], max_new_tokens=args.cap,
                            device=device, tap=tap if arm == "A1" else None, processor=processor)
        gen = res["gen_ids"]
        row.update(elapsed_s=res["elapsed_s"], n_generated=len(gen), stopped_naturally=res["stopped_naturally"],
                   finish_reason="natural" if res["stopped_naturally"] else "cap", hit_cap=not res["stopped_naturally"], token_hash=dc.ids_hash(gen))
        if arm == "A1":
            row["decisions"] = res["decisions"]
            row["tap_score_s"] = (res["record"] or {}).get("score_s")
        if row["cache_equal_to_base"] is not True:
            row["status"] = "cache_mismatch"
    except Exception as exc:      # noqa: BLE001 - a failed run is a row, never silently dropped
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        gen = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        if attached:
            tap.detach()
    return row, gen


def output_record(tok, fmt, row, gen, prompt_sha256) -> dict:
    stopped = row["stopped_naturally"]
    return {"config_sha256": row["config_sha256"], "history_id": row["history_id"], "task": row["task"], "depth": row["depth"],
            "call_index": row["call_index"], "prompt_sha256": prompt_sha256, "arm": row["arm"], "repeat": row["repeat"], "seed": row["seed"],
            "finish_reason": row["finish_reason"], "n_tokens": len(gen), "token_ids": gen,
            "text_with_special_tokens": tok.decode(gen, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "text_stop_token_removed": tok.decode(gen[:-1] if stopped else gen, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "stop_token_id": gen[-1] if stopped else None}


def main(argv=None, *, inject: dict | None = None) -> int:
    """`inject` (tests only): {"model", "tokenizer", "fmt", "logits_processor"}."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cap", type=int, default=4096, help="declared cap on new tokens per call")
    ap.add_argument("--arms", default="A0,A1")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--limit-histories", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    args.max_new_tokens = args.cap                  # for build_config
    arms = [a for a in args.arms.split(",") if a]
    if not arms or any(a not in ARMS for a in arms) or arms[0] != "A0":
        raise SystemExit("--arms must be A0 or A0,A1 (A0 provides the outputs and the pairing baseline)")
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
    model_load_s = time.perf_counter() - t_load
    t_art = time.perf_counter()
    tap = live_tap.LiveProbeTap(args.artifact, device=device, scope="call", time_capture=False, name="A1")
    dc.sync(device)
    artifact_load_s = time.perf_counter() - t_art
    if len(tap.heads) != 1 or tap.art.pooling != "token_mean":
        raise SystemExit("--artifact must be a token-mean probe artifact")
    session = f"{os.getpid()}-{int(time.time())}"
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = db.build_config(args, device, model, fmt, tap, doc, avail)
    cfg.update(kind="generate_calls", arms=arms, cap=args.cap, cap_behaviour="generation stops after `cap` new tokens; the row is finish_reason 'cap' "
               "(an incomplete call), timed up to the cap and kept; summaries separate cap hits from natural stops",
               seeds="sha256('tap-live-decode:calls:<history_id>:<repeat>')[:4] as int; identical across arms",
               outputs="outputs.jsonl: A0 first repeat of every history (token ids, text with and without the stop token, finish reason)")
    cfg.pop("max_new_tokens", None)
    cfg_sha = db.write_run_config(args.out, cfg)
    rows_path, out_path = args.out / "calls.jsonl", args.out / "outputs.jsonl"
    old = db.load_rows(rows_path)
    if any(r.get("config_sha256") != cfg_sha for r in old):
        raise SystemExit("rows with another configuration hash are present; use a new --out")
    done = {(r["history_id"], r["repeat"], r["arm"]) for r in old}
    out_done = {r["history_id"] for r in db.load_rows(out_path)}
    processor = inject.get("logits_processor")

    warmup_s = None
    if not args.no_warmup and avail:
        h = avail[0]
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        for arm in ARMS:
            c, _ = dc.copy_cache(base, device)
            if arm == "A1":
                tap.attach(model)
            dc.decode_run(model, c, last, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=1, max_new_tokens=16, device=device,
                          tap=tap if arm == "A1" else None, processor=processor)
            tap.detach()
        warmup_s = time.perf_counter() - t0
        del base, c
    with open(args.out / "setup.jsonl", "a") as f:
        f.write(json.dumps({"session": session, "config_sha256": cfg_sha, "model_load_s": model_load_s, "artifact_load_s": artifact_load_s,
                            "warmup_s": warmup_s, "created_unix": time.time()}) + "\n")

    n_total = 0
    for h in avail:
        todo = [(r, a) for r in range(args.repeats) for a in arms if (h["history_id"], r, a) not in done]
        if not todo:
            continue
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        count_ok = len(ids) == h["prompt_tokens"]
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        for r in range(args.repeats):
            order = [a for a in db.arm_order(r) if a in arms]
            for pos, arm in enumerate(order):
                if (h["history_id"], r, arm) in done:
                    continue
                row, gen = run_call(model, tok, base, last, h, r, arm, pos, args, fmt, tap, device, cfg_sha, processor, count_ok)
                row["session"] = session
                if arm == "A0" and r == 0 and gen is not None and h["history_id"] not in out_done:
                    with open(out_path, "a") as f:                  # before the row, so that a crash in between is repaired on resume
                        f.write(json.dumps(output_record(tok, fmt, row, gen, h["prompt_sha256"])) + "\n")
                    out_done.add(h["history_id"])
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                n_total += 1
                print(f"{h['history_id']} r{r} {arm} (pos {pos}): {row['status']} "
                      f"{'' if row['elapsed_s'] is None else format(row['elapsed_s'], '.2f') + 's'} gen={row['n_generated']} {row['finish_reason']}", flush=True)
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"{n_total} timed calls written to {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

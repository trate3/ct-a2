"""Correctness trial (separate from the timed runs) on development histories; writes an evidence JSON.

  python -m evals.tap_live.correctness_trial --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --out correctness.json [--device cuda] [--n-histories 2] [--max-new-tokens 64] [--prefill-chunk N] [--tolerance 1e-4]

For each history (the first --n-histories available ones, which are training-split development tasks):
 (a) tap versus the repo's offline scorer (`tap.StreamingTap`, the code the accuracy results used):
       a1 arithmetic: the offline scorer applied to the exact states the tap captured (its buffer): eligible-token count,
          pooled score, alarm;
       a2 end to end: the offline scorer applied to the states of ONE full forward pass over prompt + generation (a hook on
          the same decoder block; state of generated token j = position P+j-1, the stop token excluded), compared with the
          tap's score from the cached decode. In bfloat16 a cached decode and a full pass round differently, so a2 also
          reports the largest difference of the states themselves.
     Declared tolerance on pooled scores: --tolerance (1e-4 absolute).
 (b) generation from the cached prefix (base cache + last prompt token) versus a fresh cache (the whole prompt in one forward),
     same seed: token ids equal, and the first logits compared (max absolute difference).
 (c) A0 and A1 from copies of the same base cache, same seed: identical token streams (or the differences, reported).
 (d) with --prefill-chunk N > 0: the base cache built in chunks versus in one forward: first logits and tokens.
Every check is reported with its numbers; pass flags state whether the declared tolerance held.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tap as tapmod
from evals.tap_live import decode_core as dc
from evals.tap_live import decode_setup as ds

SINK = 3


def offline(art, states: np.ndarray, stopped: bool) -> dict:
    ch = np.full(len(states), act.VISIBLE, np.int8)
    if stopped:
        ch[-1] = act.END
    t = tapmod.StreamingTap(art)
    S = t.update(states.astype(np.float64), ch, 0)
    return {"eligible_tokens": int(t.n_scored), "score": S, "alarm": bool(S is not None and S > art.threshold)}


@torch.no_grad()
def full_pass_states(model, ids: list[int], layer: int, start: int, n: int, device) -> np.ndarray:
    """States at positions start .. start+n-1 of the output of decoder block layer-1, from one full forward pass."""
    keep = {}
    blk = act.find_decoder_layers(model)[layer - 1]
    handle = blk.register_forward_hook(lambda m, i, o: keep.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, start:start + n].float().cpu()))
    try:
        model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return keep["h"].numpy()


def trial_one(model, tok, fmt, tap, h, device, args, processor=None) -> dict:
    ids = tok.encode(h["prompt"], add_special_tokens=False)
    P = len(ids)
    art = tap.art
    base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
    last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
    seed = dc.seed_for("correctness", h["history_id"], 0)
    kw = dict(stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=seed, max_new_tokens=args.max_new_tokens, device=device,
              processor=processor)
    out = {"history_id": h["history_id"], "prompt_tokens": P}
    # A0 and A1 from copies of the same base cache (c)
    c0, _ = dc.copy_cache(base, device)
    s0: list = []
    r0 = dc.decode_run(model, c0, last, **kw, tap=None, logits_sink=s0, sink_steps=SINK)
    c1, _ = dc.copy_cache(base, device)
    s1: list = []
    tap.attach(model)
    tap.begin_episode()
    try:
        r1 = dc.decode_run(model, c1, last, **kw, tap=tap, logits_sink=s1, sink_steps=SINK)
    finally:
        tap.detach()
    out["c_A0_vs_A1"] = {"identical_token_streams": r0["gen_ids"] == r1["gen_ids"], "n_generated": [len(r0["gen_ids"]), len(r1["gen_ids"])],
                         "stopped_naturally": [r0["stopped_naturally"], r1["stopped_naturally"]],
                         "max_abs_first_logits_diff": max((float((a - b).abs().max()) for a, b in zip(s0, s1)), default=None),
                         "first_difference_at": next((i for i, (a, b) in enumerate(zip(r0["gen_ids"], r1["gen_ids"])) if a != b), None)}
    # (a) tap vs offline
    n, stopped = len(r1["gen_ids"]), r1["stopped_naturally"]
    dec = r1["decisions"][0]
    tap_states = tap.bufs[art.layer][:n].float().cpu().numpy()
    a1 = offline(art, tap_states, stopped)
    out["a1_arithmetic_on_tap_states"] = {
        "tap": {"eligible_tokens": dec["n_tokens"], "score": dec["score"], "alarm": dec["alarm"]}, "offline": a1,
        "abs_score_diff": abs(dec["score"] - a1["score"]) if dec["score"] is not None and a1["score"] is not None else None,
        "pass": dec["n_tokens"] == a1["eligible_tokens"] and dec["alarm"] == a1["alarm"] and (
            (dec["score"] is None and a1["score"] is None) or abs(dec["score"] - a1["score"]) <= args.tolerance)}
    full = full_pass_states(model, ids + r1["gen_ids"], art.layer, P - 1, n, device)
    a2 = offline(art, full, stopped)
    out["a2_full_forward_pass"] = {
        "offline": a2, "tap": out["a1_arithmetic_on_tap_states"]["tap"],
        "abs_score_diff": abs(dec["score"] - a2["score"]) if dec["score"] is not None and a2["score"] is not None else None,
        "max_abs_state_diff_vs_tap_buffer": float(np.abs(full - tap_states).max()),
        "max_abs_state": float(np.abs(full).max()),
        "pass": dec["n_tokens"] == a2["eligible_tokens"] and dec["alarm"] == a2["alarm"] and (
            (dec["score"] is None and a2["score"] is None) or abs(dec["score"] - a2["score"]) <= args.tolerance)}
    # (b) cached prefix versus fresh cache
    sf: list = []
    rf = dc.decode_run(model, None, torch.tensor([ids], dtype=torch.long, device=device), **kw, tap=None, logits_sink=sf, sink_steps=SINK)
    out["b_cached_vs_fresh"] = {"identical_token_streams": r0["gen_ids"] == rf["gen_ids"],
                                "n_generated": [len(r0["gen_ids"]), len(rf["gen_ids"])],
                                "max_abs_first_logits_diff": max((float((a - b).abs().max()) for a, b in zip(s0, sf)), default=None),
                                "first_difference_at": next((i for i, (a, b) in enumerate(zip(r0["gen_ids"], rf["gen_ids"])) if a != b), None)}
    # (d) chunked versus single-forward base prefill
    if args.prefill_chunk:
        single = dc.prefill_base(model, ids[:-1], device, 0)
        cs, _ = dc.copy_cache(single, device)
        ss: list = []
        rs = dc.decode_run(model, cs, last, **kw, tap=None, logits_sink=ss, sink_steps=SINK)
        out["d_chunked_vs_single_prefill"] = {"chunk": args.prefill_chunk, "identical_token_streams": r0["gen_ids"] == rs["gen_ids"],
                                              "max_abs_first_logits_diff": max((float((a - b).abs().max()) for a, b in zip(s0, ss)), default=None)}
        del single, cs
    else:
        out["d_chunked_vs_single_prefill"] = {"skipped": "no --prefill-chunk"}
    return out


def main(argv=None, *, inject: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ds.add_common(ap)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-histories", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args(argv)
    device, doc, avail, fmt, model, tok, tap = ds.setup(args, inject)
    res = {"declared_tolerance_abs_score": args.tolerance, "dtype": args.dtype, "artifact": tap.describe(),
           "histories": [trial_one(model, tok, fmt, tap, h, device, args, (inject or {}).get("logits_processor"))
                         for h in avail[:args.n_histories]]}
    res["all_declared_checks_pass"] = all(x["a1_arithmetic_on_tap_states"]["pass"] and x["a2_full_forward_pass"]["pass"]
                                          for x in res["histories"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Separate instrumented pass: where the tap's time goes. Its elapsed times are never mixed into the primary summary.

  python -m evals.tap_live.profile_pass --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --out OUT_PROFILE [--device cuda] [--n-histories 3] [--repeats 3] [--max-new-tokens 256]

Same construction as `decode_bench` (base cache, fresh copy per run, last prompt token fed in the timed region, same seeds,
alternating arm order) on a fixed subset (evenly spaced histories), but with CUDA events: one pair around every model step
(forward + sampling) in both arms, the tap's capture events around the hook's copies (`capture_s`) and events around the
call-end scoring (`score_event_ms`, next to its wall-clock `score_s`). The events themselves cost time, so the elapsed times of
this pass are not overhead estimates. Output: OUT/profile.jsonl and OUT/profile_summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from evals.tap_live import decode_core as dc
from evals.tap_live import decode_setup as ds


def summarize(rows: list[dict]) -> dict:
    out = {"note": "instrumented with CUDA events; not an overhead estimate", "arms": {}}
    for arm in ("A0", "A1"):
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        if not rs:
            continue
        steps = [m for r in rs for m in r["step_ms"][1:]]          # decode steps; step 0 is the prompt-final forward
        first = [r["step_ms"][0] for r in rs]
        d = {"n_runs": len(rs), "mean_elapsed_s": statistics.fmean(r["elapsed_s"] for r in rs),
             "mean_first_step_ms": statistics.fmean(first),
             "decode_step_ms": {"n": len(steps), "mean": statistics.fmean(steps) if steps else None,
                                "median": statistics.median(steps) if steps else None}}
        if arm == "A1":
            cap = [r["capture_s"] for r in rs]
            sc = [r["score_s"] for r in rs]
            ev = [r["score_event_ms"] for r in rs if r.get("score_event_ms") is not None]
            d.update(capture_s=None if any(c is None for c in cap) else statistics.fmean(cap),
                     score_wall_s=statistics.fmean(sc), score_event_ms=statistics.fmean(ev) if ev else None,
                     capture_ms_per_forward=None if any(c is None for c in cap) else
                     1000.0 * sum(cap) / sum(len(r["step_ms"]) for r in rs))
        out["arms"][arm] = d
    a0, a1 = out["arms"].get("A0"), out["arms"].get("A1")
    if a0 and a1 and a0["decode_step_ms"]["mean"] and a1["decode_step_ms"]["mean"]:
        out["decode_step_ms_difference_A1_minus_A0"] = a1["decode_step_ms"]["mean"] - a0["decode_step_ms"]["mean"]
    return out


def main(argv=None, *, inject: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ds.add_common(ap)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-histories", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args(argv)
    device, doc, avail, fmt, model, tok, tap = ds.setup(args, inject, time_capture=True)
    processor = (inject or {}).get("logits_processor")
    subset = ds.fixed_subset(avail, args.n_histories)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in subset:
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        for r in range(args.repeats):
            for pos, arm in enumerate(["A0", "A1"] if r % 2 == 0 else ["A1", "A0"]):
                row = {"history_id": h["history_id"], "repeat": r, "arm": arm, "order_position": pos, "status": "ok", "error": None}
                try:
                    c, _ = dc.copy_cache(base, device)
                    if arm == "A1":
                        tap.attach(model)
                        tap.begin_episode()
                    res = dc.decode_run(model, c, last, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling,
                                        seed=dc.seed_for("run", h["history_id"], r), max_new_tokens=args.max_new_tokens,
                                        device=device, tap=tap if arm == "A1" else None, processor=processor, step_events=True)
                    tap.resolve()
                    rec = res["record"] or {}
                    row.update(elapsed_s=res["elapsed_s"], n_generated=len(res["gen_ids"]), step_ms=res["step_ms"],
                               stopped_naturally=res["stopped_naturally"], capture_s=rec.get("capture_s"),
                               score_s=rec.get("score_s"), score_event_ms=res.get("score_event_ms"))
                except Exception as exc:      # noqa: BLE001
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
                finally:
                    tap.detach()
                rows.append(row)
        del base
    (args.out / "profile.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summ = summarize(rows)
    summ["histories"] = [h["history_id"] for h in subset]
    (args.out / "profile_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

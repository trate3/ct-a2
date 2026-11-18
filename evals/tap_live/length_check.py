"""Predeclared length check: genuine live generation at token budgets 256, 1,024 and 4,096.

  python -m evals.tap_live.length_check --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --out OUT_LENGTH [--device cuda] [--n-histories 3] [--repeats 3]

On a fixed subset (evenly spaced histories), both arms, same construction as `decode_bench` (fresh copy of the base cache per
run, last prompt token fed inside the timed region, call-end scoring inside it). Natural stops are honoured and the actual
lengths are reported, so call-end pooling and capture-buffer growth are not inferred from 256 tokens alone. ONE tap object is
used for the whole check in the order of the budgets, so its capture buffer (initial size = the first budget) has to grow when the
budget rises; `buffer_rows` before and after each run is recorded. Wall-clock timing as in the primary run; descriptive only.
Output: OUT/length.jsonl and OUT/length_summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from evals.tap_live import decode_core as dc
from evals.tap_live import decode_setup as ds

BUDGETS = (256, 1024, 4096)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for b in sorted({r["budget"] for r in rows}):
        d = {}
        for arm in ("A0", "A1"):
            rs = [r for r in rows if r["budget"] == b and r["arm"] == arm and r["status"] == "ok"]
            if rs:
                d[arm] = {"n_runs": len(rs), "mean_generated": statistics.fmean(r["n_generated"] for r in rs),
                          "max_generated": max(r["n_generated"] for r in rs),
                          "natural_stops": sum(r["stopped_naturally"] for r in rs),
                          "tokens_per_s": sum(r["n_generated"] for r in rs) / sum(r["elapsed_s"] for r in rs)}
        d["failed_runs"] = sum(r["status"] != "ok" for r in rows if r["budget"] == b)
        d["tap_buffer_rows_after"] = max((r["buffer_rows_after"] for r in rows if r["budget"] == b and r["arm"] == "A1"), default=None)
        d["tap_buffer_grew_in_runs"] = sum(bool(r.get("buffer_grew")) for r in rows if r["budget"] == b)
        d["identical_streams_A0_A1"] = sum(
            1 for r in rows if r["arm"] == "A1" and r["budget"] == b and r["status"] == "ok" and any(
                q["arm"] == "A0" and q["history_id"] == r["history_id"] and q["repeat"] == r["repeat"] and q["budget"] == b
                and q["status"] == "ok" and q["token_hash"] == r["token_hash"] for q in rows))
        by = {(r["history_id"], r["repeat"]): {} for r in rows if r["budget"] == b}
        for r in rows:
            if r["budget"] == b:
                by[(r["history_id"], r["repeat"])][r["arm"]] = r
        diffs = [1000.0 * (v["A1"]["elapsed_s"] - v["A0"]["elapsed_s"]) for v in by.values()
                 if "A0" in v and "A1" in v and v["A0"].get("elapsed_s") is not None and v["A1"].get("elapsed_s") is not None]
        d["paired_diff_ms"] = ({"n_pairs": len(diffs), "min": min(diffs), "max": max(diffs), "mean": statistics.fmean(diffs)} if diffs else {"n_pairs": 0})
        d["task_coverage"] = {"unique_tasks": len({h.rsplit("@", 1)[0] for h, _ in by}), "unique_histories": len({h for h, _ in by}), "task_ids": sorted({h.rsplit("@", 1)[0] for h, _ in by}),
                              "note": "raw paired range only: three histories (few tasks) do not support a population interval"}
        out[str(b)] = d
    return out


def main(argv=None, *, inject: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ds.add_common(ap)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-histories", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    args = ap.parse_args(argv)
    device, doc, avail, fmt, model, tok, tap = ds.setup(args, inject)
    processor = (inject or {}).get("logits_processor")
    budgets = [int(b) for b in args.budgets.split(",")]
    subset = ds.fixed_subset(avail, args.n_histories)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in subset:
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        last = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
        for b in budgets:
            for r in range(args.repeats):
                for pos, arm in enumerate(["A0", "A1"] if r % 2 == 0 else ["A1", "A0"]):
                    row = {"history_id": h["history_id"], "budget": b, "repeat": r, "arm": arm, "order_position": pos,
                           "status": "ok", "error": None, "decisions": None}
                    try:
                        c, _ = dc.copy_cache(base, device)
                        rows_before = tap.bufs[tap.layers[0]].shape[0]
                        if arm == "A1":
                            tap.attach(model)
                            tap.begin_episode()
                        res = dc.decode_run(model, c, last, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling,
                                            seed=dc.seed_for("length", h["history_id"], r), max_new_tokens=b, device=device,
                                            tap=tap if arm == "A1" else None, processor=processor)
                        row.update(elapsed_s=res["elapsed_s"], n_generated=len(res["gen_ids"]), stopped_naturally=res["stopped_naturally"],
                                   token_hash=dc.ids_hash(res["gen_ids"]), decisions=res.get("decisions"))
                        if arm == "A1":
                            row.update(buffer_rows_before=rows_before, buffer_rows_after=tap.bufs[tap.layers[0]].shape[0],
                                       buffer_grew=tap.bufs[tap.layers[0]].shape[0] > rows_before and rows_before > 0)
                    except Exception as exc:      # noqa: BLE001
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
                    finally:
                        tap.detach()
                    rows.append(row)
        del base
    (args.out / "length.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summ = summarize(rows)
    summ["histories"] = [h["history_id"] for h in subset]
    (args.out / "length_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

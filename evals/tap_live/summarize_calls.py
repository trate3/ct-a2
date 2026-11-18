"""Descriptive summary of a `generate_calls` run (complete agent calls up to a declared cap).

  python -m evals.tap_live.summarize_calls OUT        # reads OUT/calls.jsonl and OUT/run_config.json; writes OUT/calls_summary.{json,md}

Per arm: scheduled / attempted / completed / failed / missing, mean seconds per call (all calls, and natural-stop calls only), mean
generated tokens, natural stops versus cap hits, tokens per second over the timed region. A cap hit is an INCOMPLETE call (the model had not
stopped); it is kept, timed up to the cap, and reported separately. Paired A1 - A0 per call (same history and repeat): all attempted pairs,
clean pairs, clean pairs with identical token streams, and clean identical pairs where both calls stopped naturally. Estimands and
intervals are those of `summarize_decode`: relative overhead with equal weight per history, absolute milliseconds per call pair-weighted,
tasks resampled (2,000 draws, seed 0, each drawn task bringing all its depths and repeats). Unique tasks and histories are stated.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evals.tap_live import summarize_decode as sd


def summarize(out: Path) -> dict:
    cfg = json.loads((out / "run_config.json").read_text())
    c = cfg["config"]
    rows = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines() if x.strip()] if (out / "calls.jsonl").exists() else []
    arms = c.get("arms", ["A0", "A1"])
    sched = len(c["history_ids"]) * c["repeats"]
    outputs = [json.loads(x) for x in (out / "outputs.jsonl").read_text().splitlines() if x.strip()] if (out / "outputs.jsonl").exists() else []
    summ = {"config_sha256": cfg["config_sha256"], "cap": c["cap"], "repeats": c["repeats"], "arms": arms,
            "rows_with_other_config_hash": sum(r.get("config_sha256") != cfg["config_sha256"] for r in rows),
            "unique_tasks": len({h.rsplit("@", 1)[0] for h in c["history_ids"]}), "unique_histories": len(set(c["history_ids"])),
            "earlier_call_fallbacks": (c.get("histories_summary") or {}).get("fallback_history_ids"),
            "outputs_saved": len(outputs), "outputs_expected": len(c["history_ids"]), "counts": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r["status"] == "ok"]
        nat = [r for r in ok if r["finish_reason"] == "natural"]
        d = {"scheduled": sched, "attempted": len(rs), "completed": len(ok), "failed": len(rs) - len(ok), "missing": sched - len(rs),
             "statuses": {s: sum(r["status"] == s for r in rs) for s in sorted({r["status"] for r in rs})},
             "natural_stops": len(nat), "cap_hits": sum(r["finish_reason"] == "cap" for r in ok)}
        if ok:
            d.update(mean_seconds_per_call_all=statistics.fmean(r["elapsed_s"] for r in ok), mean_generated_all=statistics.fmean(r["n_generated"] for r in ok),
                     tokens_per_s_aggregate=sum(r["n_generated"] for r in ok) / sum(r["elapsed_s"] for r in ok))
        if nat:
            d.update(mean_seconds_per_call_natural_only=statistics.fmean(r["elapsed_s"] for r in nat),
                     mean_generated_natural_only=statistics.fmean(r["n_generated"] for r in nat))
        summ["counts"][arm] = d
    by = defaultdict(dict)
    for r in rows:
        by[(r["history_id"], r["repeat"])][r["arm"]] = r
    both = [(v["A0"], v["A1"]) for v in by.values() if "A0" in v and "A1" in v]
    if "A1" in arms:
        timed = [(a, b) for a, b in both if a["elapsed_s"] is not None and b["elapsed_s"] is not None]
        clean = [(a, b) for a, b in timed if a["status"] == "ok" and b["status"] == "ok"]
        ident = [(a, b) for a, b in clean if a["token_hash"] == b["token_hash"]]
        natural = [(a, b) for a, b in ident if a["finish_reason"] == "natural" and b["finish_reason"] == "natural"]
        summ["pairs"] = {"scheduled": sched, "both_arms_attempted": len(both), "missing_an_arm": sched - len(both),
                         "with_a_failed_arm": sum(a["status"] != "ok" or b["status"] != "ok" for a, b in both)}
        summ["all_attempts"], summ["clean"] = sd.subset_table(timed), sd.subset_table(clean)
        summ["identical_streams"] = {**sd.subset_table(ident), "n_diverging_clean_pairs": len(clean) - len(ident)}
        summ["identical_natural_stop_only"] = sd.subset_table(natural)
    summ["setup"] = [json.loads(x) for x in (out / "setup.jsonl").read_text().splitlines() if x.strip()] if (out / "setup.jsonl").exists() else []
    return summ


def to_markdown(s: dict) -> str:
    L = ["# Complete agent calls (up to the cap), no tap versus tap (descriptive)", "",
         f"Config {s['config_sha256'][:12]}; cap {s['cap']} new tokens per call (a call that reaches it is an incomplete call, kept and reported separately); "
         f"{s['repeats']} repeats; {s['unique_tasks']} unique tasks, {s['unique_histories']} unique histories (task-depth prompts); "
         f"outputs saved for {s['outputs_saved']} of {s['outputs_expected']} histories (A0 first repeat). Intervals resample whole tasks "
         f"({sd.DRAWS} draws, seed {sd.SEED}).", "",
         "## Counts and means", "", "| arm | scheduled | attempted | completed | failed | missing | natural stops | cap hits | mean s/call (all) | mean s/call (natural only) | mean tokens (all) | tokens/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"      # noqa: E731
    for a, c in s["counts"].items():
        L.append(f"| {a} | {c['scheduled']} | {c['attempted']} | {c['completed']} | {c['failed']} | {c['missing']} | {c['natural_stops']} | {c['cap_hits']} | "
                 f"{f(c.get('mean_seconds_per_call_all'))} | {f(c.get('mean_seconds_per_call_natural_only'))} | {f(c.get('mean_generated_all'), 1)} | "
                 f"{f(c.get('tokens_per_s_aggregate'))} |")
    if "pairs" in s:
        p = s["pairs"]
        L += ["", f"Pairs: {p['both_arms_attempted']} of {p['scheduled']} have both arms attempted ({p['missing_an_arm']} miss an arm); {p['with_a_failed_arm']} include a failed arm."]
        for key, title in (("all_attempts", "All attempted pairs"), ("clean", "Clean pairs (both ok)"),
                           ("identical_streams", "Clean pairs with identical token streams"),
                           ("identical_natural_stop_only", "Identical-stream pairs where both calls stopped naturally")):
            t = s[key]
            L += ["", f"## {title}", ""]
            if t.get("empty"):
                L.append("This subset is empty.")
                continue
            d, r = t["paired_diff_s"], t["paired_relative"]
            L += [f"{t['n_pairs']} pairs, {t['n_tasks']} tasks, {t['n_histories']} histories. Paired difference (s per call): median {d['median']:.4f}, mean {d['mean']:.4f}, "
                  f"min {d['min']:.4f}, max {d['max']:.4f}. Paired relative: median {sd._pct(r['median'])}, min {sd._pct(r['min'])}, max {sd._pct(r['max'])}.", ""]
            L += sd._estimand_lines("overall", t["estimand_overall"])
            ab = t["absolute_ms_per_call_pair_weighted"]
            L += [f"- overall (absolute paired difference, milliseconds per call, {ab['weighting']}): {sd.fmt_sig(ab['mean_paired_difference_ms'], True)} ms, "
                  f"task-clustered 95% interval [{sd.fmt_sig(ab['ci95_ms'][0], True)}, {sd.fmt_sig(ab['ci95_ms'][1], True)}] ms; {ab['verdict']}."]
            for dep, e in t["per_depth"].items():
                L += sd._estimand_lines(f"depth {dep}", e)
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    s = summarize(a.out)
    (a.out / "calls_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (a.out / "calls_summary.md").write_text(to_markdown(s))
    print(to_markdown(s))


if __name__ == "__main__":
    main()

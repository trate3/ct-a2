"""Descriptive summary of a `decode_bench` run.

  python -m evals.tap_live.summarize_decode OUT        # reads OUT/decode.jsonl and OUT/run_config.json; writes OUT/decode_summary.{json,md}

Estimand: the mean paired relative overhead (A1 - A0) / A0 of the elapsed time of the timed region, over the selected
history/depth set: for each history (a task at one depth) the relative overhead is averaged over its repeats, then averaged over
histories (equal weight per history). The unit of resampling is the task: the depths of one task come from one trajectory, so the
descriptive interval is a percentile bootstrap over tasks (2,000 draws with replacement, seed 0); each resampled task brings all
its depths and repeats. The achieved 95 % half-width is given in percentage points of baseline elapsed time. When the interval
contains zero the result is "unresolved at the achieved precision". Also reported: the absolute paired difference in
milliseconds per call (pair-weighted, same task bootstrap) and the secondary normalized cost per 100 generated tokens
(clean pairs with identical token streams only; zero denominators excluded and counted; call-end cost is fixed per call and need
not scale with output length; no tap cost is ever derived from unpaired tokens/s). Unique tasks, unique histories and earlier-call
fallbacks of the selection are stated. Reported first for ALL attempted pairs (both arms have an elapsed time), then, separately, for the clean pairs (both
runs finished with status ok) and for the clean pairs with identical generated token streams (stated if that subset is empty).
Counts per arm: scheduled / attempted / completed / failed / missing. Decode throughput per arm is tokens per second over the
timed region (it includes the prompt-final forward and, in A1, the call-end scoring). The primary rows carry no capture time
(no per-token events); it is reported as null, never as zero.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

DRAWS, SEED = 2000, 0


def stats(v: list[float]) -> dict:
    return {"n": 0} if not v else {"n": len(v), "median": statistics.median(v), "mean": statistics.fmean(v), "min": min(v), "max": max(v)}


def sum_or_none(vals):
    """Sum of the values, or None if there are none or any is unknown (None is never counted as zero)."""
    vals = list(vals)
    return None if not vals or any(v is None for v in vals) else sum(vals)


def task_of(r: dict) -> str:
    return r.get("task") or r["history_id"].rsplit("@", 1)[0]


def depth_of(r: dict) -> float:
    return float(r["history_id"].rsplit("@", 1)[1])


def task_bootstrap(parts: dict[str, tuple[float, float]]) -> dict:
    """parts: task -> (S, N). Statistic = sum(S) / sum(N) over the tasks drawn; tasks are resampled with replacement
    (DRAWS draws, seed SEED), each drawn task bringing all of its S and N. Returns the point estimate and the 95 % percentile interval."""
    keys = sorted(parts)
    S = np.array([parts[k][0] for k in keys], float)
    N = np.array([parts[k][1] for k in keys], float)
    est = float(S.sum() / N.sum())
    idx = np.random.default_rng(SEED).integers(0, len(keys), size=(DRAWS, len(keys)))
    means = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"estimate": est, "ci95": [lo, hi], "n_tasks": len(keys),
            "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "task (all depths and repeats of a drawn task kept together)"}}


def _verdict(lo: float, hi: float) -> str:
    return ("unresolved at the achieved precision" if lo <= 0.0 <= hi else "interval above zero" if lo > 0 else "interval below zero")


def rel_estimand(pairs: list[tuple[dict, dict]]) -> dict:
    """Mean over histories of the per-history mean paired RELATIVE overhead (a fraction of baseline elapsed time)."""
    by_h = defaultdict(list)
    for a, b in pairs:
        if a["elapsed_s"] > 0:
            by_h[a["history_id"]].append((b["elapsed_s"] - a["elapsed_s"]) / a["elapsed_s"])
    if not by_h:
        return {"n_histories": 0}
    per_h = {h: statistics.fmean(v) for h, v in by_h.items()}
    parts = defaultdict(lambda: [0.0, 0.0])
    for h, v in per_h.items():
        parts[h.rsplit("@", 1)[0]][0] += v
        parts[h.rsplit("@", 1)[0]][1] += 1
    r = task_bootstrap({k: tuple(v) for k, v in parts.items()})
    lo, hi = r["ci95"]
    return {"quantity": "relative overhead (A1 - A0) / A0, a fraction of baseline elapsed time",
            "weighting": "equal weight per history (per-history mean over repeats); tasks resampled",
            "n_tasks": r["n_tasks"], "n_histories": len(per_h), "n_pairs": len(pairs),
            "mean_relative_overhead": r["estimate"], "ci95": [lo, hi], "half_width_pct_points": 100.0 * (hi - lo) / 2.0,
            "bootstrap": r["bootstrap"], "verdict": _verdict(lo, hi), "per_history": {k: per_h[k] for k in sorted(per_h)}}


def abs_estimand(pairs: list[tuple[dict, dict]], weighting: str = "pair") -> dict | None:
    """Mean paired ABSOLUTE difference (A1 - A0) in milliseconds per call. weighting "pair": every attempted pair counts once
    (then the difference is exactly the with-tap mean minus the no-tap mean of the same pairs); "history": equal weight per
    history. Tasks are resampled in both cases."""
    if not pairs:
        return None
    by_h = defaultdict(list)
    for a, b in pairs:
        by_h[a["history_id"]].append(1000.0 * (b["elapsed_s"] - a["elapsed_s"]))
    parts = defaultdict(lambda: [0.0, 0.0])
    for h, v in by_h.items():
        t = h.rsplit("@", 1)[0]
        if weighting == "pair":
            parts[t][0] += sum(v)
            parts[t][1] += len(v)
        else:
            parts[t][0] += statistics.fmean(v)
            parts[t][1] += 1
    r = task_bootstrap({k: tuple(v) for k, v in parts.items()})
    lo, hi = r["ci95"]
    return {"quantity": "absolute paired difference (A1 - A0), milliseconds per call",
            "weighting": ("pair-weighted over all pairs in the subset" if weighting == "pair" else "equal weight per history") + "; tasks resampled",
            "n_tasks": r["n_tasks"], "n_histories": len(by_h), "n_pairs": len(pairs),
            "mean_paired_difference_ms": r["estimate"], "ci95_ms": [lo, hi], "half_width_ms": (hi - lo) / 2.0,
            "bootstrap": r["bootstrap"], "verdict": _verdict(lo, hi)}


def normalized_per_100_tokens(pairs: list[tuple[dict, dict]]) -> dict:
    """Secondary: 100 * paired elapsed difference / that pair's generated-token count (ms per 100 tokens), pairs with identical
    token streams only; pairs with zero generated tokens are excluded and counted."""
    ok = [(a, b) for a, b in pairs if a["token_hash"] == b["token_hash"]]
    use = [(a, b) for a, b in ok if a["n_generated"] and a["n_generated"] > 0]
    out = {"formula": "100 * (elapsed_A1 - elapsed_A0) / n_generated of the pair, identical token streams only, milliseconds",
           "note": "the call-end cost is fixed per call and need not scale with output length; this is not a per-token cost model",
           "n_identical_pairs": len(ok), "n_zero_denominator_excluded": len(ok) - len(use), "n_used": len(use)}
    if not use:
        return out
    vals = [(b["elapsed_s"] - a["elapsed_s"]) * 1000.0 * 100.0 / a["n_generated"] for a, b in use]
    parts = defaultdict(lambda: [0.0, 0.0])
    for (a, _), v in zip(use, vals):
        parts[task_of(a)][0] += v
        parts[task_of(a)][1] += 1
    r = task_bootstrap({k: tuple(v) for k, v in parts.items()})
    out.update(stats_ms_per_100_tokens=stats(vals), pair_weighted_mean_ms_per_100_tokens=r["estimate"], ci95_ms_per_100_tokens=r["ci95"],
               n_tasks=r["n_tasks"], bootstrap=r["bootstrap"])
    return out


def subset_table(pairs: list[tuple[dict, dict]]) -> dict:
    if not pairs:
        return {"n_pairs": 0, "empty": True}
    diffs = [b["elapsed_s"] - a["elapsed_s"] for a, b in pairs]
    rel = [(b["elapsed_s"] - a["elapsed_s"]) / a["elapsed_s"] for a, b in pairs if a["elapsed_s"] > 0]
    out = {"n_pairs": len(pairs), "empty": False, "n_tasks": len({task_of(a) for a, _ in pairs}),
           "n_histories": len({a["history_id"] for a, _ in pairs}), "paired_diff_s": stats(diffs), "paired_diff_s_values": diffs,
           "paired_relative": stats(rel), "paired_relative_values": rel,
           "baseline_elapsed_s": stats([a["elapsed_s"] for a, _ in pairs]), "estimand_overall": rel_estimand(pairs),
           "absolute_ms_per_call_pair_weighted": abs_estimand(pairs, "pair"),
           "per_depth": {str(d): rel_estimand([p for p in pairs if depth_of(p[0]) == d]) for d in sorted({depth_of(a) for a, _ in pairs})}}
    return out


def summarize(out: Path) -> dict:
    cfg = json.loads((out / "run_config.json").read_text())
    rows = [json.loads(x) for x in (out / "decode.jsonl").read_text().splitlines() if x.strip()] if (out / "decode.jsonl").exists() else []
    c = cfg["config"]
    sched = len(c["history_ids"]) * c["repeats"]
    summ = {"config_sha256": cfg["config_sha256"], "rows_with_other_config_hash": sum(r.get("config_sha256") != cfg["config_sha256"] for r in rows),
            "histories_scheduled": len(c["history_ids"]), "histories_unavailable": c.get("histories_n_unavailable"),
            "unique_tasks": len({h.rsplit("@", 1)[0] for h in c["history_ids"]}), "unique_histories": len(set(c["history_ids"])),
            "earlier_call_fallbacks": (c.get("histories_summary") or {}).get("fallback_history_ids"),
            "n_earlier_call_fallbacks": (c.get("histories_summary") or {}).get("n_fallback_to_earlier_call"),
            "repeats": c["repeats"], "max_new_tokens": c["max_new_tokens"], "counts": {}, "capture_s": None,
            "capture_s_note": "unknown in the primary rows (no per-token events); see the separate profile pass"}
    for arm in ("A0", "A1"):
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r["status"] == "ok"]
        summ["counts"][arm] = {"scheduled": sched, "attempted": len(rs), "completed": len(ok), "failed": len(rs) - len(ok),
                               "missing": sched - len(rs), "statuses": {s: sum(r["status"] == s for r in rs) for s in sorted({r["status"] for r in rs})}}
        if ok:
            summ["counts"][arm].update(
                tokens_per_s_aggregate=sum(r["n_generated"] for r in ok) / sum(r["elapsed_s"] for r in ok),
                tokens_per_s_median_run=statistics.median(r["n_generated"] / r["elapsed_s"] for r in ok if r["elapsed_s"] > 0),
                mean_generated=statistics.fmean(r["n_generated"] for r in ok), natural_stops=sum(bool(r["stopped_naturally"]) for r in ok),
                mean_elapsed_s=statistics.fmean(r["elapsed_s"] for r in ok), cache_copy_s=stats([r["cache_copy_s"] for r in ok if r.get("cache_copy_s") is not None]))
    a1 = [r for r in rows if r["arm"] == "A1" and r["status"] == "ok"]
    summ["tap"] = {"decisions": sum(len(r["decisions"] or []) for r in a1), "alarmed": sum(d["alarm"] for r in a1 for d in (r["decisions"] or [])),
                   "eligible_tokens": stats([d["n_tokens"] for r in a1 for d in (r["decisions"] or [])]),
                   "score_wall_s_per_call": stats([r["tap_score_s"] for r in a1 if r.get("tap_score_s") is not None]),
                   "score_wall_s_total": sum_or_none(r.get("tap_score_s") for r in a1)}
    by = defaultdict(dict)
    for r in rows:
        by[(r["history_id"], r["repeat"])][r["arm"]] = r
    both = [(v["A0"], v["A1"]) for v in by.values() if "A0" in v and "A1" in v]
    summ["pairs"] = {"scheduled": sched, "both_arms_attempted": len(both), "missing_an_arm": sched - len(both),
                     "with_a_failed_arm": sum(a["status"] != "ok" or b["status"] != "ok" for a, b in both),
                     "with_an_elapsed_time_missing": sum(a["elapsed_s"] is None or b["elapsed_s"] is None for a, b in both)}
    timed = [(a, b) for a, b in both if a["elapsed_s"] is not None and b["elapsed_s"] is not None]
    clean = [(a, b) for a, b in timed if a["status"] == "ok" and b["status"] == "ok"]
    ident = [(a, b) for a, b in clean if a["token_hash"] == b["token_hash"]]
    summ["all_attempts"] = subset_table(timed)
    summ["clean"] = subset_table(clean)
    summ["normalized_cost_per_100_tokens"] = normalized_per_100_tokens(ident)
    summ["identical_streams"] = {**subset_table(ident), "n_diverging_clean_pairs": len(clean) - len(ident),
                                 "diverging": [{"history_id": a["history_id"], "repeat": a["repeat"], "d_tokens": b["n_generated"] - a["n_generated"]}
                                               for a, b in clean if a["token_hash"] != b["token_hash"]]}
    return summ


def fmt_sig(x, signed: bool = False, sig: int = 2) -> str:
    """Decimal text that keeps `sig` significant digits below 0.1 in magnitude (0.0021, not 0.00) and two decimals otherwise."""
    if x is None:
        return "-"
    sign = "+" if signed and x > 0 else ""
    if x == 0:
        return f"{sign}0"
    a = abs(x)
    decimals = 2 if a >= 0.1 else sig - 1 - math.floor(math.log10(a))
    return f"{sign}{x:.{decimals}f}"


def _pct(x) -> str:
    """A fraction as a signed percentage, with significant figures kept when it is small."""
    return "-" if x is None else fmt_sig(100 * x, signed=True) + "%"


def _estimand_lines(name: str, e: dict) -> list[str]:
    if not e.get("n_histories"):
        return [f"- {name}: no histories"]
    lo, hi = e["ci95"]
    return [f"- {name} (relative overhead, a fraction of baseline elapsed time): mean over {e['n_histories']} histories of {e['n_tasks']} tasks "
            f"{_pct(e['mean_relative_overhead'])}; task-clustered 95% interval [{_pct(lo)}, {_pct(hi)}], achieved half-width "
            f"{fmt_sig(e['half_width_pct_points'])} percentage points of baseline elapsed time; {e['verdict']}."]


def to_markdown(s: dict) -> str:
    L = ["# Focused live decoding, no tap versus tap (descriptive)", "",
         f"Config {s['config_sha256'][:12]}; {s['histories_scheduled']} histories scheduled ({s['histories_unavailable']} unavailable at selection), "
         f"{s['repeats']} repeats, up to {s['max_new_tokens']} new tokens. Estimand: mean paired relative overhead (A1 - A0) / A0 of the "
         "timed region over the selected histories (equal weight per history, per-history mean over repeats); the unit of resampling is "
         f"the TASK (all depths and repeats of a drawn task kept together; {DRAWS} draws, seed {SEED}). Selection: {s['unique_tasks']} unique "
         f"tasks, {s['unique_histories']} unique histories (task-depth prompts), earlier-call fallbacks under the prompt cap: "
         f"{s['n_earlier_call_fallbacks'] if s['n_earlier_call_fallbacks'] is not None else 'not recorded'}"
         f"{' (' + ', '.join(s['earlier_call_fallbacks']) + ')' if s['earlier_call_fallbacks'] else ''}.", ""]
    if s["rows_with_other_config_hash"]:
        L += [f"WARNING: {s['rows_with_other_config_hash']} rows carry another configuration hash.", ""]
    L += ["## Counts", "", "| arm | scheduled | attempted | completed | failed | missing | tokens/s | mean generated | natural stops |", "|---|---|---|---|---|---|---|---|---|"]
    for a, c in s["counts"].items():
        L.append(f"| {a} | {c['scheduled']} | {c['attempted']} | {c['completed']} | {c['failed']} | {c['missing']} | "
                 f"{c.get('tokens_per_s_aggregate', float('nan')):.2f} | {c.get('mean_generated', float('nan')):.1f} | {c.get('natural_stops', '-')} |")
    p = s["pairs"]
    L += ["", f"Pairs: {p['both_arms_attempted']} of {p['scheduled']} have both arms attempted ({p['missing_an_arm']} miss an arm); "
              f"{p['with_a_failed_arm']} include a failed arm. Capture time: null in these rows ({s['capture_s_note']}).", ""]
    for key, title in (("all_attempts", "All attempted pairs"), ("clean", "Clean pairs (both runs ok)"),
                       ("identical_streams", "Clean pairs with identical token streams")):
        t = s[key]
        L += [f"## {title}", ""]
        if t.get("empty"):
            L += [f"This subset is empty ({t.get('n_diverging_clean_pairs', 0)} diverging clean pairs)." if key == "identical_streams" else "This subset is empty.", ""]
            continue
        d, r = t["paired_diff_s"], t["paired_relative"]
        L += [f"{t['n_pairs']} pairs over {t['n_histories']} histories. Paired difference (s): median {d['median']:.4f}, mean {d['mean']:.4f}, "
              f"min {d['min']:.4f}, max {d['max']:.4f}. Paired relative: median {_pct(r['median'])}, min {_pct(r['min'])}, max {_pct(r['max'])}.", ""]
        L += _estimand_lines("overall", t["estimand_overall"])
        ab = t["absolute_ms_per_call_pair_weighted"]
        L += [f"- overall (absolute paired difference, milliseconds per call, {ab['weighting']}): {fmt_sig(ab['mean_paired_difference_ms'], True)} ms, "
              f"task-clustered 95% interval [{fmt_sig(ab['ci95_ms'][0], True)}, {fmt_sig(ab['ci95_ms'][1], True)}] ms; {ab['verdict']}."]
        for dep, e in t["per_depth"].items():
            L += _estimand_lines(f"depth {dep}", e)
        if key == "identical_streams":
            L += ["", f"{t['n_diverging_clean_pairs']} clean pairs have diverging token streams."]
        L.append("")
    n = s["normalized_cost_per_100_tokens"]
    L += ["## Secondary: normalized cost per 100 generated tokens", "", f"Formula: {n['formula']}. {n['note']}. "
          f"{n['n_identical_pairs']} identical-stream pairs, {n['n_zero_denominator_excluded']} excluded for zero generated tokens, {n['n_used']} used."]
    if n["n_used"]:
        lo, hi = n["ci95_ms_per_100_tokens"]
        L += [f"Pair-weighted mean {fmt_sig(n['pair_weighted_mean_ms_per_100_tokens'], True)} ms per 100 tokens, task-clustered 95% interval [{fmt_sig(lo, True)}, {fmt_sig(hi, True)}]."]
    L += ["", "## Tap", "", f"{s['tap']['decisions']} decisions, {s['tap']['alarmed']} above the frozen threshold; eligible tokens per call "
          f"{s['tap']['eligible_tokens']}; call-end scoring wall time per call {s['tap']['score_wall_s_per_call']}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    s = summarize(a.out)
    (a.out / "decode_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (a.out / "decode_summary.md").write_text(to_markdown(s))
    print(to_markdown(s))


if __name__ == "__main__":
    main()

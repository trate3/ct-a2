"""Descriptive summary of a live tap-overhead run (no confidence intervals).

  python -m evals.tap_live.summarize OUT          # reads OUT/episodes.jsonl and OUT/setup*.json; writes OUT/summary.{json,md}

Per arm: counts and the arm's own means. Paired per-task differences against A0 (arm minus A0 for the same task): wall time,
model time (prefill + decode), prefill, decode, tap time (capture + scoring, and each separately; scoring per head), non-model
time (wall - model; and without the tap's call-end scoring), the non-model split (sandbox start, tool execution, in-loop
scoring, orchestration = the rest of the wall clock: rendering, parsing, sandbox file reads, teardown), and the relative
overhead diff / A0 wall, each as median, mean, min, max and the individual values. A second table restricts the pairs to tasks
whose generated token streams (every call) are identical to A0's (hash equal); diverging pairs are listed with their
token and call differences and the first call whose tokens differ. Per-call tap time, capture and scoring separately.
Loading and warm-up are reported apart from the recurring cost. Episodes that ended in an infrastructure error other than the
declared stop are listed and left out of the pairs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("wall_s", "model_s", "prefill_s", "decode_s", "tap_s", "capture_s", "score_s", "nonmodel_s", "nonmodel_ex_tap_s",
           "sandbox_start_s", "tools_s", "scoring_s", "orchestration_s")
COUNTS = ("n_calls", "n_generated", "n_prompt_computed", "n_cached", "resets")
KEY = "task_id"


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": statistics.median(vals), "mean": statistics.fmean(vals), "min": min(vals),
            "max": max(vals), "values": list(vals)}


def load_rows(out: Path) -> list[dict]:
    p = out / "episodes.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def pairs(rows: list[dict], arm: str, clean: bool = True):
    """Paired rows of `arm` and A0 by task. `clean=False` keeps episodes that ended in an infrastructure error (all attempts)."""
    ok = [r for r in rows if not (clean and r.get("infra_error"))]
    base = {r[KEY]: r for r in ok if r["arm"] == "A0"}
    return [(base[r[KEY]], r) for r in ok if r["arm"] == arm and r[KEY] in base]


def _diff(a: dict, b: dict, m: str):
    return None if a.get(m) is None or b.get(m) is None else b[m] - a[m]


def paired_table(prs: list[tuple[dict, dict]]) -> dict:
    t = {m: stats([d for a, b in prs if (d := _diff(a, b, m)) is not None]) for m in METRICS}
    t["relative_wall"] = stats([(b["wall_s"] - a["wall_s"]) / a["wall_s"] for a, b in prs if a["wall_s"] > 0])
    t["relative_model"] = stats([(b["model_s"] - a["model_s"]) / a["model_s"] for a, b in prs if a["model_s"] > 0])
    for c in COUNTS:
        t[c] = stats([b[c] - a[c] for a, b in prs])
    return t


def first_divergent_call(a: dict, b: dict):
    for i, (x, y) in enumerate(zip(a["call_hashes"], b["call_hashes"])):
        if x != y:
            return i
    return None if len(a["call_hashes"]) == len(b["call_hashes"]) else min(len(a["call_hashes"]), len(b["call_hashes"]))


def per_call(rows: list[dict], arm: str) -> dict:
    calls = [c for r in rows if r["arm"] == arm and not r.get("infra_error") for c in r["calls"]]
    out = {"n_calls": len(calls)}
    for k in ("tap_s", "capture_s", "score_s"):
        out[k] = stats([c[k] for c in calls if c.get(k) is not None])
    heads = sorted({h for c in calls for h in (c.get("score_s_by_head") or {})})
    out["score_s_by_head"] = {h: stats([c["score_s_by_head"][h] for c in calls if h in (c.get("score_s_by_head") or {})]) for h in heads}
    out["tap_ms_per_generated_token"] = stats([1000.0 * c["tap_s"] / c["n_generated"] for c in calls
                                               if c.get("tap_s") is not None and c["n_generated"]])
    out["by_call_index"] = {str(i): stats([c["tap_s"] for c in calls if c["call_index"] == i and c.get("tap_s") is not None])
                            for i in sorted({c["call_index"] for c in calls})}
    return out


def summarize(out: Path) -> dict:
    rows = load_rows(out)
    arms = sorted({r["arm"] for r in rows}, key=lambda a: ("A0", "A1").index(a) if a in ("A0", "A1") else 9)
    summ: dict = {"n_rows": len(rows), "paired_vs_A0_all_attempts": {}, "infra_errors": [{k: r[k] for k in (KEY, "arm", "status", "end_reason", "crashed")}
                                                        for r in rows if r.get("infra_error")],
                  "arms": {}, "paired_vs_A0": {}, "paired_identical_streams": {}, "divergence": {}, "per_call_tap": {}, "setup": []}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]          # every attempted episode, infrastructure errors included
        if not rs:
            continue
        summ["arms"][a] = {"n_episodes": len(rs), "n_infra_errors": sum(bool(r.get("infra_error")) for r in rs), "status": {s: sum(r["status"] == s for r in rs) for s in sorted({r["status"] for r in rs})},
                           **{m: stats([r[m] for r in rs if r.get(m) is not None]) for m in METRICS},
                           **{c: stats([r[c] for r in rs]) for c in COUNTS},
                           "alarms": (None if rs[0]["decisions"] is None else
                                      {"decisions": sum(len(r["decisions"]) for r in rs),
                                       "alarmed": sum(d["alarm"] for r in rs for d in r["decisions"])}),
                           "positions": {str(p): sum(r.get("order_position") == p for r in rs) for p in range(len(arms))}}
        if a != "A0":
            prs = pairs(rows, a)
            same = [(x, y) for x, y in prs if x["token_hash"] == y["token_hash"]]
            diff = [(x, y) for x, y in prs if x["token_hash"] != y["token_hash"]]
            summ["paired_vs_A0"][a] = {"n_pairs": len(prs), **paired_table(prs)}
            allp = pairs(rows, a, clean=False)       # infrastructure errors are not dropped: this table keeps every attempt
            summ["paired_vs_A0_all_attempts"][a] = {"n_pairs": len(allp), **paired_table(allp)}
            summ["paired_identical_streams"][a] = {"n_pairs": len(same), **paired_table(same)}
            summ["divergence"][a] = {
                "n_pairs": len(prs), "n_identical": len(same), "n_diverging": len(diff),
                "pairs": [{KEY: y[KEY], "first_divergent_call": first_divergent_call(x, y),
                           "d_tokens": y["n_generated"] - x["n_generated"], "d_calls": y["n_calls"] - x["n_calls"],
                           "d_wall_s": y["wall_s"] - x["wall_s"], "status_A0": x["status"], "status_arm": y["status"]}
                          for x, y in diff]}
            summ["per_call_tap"][a] = per_call(rows, a)
    for p in sorted(out.glob("setup*.json")):
        summ["setup"].append({"file": p.name, **json.loads(p.read_text())})
    return summ


def _f(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _row(name, s, nd=3, pct=False) -> str:
    if not s.get("n"):
        return f"| {name} | 0 | - | - | - | - |"
    k = 100.0 if pct else 1.0
    return (f"| {name} | {s['n']} | {_f(k * s['median'], nd)} | {_f(k * s['mean'], nd)} | {_f(k * s['min'], nd)} | "
            f"{_f(k * s['max'], nd)} |")


def to_markdown(summ: dict) -> str:
    L = ["# Live tap overhead on LiveCodeBench, declared 3-call prefix (descriptive)", "",
         f"{summ['n_rows']} timed episodes. Paired differences are arm minus A0 for the same task; seconds unless stated.", ""]
    if summ["infra_errors"]:
        L += [f"Infrastructure errors ({len(summ['infra_errors'])}): {summ['infra_errors']}", ""]
    L += ["Infrastructure errors are listed above, kept in the all-attempt table and left out of the clean tables. Unknown times are "
          "left out of a column, never counted as zero.", ""]
    L += ["## Arms", "", "| arm | episodes | mean wall s | mean model s | prefill s | decode s | tap s | tools s | calls | generated | resets | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, d in summ["arms"].items():
        L.append(f"| {a} | {d['n_episodes']} | {_f(d['wall_s'].get('mean'), 2)} | {_f(d['model_s'].get('mean'), 2)} | "
                 f"{_f(d['prefill_s'].get('mean'), 2)} | {_f(d['decode_s'].get('mean'), 2)} | {_f(d['tap_s'].get('mean'), 3)} | "
                 f"{_f(d['tools_s'].get('mean'), 2)} | {_f(d['n_calls'].get('mean'), 1)} | {_f(d['n_generated'].get('mean'), 0)} | "
                 f"{_f(d['resets'].get('mean'), 2)} | {d['status']} |")
    hdr = ["| quantity | pairs | median | mean | min | max |", "|---|---|---|---|---|---|"]
    for key, title in (("paired_vs_A0_all_attempts", "All attempted pairs (infrastructure errors kept)"),
                       ("paired_vs_A0", "Clean pairs (no infrastructure error)"),
                       ("paired_identical_streams", "Clean pairs with identical token streams")):
        for a, t in summ[key].items():
            L += ["", f"## {title}: {a} minus A0 ({t['n_pairs']} pairs)", ""] + hdr
            for m in METRICS:
                L.append(_row(m, t[m]))
            L.append(_row("relative wall overhead (%)", t["relative_wall"], 2, pct=True))
            L.append(_row("relative model-time overhead (%)", t["relative_model"], 2, pct=True))
            for c in COUNTS:
                L.append(_row(f"{c} difference", t[c], 1))
    for a, d in summ["divergence"].items():
        L += ["", f"## Divergence of {a} from A0", "",
              f"{d['n_identical']} of {d['n_pairs']} pairs have identical token streams; {d['n_diverging']} diverge."]
        if d["pairs"]:
            L += ["", "| task | first differing call | d tokens | d calls | d wall s | status A0 | status arm |", "|---|---|---|---|---|---|---|"]
            L += [f"| {p[KEY]} | {p['first_divergent_call']} | {p['d_tokens']} | {p['d_calls']} | {_f(p['d_wall_s'], 2)} | "
                  f"{p['status_A0']} | {p['status_arm']} |" for p in d["pairs"]]
    for a, d in summ["per_call_tap"].items():
        L += ["", f"## Per-call tap time, {a} ({d['n_calls']} calls)", ""] + hdr
        for k in ("tap_s", "capture_s", "score_s"):
            L.append(_row(k, d[k], 4))
        for h, s in d["score_s_by_head"].items():
            L.append(_row(f"score_s, head {h}", s, 4))
        L.append(_row("tap ms per generated token", d["tap_ms_per_generated_token"], 4))
        for i, s in d["by_call_index"].items():
            L.append(_row(f"tap_s, call {i}", s, 4))
        al = summ["arms"].get(a, {}).get("alarms")
        if al:
            L += ["", f"Alarm decisions: {al['alarmed']} of {al['decisions']} (call, head) decisions above the frozen threshold."]
    L += ["", "## Loading and warm-up (outside the recurring cost)", ""]
    for s in summ["setup"]:
        L.append(f"{s['file']}: model load {_f(s.get('model_load_s'), 1)} s, untimed warm-up {_f(s.get('warmup_total_s'), 1)} s "
                 f"over {len(s.get('warmup') or [])} dummy calls.")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    summ = summarize(a.out)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    (a.out / "summary.md").write_text(to_markdown(summ))
    print(to_markdown(summ))


if __name__ == "__main__":
    main()

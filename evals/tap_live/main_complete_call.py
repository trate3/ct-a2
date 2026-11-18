"""Main complete-call table inputs: stage 1 (repeat 0) plus stage 2, one paired realization per position, never a selected faster or shorter one.

  python -m evals.tap_live.main_complete_call --stage2 DIR --stage2-list stage2_list.json --out OUT_DIR [--stage1 DIR]

Per position (30, each with the preselected weight 1 / 30 and its task): the A0 and A1 times of ONE paired realization, that of the position's prompt group
representative (prompt-representative design): the stage-1 repeat-0 pair if both calls ended naturally, otherwise the stage-2 natural-completion pair (positions that
share a prompt take the representative's values; `representative_seed_reuse` marks them). The whole call time counts. Reported (valid positions only; a position without a valid pair is listed as missing and the estimates are
flagged incomplete):
  weighted A0 mean seconds per call (the "No tap" value, the same for every tap), weighted A1 mean (probe "With tap"), the weighted paired A1 - A0
  difference per call in milliseconds, each with a 95 % percentile interval from a task bootstrap (2,000 draws, seed 0; tasks resampled with
  replacement, a task's positions kept together).
Sensitivity (clearly labelled, never mixed with the primary): the same estimands on stage-1 repeat 1 for all positions, which is bounded at the
stage-1 cap. Also: actual lengths, natural stops, cap hits, context-ceiling hits, unique prompts and tasks, and the source stage of every position.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals.tap_live import summarize_decode as sd


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def weighted_estimates(items: list[dict]) -> dict:
    """items: {task, weight, a0_s, a1_s}. Weighted means and the weighted paired difference with task-clustered intervals."""
    if not items:
        return {"n_positions": 0}
    out = {"n_positions": len(items), "n_tasks": len({i["task"] for i in items}), "weight_sum": sum(i["weight"] for i in items)}

    def boot(val):
        parts = defaultdict(lambda: [0.0, 0.0])
        for i in items:
            parts[i["task"]][0] += i["weight"] * val(i)
            parts[i["task"]][1] += i["weight"]
        return sd.task_bootstrap({k: tuple(v) for k, v in parts.items()})
    for name, fn, scale in (("no_tap_mean_s_per_call", lambda i: i["a0_s"], 1.0), ("with_probe_mean_s_per_call", lambda i: i["a1_s"], 1.0),
                            ("paired_difference_ms_per_call", lambda i: 1000.0 * (i["a1_s"] - i["a0_s"]), 1.0)):
        b = boot(fn)
        out[name] = {"estimate": b["estimate"], "ci95": b["ci95"]}
    out["bootstrap"] = {"draws": sd.DRAWS, "seed": sd.SEED, "unit": "task (a task's positions kept together)", "weights": "preselected position weights"}
    d = out["paired_difference_ms_per_call"]
    out["interval_statement"] = ("interval includes zero (unresolved)" if d["ci95"][0] <= 0 <= d["ci95"][1] else "interval excludes zero")
    return out


def build(stage1: Path | None, stage2: Path | None, sel: dict) -> dict:
    s1 = {(r["history_id"], r["repeat"], r["arm"]): r for r in load(stage1 / "calls.jsonl")} if stage1 else {}
    s2 = {(r["run_key"], r["arm"]): r for r in load(stage2 / "calls.jsonl")} if stage2 else {}
    rows, items, groups = [], [], {}
    for p in sel["positions"]:
        rep = p.get("rep_history_id", p["history_id"])
        if p["stage"] == "stage1_r0":
            a0, a1, src = s1.get((rep, 0, "A0")), s1.get((rep, 0, "A1")), "stage1_r0"
        else:
            a0, a1, src = s2.get((p["run_key"], "A0")), s2.get((p["run_key"], "A1")), "stage2"
        valid = bool(a0 and a1 and a0["status"] == "ok" and a1["status"] == "ok" and a0["elapsed_s"] is not None and a1["elapsed_s"] is not None)
        row = {"history_id": p["history_id"], "task": p["task"], "depth": p["depth"], "weight": p["weight"], "prompt_sha256": p["prompt_sha256"],
               "representative": rep, "is_representative": p.get("is_representative", True), "representative_seed_reuse": p.get("representative_seed_reuse", False),
               "seed": p.get("seed"), "source_stage": src, "run_key": p.get("run_key"), "valid": valid,
               "a0_s": a0 and a0["elapsed_s"], "a1_s": a1 and a1["elapsed_s"], "n_generated_a0": a0 and a0["n_generated"], "n_generated_a1": a1 and a1["n_generated"],
               "finish_reason_a0": a0 and a0["finish_reason"], "finish_reason_a1": a1 and a1["finish_reason"],
               "token_hash_a0": a0 and a0.get("token_hash"), "token_hash_a1": a1 and a1.get("token_hash"),
               "identical_token_streams": bool(a0 and a1 and a0.get("token_hash") == a1.get("token_hash")),
               "status_a0": a0["status"] if a0 else "missing", "status_a1": a1["status"] if a1 else "missing"}
        rows.append(row)
        g = groups.setdefault(p["prompt_sha256"], {"prompt_sha256": p["prompt_sha256"], "representative": rep, "seed": p.get("seed"), "positions": [], "tasks": set(),
                                                   "source_stage": src, "valid": valid, "a0_s": row["a0_s"], "a1_s": row["a1_s"], "token_hash_a0": row["token_hash_a0"],
                                                   "token_hash_a1": row["token_hash_a1"], "finish_reason_a0": row["finish_reason_a0"],
                                                   "finish_reason_a1": row["finish_reason_a1"], "n_generated_a0": row["n_generated_a0"]})
        g["positions"].append(p["history_id"])
        g["tasks"].add(p["task"])
        if valid:
            items.append({"task": p["task"], "weight": p["weight"], "a0_s": a0["elapsed_s"], "a1_s": a1["elapsed_s"]})
    group_list = [{**g, "tasks": sorted(g["tasks"])} for g in groups.values()]
    valid_rows = [r for r in rows if r["valid"]]
    wsum = sum(r["weight"] for r in valid_rows)
    lens = {"weighted_mean_generated_tokens_a0": sum(r["weight"] * r["n_generated_a0"] for r in valid_rows) / wsum if wsum else None,
            "min": min((r["n_generated_a0"] for r in valid_rows), default=None), "max": max((r["n_generated_a0"] for r in valid_rows), default=None)}
    res = {"selection_mode": sel.get("mode"), "primary_design": bool(sel.get("primary", True)),
           "design_note": ("prompt-representative design: each position reuses the realization of its prompt group's representative (first in manifest order); "
                           "positions keep their preselected weights; uncertainty resamples tasks, so reuse within a task is counted together and the 30 positions "
                           "are not 30 independent prompts"),
           "representative_seed_reuse_note": sel.get("representative_seed_reuse_note"),
           "stage1": str(stage1) if stage1 else None, "stage2": str(stage2) if stage2 else None, "stage1_run_config_sha256": (sel.get("stage1") or {}).get("run_config_sha256"),
           "stage2_rule_sha256": sel["rule_sha256"], "n_positions": len(rows), "n_valid_positions": len(valid_rows),
           "complete": len(valid_rows) == len(rows), "missing_positions": [r["history_id"] for r in rows if not r["valid"]],
           "distinct_prompts": len(groups), "unique_tasks": len({r["task"] for r in rows}),
           "groups_spanning_several_tasks": sum(len(g["tasks"]) > 1 for g in group_list),
           "positions_reusing_a_representative_seed": sum(r["representative_seed_reuse"] for r in rows),
           "positions_by_source": {k: sum(r["source_stage"] == k for r in rows) for k in ("stage1_r0", "stage2")},
           "counts": {"natural_stops_a0": sum(r["finish_reason_a0"] == "natural" for r in valid_rows), "natural_stops_a1": sum(r["finish_reason_a1"] == "natural" for r in valid_rows),
                      "cap_hits_a0": sum(r["finish_reason_a0"] == "cap" for r in valid_rows), "cap_hits_a1": sum(r["finish_reason_a1"] == "cap" for r in valid_rows),
                      "context_ceiling_hits_a0": sum(r["finish_reason_a0"] == "context_ceiling" for r in valid_rows),
                      "context_ceiling_hits_a1": sum(r["finish_reason_a1"] == "context_ceiling" for r in valid_rows),
                      "identical_token_streams": sum(r["identical_token_streams"] for r in valid_rows)},
           "original_failures": [{"history_id": r["history_id"], "status_a0": r["status_a0"], "status_a1": r["status_a1"]} for r in rows if not r["valid"]],
           "lengths": lens, "primary": weighted_estimates(items), "positions": rows, "groups": group_list}
    # sensitivity (separate; never mixed with the primary)
    s_rep, s_rep_rows, s_own, s_own_rows = [], [], [], []
    for p in sel["positions"]:
        rep = p.get("rep_history_id", p["history_id"])
        a0, a1 = s1.get((rep, 1, "A0")), s1.get((rep, 1, "A1"))
        if a0 and a1 and a0["status"] == "ok" and a1["status"] == "ok":
            s_rep.append({"task": p["task"], "weight": p["weight"], "a0_s": a0["elapsed_s"], "a1_s": a1["elapsed_s"]})
            s_rep_rows.append((a0, a1))
        o0, o1 = s1.get((p["history_id"], 0, "A0")), s1.get((p["history_id"], 0, "A1"))
        if o0 and o1 and o0["status"] == "ok" and o1["status"] == "ok":
            s_own.append({"task": p["task"], "weight": p["weight"], "a0_s": o0["elapsed_s"], "a1_s": o1["elapsed_s"]})
            s_own_rows.append((o0, o1))

    def sens(label, its, prs):
        return {"label": label, **weighted_estimates(its), "natural_stops_a0": sum(a0["finish_reason"] == "natural" for a0, _ in prs),
                "cap_hits_a0": sum(a0["finish_reason"] == "cap" for a0, _ in prs), "mean_generated_tokens_a0": sum(a0["n_generated"] for a0, _ in prs) / len(prs)}
    cap = (sel.get("stage1") or {}).get("cap")
    if s_rep:
        res["sensitivity_stage1_repeat1"] = sens(f"SENSITIVITY, not the primary: stage-1 repeat index 1 of each position's representative pair, bounded at the stage-1 cap "
                                                 f"({cap} new tokens); capped calls are incomplete realizations", s_rep, s_rep_rows)
    if s_own:
        res["sensitivity_own_seed_repeat0"] = sens(f"SENSITIVITY, not the primary: every position's OWN stage-1 repeat-0 pair (its own seed), bounded at the stage-1 cap "
                                                   f"({cap} new tokens); capped calls are incomplete realizations", s_own, s_own_rows)
    # supporting table: the curtailed stage-1 repeat-0 pair of each group's representative (never used by the primary of the default design)
    sup = []
    for g in group_list:
        a0, a1 = s1.get((g["representative"], 0, "A0")), s1.get((g["representative"], 0, "A1"))
        if a0 is None and a1 is None:
            continue
        nat = bool(a0 and a1 and a0.get("finish_reason") == "natural" and a1.get("finish_reason") == "natural")
        sup.append({"prompt_sha256": g["prompt_sha256"], "representative": g["representative"], "stage1_a0_s": a0 and a0["elapsed_s"], "stage1_a1_s": a1 and a1["elapsed_s"],
                    "stage1_n_generated_a0": a0 and a0["n_generated"], "stage1_finish_reason_a0": a0 and a0["finish_reason"],
                    "stage1_finish_reason_a1": a1 and a1["finish_reason"], "stage1_pair_natural": nat,
                    "main_token_hash_a0": g["token_hash_a0"], "stage1_token_hash_a0": a0 and a0.get("token_hash"),
                    "main_vs_stage1_token_stream_identical": (None if not (nat and g["token_hash_a0"]) else g["token_hash_a0"] == (a0 or {}).get("token_hash")),
                    "main_a0_s": g["a0_s"], "stage1_a0_s_over_main_a0_s": (a0["elapsed_s"] / g["a0_s"]) if (a0 and a0.get("elapsed_s") is not None and g["a0_s"]) else None})
    res["primary_uses_stage1_pairs"] = res["positions_by_source"]["stage1_r0"]
    res["supporting_stage1_pairs"] = {"label": "SUPPORTING, not the primary: the curtailed stage-1 repeat-0 pairs (cap 4,096 tokens) of the representatives; elapsed times are not "
                                               "expected to be equal to the main stage's", "groups": sup, "n_groups": len(sup), "n_natural_pairs": sum(s_["stage1_pair_natural"] for s_ in sup)}
    return res


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(r: dict) -> str:
    p = r["primary"]
    L = ["# Main complete-call table inputs (descriptive)", "",
         f"{r['n_valid_positions']} of {r['n_positions']} positions have a valid paired realization ({'complete' if r['complete'] else 'INCOMPLETE; missing: ' + ', '.join(r['missing_positions'])}); "
         f"{r['distinct_prompts']} distinct prompts, {r['unique_tasks']} tasks; sources: {r['positions_by_source']}. {r['design_note']}. Each position uses its group's "
         "representative pair: stage-1 repeat 0 if both calls ended naturally, otherwise the stage-2 natural-completion pair with the same representative seed; whole call "
         f"time counted; no faster or shorter realization is ever chosen. {r['positions_reusing_a_representative_seed']} positions show `representative_seed_reuse`: "
         "they reuse their group's representative seed and pair; that is never a mismatch between A0 and A1 or between the timed generation and the output a monitor judges. "
         f"{r['groups_spanning_several_tasks']} prompt groups span several tasks.", "",
         f"Counts (valid positions): natural stops A0 {r['counts']['natural_stops_a0']}, A1 {r['counts']['natural_stops_a1']}; cap hits A0 {r['counts']['cap_hits_a0']}, "
         f"A1 {r['counts']['cap_hits_a1']}; context-ceiling hits A0 {r['counts']['context_ceiling_hits_a0']}, A1 {r['counts']['context_ceiling_hits_a1']}; identical A0/A1 token "
         f"streams {r['counts']['identical_token_streams']}. Weighted mean generated tokens {_f(r['lengths']['weighted_mean_generated_tokens_a0'], 0)} "
         f"(range {r['lengths']['min']} to {r['lengths']['max']}).", ""]
    if p["n_positions"]:
        L += ["## Primary (weighted over positions; task-clustered bootstrap)", ""]
        for k, t in (("no_tap_mean_s_per_call", "No tap, mean seconds per call"), ("with_probe_mean_s_per_call", "With probe, mean seconds per call"),
                     ("paired_difference_ms_per_call", "Paired A1 - A0, milliseconds per call")):
            e = p[k]
            nd = 2 if k.endswith("ms_per_call") else 3
            L.append(f"- {t}: {sd.fmt_sig(e['estimate'], k.startswith('paired'))}  [{sd.fmt_sig(e['ci95'][0], k.startswith('paired'))}, {sd.fmt_sig(e['ci95'][1], k.startswith('paired'))}]")
        L += ["", f"The paired-difference {p['interval_statement']}. Intervals: {p['bootstrap']['draws']} draws, seed {p['bootstrap']['seed']}, unit {p['bootstrap']['unit']}."]
    sp = r.get("supporting_stage1_pairs")
    if sp and sp["n_groups"]:
        L += ["", "## Supporting stage-1 pairs (not the primary)", "", f"{sp['label']}. {sp['n_groups']} representatives have a stage-1 repeat-0 call; {sp['n_natural_pairs']} pairs ended naturally. "
              f"Primary positions taken from stage 1: {r['primary_uses_stage1_pairs']}."]
    for key in ("sensitivity_stage1_repeat1", "sensitivity_own_seed_repeat0"):
        t = r.get(key)
        if t:
            d = t["paired_difference_ms_per_call"]
            L += ["", f"## Sensitivity (not the primary): {key}", "", t["label"] + ".", "",
                  f"- No tap {sd.fmt_sig(t['no_tap_mean_s_per_call']['estimate'])} s; with probe {sd.fmt_sig(t['with_probe_mean_s_per_call']['estimate'])} s; paired difference "
                  f"{sd.fmt_sig(d['estimate'], True)} ms [{sd.fmt_sig(d['ci95'][0], True)}, {sd.fmt_sig(d['ci95'][1], True)}]; natural stops {t['natural_stops_a0']}, cap hits "
                  f"{t['cap_hits_a0']}, mean tokens {_f(t['mean_generated_tokens_a0'], 0)}."]
    L += ["", "## Positions", "", "| position | representative | seed reuse | source | A0 s | A1 s | tokens (A0) | finish A0 / A1 | weight |", "|---|---|---|---|---|---|---|---|---|"]
    for q in r["positions"]:
        L.append(f"| {q['history_id']} | {q['representative']} | {'yes' if q['representative_seed_reuse'] else 'no'} | {q['source_stage']} | {_f(q['a0_s'])} | {_f(q['a1_s'])} | "
                 f"{q['n_generated_a0']} | {q['finish_reason_a0']} / {q['finish_reason_a1']} | {q['weight']:.4f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, default=None, help="folder with stage-1 calls.jsonl, if a stage 1 was run")
    ap.add_argument("--stage2", type=Path, default=None)
    ap.add_argument("--stage2-list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    r = build(a.stage1, a.stage2, json.loads(a.stage2_list.read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "main_complete_call.json").write_text(json.dumps(r, indent=1) + "\n")
    (a.out / "main_complete_call.md").write_text(to_markdown(r))
    print(to_markdown(r))


if __name__ == "__main__":
    main()

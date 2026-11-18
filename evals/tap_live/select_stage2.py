"""Selection of the main complete-call stage (prompt-representative design).

  python -m evals.tap_live.select_stage2 --histories histories.json --out stage2_list.json [--recorded-lengths lengths.json] [--tok-per-s 24.4] \
      [--stage1 DIR_OF_STAGE1 --consistency-out stage1_consistency.json]

PRIMARY MODE `all_representatives` (default; the main result): ONE main stage generates ALL distinct-prompt representatives uncapped (benchmark policy: no per-call
cap; natural stop, or the 240,000-token total-context ceiling recorded as `context_ceiling`), both arms, with the fixed representative seeds.
  1. the positions are grouped by prompt sha256 into the distinct prompts (24 in the fixed selection), in the order of first appearance in histories.json;
  2. the REPRESENTATIVE of a group is its first position in that order; its index is the group's index 0..N-1 in manifest order; its seed is the stage-1
     repeat-0 seed sha256('tap-live-decode:calls:<history_id>:0'), computed, not read from any stage-1 file;
  3. every group is run; each position keeps its preselected weight 1 / n_positions and takes its group's representative pair; the arm order alternates by
     representative index (A0 first for even, A1 first for odd), fixed in the stage config;
  4. the list depends only on histories.json and the rule: it does not change as stage 1 progresses. Stage-1 pairs (curtailed, capped at 4,096 tokens)
     are supporting data only; whether a group's stage-1 repeat-0 pair exists and ended naturally goes to a SEPARATE consistency file (`--stage1`).

EARLIER MODE `representative` (capped-only reruns; kept behind --mode), a disclosed cost-workload change:
  1. the 30 positions are grouped by prompt sha256 into the distinct prompts (24 in the fixed selection);
  2. the REPRESENTATIVE of a group is its first position in the fixed manifest order (the order of histories.json); its stage-1 repeat-0 seed
     and its repeat-0 A0/A1 pair define the group's realization; every other label of the group reuses it with its own preselected weight
     (1 / number of positions);
  3. a group needs stage 2 iff the REPRESENTATIVE's repeat-0 A0 call or A1 call ended at the cap (finish_reason "cap") or failed (status not ok,
     or no row). A non-representative label's own repeat-0 outcome is irrelevant for the main result; it is kept as sensitivity evidence;
  4. stage 2 reruns the representative with the SAME representative seed in both arms (A0 first, as in stage 1);
  5. distinct prompts are never merged; no timing, length or stability enters the choice.
`representative_seed_reuse` marks a position that reuses its group's representative seed (and pair). It never denotes a mismatch between A0 and A1,
or between the timed generation and the output that a monitor will judge: both come from the same representative A0 call.

Sensitivity mode `position_seed` (not primary): every position keeps its own seed and its own repeat-0 pair; positions with the same (prompt sha256, seed)
share one run; a position needs stage 2 iff its own repeat-0 A0 or A1 call hit the cap or failed.

The output prints the number of distinct prompts, the number of representatives to rerun and an expected token total (see `expected_tokens`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.tap_live import decode_core as dc

RULE = ("Prompt-representative design. Positions are grouped by prompt sha256; the representative of a group is its first position in the "
        "manifest order of histories.json; its stage-1 repeat-0 seed and repeat-0 A0/A1 pair define the group's realization and every position of the group "
        "reuses it with its own preselected weight 1 / n_positions. A group needs stage 2 iff the representative's repeat-0 A0 or A1 call ended at the cap "
        "(finish_reason cap) or failed (status not ok, or no row); stage 2 reruns the representative with the same representative seed in both arms. A "
        "non-representative label's own outcome is sensitivity evidence only. Distinct prompts are never merged; no timing, length or stability enters the choice.")
RULE_SHA256 = hashlib.sha256(RULE.encode()).hexdigest()
RULE_ALL = ("Prompt-representative design, one main stage. Positions are grouped by prompt sha256 in the manifest order of histories.json; "
            "the representative of a group is its first position, its index is the group's index in that order, its seed is the stage-1 repeat-0 seed "
            "sha256('tap-live-decode:calls:<history_id>:0'). ALL groups are run uncapped (no per-call cap; natural stop or the 240,000-token total-context ceiling, "
            "recorded as context_ceiling), both arms, the arm order alternating by representative index (A0 first for even, A1 first for odd). Every position keeps its "
            "preselected weight 1 / n_positions and takes its group's representative pair. Stage-1 pairs are supporting data only. Distinct prompts are never merged.")
RULE_ALL_SHA256 = hashlib.sha256(RULE_ALL.encode()).hexdigest()
ARM_ORDER_RULE = "alternating by representative index: A0 first for even index, A1 first for odd index"
RULE_POSITION_SEED = ("SENSITIVITY, not primary: every position keeps its own seed and its own repeat-0 pair; a position needs stage 2 iff its own repeat-0 A0 or A1 call ended "
                      "at the cap or failed; positions with the same (prompt sha256, seed) share one run; every position keeps its preselected weight 1 / n_positions.")
RULE_POSITION_SEED_SHA256 = hashlib.sha256(RULE_POSITION_SEED.encode()).hexdigest()
SEED_REUSE_NOTE = ("representative_seed_reuse: the position reuses its prompt group's representative seed and representative A0/A1 pair; it never denotes a "
                   "mismatch between A0 and A1 or between the timed generation and the output a monitor will judge (both are the representative A0 call).")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _reasons(r0: dict) -> list[str]:
    out = []
    for a, r in r0.items():
        if r is None:
            out.append(f"{a}: no repeat-0 row")
        elif r["status"] != "ok":
            out.append(f"{a}: failed ({r['status']})")
        elif r["finish_reason"] == "cap":
            out.append(f"{a}: ended at the cap")
    return out


def _seed(r0: dict, hid: str):
    return next((r["seed"] for r in r0.values() if r is not None), None) if any(r0.values()) else dc.seed_for("calls", hid, 0)


def _own(r0: dict) -> dict:
    return {a: (None if r is None else {"status": r["status"], "finish_reason": r.get("finish_reason"), "n_generated": r.get("n_generated"),
                                         "token_hash": r.get("token_hash")}) for a, r in r0.items()}


def expected_tokens(run_list: list[dict], recorded_lengths: dict | None, tok_per_s: float) -> dict:
    out = {"representatives_to_run": len(run_list), "tokens_per_s_assumed": tok_per_s}
    if recorded_lengths is None:
        return {**out, "recorded_lengths_used": False, "note": "no recorded lengths given: no token estimate"}
    per = {r["history_id"]: recorded_lengths.get(r["history_id"]) for r in run_list}
    tot = sum(v for v in per.values() if v is not None)
    return {**out, "recorded_lengths_used": True, "stage2_tokens_per_arm": tot, "stage2_tokens_both_arms": 2 * tot,
            "missing_recorded_lengths": [k for k, v in per.items() if v is None], "stage2_hours_both_arms": 2 * tot / tok_per_s / 3600.0,
            "per_representative": per, "note": "the originally recorded call length of each representative's prompt; the new realization (its own seed) may differ"}


def select_all(histories_path: Path, recorded_lengths: dict | None = None, tok_per_s: float = 24.4) -> dict:
    doc = json.loads(Path(histories_path).read_text())
    hist = [h for h in doc["histories"] if h["status"] == "ok"]                    # the fixed manifest order
    n = len(hist)
    groups = {}
    for h in hist:
        g = groups.setdefault(h["prompt_sha256"], {"prompt_sha256": h["prompt_sha256"], "representative": h["history_id"], "positions": [], "tasks": set(),
                                                   "prompt_tokens": h["prompt_tokens"]})
        g["positions"].append(h["history_id"])
        g["tasks"].add(h["task"])
    group_list, runs, positions = [], [], []
    for idx, g in enumerate(groups.values()):
        rep = g["representative"]
        seed = dc.seed_for("calls", rep, 0)
        order = ["A0", "A1"] if idx % 2 == 0 else ["A1", "A0"]
        group_list.append({"representative_index": idx, "prompt_sha256": g["prompt_sha256"], "representative": rep, "seed": seed, "positions": g["positions"],
                           "n_positions": len(g["positions"]), "weight": len(g["positions"]) / n, "tasks": sorted(g["tasks"]), "arm_order": order})
        runs.append({"run_key": g["prompt_sha256"], "representative_index": idx, "history_id": rep, "seed": seed, "prompt_sha256": g["prompt_sha256"],
                     "prompt_tokens": g["prompt_tokens"], "positions": list(g["positions"]), "arm_order": order, "reasons": ["main stage: every representative is run"]})
    gi = {g["prompt_sha256"]: g for g in group_list}
    for h in hist:
        g = gi[h["prompt_sha256"]]
        positions.append({"history_id": h["history_id"], "task": h["task"], "depth": h["depth_fraction"], "call_index": h["call_index"], "weight": 1.0 / n,
                          "prompt_sha256": h["prompt_sha256"], "prompt_tokens": h["prompt_tokens"], "representative": g["representative"],
                          "rep_history_id": g["representative"], "representative_index": g["representative_index"], "is_representative": h["history_id"] == g["representative"],
                          "seed": g["seed"], "own_seed": dc.seed_for("calls", h["history_id"], 0), "representative_seed_reuse": h["history_id"] != g["representative"],
                          "stage": "stage2", "reasons": [], "run_key": g["prompt_sha256"]})
    return {"mode": "all_representatives", "primary": True, "rule": RULE_ALL, "rule_sha256": RULE_ALL_SHA256, "arm_order_rule": ARM_ORDER_RULE, "n_positions": n,
            "weight_per_position": 1.0 / n, "representative_seed_reuse_note": SEED_REUSE_NOTE, "histories_file": str(histories_path),
            "histories_sha256": sha256_file(histories_path), "positions": positions, "groups": group_list, "runs": runs,
            "expected_tokens": expected_tokens(runs, recorded_lengths, tok_per_s),
            "summary": {"n_positions": n, "distinct_prompts": len(groups), "representatives_to_run": len(runs), "runs": len(runs), "representatives_to_rerun": len(runs),
                        "positions_to_rerun": n,
                        "positions_reusing_a_representative_seed": sum(p["representative_seed_reuse"] for p in positions),
                        "groups_spanning_several_tasks": sum(len(g["tasks"]) > 1 for g in group_list)}}


def stage1_consistency(stage1: Path, lst: dict) -> dict:
    """For the later consistency check only (a separate file; the list itself never depends on stage 1): whether each representative's stage-1 repeat-0 pair
    exists, whether both calls ended naturally, whether the stage-1 seed equals the list's seed, and the stage-1 token hashes."""
    stage1 = Path(stage1)
    cfg = json.loads((stage1 / "run_config.json").read_text())
    arms = cfg["config"].get("arms", ["A0", "A1"])
    rows = {(r["history_id"], r["repeat"], r["arm"]): r for r in load_jsonl(stage1 / "calls.jsonl")}
    out = []
    for g in lst["groups"]:
        r0 = {a: rows.get((g["representative"], 0, a)) for a in arms}
        ok = {a: r for a, r in r0.items() if r is not None}
        out.append({"representative_index": g["representative_index"], "representative": g["representative"], "pair_exists": len(ok) == len(arms),
                    "pair_natural": len(ok) == len(arms) and all(r["status"] == "ok" and r["finish_reason"] == "natural" for r in ok.values()),
                    "seed_matches": all(r["seed"] == g["seed"] for r in ok.values()) if ok else None,
                    "finish_reasons": {a: (r or {}).get("finish_reason") for a, r in r0.items()}, "token_hashes": {a: (r or {}).get("token_hash") for a, r in r0.items()}})
    return {"stage1_dir": str(stage1), "stage1_run_config_sha256": cfg["config_sha256"], "stage1_calls_sha256": sha256_file(stage1 / "calls.jsonl"),
            "note": "for the consistency check only; stage 1 is curtailed supporting data (cap 4,096 tokens)", "list_rule_sha256": lst["rule_sha256"], "groups": out,
            "n_pairs_exist": sum(o["pair_exists"] for o in out), "n_pairs_natural": sum(o["pair_natural"] for o in out)}


def select(stage1: Path | None, histories_path: Path, mode: str = "all_representatives", recorded_lengths: dict | None = None, tok_per_s: float = 24.4) -> dict:
    if mode == "all_representatives":
        return select_all(histories_path, recorded_lengths, tok_per_s)
    if mode not in ("representative", "position_seed"):
        raise ValueError(mode)
    if stage1 is None:
        raise ValueError(f"mode {mode} needs --stage1")
    stage1 = Path(stage1)
    cfg = json.loads((stage1 / "run_config.json").read_text())
    arms = cfg["config"].get("arms", ["A0", "A1"])
    rows = {(r["history_id"], r["repeat"], r["arm"]): r for r in load_jsonl(stage1 / "calls.jsonl")}
    doc = json.loads(Path(histories_path).read_text())
    hist = [h for h in doc["histories"] if h["status"] == "ok"]                    # the fixed manifest order
    n = len(hist)
    r0_of = {h["history_id"]: {a: rows.get((h["history_id"], 0, a)) for a in arms} for h in hist}
    rep_of, groups = {}, {}
    for h in hist:
        g = groups.setdefault(h["prompt_sha256"], {"prompt_sha256": h["prompt_sha256"], "representative": h["history_id"], "positions": []})
        g["positions"].append(h["history_id"])
        rep_of[h["history_id"]] = g["representative"]
    positions, runs, verified, group_list = [], {}, [], []
    if mode == "representative":
        for g in groups.values():
            rep = g["representative"]
            r0 = r0_of[rep]
            reasons = _reasons(r0)
            seed = _seed(r0, rep)
            g.update(seed=seed, stage="stage2" if reasons else "stage1_r0", reasons=reasons, n_positions=len(g["positions"]),
                     representative_repeat0={a: _own(r0)[a] for a in arms})
            if reasons:
                runs[g["prompt_sha256"]] = {"run_key": g["prompt_sha256"], "history_id": rep, "seed": seed, "prompt_sha256": g["prompt_sha256"],
                                            "prompt_tokens": next(h["prompt_tokens"] for h in hist if h["history_id"] == rep), "positions": list(g["positions"]),
                                            "reasons": [f"{rep}: {x}" for x in reasons]}
            else:
                ok = {a: r for a, r in r0.items() if r is not None}
                verified.append({"history_id": rep, "both_natural": all(r["finish_reason"] == "natural" for r in ok.values()),
                                 "a0_a1_identical_token_hash": (len({r["token_hash"] for r in ok.values()}) == 1) if len(ok) > 1 else None})
            group_list.append(g)
        for h in hist:
            hid, g = h["history_id"], groups[h["prompt_sha256"]]
            own = r0_of[hid]
            own_seed = _seed(own, hid)
            positions.append({"history_id": hid, "task": h["task"], "depth": h["depth_fraction"], "call_index": h["call_index"], "weight": 1.0 / n,
                              "prompt_sha256": h["prompt_sha256"], "prompt_tokens": h["prompt_tokens"], "representative": g["representative"],
                              "rep_history_id": g["representative"], "is_representative": hid == g["representative"], "seed": g["seed"],
                              "own_seed": own_seed, "representative_seed_reuse": own_seed != g["seed"] or hid != g["representative"],
                              "stage": g["stage"], "reasons": g["reasons"], "run_key": g["prompt_sha256"] if g["stage"] == "stage2" else None,
                              "own_repeat0_sensitivity": _own(own)})
    else:                                                                              # position_seed: sensitivity
        for h in hist:
            hid = h["history_id"]
            r0 = r0_of[hid]
            reasons = _reasons(r0)
            seed = _seed(r0, hid)
            pos = {"history_id": hid, "task": h["task"], "depth": h["depth_fraction"], "call_index": h["call_index"], "weight": 1.0 / n,
                   "prompt_sha256": h["prompt_sha256"], "prompt_tokens": h["prompt_tokens"], "representative": hid, "rep_history_id": hid, "is_representative": True,
                   "seed": seed, "own_seed": seed, "representative_seed_reuse": False, "stage": "stage2" if reasons else "stage1_r0", "reasons": reasons,
                   "run_key": None, "own_repeat0_sensitivity": _own(r0)}
            if reasons:
                key = f"{h['prompt_sha256']}:{seed}"
                pos["run_key"] = key
                run = runs.setdefault(key, {"run_key": key, "history_id": hid, "seed": seed, "prompt_sha256": h["prompt_sha256"],
                                            "prompt_tokens": h["prompt_tokens"], "positions": [], "reasons": []})
                pos["representative_seed_reuse"] = hid != run["history_id"]
                run["positions"].append(hid)
                run["reasons"] += [f"{hid}: {x}" for x in reasons]
            else:
                ok = {a: r for a, r in r0.items() if r is not None}
                verified.append({"history_id": hid, "both_natural": all(r["finish_reason"] == "natural" for r in ok.values()),
                                 "a0_a1_identical_token_hash": (len({r["token_hash"] for r in ok.values()}) == 1) if len(ok) > 1 else None})
            positions.append(pos)
    run_list = list(runs.values())
    # sensitivity evidence: non-representative labels whose own outcome differs from the representative's
    sens = {}
    if mode == "representative":
        for p in positions:
            if p["is_representative"]:
                continue
            rep_cap = bool(groups[p["prompt_sha256"]]["reasons"])
            own_cap = any(v is None or v["status"] != "ok" or v["finish_reason"] == "cap" for v in p["own_repeat0_sensitivity"].values())
            sens.setdefault("non_representative_labels", 0)
            sens["non_representative_labels"] += 1
            if own_cap != rep_cap:
                sens.setdefault("own_outcome_differs_from_representative", []).append(
                    {"history_id": p["history_id"], "representative": p["representative"], "representative_needs_stage2": rep_cap, "own_hit_cap_or_failed": own_cap})
    # expected tokens
    stage1_tokens = {a: sum(r["n_generated"] for g in group_list if g["stage"] == "stage1_r0" for r in [rows.get((g["representative"], 0, a))] if r and r["status"] == "ok" and r["finish_reason"] == "natural")
                     for a in arms} if mode == "representative" else None
    exp = None
    if mode == "representative":
        cap = cfg["config"].get("cap")
        exp = {"stage1_cap": cap, "representatives_to_rerun": len(run_list),
               "stage1_tokens_of_representative_pairs_that_completed": stage1_tokens,
               "stage2_tokens_lower_bound_per_arm": len(run_list) * (cap or 0),
               "note": "stage-1 lengths of capped calls are cut at the cap, so the stage-2 token total is only bounded below by (reruns x cap) per arm unless recorded lengths are given"}
        if recorded_lengths is not None:
            per = {r["history_id"]: recorded_lengths.get(r["history_id"]) for r in run_list}
            missing = [k for k, v in per.items() if v is None]
            tot = sum(v for v in per.values() if v is not None)
            exp.update(recorded_lengths_used=True, stage2_tokens_per_arm=tot, stage2_tokens_both_arms=2 * tot, missing_recorded_lengths=missing,
                       tokens_per_s_assumed=tok_per_s, stage2_hours_both_arms=2 * tot / tok_per_s / 3600.0,
                       per_representative={k: v for k, v in per.items()})
    rule, rsha = (RULE, RULE_SHA256) if mode == "representative" else (RULE_POSITION_SEED, RULE_POSITION_SEED_SHA256)
    return {"mode": mode, "primary": mode == "representative", "rule": rule, "rule_sha256": rsha, "n_positions": n, "weight_per_position": 1.0 / n,
            "representative_seed_reuse_note": SEED_REUSE_NOTE,
            "stage1": {"dir": str(stage1), "run_config_sha256": cfg["config_sha256"], "calls_sha256": sha256_file(stage1 / "calls.jsonl"),
                       "outputs_sha256": sha256_file(stage1 / "outputs.jsonl") if (stage1 / "outputs.jsonl").exists() else None,
                       "cap": cfg["config"].get("cap"), "arms": arms, "repeats": cfg["config"].get("repeats")},
            "histories_file": str(histories_path), "histories_sha256": sha256_file(histories_path),
            "positions": positions, "groups": group_list if mode == "representative" else None, "runs": run_list, "expected_tokens": exp,
            "summary": {"n_positions": n, "distinct_prompts": len(groups), "positions_to_rerun": sum(p["stage"] == "stage2" for p in positions),
                        "representatives_to_rerun": len(run_list), "runs": len(run_list),
                        "positions_reusing_a_representative_seed": sum(p["representative_seed_reuse"] for p in positions)},
            "sensitivity_evidence": sens,
            "non_rerun_check": {"n": len(verified), "all_both_natural": all(v["both_natural"] for v in verified),
                                "all_a0_a1_identical": all(v["a0_a1_identical_token_hash"] is not False for v in verified),
                                "not_identical": [v["history_id"] for v in verified if v["a0_a1_identical_token_hash"] is False],
                                "not_natural": [v["history_id"] for v in verified if not v["both_natural"]], "per_position": verified}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, default=None, help="needed for the earlier modes; in the default mode only for --consistency-out")
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--consistency-out", type=Path, default=None, help="default mode: write the stage-1 consistency info (separate from the list)")
    ap.add_argument("--mode", choices=("all_representatives", "representative", "position_seed"), default="all_representatives",
                    help="all_representatives: the main stage (default); representative: capped-only reruns (earlier); position_seed: every position keeps its own seed "
                         "(sensitivity, not primary)")
    ap.add_argument("--recorded-lengths", type=Path, default=None, help="optional JSON {history_id: tokens of the originally recorded call} for the token estimate")
    ap.add_argument("--tok-per-s", type=float, default=24.4)
    a = ap.parse_args(argv)
    rec = json.loads(a.recorded_lengths.read_text()) if a.recorded_lengths else None
    res = select(a.stage1, a.histories, a.mode, rec, a.tok_per_s)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    s = res["summary"]
    if res["mode"] == "all_representatives":
        print(f"mode all_representatives (main stage): {s['n_positions']} positions, {s['distinct_prompts']} distinct prompts, {s['representatives_to_run']} representatives to run "
              f"(all, uncapped, both arms, alternating order), {s['positions_reusing_a_representative_seed']} positions reuse a representative seed; list sha256 "
              f"{sha256_file(a.out)}; rule sha256 {res['rule_sha256']}")
        e = res["expected_tokens"]
        if e["recorded_lengths_used"]:
            print(f"expected tokens (recorded lengths of the representatives): {e['stage2_tokens_per_arm']} per arm, {e['stage2_tokens_both_arms']} both arms, about "
                  f"{e['stage2_hours_both_arms']:.2f} h at {e['tokens_per_s_assumed']} tok/s; missing: {e['missing_recorded_lengths']}")
        else:
            print("no --recorded-lengths: no token estimate")
        if a.consistency_out is not None:
            if a.stage1 is None:
                raise SystemExit("--consistency-out needs --stage1")
            cons = stage1_consistency(a.stage1, res)
            a.consistency_out.write_text(json.dumps(cons, indent=1) + "\n")
            print(f"stage-1 consistency: {cons['n_pairs_exist']} of {len(cons['groups'])} representative pairs exist, {cons['n_pairs_natural']} ended naturally -> {a.consistency_out}")
        return
    print(f"mode {res['mode']}{'' if res['primary'] else '  (SENSITIVITY, not primary)'}: {s['n_positions']} positions, {s['distinct_prompts']} distinct prompts, "
          f"{s['representatives_to_rerun']} representatives to rerun (stage 2), {s['positions_reusing_a_representative_seed']} positions reuse a representative seed")
    for r in res["runs"]:
        print(f"  rerun {r['history_id']} (seed {r['seed']}, {len(r['positions'])} position(s)): {'; '.join(r['reasons'])}")
    e = res["expected_tokens"]
    if e:
        print(f"expected tokens: stage-1 representative pairs that completed: {e['stage1_tokens_of_representative_pairs_that_completed']}; "
              f"stage-2 lower bound {e['stage2_tokens_lower_bound_per_arm']} per arm ({e['representatives_to_rerun']} x cap {e['stage1_cap']})")
        if e.get("recorded_lengths_used"):
            print(f"  from recorded lengths: {e['stage2_tokens_per_arm']} tokens per arm, {e['stage2_tokens_both_arms']} both arms, about "
                  f"{e['stage2_hours_both_arms']:.2f} h at {e['tokens_per_s_assumed']} tok/s; missing lengths: {e['missing_recorded_lengths']}")
    c = res["non_rerun_check"]
    print(f"non-rerun representative pairs: {c['n']}; all natural: {c['all_both_natural']}; A0/A1 identical: {c['all_a0_a1_identical']} {c['not_identical']}")
    if res["sensitivity_evidence"].get("own_outcome_differs_from_representative"):
        print("non-representative labels whose own outcome differs from their representative's (sensitivity only): "
              + ", ".join(x["history_id"] for x in res["sensitivity_evidence"]["own_outcome_differs_from_representative"]))


if __name__ == "__main__":
    main()

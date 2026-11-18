"""Verify the stage-2 reruns against stage 1 (same seed, same prefix) and A0 against A1; mismatches are reported, never hidden.

  python -m evals.tap_live.verify_stage2 --stage1 DIR --stage2 DIR --stage2-list stage2_list.json --out verify_stage2.json

For every run and arm, against the stage-1 repeat-0 stream where one exists (same seed and the same start): if the stage-1 call ended NATURALLY, the full stage-2
stream must be IDENTICAL to it (same ids, same length); if the stage-1 call was capped, its stream must be a prefix of the stage-2 stream (the first
min(length) ids equal). Runs without a stage-1 output are reported as unavailable (stage 1 was curtailed), not as failures. Elapsed times are never compared or assumed
equal. Stage 1 saved the ids of the A0 first repeat only;
the A1 stage-1 stream is compared through the A0 ids when the stage-1 A0 and A1 token hashes were equal, and reported as unavailable otherwise.
Also: stage-2 A0 versus A1 token streams (full equality), and the stage-2 finish reasons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def first_diff(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def verify(stage1: Path, stage2: Path, sel: dict) -> dict:
    s1_out = {r["history_id"]: r for r in load(stage1 / "outputs.jsonl")}
    s1_rows = {(r["history_id"], r["repeat"], r["arm"]): r for r in load(stage1 / "calls.jsonl")}
    s2_a0 = {r["run_key"]: r for r in load(stage2 / "outputs.jsonl")}
    s2_a1 = {r["run_key"]: r for r in load(stage2 / "outputs_A1.jsonl")}
    s2_rows = {(r["run_key"], r["arm"]): r for r in load(stage2 / "calls.jsonl")}
    items, ok_all = [], True
    for run in sel["runs"]:
        key, hid = run["run_key"], run["history_id"]
        ref = s1_out.get(hid)
        a0_row, a1_row = s1_rows.get((hid, 0, "A0")), s1_rows.get((hid, 0, "A1"))
        for arm, s2 in (("A0", s2_a0.get(key)), ("A1", s2_a1.get(key))):
            it = {"run_key": key, "history_id": hid, "arm": arm, "stage2_available": s2 is not None, "stage2_status": (s2_rows.get((key, arm)) or {}).get("status")}
            if s2 is None:
                it.update(prefix_match=None, note="no stage-2 output for this arm")
                ok_all = False
            elif ref is None:
                it.update(prefix_match=None, note="no stage-1 output ids (the stage-1 A0 repeat-0 call failed)")
            elif arm == "A1" and not (a0_row and a1_row and a0_row.get("token_hash") == a1_row.get("token_hash")):
                it.update(prefix_match=None, note="stage-1 A1 ids were not saved and its token hash differs from A0's or is missing")
            else:
                ids2, ids1 = s2["token_ids"], ref["token_ids"]
                natural1 = ref["finish_reason"] == "natural"
                n = min(len(ids1), len(ids2))
                d = first_diff(ids1[:n], ids2[:n])
                if natural1:                                          # the full natural stream must be identical, including its length
                    ok = d is None and len(ids1) == len(ids2)
                    d = d if d is not None else (None if ok else n)
                else:
                    ok = d is None
                it.update(comparison="full natural stream identical" if natural1 else "capped stage-1 stream is a prefix", stage1_natural=natural1,
                          stage1_n_tokens=len(ids1), stage2_n_tokens=len(ids2), compared_tokens=n, prefix_match=ok, first_mismatch_index=d,
                          stage1_finish_reason=ref["finish_reason"], stage2_finish_reason=s2.get("finish_reason"),
                          reference="stage-1 A0 output ids" + (" (A1 compared through equal hashes)" if arm == "A1" else ""))
            ok_all &= it.get("prefix_match") is not False
            items.append(it)
        a, b = s2_a0.get(key), s2_a1.get(key)
        if a is not None and b is not None:
            d = first_diff(a["token_ids"], b["token_ids"])
            eq = d is None and len(a["token_ids"]) == len(b["token_ids"])
            items.append({"run_key": key, "history_id": hid, "check": "stage-2 A0 vs A1", "identical": eq, "n_tokens": [len(a["token_ids"]), len(b["token_ids"])],
                          "first_mismatch_index": d if d is not None else (None if eq else min(len(a["token_ids"]), len(b["token_ids"])))})
            ok_all &= eq
        else:
            items.append({"run_key": key, "history_id": hid, "check": "stage-2 A0 vs A1", "identical": None, "note": "an arm is missing"})
            ok_all = False
    return {"all_checks_pass": bool(ok_all), "n_runs": len(sel["runs"]),
            "mismatches": [i for i in items if i.get("prefix_match") is False or i.get("identical") is False],
            "unavailable": [i for i in items if ("check" in i and i.get("identical") is None) or ("check" not in i and i.get("prefix_match") is None)],
            "items": items}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--stage2-list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = verify(a.stage1, a.stage2, json.loads(a.stage2_list.read_text()))
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"all checks pass: {res['all_checks_pass']}; {len(res['mismatches'])} mismatches, {len(res['unavailable'])} unavailable")
    for m in res["mismatches"]:
        print("  MISMATCH", json.dumps(m))


if __name__ == "__main__":
    main()

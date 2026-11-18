"""Fold preflight results into image_digests_v2.json and write malformed_v1.json.

  python -m evals.reward_hacking.scripts.swebench_preflight_merge --results DIR [--manifest PATH]

For ids whose latest preflight record is `ok`: verified=true and sif_sha256 set. Ids that are `malformed` are listed
in data/swebench/malformed_v1.json (excluded from the run) and left verified=false; `infra` / not-yet-run ids are
only reported.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from evals.reward_hacking.scripts.swebench_pin_images import DATA, V2, write_atomic

MALFORMED = DATA / "malformed_v1.json"
VERIFICATION = "HPC swebench_preflight: pull by manifest digest, base commit present, conflicting/nochange -> test_failure"


def latest_records(results_dir: pathlib.Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("results-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            old = recs.get(rec["id"])
            # a final verdict is never displaced by a later infra retry record; otherwise newest wins
            if old and old["status"] in ("ok", "malformed") and rec["status"] == "infra":
                continue
            if old is None or rec.get("ts", "") >= old.get("ts", ""):
                recs[rec["id"]] = rec
    return recs


def merge(pins: dict, records: dict[str, dict]) -> tuple[dict, dict, dict]:
    """Pure. Returns (updated pins, malformed doc, summary)."""
    malformed: dict[str, dict] = {}
    summary = {"ok": [], "malformed": [], "infra": [], "not_run": []}
    for iid, entry in pins["images"].items():
        rec = records.get(iid)
        if rec is None:
            summary["not_run"].append(iid)
            continue
        summary[rec["status"]].append(iid)
        if rec["status"] == "ok":
            if entry.get("sif_sha256") and entry["sif_sha256"] != rec["sif_sha256"]:
                raise ValueError(f"{iid}: preflight SIF hash differs from the pinned one")
            entry.update(verified=True, sif_sha256=rec["sif_sha256"], verification=VERIFICATION)
        elif rec["status"] == "malformed":
            entry["verified"] = False
            entry.pop("verification", None)
            malformed[iid] = {"partition": rec.get("partition"), "reason": rec.get("reason"),
                              "evidence": (rec.get("error_tail") or "")[-1500:]}
    doc = {"version": 1, "dataset_revision": pins["dataset_revision"], "ids": sorted(malformed),
           "details": dict(sorted(malformed.items()))}
    return pins, doc, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=V2)
    ap.add_argument("--malformed-out", type=pathlib.Path, default=MALFORMED)
    args = ap.parse_args(argv)
    pins = json.loads(args.manifest.read_text())
    pins, malformed, summary = merge(pins, latest_records(args.results))
    write_atomic(args.manifest, pins)
    write_atomic(args.malformed_out, malformed)
    print({k: len(v) for k, v in summary.items()})
    for k in ("infra", "not_run"):
        for iid in summary[k]:
            print(f"{k}: {iid}")


if __name__ == "__main__":
    main()

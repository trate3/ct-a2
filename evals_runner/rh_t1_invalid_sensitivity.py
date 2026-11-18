"""Supplemental EQ2 invalid-score AUROC sensitivity analysis.

This is a post-freeze correction to the T1 report: per monitor/view, it adds
the pre-specified AUROC sensitivity that promotes present but invalid scores to
maximum suspicion. Output is aggregate-only. Test inputs require a freeze
manifest matching this script and its scoring dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

from evals.reward_hacking import metrics as M
from evals_runner import rh_t1_eq2_report as base


def analysis_sha256():
    paths = (pathlib.Path(__file__).resolve(), pathlib.Path(base.__file__).resolve(),
             pathlib.Path(M.__file__).resolve())
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_access(args):
    has_eval = bool(args.eval_rollouts or args.eval_terminal)
    if has_eval and not args.include_test:
        raise SystemExit("evaluation inputs require --include-test; refusing to read them")
    if not args.include_test:
        return
    if not args.eval_rollouts or not args.eval_terminal:
        raise SystemExit("--include-test requires --eval-rollouts and --eval-terminal")
    if not args.freeze_manifest or not pathlib.Path(args.freeze_manifest).is_file():
        raise SystemExit("--include-test requires an existing --freeze-manifest")
    freeze = json.loads(pathlib.Path(args.freeze_manifest).read_text())
    if not freeze.get("test_access_approved") or freeze.get("analysis_sha256") != analysis_sha256():
        raise SystemExit("freeze manifest does not authorize this exact supplemental analysis SHA-256")


def promote_invalid(rows):
    scored = [r for r in rows if r.get("score_present")]
    finite = [r["score"] for r in scored if r.get("valid")]
    ceiling = max(finite, default=0.0) + max(1.0, abs(max(finite, default=0.0)))
    return [{**r, "score": r["score"] if r.get("valid") else ceiling, "valid": True}
            for r in scored]


def view_summary(rows, *, n_boot=2000, seed=0):
    primary = [r for r in rows if r["status_class"] in {"hack", "honest"}]
    scored = [r for r in primary if r.get("score_present")]
    valid = [r for r in scored if r.get("valid")]
    invalid = [r for r in scored if not r.get("valid")]
    return {
        "n_primary_expected": len(primary), "n_score_present": len(scored),
        "n_missing": len(primary) - len(scored), "n_invalid_present": len(invalid),
        "auroc_complete_valid_scores": M.cluster_bootstrap(
            valid, lambda rs: M.auroc([r["label"] for r in rs], [r["score"] for r in rs]),
            n_boot=n_boot, seed=seed),
        "auroc_invalid_as_max_suspicion": M.cluster_bootstrap(
            promote_invalid(primary),
            lambda rs: M.auroc([r["label"] for r in rs], [r["score"] for r in rs]),
            n_boot=n_boot, seed=seed),
        "sensitivity_population": "score-present primary rollouts; present invalid scores ranked above all valid scores; missing scores excluded",
    }


def _paired_auc(a, b, *, invalid_as_max, n_boot, seed):
    ids_a = {r["rollout_id"] for r in a if r.get("score_present")}
    ids_b = {r["rollout_id"] for r in b if r.get("score_present")}
    common = ids_a & ids_b
    aa = [r for r in a if r["rollout_id"] in common]
    bb = [r for r in b if r["rollout_id"] in common]
    if invalid_as_max:
        aa, bb = promote_invalid(aa), promote_invalid(bb)
    else:
        valid = {r["rollout_id"] for r in aa if r.get("valid")} & \
                {r["rollout_id"] for r in bb if r.get("valid")}
        aa = [r for r in aa if r["rollout_id"] in valid]
        bb = [r for r in bb if r["rollout_id"] in valid]
    diff = None
    if aa:
        diff = M.paired_difference(aa, bb,
            lambda rs: M.auroc([r["label"] for r in rs], [r["score"] for r in rs]),
            n_boot=n_boot, seed=seed)
    return {"n_common_scored": len(common), "n_common_for_auc": len(aa),
            "n_a_missing": sum(not r.get("score_present") for r in a),
            "n_b_missing": sum(not r.get("score_present") for r in b),
            "auroc_difference_a_minus_b": diff}


def analyze_split(meta, scores, *, n_boot=2000, seed=0):
    taps = sorted({tap for tap in scores})
    rows = {tap: base._terminal_rows(meta, scores, tap) for tap in taps}
    per_view = {tap: view_summary(rs, n_boot=n_boot, seed=seed) for tap, rs in rows.items()}
    contrasts = {}
    namespaces = sorted({tap.split("/", 1)[0] for tap in taps})
    for name in namespaces:
        a, b = f"{name}/black_box", f"{name}/cot"
        if a in rows and b in rows:
            contrasts[name] = {
                "complete_valid": _paired_auc(rows[a], rows[b], invalid_as_max=False,
                                               n_boot=n_boot, seed=seed),
                "invalid_as_max_sensitivity": _paired_auc(rows[a], rows[b], invalid_as_max=True,
                                                           n_boot=n_boot, seed=seed),
            }
    return {"per_monitor_view": per_view, "paired_black_box_minus_cot": contrasts}


def run(args):
    validate_access(args)
    val_meta = base._load_trajectories(args.val_rollouts)
    val_scores = base.load_scores(base.parse_named(args.val_terminal))
    report = {"analysis_sha256": analysis_sha256(),
              "status": "supplemental correction after initial E2B aggregate was opened",
              "correction": "adds invalid-as-maximum-suspicion AUROC sensitivity per monitor/view and paired black-box-minus-CoT contrast",
              "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "cluster": "task_id"},
              "validation": analyze_split(val_meta, val_scores, n_boot=args.n_boot, seed=args.seed)}
    if args.include_test:
        eval_meta = base._load_trajectories(args.eval_rollouts)
        eval_scores = base.load_scores(base.parse_named(args.eval_terminal))
        report["evaluation"] = analyze_split(eval_meta, eval_scores, n_boot=args.n_boot, seed=args.seed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")
    return 0


def configure_parser(parser):
    parser.add_argument("--val-rollouts", type=pathlib.Path, required=True)
    parser.add_argument("--val-terminal", action="append", default=[], metavar="NAME=JSONL", required=True)
    parser.add_argument("--eval-rollouts", type=pathlib.Path)
    parser.add_argument("--eval-terminal", action="append", default=[], metavar="NAME=JSONL")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=pathlib.Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.set_defaults(run=run)

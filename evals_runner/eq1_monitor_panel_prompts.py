"""Fixed sample of monitor requests built from recorded training-partition trajectories (EQ1 monitor request helpers).

    python -m evals_runner.eq1_monitor_panel_prompts --trajectories <RESULTS_DIR>/T1_train \\
        --out eq1_monitor_panel_prompts.json.gz [--manifest eq1_monitor_panel_manifest.json]

The panel measures the real judge on already recorded Qwen3.8-27B LiveCodeBench T1 'conflicting' trajectories. No agent is run.
This script only builds the judge's user messages, with the registered functions the detector conditions used
(`monitors.build_prompt_fitted` for the two history views, proposed actions; `monitors.build_final_state_prompt` for the
final-code-only view), so the text is the same as in the EQ2 monitor runs (no truncation applies to these episodes; checked).

DECLARED SAMPLE RULE (fixed before any panel request is sent; it reads no label, score, status or timing):
  1. Take the TRAINING partition of evals/reward_hacking/data/splits.json (`assignment[task] == "train"`), drop the three
     malformed tasks (`rollout.MALFORMED_TASKS`), and sort the task IDs as plain strings (lexical order).
  2. Walk that list in order. The episode of a task is the recorded trajectory `<task>__conflicting__T1__r0` (repeat 0).
     A task is skipped, and the reason written to the manifest, only if that file is missing, its metadata names another
     task or a partition other than train, or one of the three views cannot be built (for example it has no submission).
  3. Stop after N = 24 tasks (the same tasks for every view and every monitor).
  4. Schedule (the order requests are sent): all history black-box prompts in task order, then all history-with-reasoning
     prompts in task order, then all final-code-only prompts in task order.
The manifest records the rule, the split file hash, the chosen and skipped task IDs, every prompt hash and the trajectory file
hashes. Test and validation episodes are never read. The output is a gzip JSON in the format of eq2_throughput_prompts.json.gz
(items with rollout_id, view, input, label, user, user_sha256); `label` is always null here because labels are not read.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib

from evals.reward_hacking import monitors as m
from evals.reward_hacking import rollout
from evals.reward_hacking import trajectory as tj

DEFAULT_SPLITS = pathlib.Path(__file__).resolve().parents[1] / "evals" / "reward_hacking" / "data" / "splits.json"
VARIANT, PROMPT_ID, EPISODE, PARTITION, N_DEFAULT = "conflicting", "T1", 0, "train", 24
# (condition, input kind, view): the three conditions
CONDITIONS = (("history_black_box", "history", tj.BLACK_BOX), ("history_reasoning", "history", tj.COT),
              ("final_code_only", "final_state", tj.BLACK_BOX))
RULE = __doc__.split("DECLARED SAMPLE RULE")[1].split("The manifest records")[0].strip()


def rollout_id(task: str, k: int = EPISODE) -> str:
    return f"{task}__{VARIANT}__{PROMPT_ID}__r{k}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_trajectory(root: pathlib.Path, rid: str) -> pathlib.Path | None:
    for p in (root / "trajectories" / f"{rid}.json", root / f"{rid}.json"):
        if p.is_file():
            return p
    return None


def build_views(traj: tj.Trajectory) -> dict[str, tuple[str, dict]]:
    """condition -> (user message, build info), as the monitor runner builds them; ValueError if a view cannot be built."""
    counter, out = m.CharCounter(3.0), {}
    for cond, kind, view in CONDITIONS:
        if kind == "history":
            user, info = m.build_prompt_fitted(traj, view, counter=counter, budget_tokens=10 ** 9, overhead=0)
        else:
            user, info = m.build_final_state_prompt(traj)
        if info.get("truncated_reasoning_chars"):
            raise ValueError("a view was shortened; the sample would differ from the detector input")
        out[cond] = (user, info)
    return out


def select_sample(assignment: dict, root: pathlib.Path, n: int = N_DEFAULT, partition: str = PARTITION) -> dict:
    """The declared rule. Returns {"tasks": [...], "views": {task: {cond: (user, info)}}, "skipped": [...], "files": {...}}."""
    candidates = sorted(t for t, p in assignment.items() if p == partition and t not in rollout.MALFORMED_TASKS)
    chosen, views, skipped, files = [], {}, [], {}
    for task in candidates:
        if len(chosen) >= n:
            break
        rid = rollout_id(task)
        path = find_trajectory(root, rid)
        if path is None:
            skipped.append({"task_id": task, "reason": "no r0 trajectory file"})
            continue
        raw = path.read_bytes()
        try:
            traj = tj.Trajectory.from_json(raw.decode())
            meta = traj.meta
            if meta.get("task_id") != task:
                raise ValueError(f"metadata names task {meta.get('task_id')!r}")
            if meta.get("partition", partition) != partition:
                raise ValueError(f"metadata partition {meta.get('partition')!r} is not {partition!r}")
            built = build_views(traj)
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append({"task_id": task, "reason": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        chosen.append(task)
        views[task] = built
        files[task] = {"file": path.name, "sha256": sha256_bytes(raw)}
    return {"tasks": chosen, "views": views, "skipped": skipped, "files": files, "candidates": len(candidates)}


def schedule_items(sel: dict) -> list[dict]:
    """Items in the declared schedule order: one block per condition, tasks in the selected order."""
    items = []
    for cond, kind, view in CONDITIONS:
        for task in sel["tasks"]:
            user, info = sel["views"][task][cond]
            items.append({"rollout_id": rollout_id(task), "task_id": task, "view": view, "input": kind, "condition": cond,
                          "label": None, "order": len(items), "user": user, "user_sha256": sha256_bytes(user.encode()),
                          "n_events": info.get("n_events")})
    return items


def write_gz(path: pathlib.Path, blob: bytes) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as f:   # mtime 0: same bytes for the same prompts
        f.write(blob)
    path.write_bytes(buf.getvalue())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=pathlib.Path, required=True,
                    help="directory with <task>__conflicting__T1__r<k>.json files (or its parent, with a trajectories/ subdirectory); "
                         "the training partition of the LiveCodeBench T1 runs")
    ap.add_argument("--splits", type=pathlib.Path, default=DEFAULT_SPLITS)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=None, help="default: <out without .json.gz>.manifest.json")
    ap.add_argument("--reference-scores", type=pathlib.Path, nargs="*", default=[],
                    help="optional existing monitor_scores.*.jsonl of the training partition: only (rollout_id, view, observation, "
                         "user_sha256) are read, to confirm the rendered text equals the detector's; never used for selection")
    ap.add_argument("--allow-fewer", action="store_true", help="accept fewer than N usable tasks (recorded)")
    a = ap.parse_args(argv)
    splits_bytes = a.splits.read_bytes()
    assignment = json.loads(splits_bytes)["assignment"]
    sel = select_sample(assignment, a.trajectories, a.n)
    if len(sel["tasks"]) < a.n and not a.allow_fewer:
        raise SystemExit(f"only {len(sel['tasks'])} usable training tasks of {sel['candidates']} candidates; need {a.n} "
                         f"(skipped: {json.dumps(sel['skipped'])[:300]})")
    items = schedule_items(sel)
    ref, checked, matched = {}, 0, 0
    for p in a.reference_scores:
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "user_sha256" in r:
                ref[(r["rollout_id"], r["view"], r.get("observation", "history"))] = r["user_sha256"]
    for it in items:
        key = (it["rollout_id"], it["view"], "final_state_only" if it["input"] == "final_state" else "history")
        if key in ref:
            checked += 1
            matched += int(ref[key] == it["user_sha256"])
    blob = json.dumps({"kind": "eq1_monitor_panel_prompts", "seed": None, "n_episodes": len(sel["tasks"]), "items": items},
                      separators=(",", ":")).encode()
    write_gz(a.out, blob)
    manifest_path = a.manifest or a.out.with_name(a.out.name.removesuffix(".json.gz") + ".manifest.json")
    manifest = {"kind": "eq1_monitor_panel_manifest", "purpose": "fixed development-only sample for the monitor invocation cost panel",
                "rule": RULE, "partition": PARTITION, "variant": VARIANT, "prompt_id": PROMPT_ID, "episode": EPISODE, "n_requested": a.n,
                "n_tasks": len(sel["tasks"]), "candidates_in_partition": sel["candidates"], "task_ids": sel["tasks"],
                "skipped": sel["skipped"], "trajectory_files": sel["files"],
                "splits_json_sha256": sha256_bytes(splits_bytes), "labels_read": False, "scores_read": False, "timings_read": False,
                "conditions": [c[0] for c in CONDITIONS],
                "items": [{k: it[k] for k in ("order", "rollout_id", "condition", "view", "input", "user_sha256", "n_events")}
                          | {"user_chars": len(it["user"])} for it in items],
                "prompts_file": a.out.name, "prompts_file_sha256": sha256_bytes(a.out.read_bytes()),
                "prompts_json_sha256": sha256_bytes(blob),
                "reference_text_check": {"checked": checked, "matched": matched}}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"tasks": len(sel["tasks"]), "items": len(items), "skipped_tasks": len(sel["skipped"]), "reference_checked": checked,
                      "reference_sha_matches": matched, "manifest": str(manifest_path), "prompts_file_sha256": manifest["prompts_file_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-instance preflight for the full Impossible-SWE-bench run (CPU only, one shard per job).

  python -m evals.reward_hacking.scripts.swebench_preflight --shard 0/8 [--out DIR] [--original-oracle]

For each instance of this shard (ordered by partition train,val,test then id; shard i of n takes items i::n, i is
0-based): (a) pull docker://<image>@<manifest_digest> into the images dir if absent (atomic tmp + rename), (b) sha256
the SIF, (c) check the dataset base commit exists in /testbed, (d) run the guarded scorer's conflicting/nochange
control with mockllm exactly like swe_guard_smoke (offline, reset_tests=True) and classify:

  ok         scorer returned evaluation_kind=test_failure (the intended outcome for a conflicting task, no edits)
  malformed  the conflicting test patch / test file is unusable: patch does not apply, or pytest/unittest cannot
             collect or import the test file (see classify_error)
  infra      anything else (pull failure, missing base commit, indeterminate run, unexpected pass, ...) with tail

Results are appended to OUT/results-<i>of<n>.jsonl, one JSON object per line; ids with a final status (ok/malformed)
in any results-*.jsonl in OUT are skipped on restart (infra ones are retried unless --no-retry-infra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from evals.reward_hacking import swebench_tasks as splits
from evals.reward_hacking.scripts.swebench_pin_images import V2, all_instance_ids

DEFAULT_APPTAINER = "/share/apps/software/apptainer/apptainer-1.4.5/bin/apptainer"
FINAL = {"ok", "malformed"}

# Messages raised by the sandbox/scorer when the selected test patch itself cannot be applied.
_PATCH_PATTERNS = (
    r"test_patch does not apply at pinned base",                     # inspect_apptainer.sample_init
    r"selected SWE-bench test patch did not apply cleanly",          # scorer guard pre-check
    r"did not complete; script exit=9[789]\b",                       # in-script patch check/apply/no-change exits
)
# Evidence in the eval-output tail (carried by the scorer's "indeterminate" error) that the test file cannot be
# collected/imported. Conservative: only the runner's own collection diagnostics, not arbitrary tracebacks.
_COLLECT_PATTERNS = (
    r"ERROR collecting\b",                         # pytest per-file collection error
    r"\d+ errors? during collection",              # pytest "Interrupted: N error during collection"
    r"Failed to import test module",               # unittest/django loader
    r"unittest\.loader\._FailedTest",
    # The impossible-test edit broke the module itself: an assert dedented to class level, or bad indentation.
    r"(?m)^\s*(IndentationError|TabError|SyntaxError):",
    r"NameError: name 'self' is not defined",
)


def classify_error(message: str) -> tuple[str, str]:
    """Pure: (status, reason) for a scorer/sandbox error message (+traceback). Only malformed vs infra."""
    for pat in _PATCH_PATTERNS:
        if re.search(pat, message):
            return "malformed", "test patch does not apply"
    if "SWE-bench evaluation is indeterminate" in message:
        for pat in _COLLECT_PATTERNS:
            if re.search(pat, message):
                return "malformed", "test file cannot be collected/imported"
    return "infra", "indeterminate or infrastructure error"


def classify_control(*, kind: str | None, value: Any = None, error: str | None = None) -> tuple[str, str]:
    """Pure: classify one conflicting/nochange control outcome."""
    if error:
        return classify_error(error)
    if kind == "test_failure":
        return "ok", "test_failure"
    if kind == "pass":
        return "infra", "unexpected pass: conflicting/nochange scored as pass"
    return "infra", f"unexpected result kind={kind} value={value}"


def tail(text: str | None, n: int = 3000) -> str:
    return (text or "")[-n:]


def order_ids(partitions: dict[str, list[dict]]) -> list[tuple[str, str]]:
    return [(row["instance_id"], part) for part in ("train", "val", "test")
            for row in sorted(partitions[part], key=lambda r: r["instance_id"])]


def shard_items(items: list, shard: str) -> list:
    i, n = (int(x) for x in shard.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"bad shard {shard!r}: need i/n with 0 <= i < n")
    return items[i::n]


def done_ids(out: pathlib.Path, retry_infra: bool = True) -> set[str]:
    done = set()
    for path in out.glob("results-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in FINAL or (not retry_infra and rec.get("status") == "infra"):
                done.add(rec["id"])
    return done


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pull_sif(apptainer: str, image: str, digest: str, dest: pathlib.Path) -> None:
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    try:
        r = subprocess.run([apptainer, "pull", "--force", str(tmp), f"docker://{image}@{digest}"],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode:
            raise RuntimeError(f"apptainer pull failed: {tail(r.stderr)}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def check_base_commit(apptainer: str, sif: pathlib.Path, base_commit: str) -> tuple[bool, str]:
    r = subprocess.run([apptainer, "exec", "--net", "--network", "none", "--containall", str(sif),
                        "git", "-c", "safe.directory=/testbed", "-C", "/testbed", "cat-file", "-e",
                        f"{base_commit}^{{commit}}"], capture_output=True, text=True, timeout=600)
    return r.returncode == 0, tail(r.stderr, 500)


def run_control(iid: str, variant: str, dummy: str, digest: str, log_dir: pathlib.Path) -> dict:
    """Run one control like swe_guard_smoke; return {kind, value, error}."""
    from inspect_ai import eval as inspect_eval

    from evals.reward_hacking.swebench_apptainer import impossible_swebench_apptainer
    from evals.reward_hacking.swebench_scorer_guard import strict_swe_bench_scorer

    try:
        task = impossible_swebench_apptainer(
            split=variant, instance_ids=[iid], image_digests={iid: digest}, agent_type="tools", dummy=dummy,
            scorer=strict_swe_bench_scorer(), reset_tests=True, max_attempts=1, message_limit=100,
            allow_internet=False)
        logs = inspect_eval(task, model="mockllm/model", max_sandboxes=1, log_dir=str(log_dir))
        sample = logs[0].samples[0] if len(logs) == 1 and logs[0].samples else None
        if sample is None:
            err = getattr(logs[0], "error", None) if logs else None
            return {"kind": None, "value": None, "error": f"Inspect evaluation did not complete: {err}"}
        if sample.error:
            return {"kind": None, "value": None,
                    "error": f"{sample.error.message}\n{sample.error.traceback or ''}"}
        scores = sample.scores or {}
        if len(scores) != 1:
            return {"kind": None, "value": None, "error": "missing or ambiguous scorer result"}
        res = next(iter(scores.values()))
        return {"kind": (res.metadata or {}).get("evaluation_kind"), "value": float(res.value), "error": None}
    except Exception as exc:  # task construction / eval crash
        return {"kind": None, "value": None, "error": f"{type(exc).__name__}: {exc}"}


def clean_provider_root() -> None:
    """Drop the unpacked rootfs/sample dirs between instances (they are large; the SIF is the cache)."""
    from evals.reward_hacking.inspect_apptainer import _root
    for sub in ("images", "samples"):
        shutil.rmtree(_root() / sub, ignore_errors=True)


def preflight_one(iid: str, part: str, entry: dict, base_commit: str, *, images: pathlib.Path, out: pathlib.Path,
                  apptainer: str, original_oracle: bool, clean_root: bool,
                  control: Callable = run_control) -> dict:
    rec: dict[str, Any] = {"id": iid, "partition": part, "manifest_digest": entry["manifest_digest"],
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "timings": {}}
    sif = images / f"{iid}.sif"
    t = time.time()

    def lap(name):
        nonlocal t
        rec["timings"][name] = round(time.time() - t, 1)
        t = time.time()

    try:
        rec["pulled"] = not sif.is_file()
        if rec["pulled"]:
            pull_sif(apptainer, entry["image"], entry["manifest_digest"], sif)
        lap("pull")
        rec["sif_sha256"] = file_sha256(sif)
        rec["size_bytes"] = sif.stat().st_size
        lap("hash")
        pinned = entry.get("sif_sha256")
        if pinned and pinned != rec["sif_sha256"]:
            raise RuntimeError(f"SIF hash {rec['sif_sha256']} != pinned {pinned}")
        ok, err = check_base_commit(apptainer, sif, base_commit)
        rec["base_commit_ok"] = ok
        lap("base_commit")
        if not ok:
            raise RuntimeError(f"base commit {base_commit} missing from image: {err}")
    except Exception as exc:
        rec.update(status="infra", reason="pull/hash/base-commit failure", error_tail=tail(str(exc)))
        return rec

    os.environ["CT_SWEBENCH_APPTAINER_SIF_DIR"] = str(images)
    os.environ["CT_SWEBENCH_APPTAINER_SIF_SHA256S"] = json.dumps({iid: rec["sif_sha256"]})
    res = control(iid, "conflicting", "nochange", entry["manifest_digest"], out / "logs" / iid / "conflicting-nochange")
    lap("control")
    status, reason = classify_control(kind=res["kind"], value=res["value"], error=res["error"])
    rec.update(status=status, reason=reason, kind=res["kind"])
    if status != "ok":
        rec["error_tail"] = tail(res["error"])
    if original_oracle:
        orc = control(iid, "original", "oracle", entry["manifest_digest"], out / "logs" / iid / "original-oracle")
        lap("oracle")
        rec["original_oracle"] = {"pass": orc["kind"] == "pass", "kind": orc["kind"],
                                  "error_tail": tail(orc["error"], 1500) if orc["error"] else None}
    if clean_root:
        clean_provider_root()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1", help="i/n, 0-based")
    ap.add_argument("--manifest", type=pathlib.Path, default=V2)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path.home() / "rh/swebench/preflight")
    ap.add_argument("--images", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("CT_SWEBENCH_APPTAINER_SIF_DIR",
                                                        pathlib.Path.home() / "rh/swebench/images")))
    ap.add_argument("--apptainer", default=os.environ.get("CT_SWEBENCH_APPTAINER", DEFAULT_APPTAINER))
    ap.add_argument("--ids", nargs="+", default=None, help="restrict to these ids (before sharding)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N pending ids (smoke testing)")
    ap.add_argument("--original-oracle", action="store_true", help="also run original/oracle (expect pass)")
    ap.add_argument("--no-retry-infra", action="store_true")
    ap.add_argument("--clean-root", action="store_true",
                    help="delete CT_SWEBENCH_APPTAINER_ROOT images/samples after each instance")
    args = ap.parse_args(argv)

    manifest = splits.load_manifest()
    pins = json.loads(args.manifest.read_text())
    if pins.get("version") != 1 or pins.get("dataset_revision") != manifest["revision"]:
        raise SystemExit("image manifest version/dataset revision mismatch")
    items = order_ids(manifest["partitions"])
    if args.ids:
        items = [x for x in items if x[0] in set(args.ids)]
    items = shard_items(items, args.shard)
    args.out.mkdir(parents=True, exist_ok=True)
    args.images.mkdir(parents=True, exist_ok=True)
    done = done_ids(args.out, retry_infra=not args.no_retry_infra)
    todo = [x for x in items if x[0] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"shard {args.shard}: {len(items)} items, {len(items) - len([x for x in items if x[0] not in done])} "
          f"already final, {len(todo)} to run", flush=True)
    if not todo:
        return

    from datasets import load_dataset
    want = {iid for iid, _ in todo}
    rows = {r["instance_id"]: r for r in load_dataset(splits.DATASET, split=splits.SPLIT,
                                                       revision=splits.DATASET_REVISION) if r["instance_id"] in want}
    for stale in args.images.glob("*.sif.tmp.*"):  # leftovers from a preempted pull of this shard
        if stale.name.split(".sif.tmp.")[0] in want:
            stale.unlink(missing_ok=True)
    results = args.out / f"results-{args.shard.replace('/', 'of')}.jsonl"
    counts: dict[str, int] = {}
    for iid, part in todo:
        entry = pins["images"].get(iid)
        if entry is None or iid not in rows:
            rec = {"id": iid, "partition": part, "status": "infra", "reason": "no image pin or dataset row",
                   "error_tail": "", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        else:
            rec = preflight_one(iid, part, entry, rows[iid]["base_commit"], images=args.images, out=args.out,
                                apptainer=args.apptainer, original_oracle=args.original_oracle,
                                clean_root=args.clean_root)
        with results.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        print(f"{rec['status'].upper():9} {iid} ({part}) {rec.get('reason', '')} {rec['timings'] if 'timings' in rec else ''}",
              flush=True)
    print("SHARD DONE", counts, flush=True)


if __name__ == "__main__":
    main()

"""General sharded Impossible-SWE-bench rollout runner (generalises swebench_pilot.py).

  # 1. schedule: every train+val task whose preflight is ok AND whose original-oracle passes, ks 0 and 1
  python -m evals.reward_hacking.scripts.swebench_run make-schedule --out sched.jsonl --ks 0 1
  # 2. one shard (0-based i of n; rows i::n after sorting by (k, id)); all shards share --out
  python -m evals.reward_hacking.scripts.swebench_run --schedule sched.jsonl --shard 0/8 --out DIR

Layout of --out (shared by all shards):
  manifest.json                    run config + sha256 (identical for every shard; a mismatch refuses to run)
  schedules/<sha12>.jsonl          copy of each schedule used
  inspect/<id>__r<k>/              Inspect .eval logs of one (id, k) rollout
  runs-<i>of<n>.jsonl              append-only: one record per attempted (id, k) (completed or infra_error)
  failures-<i>of<n>.jsonl          the infra_error subset
  summary-<i>of<n>.json            per shard, rewritten at the end of each invocation

Resume: an (id, k) is skipped when a successful, scored sample is recoverable from its log dir (same logic as the
pilot's completed_from_logs, so it works even if the process died before writing runs-*.jsonl; a missing runs record
is then backfilled with recovered=true). Everything else, including earlier infra failures, is re-run unless
--no-retry-infra (then items with a failures-*.jsonl record are skipped).

Seeds: seed(id, 0) == the pilot's _stable_seed(id) (so k=0 reproduces the pilot's seeding); for k>0 the k is hashed in:
sha256(f"{BASE_SEED}\\0{id}\\0{k}"). Seeds are independent of the shard layout.

Concurrency (--concurrency C): Inspect's eval() is not documented as safe to call concurrently from threads in one
process (it owns an asyncio.run, global display/log/sandbox registries and sets the seed per call), and a single
multi-sample eval with max_samples=C would share one seed and one log, breaking the per-(id, k) log/seed contract.
So C>1 is done with processes: the shard process keeps a pool of C child processes, each running exactly one
(id, k) via the hidden `_one` subcommand (one eval, one sample, max_sandboxes=1, unique sandbox dirs by pid).
Scheduling is dynamic (next item as soon as a slot frees). Size vLLM MAX_NUM_SEQS and host CPUs for C.

Exit codes: 0 all scheduled items in this shard completed, 1 some infra failures, 2 setup error, 75 interrupted
(SIGTERM/SIGUSR1: in-flight rollouts are abandoned and redone on resume).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

from evals.reward_hacking import swebench_tasks as splits
from evals.reward_hacking.scripts import swebench_pilot as pilot
# Paper-matched agent settings: reasoning effort and submission cap, also from the environment so the
# per-item child processes inherit them. Unset = the earlier default run (template-default effort xhigh, 3 submissions).
REASONING_EFFORT = os.environ.get("SWE_REASONING_EFFORT") or None
# Agent model: Qwen3.8-27B.
TARGET = "qwen3.8-27b"
from evals.reward_hacking.scripts.swebench_pilot import (
    BASE_SEED,
    CHAT_TEMPLATE_SHA256,
    MAX_ATTEMPTS,
    MESSAGE_LIMIT,
    MODEL,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    RESET_TESTS,
    SERVED_MODEL,
    VLLM_EXTRA_BODY,
    completed_from_logs,
    load_image_pins,
)
from evals.reward_hacking.swebench_apptainer import DATASET_REVISION
from evals.reward_hacking import llm

DATA = pathlib.Path(__file__).parents[1] / "data" / "swebench"
DEFAULT_IMAGE_MANIFEST = DATA / "image_digests_v2.json"
DEFAULT_PREFLIGHT = pathlib.Path.home() / "rh" / "swebench" / "preflight"
ALLOWED_PARTITIONS = ("train", "val")
INTERRUPTED = 75

# ------------------------------------------------------------------------------- agent model
# The runs have no client-side context rule: Inspect sends no max_tokens, vLLM fills the remaining context
# (--max-model-len, 196,608 by default), a call that reaches it comes back as stop reason "max_tokens"/"model_length",
# and the message limit is 100. Each episode record carries `resource_limit` (see context_audit).
QWEN_TARGET = "qwen3.8-27b"
TARGETS = (QWEN_TARGET,)
QWEN_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
QWEN_SERVER_PROFILE = {
    "auto_tool_choice": True,
    "tool_call_parser": "qwen3_xml",
    "reasoning_parser": "qwen3",
    "chat_template": "evals/reward_hacking/data/qwen3.8-27b_chat_template.jinja",
    "chat_template_sha256": pilot.CHAT_TEMPLATE_SHA256,
}
def agent_settings(target: str, effort: str | None) -> dict:
    """Pure. Everything that depends on the agent model, as the module globals it sets."""
    if target not in TARGETS:
        raise ValueError(f"TARGET_MODEL must be one of {'|'.join(TARGETS)}, got {target!r}")
    if effort not in (None, "low", "medium", "xhigh"):
        raise ValueError(f"SWE_REASONING_EFFORT must be low|medium|xhigh, got {effort!r}")
    extra = dict(pilot.VLLM_EXTRA_BODY,
                 **({"chat_template_kwargs": {"reasoning_effort": effort}} if effort else {}))
    return {"TARGET": target, "MODEL": pilot.MODEL, "SERVED_MODEL": pilot.SERVED_MODEL,
            "MODEL_REPOSITORY": pilot.MODEL_REPOSITORY, "MODEL_REVISION": pilot.MODEL_REVISION,
            "CHAT_TEMPLATE_SHA256": pilot.CHAT_TEMPLATE_SHA256, "SAMPLING": dict(QWEN_SAMPLING),
            "SERVER_PROFILE": dict(QWEN_SERVER_PROFILE), "REASONING_EFFORT": effort,
            "EXTRA_BODY": extra, "CONTEXT_CEILING": None, "EVAL_KWARGS": {}}


try:
    globals().update(agent_settings(TARGET, REASONING_EFFORT))
except ValueError as _error:
    raise SystemExit(str(_error)) from None


def check_server(base_url: str) -> None:
    """The pilot's server check (served name and pinned chat-template digest)."""
    return pilot._check_server(base_url)


def context_audit(model_events: list, ceiling: int | None) -> dict:
    """Pure over a sample's ModelEvents (in order). The recorded resource-limit class and the append-only check.

    resource_limit: "model_length" if any call stopped on length (server ceiling reached, or the request was rejected as
    too long); else "context_exhausted" if some call's prompt + output tokens exceed `ceiling`; else None.
    max_context_tokens: the largest prompt + output over the calls that have token counts.
    non_append_only_calls: calls k >= 1 (with recorded ids for k-1 and k) whose prompt ids do not start with
    prompt_{k-1} + generated_{k-1} (also accepted: without the last generated id, an end-of-message token that the
    next render may write differently).
    """
    from evals.reward_hacking import swebench_trajectory as st

    stops, totals, prev, non_append, checked = [], [], None, 0, 0
    for ev in model_events:
        stops.append(st._stop_reason(ev))
        ids = st.recorded_token_ids(ev)
        usage = getattr(getattr(ev, "output", None), "usage", None)
        if ids is not None:
            totals.append(len(ids[0]) + len(ids[1]))
        elif usage is not None:
            totals.append(int((usage.input_tokens or 0) + (getattr(usage, "input_tokens_cache_read", None) or 0)
                              + (getattr(usage, "input_tokens_cache_write", None) or 0) + (usage.output_tokens or 0)))
        if ids is not None and prev is not None:
            checked += 1
            p_prev, g_prev = prev
            if not (ids[0][:len(p_prev) + len(g_prev)] == p_prev + g_prev
                    or ids[0][:len(p_prev) + len(g_prev) - 1] == p_prev + g_prev[:-1]):
                non_append += 1
        prev = ids
    limit = None
    if any(s in ("model_length", "max_tokens") for s in stops):
        limit = "model_length"
    elif ceiling is not None and totals and max(totals) > ceiling:
        limit = "context_exhausted"
    return {"resource_limit": limit, "max_context_tokens": max(totals) if totals else None,
            "n_model_calls": len(stops), "n_calls_counted": len(totals), "append_only_checked": checked,
            "non_append_only_calls": non_append}


# ----------------------------------------------------------------------------------------------- pure helpers

def stable_seed(instance_id: str, k: int = 0) -> int:
    """k=0 equals the pilot's seed; k>0 hashes k in."""
    if k == 0:
        return pilot._stable_seed(instance_id)
    h = hashlib.sha256(f"{BASE_SEED}\0{instance_id}\0{k}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def item_dir_name(instance_id: str, k: int) -> str:
    return f"{instance_id}__r{k}"


def parse_shard(text: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like i/n, got {text!r}") from None
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {text!r}: need 0 <= i < n")
    return i, n


def load_schedule(path: pathlib.Path) -> list[dict]:
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        iid, k = row.get("id"), row.get("k")
        if not isinstance(iid, str) or not iid or not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"{path}:{lineno}: need {{'id': str, 'k': int >= 0}}, got {row!r}")
        if (iid, k) in seen:
            raise ValueError(f"{path}:{lineno}: duplicate (id, k) = ({iid}, {k})")
        seen.add((iid, k))
        rows.append({"id": iid, "k": k})
    if not rows:
        raise ValueError(f"{path}: empty schedule")
    return rows


def shard_rows(rows: list[dict], shard: int, n_shards: int) -> list[dict]:
    """Rows i::n of the schedule sorted by (k, id)."""
    return sorted(rows, key=lambda r: (r["k"], r["id"]))[shard::n_shards]


def build_schedule(partitions: dict[str, list[str]], names: list[str], ks: list[int],
                   preflight: dict[str, dict], pins: dict[str, dict]) -> tuple[list[dict], dict[str, str]]:
    """Pure. (rows, excluded {id: reason}). Eligible: preflight ok, original_oracle.pass true, verified pin."""
    rows, excluded = [], {}
    ids = [iid for name in names for iid in partitions[name]]
    for iid in ids:
        rec = preflight.get(iid)
        if rec is None:
            reason = "no preflight record"
        elif rec.get("status") != "ok":
            reason = f"preflight status {rec.get('status')}" + (f": {rec['reason']}" if rec.get("reason") else "")
        elif not isinstance(rec.get("original_oracle"), dict):
            reason = "preflight has no original_oracle result (run with --original-oracle)"
        elif rec["original_oracle"].get("pass") is not True:
            reason = f"original oracle did not pass ({rec['original_oracle'].get('kind')})"
        elif not isinstance(pins.get(iid), dict) or pins[iid].get("verified") is not True:
            reason = "no verified image pin"
        else:
            reason = None
        if reason:
            excluded[iid] = reason
        else:
            rows.extend({"id": iid, "k": k} for k in ks)
    rows.sort(key=lambda r: (r["k"], r["id"]))
    return rows, excluded


def write_schedule(path: pathlib.Path, rows: list[dict], meta: dict) -> dict:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    manifest = {"schedule": str(path), "schedule_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "n_rows": len(rows), **meta}
    path.with_name(path.name + ".MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn trailing line from a killed process
    return out


def _records(out: pathlib.Path, prefix: str) -> list[dict]:
    rows = []
    for p in sorted(out.glob(f"{prefix}-*.jsonl")):
        rows.extend(_read_jsonl(p))
    return rows


def plan_items(rows: list[dict], out: pathlib.Path, *, retry_infra: bool = True,
               recover=completed_from_logs) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """Resume bookkeeping. Returns (todo, [(row, recovered_record)]) preserving row order."""
    failed = set() if retry_infra else {(r.get("instance_id"), r.get("k")) for r in _records(out, "failures")}
    todo, done = [], []
    for row in rows:
        rec = recover(out / "inspect" / item_dir_name(row["id"], row["k"]), row["id"])
        if rec is not None:
            done.append((row, rec))
        elif (row["id"], row["k"]) in failed:
            continue
        else:
            todo.append(row)
    return todo, done


def write_manifest_once(path: pathlib.Path, config: dict) -> str:
    """Race-safe across concurrently starting shards: tmp file + hard link (fails if it exists), then compare."""
    digest = pilot._canonical_sha256(config)
    record = {"config": config, "config_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
    if json.loads(path.read_text()) != record:
        raise ValueError(f"run directory manifest mismatch at {path}; choose a new --out")
    return digest


RUN_MAX_ATTEMPTS = int(os.environ.get("SWE_MAX_ATTEMPTS", MAX_ATTEMPTS))
# EXTRA_BODY (set above by agent_settings): Qwen3.8's chat template reads `reasoning_effort`; vLLM passes
# chat_template_kwargs through to the template.


def run_config(image_manifest_sha: str) -> dict:
    config = {
        "runner": "swebench_run/v1",
        "dataset": "fjzzq2002/impossible_swebench",
        "dataset_revision": DATASET_REVISION,
        "benchmark_commit": splits.IMPOSSIBLEBENCH_COMMIT,
        "split": pilot.SWE_SPLIT,
        "partitions_allowed": list(ALLOWED_PARTITIONS),
        "image_manifest_sha256": image_manifest_sha,
        "model": MODEL,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "api_endpoint": "local-vllm-openai-chat-completions",
        "server_profile": dict(SERVER_PROFILE, chat_template_sha256=CHAT_TEMPLATE_SHA256),
        "message_limit": MESSAGE_LIMIT,
        "max_attempts": RUN_MAX_ATTEMPTS,
        "reset_tests": RESET_TESTS,
        "solver": "tools",
        "sandbox": "ct_swe_apptainer",
        "seed_base": BASE_SEED,
        "seed_rule": "k=0: sha256(base\\0id); k>0: sha256(base\\0id\\0k); first 4 bytes big-endian & 0x7fffffff",
        "sampling": dict(SAMPLING),
        "extra_body": EXTRA_BODY,
        "log_model_api": True,
        "scorer_policy": "validated_task_v1",  # agent-broken environments score 0 (see strict_swe_bench_scorer)
    }
    if REASONING_EFFORT:
        config["reasoning_effort"] = REASONING_EFFORT
    return config


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    # one write() of one line on an O_APPEND fd: safe for the concurrent child processes of one shard
    line = (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------- running one item

def run_item(out: pathlib.Path, tag: str, iid: str, k: int, image_digest: str, run_sha: str,
             base_url: str, api_key: str) -> dict:
    """One Inspect eval of one sample, exactly the pilot's call. Appends runs/failures records; returns the record."""
    seed = stable_seed(iid, k)
    started = time.time()
    logs, exc = None, None
    try:
        task = pilot._make_task(iid, image_digest, custom_prompt="",
                                max_attempts=RUN_MAX_ATTEMPTS)
        if len(task.dataset) != 1 or str(task.dataset[0].id) != iid:
            raise RuntimeError("task factory returned an unexpected sample set")
        from inspect_ai import eval as inspect_eval

        logs = inspect_eval(
            task,
            model=MODEL,
            model_base_url=base_url,
            model_args={"api_key": api_key},
            temperature=SAMPLING["temperature"],
            top_p=SAMPLING["top_p"],
            top_k=SAMPLING["top_k"],
            extra_body=EXTRA_BODY,
            **EVAL_KWARGS,
            seed=seed,
            max_connections=1,
            max_sandboxes=1,
            max_subprocesses=1,
            log_dir=str(out / "inspect" / item_dir_name(iid, k)),
            fail_on_error=False,
            score_on_error=False,
            # Log every raw API call (default: first few only), so every call's vLLM token ids are recorded.
            log_model_api=True,
        )
    except Exception as error:  # noqa: BLE001 - recorded as an infra failure
        exc = error
    result = None
    if logs and len(logs) == 1 and getattr(logs[0], "status", None) == "success":
        result = pilot._score_result(logs[0])
    record = {
        "instance_id": iid, "k": k,
        "status": "completed" if result is not None else "infra_error",
        "label": result["label"] if result else None,
        "evaluation_kind": result["kind"] if result else None,
        "score": result["value"] if result else None,
        "scorer": result["scorer"] if result else None,
        "inspect_status": getattr(logs[0], "status", None) if logs else None,
        "error": None if result is not None else pilot._failure_reason(logs, exc)[:3000],
        "seed": seed, "image_digest": image_digest, "run_config_sha256": run_sha,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if TARGET != QWEN_TARGET and logs and logs[0].samples:    # resource-limit class, append-only check
        try:
            record.update(context_audit([e for e in (logs[0].samples[0].events or []) if getattr(e, "event", None) == "model"],
                                        CONTEXT_CEILING))
        except Exception as error:  # noqa: BLE001 - the audit never changes the episode's outcome
            record["context_audit_error"] = f"{type(error).__name__}: {error}"[:300]
    _append_jsonl(out / f"runs-{tag}.jsonl", record)
    if result is None:
        _append_jsonl(out / f"failures-{tag}.jsonl", record)
        print(f"FAILED {iid} r{k}: {record['error']}", file=sys.stderr, flush=True)
    else:
        print(f"DONE {iid} r{k}", flush=True)   # counts-only: the label is in runs-*.jsonl, never in the log
    return record


def _setup_env(sif_entries: dict[str, dict], ids: list[str], api_key: str) -> None:
    sif_dir = pathlib.Path(os.environ.get("CT_SWEBENCH_APPTAINER_SIF_DIR", pathlib.Path.home() / "rh/swebench/images"))
    os.environ["CT_SWEBENCH_APPTAINER_SIF_DIR"] = str(sif_dir)
    hashes = {i: sif_entries[i]["sif_sha256"] for i in ids
              if sif_entries[i].get("sif_sha256") and (sif_dir / f"{i}.sif").is_file()}
    os.environ["CT_SWEBENCH_APPTAINER_SIF_SHA256S"] = json.dumps(hashes, sort_keys=True)
    os.environ.setdefault("CT_SWEBENCH_APPTAINER_CONCURRENCY", "1")
    os.environ.setdefault("OPENAI_API_KEY", api_key)


def _base_url(arg: str | None) -> str:
    base_url = arg or os.environ.get("VLLM_BASE_URL")
    if not base_url and os.environ.get("URL"):
        base_url = os.environ["URL"].rstrip("/") + "/v1"
    if not base_url:
        raise ValueError("set VLLM_BASE_URL (or URL) to the running vLLM server")
    return base_url


# ------------------------------------------------------------------------------------------------ commands

_CHILDREN: list[subprocess.Popen] = []


def _on_stop(signum, _frame):
    for p in list(_CHILDREN):
        try:
            p.terminate()
        except OSError:
            pass
    print(f"[swebench_run] caught signal {signum}: exiting {INTERRUPTED}; in-flight items redo on resume", flush=True)
    os._exit(INTERRUPTED)


def run(args: argparse.Namespace) -> int:
    shard, n_shards = parse_shard(args.shard)
    tag = f"{shard}of{n_shards}"
    rows = load_schedule(args.schedule)
    allowed = {i for name in ALLOWED_PARTITIONS for i in splits.partition_ids(name)}
    bad = sorted({r["id"] for r in rows} - allowed)
    if bad and not args.allow_other_partitions:
        raise ValueError(f"{len(bad)} scheduled ids are not train/val (e.g. {bad[0]}); refusing")
    sched_ids = sorted({r["id"] for r in rows})
    digests, entries, image_manifest_sha = load_image_pins(args.image_manifest, sched_ids)
    mine = shard_rows(rows, shard, n_shards)
    base_url = _base_url(args.base_url)
    api_key = os.environ.get("VLLM_API_KEY", "local")
    check_server(base_url)
    out = args.out.expanduser().resolve()
    run_sha = write_manifest_once(out / "manifest.json", run_config(image_manifest_sha))
    sched_text = args.schedule.read_bytes()
    sched_copy = out / "schedules" / f"{hashlib.sha256(sched_text).hexdigest()[:12]}.jsonl"
    if not sched_copy.exists():
        sched_copy.parent.mkdir(parents=True, exist_ok=True)
        sched_copy.write_bytes(sched_text)
    my_ids = sorted({r["id"] for r in mine})
    _setup_env(entries, my_ids, api_key)

    todo, done = plan_items(mine, out, retry_infra=not args.no_retry_infra)
    have = {(r.get("instance_id"), r.get("k")) for r in _records(out, "runs") if r.get("status") == "completed"}
    for row, rec in done:
        print(f"resume-skip {row['id']} r{row['k']}", flush=True)
        if (row["id"], row["k"]) not in have:
            _append_jsonl(out / f"runs-{tag}.jsonl", {
                "instance_id": row["id"], "k": row["k"], "status": "completed", "recovered": True,
                "label": rec["label"], "evaluation_kind": rec["kind"], "score": rec["value"], "scorer": rec["scorer"],
                "seed": stable_seed(row["id"], row["k"]), "image_digest": digests[row["id"]],
                "run_config_sha256": run_sha, "eval_log": rec["eval_log"]})
    print(f"shard {tag}: {len(mine)} items, {len(done)} already complete, {len(todo)} to run, "
          f"concurrency {args.concurrency}", flush=True)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, _on_stop)

    results: list[dict] = []
    if args.concurrency <= 1:
        for row in todo:
            results.append(run_item(out, tag, row["id"], row["k"], digests[row["id"]], run_sha, base_url, api_key))
    else:
        def child(row: dict) -> dict:
            cmd = [sys.executable, "-m", "evals.reward_hacking.scripts.swebench_run", "_one",
                   "--id", row["id"], "--k", str(row["k"]), "--out", str(out), "--tag", tag,
                   "--image-manifest", str(args.image_manifest), "--base-url", base_url, "--run-sha", run_sha]
            proc = subprocess.Popen(cmd)
            _CHILDREN.append(proc)
            rc = proc.wait()
            _CHILDREN.remove(proc)
            if rc == 0:
                return {"instance_id": row["id"], "k": row["k"], "status": "completed"}
            rec = {"instance_id": row["id"], "k": row["k"], "status": "infra_error", "run_config_sha256": run_sha,
                   "error": f"worker process exited rc={rc}", "seed": stable_seed(row["id"], row["k"]),
                   "image_digest": digests[row["id"]]}
            if rc != 1:  # rc 1 = the worker itself already recorded an infra_error
                _append_jsonl(out / f"runs-{tag}.jsonl", rec)
                _append_jsonl(out / f"failures-{tag}.jsonl", rec)
            return rec

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(child, todo))

    failures = sum(1 for r in results if r.get("status") == "infra_error")
    completed = {f"{r['id']}__r{r['k']}": rec for r, rec in done}
    for r in results:
        if r.get("status") == "completed":
            completed[item_dir_name(r["instance_id"], r["k"])] = r
    summary = {"run_config_sha256": run_sha, "shard": tag, "n_items": len(mine), "already_complete": len(done),
               "attempted_in_this_invocation": len(todo), "failures_in_this_invocation": failures,
               "completed_in_this_invocation": len(results) - failures,
               "skipped_for_no_retry_infra": len(mine) - len(done) - len(todo), "completed_items": completed}
    (out / f"summary-{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return 1 if failures else 0


def run_one(args: argparse.Namespace) -> int:
    digests, entries, _ = load_image_pins(args.image_manifest, [args.id])
    api_key = os.environ.get("VLLM_API_KEY", "local")
    _setup_env(entries, [args.id], api_key)
    signal.signal(signal.SIGTERM, _on_stop)
    rec = run_item(args.out, args.tag, args.id, args.k, digests[args.id], args.run_sha, args.base_url, api_key)
    return 0 if rec["status"] == "completed" else 1


def make_schedule(args: argparse.Namespace) -> int:
    from evals.reward_hacking.scripts.swebench_preflight_merge import latest_records

    manifest = splits.load_manifest()
    parts = {name: splits.partition_ids(name, manifest=manifest) for name in ("train", "val", "test")}
    names = args.partitions
    if any(n not in ALLOWED_PARTITIONS for n in names) and not args.allow_test:
        raise ValueError("only train/val partitions may be scheduled (pass --allow-test deliberately for test)")
    raw = json.loads(args.image_manifest.read_text())
    preflight = latest_records(args.preflight)
    rows, excluded = build_schedule(parts, names, args.ks, preflight, raw["images"])
    counts = {name: sum(1 for r in rows if r["id"] in set(parts[name]) and r["k"] == args.ks[0]) for name in names}
    meta = {"partitions": names, "ks": args.ks, "tasks_per_partition": counts,
            "n_tasks": len({r["id"] for r in rows}), "n_excluded": len(excluded), "excluded": dict(sorted(excluded.items())),
            "preflight_dir": str(args.preflight), "n_preflight_records": len(preflight),
            "image_manifest": str(args.image_manifest),
            "image_manifest_sha256": hashlib.sha256(args.image_manifest.read_bytes()).hexdigest(),
            "splits_manifest_sha256": hashlib.sha256(splits.SPLITS_PATH.read_bytes()).hexdigest()}
    m = write_schedule(args.out, rows, meta)
    print(f"wrote {args.out}: {m['n_rows']} rows ({m['n_tasks']} tasks x ks {args.ks}), excluded {len(excluded)}; "
          f"sha256 {m['schedule_sha256']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["make-schedule"]:
            p = argparse.ArgumentParser(prog="swebench_run make-schedule")
            p.add_argument("--out", type=pathlib.Path, required=True)
            p.add_argument("--partitions", type=lambda s: s.split(","), default=list(ALLOWED_PARTITIONS))
            p.add_argument("--ks", type=int, nargs="+", default=[0])
            p.add_argument("--preflight", type=pathlib.Path, default=DEFAULT_PREFLIGHT,
                           help="dir of preflight results-*.jsonl (must include --original-oracle results)")
            p.add_argument("--image-manifest", type=pathlib.Path, default=DEFAULT_IMAGE_MANIFEST)
            p.add_argument("--allow-test", action="store_true")
            a = p.parse_args(argv[1:])
            if len(set(a.ks)) != len(a.ks) or min(a.ks) < 0:
                raise ValueError("--ks must be distinct non-negative ints")
            return make_schedule(a)
        if argv[:1] == ["_one"]:
            p = argparse.ArgumentParser(prog="swebench_run _one")
            p.add_argument("--id", required=True)
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--out", type=pathlib.Path, required=True)
            p.add_argument("--tag", required=True)
            p.add_argument("--image-manifest", type=pathlib.Path, required=True)
            p.add_argument("--base-url", required=True)
            p.add_argument("--run-sha", required=True)
            return run_one(p.parse_args(argv[1:]))
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--schedule", type=pathlib.Path, required=True)
        p.add_argument("--shard", default="0/1", help="i/n, 0-based")
        p.add_argument("--out", type=pathlib.Path, required=True)
        p.add_argument("--image-manifest", type=pathlib.Path, default=DEFAULT_IMAGE_MANIFEST)
        p.add_argument("--base-url", help="defaults to VLLM_BASE_URL or URL + /v1")
        p.add_argument("--concurrency", type=int, default=1, help="child processes running one (id, k) each")
        p.add_argument("--no-retry-infra", action="store_true")
        p.add_argument("--allow-other-partitions", action="store_true", help="allow ids outside train/val")
        a = p.parse_args(argv)
        if a.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")
        return run(a)
    except Exception as exc:  # noqa: BLE001
        print(f"swebench_run setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

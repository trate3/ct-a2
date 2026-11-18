"""Run the frozen ImpossibleBench SWE training pilot on the existing vLLM server.

No test-partition IDs are accepted. Inspect .eval logs are kept per instance;
completed, scored samples are discovered from those logs and skipped on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log

from evals.reward_hacking import swebench_tasks as splits
from evals.reward_hacking.swebench_apptainer import (
    DATASET_REVISION,
    dockerhub_image_for_id,
    impossible_swebench_apptainer,
)
from evals.reward_hacking.swebench_scorer_guard import strict_swe_bench_scorer


DEFAULT_IMAGE_MANIFEST = pathlib.Path(__file__).parents[1] / "data" / "swebench" / "image_digests_v1.json"
MODEL = "vllm/qwen3.8-27b"
SERVED_MODEL = "qwen3.8-27b"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHAT_TEMPLATE_PATH = pathlib.Path(__file__).parents[1] / "data" / "qwen3.8-27b_chat_template.jinja"
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
BASE_SEED = 20261104
# the run plan (Muse diagnostic screens): SWE_MESSAGE_LIMIT overrides the limit; unset it is 100, as in every earlier run.
MESSAGE_LIMIT = int(os.environ.get("SWE_MESSAGE_LIMIT") or 100)
MAX_ATTEMPTS = 3
RESET_TESTS = True


VLLM_EXTRA_BODY = {"top_k": 20, "return_token_ids": True}


def pilot_ids(*, selected_id: str | None = None, all_ids: bool = False) -> list[str]:
    manifest = splits.load_manifest()
    allowed = splits.pilot_ids(manifest=manifest)
    if all_ids == (selected_id is not None):
        raise ValueError("select exactly one of --id ID or --all")
    if selected_id is not None:
        if selected_id not in allowed:
            raise ValueError(f"{selected_id!r} is not in the frozen training pilot IDs")
        return [selected_id]
    return allowed


def load_image_pins(path: pathlib.Path, ids: list[str]) -> tuple[dict[str, str], dict[str, dict], str]:
    """Require a verified immutable manifest ref for every selected task."""
    raw = json.loads(path.read_text())
    if raw.get("version") != 1 or raw.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("image pin manifest version/dataset revision mismatch")
    entries = raw.get("images")
    if not isinstance(entries, dict):
        raise ValueError("image pin manifest has no images mapping")
    selected: dict[str, dict] = {}
    digest_map: dict[str, str] = {}
    for iid in ids:
        entry = entries.get(iid)
        if not isinstance(entry, dict) or entry.get("verified") is not True:
            raise ValueError(f"missing verified image pin for pilot ID {iid}")
        expected_name = splits.image_name(iid).removesuffix(":latest")
        if entry.get("image") != expected_name:
            raise ValueError(f"image name mismatch for {iid}: {entry.get('image')} != {expected_name}")
        digest = entry.get("manifest_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"missing full OCI manifest digest for {iid}")
        sif_sha = entry.get("sif_sha256")
        if sif_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sif_sha)):
            raise ValueError(f"invalid cached SIF hash for {iid}")
        selected[iid] = entry
        digest_map[iid] = digest
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest_map, selected, manifest_sha


def _stable_seed(instance_id: str) -> int:
    h = hashlib.sha256(f"{BASE_SEED}\0{instance_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _check_server(base_url: str) -> None:
    if hashlib.sha256(CHAT_TEMPLATE_PATH.read_bytes()).hexdigest() != CHAT_TEMPLATE_SHA256:
        raise ValueError("Qwen3.8 chat-template digest changed; refusing to run with a different prompt format")
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer local"})
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.loads(response.read())
    served = {str(row.get("id")) for row in body.get("data", [])}
    if SERVED_MODEL not in served:
        raise ValueError(f"vLLM endpoint does not serve {SERVED_MODEL}: {sorted(served)}")


def _score_result(log) -> dict[str, Any] | None:
    if not log.samples:
        return None
    for scorer, score in (log.samples[0].scores or {}).items():
        metadata = getattr(score, "metadata", None) or {}
        kind = metadata.get("evaluation_kind")
        if kind in {"pass", "test_failure", "flagged"}:
            value = float(score.value)
            label = 1 if kind == "pass" else 0
            return {"scorer": scorer, "kind": kind, "value": value, "label": label}
    return None


def completed_from_logs(log_dir: pathlib.Path, instance_id: str) -> dict[str, Any] | None:
    """Recover a completed sample even if the runner died before writing runs.jsonl."""
    if not log_dir.exists():
        return None
    for info in list_eval_logs(str(log_dir), recursive=True, descending=True):
        try:
            log = read_eval_log(info)
        except Exception:
            continue
        if getattr(log, "status", None) != "success" or not log.samples:
            continue
        sample = log.samples[0]
        if str(sample.id) != instance_id:
            continue
        score = _score_result(log)
        if score is not None:
            return {"instance_id": instance_id, **score, "eval_log": str(info.name)}
    return None


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_or_write_run_manifest(path: pathlib.Path, value: dict) -> str:
    digest = _canonical_sha256(value)
    record = {"config": value, "config_sha256": digest}
    if path.exists():
        old = json.loads(path.read_text())
        if old != record:
            raise ValueError(f"run directory manifest mismatch at {path}; choose a new --out")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return digest


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _failure_reason(logs: list | None, exc: Exception | None = None) -> str:
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"
    if not logs:
        return "Inspect returned no EvalLog"
    log = logs[0]
    error = getattr(log, "error", None)
    return str(getattr(error, "message", error) or f"Inspect status={getattr(log, 'status', None)}")


SWE_SPLIT = os.environ.get("SWE_SPLIT", "conflicting")   # the run plan: "oneoff" runs the one-off variant of the same instances
if SWE_SPLIT not in ("conflicting", "oneoff"):
    raise ValueError(f"SWE_SPLIT must be conflicting or oneoff, got {SWE_SPLIT!r}")


def _make_task(instance_id: str, image_digest: str, custom_prompt: str = "", max_attempts: int = MAX_ATTEMPTS):
    return impossible_swebench_apptainer(
        split=SWE_SPLIT,
        instance_ids=[instance_id],
        image_digests={instance_id: image_digest},
        agent_type="tools",
        max_attempts=max_attempts,
        message_limit=MESSAGE_LIMIT,
        allow_internet=False,
        hide_tests=False,
        reset_tests=RESET_TESTS,
        scorer=strict_swe_bench_scorer(reset_patch=False, reset_tests=RESET_TESTS, validated_task=True),
        custom_prompt=custom_prompt,
    )


def run(args: argparse.Namespace) -> int:
    ids = pilot_ids(selected_id=args.id, all_ids=args.all)
    run_ids = ids
    pytest_id = "pytest-dev__pytest-5787"
    if args.all and pytest_id in ids:
        run_ids = [pytest_id, *(iid for iid in ids if iid != pytest_id)]
    all_ids = splits.pilot_ids()
    all_image_digests, all_image_entries, image_manifest_sha = load_image_pins(args.image_manifest, all_ids)
    image_digests = {iid: all_image_digests[iid] for iid in ids}
    image_entries = {iid: all_image_entries[iid] for iid in ids}
    base_url = args.base_url or os.environ.get("VLLM_BASE_URL")
    if not base_url:
        server_url = os.environ.get("URL")
        if server_url:
            base_url = server_url.rstrip("/") + "/v1"
    if not base_url:
        raise ValueError("set VLLM_BASE_URL (or URL) to the running vLLM server")
    api_key = os.environ.get("VLLM_API_KEY", "local")
    _check_server(base_url)
    out = args.out.expanduser().resolve()
    inspect_root = out / "inspect"
    config = {
        "dataset": "fjzzq2002/impossible_swebench",
        "dataset_revision": DATASET_REVISION,
        "benchmark_commit": splits.IMPOSSIBLEBENCH_COMMIT,
        "split": "conflicting",
        "partition": "train_pilot_only",
        # Stable across `--id X` followed by `--all`: completed .eval logs for
        # X can be resumed while the remaining pilot tasks run.
        "ids": all_ids,
        "image_manifest_sha256": image_manifest_sha,
        "image_digests": all_image_digests,
        "model": MODEL,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "api_endpoint": "local-vllm-openai-chat-completions",
        "server_profile": {
            "auto_tool_choice": True,
            "tool_call_parser": "qwen3_xml",
            "reasoning_parser": "qwen3",
            "chat_template": "evals/reward_hacking/data/qwen3.8-27b_chat_template.jinja",
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        },
        "message_limit": MESSAGE_LIMIT,
        "max_attempts": MAX_ATTEMPTS,
        "reset_tests": RESET_TESTS,
        "solver": "tools",
        "sandbox": "ct_swe_apptainer",
        "seed_base": BASE_SEED,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        # Inspect's OpenAI-compatible provider drops top_k, so it goes to vLLM in extra_body; return_token_ids
        # makes the logged raw responses carry prompt and generated token ids for token-exact probe extraction.
        "extra_body": VLLM_EXTRA_BODY,
    }
    run_sha = _load_or_write_run_manifest(out / "manifest.json", config)
    out.mkdir(parents=True, exist_ok=True)

    sif_dir = pathlib.Path(os.environ.get("CT_SWEBENCH_APPTAINER_SIF_DIR", pathlib.Path.home() / "rh/swebench/images"))
    os.environ["CT_SWEBENCH_APPTAINER_SIF_DIR"] = str(sif_dir)
    sif_hashes = {
        iid: image_entries[iid]["sif_sha256"]
        for iid in ids
        if image_entries[iid].get("sif_sha256") and (sif_dir / f"{iid}.sif").is_file()
    }
    os.environ["CT_SWEBENCH_APPTAINER_SIF_SHA256S"] = json.dumps(sif_hashes, sort_keys=True)
    os.environ.setdefault("CT_SWEBENCH_APPTAINER_CONCURRENCY", "1")
    # Inspect's OpenAI-compatible provider reads OPENAI_API_KEY; the Slurm
    # endpoint convention exposes the same local token as VLLM_API_KEY.
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    failures = 0
    for instance_id in run_ids:
        per_id_logs = inspect_root / instance_id
        recovered = completed_from_logs(per_id_logs, instance_id)
        if recovered is not None:
            print(f"resume-skip {instance_id}")
            continue

        seed = _stable_seed(instance_id)
        started = time.time()
        logs = None
        exc: Exception | None = None
        try:
            task = _make_task(instance_id, image_digests[instance_id])
            if len(task.dataset) != 1 or str(task.dataset[0].id) != instance_id:
                raise RuntimeError("task factory returned an unexpected sample set")
            from inspect_ai import eval as inspect_eval

            logs = inspect_eval(
                task,
                model=MODEL,
                model_base_url=base_url,
                model_args={"api_key": api_key},
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                extra_body=VLLM_EXTRA_BODY,
                seed=seed,
                max_connections=1,
                max_sandboxes=1,
                max_subprocesses=1,
                log_dir=str(per_id_logs),
                fail_on_error=False,
                score_on_error=False,
                # Log every raw API call (default: first few only), so every call's vLLM token ids are recorded.
                log_model_api=True,
            )
        except Exception as error:
            exc = error

        result = None
        if logs and len(logs) == 1:
            log = logs[0]
            result = _score_result(log) if getattr(log, "status", None) == "success" else None
        record = {
            "instance_id": instance_id,
            "status": "completed" if result is not None else "infra_error",
            "label": result["label"] if result is not None else None,
            "evaluation_kind": result["kind"] if result is not None else None,
            "score": result["value"] if result is not None else None,
            "scorer": result["scorer"] if result is not None else None,
            "inspect_status": getattr(logs[0], "status", None) if logs else None,
            "error": None if result is not None else _failure_reason(logs, exc),
            "seed": seed,
            "image_digest": image_digests[instance_id],
            "run_config_sha256": run_sha,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        _append_jsonl(out / "runs.jsonl", record)
        if result is None:
            failures += 1
            _append_jsonl(out / "failures.jsonl", record)
            print(f"FAILED {instance_id}: {record['error']}", file=sys.stderr)
        else:
            print(f"DONE {instance_id}")

    summary = {
        "run_config_sha256": run_sha,
        "selected_ids": ids,
        "completed": {iid: completed_from_logs(inspect_root / iid, iid) for iid in ids},
        "failures_in_this_invocation": failures,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    known = splits.pilot_ids()
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--id", choices=known, help="run one frozen training pilot instance")
    choice.add_argument("--all", action="store_true", help="run all four frozen training pilot instances")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--image-manifest", type=pathlib.Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--base-url", help="vLLM OpenAI API base URL; defaults to VLLM_BASE_URL or URL + /v1")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"swebench pilot setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

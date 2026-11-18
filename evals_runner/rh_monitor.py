"""Reward hacking: score finished rollouts with the matched black-box and CoT monitors (EQ2).

Reads `<rollouts>/trajectories/*.json` and appends one row per (rollout, view) to the monitor's scores file;
reruns skip rows already present. Rollouts without a label (infrastructure errors) and malformed tasks are
skipped. Terminal decision point only (labelled submission, proposed actions).

    python -m evals_runner rh-monitor --rollouts DIR [DIR ...] --views black_box,cot \
        [--monitor-model qwen|glm] [--api completion|chat] [--engine llamacpp|vllm --url ...]

`--monitor-model NAME` is a key of `monitors.MONITORS` (the registry). Besides qwen and glm there are small open
monitors served by vLLM (deploy/second_machine/slurm/monitor_vllm.sbatch): gemma4-e2b, gemma4-e4b, cotguard-rl, cotguard-sft
(all with the SAME matched user message) and cotguard-rl-native, cotguard-sft-native (CoT-Guard's own prompt and A/B/C/D
verdict; binary score, so AUROC is degenerate). Scores/manifest: monitor_scores.<NAME>.jsonl / monitor_manifest.<NAME>.json.
Their context budget is counted in tokens with the model's own tokenizer (`--tokenizer DIR|HF_ID`, default
$RH_HOME/models/<weights>), falling back to a chars/token estimate (recorded as `token_count_method`); only private
reasoning is ever shortened. `--yarn on|off` overrides the registry's YaRN setting (CoT-Guard); it is recorded.

Monitor models (same instruction, user message, score parsing, sampling and context budget for both):
  qwen  the target model; default api `completion` (our pinned template, thinking off, greedy). Files:
        monitor_scores.jsonl, monitor_manifest.json (the pre-EQ2 names, kept for compatibility).
  glm   GLM-5.3-Flash on llama.cpp `--jinja`; default api `chat` (/v1/chat/completions, the GGUF's own
        template, thinking disabled). Files: monitor_scores.glm.jsonl, monitor_manifest.glm.json.

The manifest records the engine, model identity (for chat: model path, build, chat-template sha256), config
and instruction digest; a changed engine/identity refuses to resume. Fail-closed accounting (the run plan 5): a row
whose output has no parseable `SCORE:` is written with score=null and counted (`counts` in the manifest and
the final line); a record that does not fit is written as an `error` row and counted. SIGTERM/SIGUSR1 stop
after the current row (status 75); rows are appended one at a time and a torn last line is repaired on the
next run.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from evals.reward_hacking import llm, monitors, rollout
from evals.reward_hacking import trajectory as tj
from evals_runner.rh_rollouts import DEFAULT_OUT
from evals_runner.gpu_cost import GpuBusySampler


MONITOR_MODELS = tuple(monitors.MONITORS)
# Identity fields whose change means these scores would no longer come from one monitor.
_IDENTITY_KEYS = {"completion": ("engine",), "chat": ("engine", "model_path", "build_info", "chat_template_sha256"),
                  "chat-vllm": ("engine", "model_alias", "engine_version", "max_model_len", "chat_template_sha256",
                                "revision")}
# Registry fields whose change means a different monitor (a changed YaRN setting changes the model's outputs).
_SPEC_KEYS = ("hf_id", "revision", "prompt_style", "hf_overrides", "max_model_len", "max_prompt_tokens", "max_tokens",
              "enable_thinking", "sampling", "request_extra")
# Fields added with the run plan: a manifest written before they existed counts as holding the default.
_SPEC_KEYS_LATE = {"assistant_prefill": "", "fail_closed_no_marker": False}


def scores_path(rollouts: pathlib.Path, model: str) -> pathlib.Path:
    return rollouts / ("monitor_scores.jsonl" if model == "qwen" else f"monitor_scores.{model}.jsonl")


def manifest_path(rollouts: pathlib.Path, model: str) -> pathlib.Path:
    return rollouts / ("monitor_manifest.json" if model == "qwen" else f"monitor_manifest.{model}.json")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rollouts", type=pathlib.Path, nargs="+", default=[DEFAULT_OUT],
                        help="one or more rollout dirs (e.g. train+val and test); each gets its own scores/manifest")
    parser.add_argument("--monitor-model", choices=MONITOR_MODELS, default="qwen")
    parser.add_argument("--api", choices=("completion", "chat"), default=None,
                        help="default: the registry's (completion for qwen, chat otherwise)")
    parser.add_argument("--tokenizer", default=None,
                        help="local dir or HF id of the monitor's tokenizer for exact token budgets "
                             "(registry monitors; default $RH_HOME/models/<weights> if present, else the HF id)")
    parser.add_argument("--yarn", choices=("auto", "on", "off"), default="auto",
                        help="CoT-Guard: YaRN context extension (auto = registry default: on)")
    parser.add_argument("--views", default="black_box,cot")
    parser.add_argument("--engine", choices=llm.ENGINES, default="llamacpp")
    parser.add_argument("--url", default=llm.DEFAULT_URL)
    parser.add_argument("--workers", type=int, default=1,
                        help="monitor calls in flight at once (greedy decoding; rows are order-independent)")
    parser.add_argument("--measure-gpu-busy", action="store_true",
                        help="sample NVML GPU utilization per check; requires --workers 1 and a dedicated monitor GPU")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--partitions", default=None,
                        help="comma-separated split partitions to score (e.g. val,test: thresholds come from val and "
                             "results from test, so train rows are not needed); default all")
    parser.add_argument("--actions", choices=("proposed", "executed"), default="proposed",
                        help="model-level view (proposed actions, default) or agent-level view of what the agent "
                             "executed (EQ3 positive control); executed scores go to monitor_scores.<model>.executed.jsonl")
    parser.add_argument("--final-state-only", action="store_true",
                        help="score only the original task instruction and submitted func.py/test.py; writes separate final-state files")
    parser.add_argument("--final-state-eval", action="store_true",
                        help="like --final-state-only, plus the scorer's result (passed + explanation) for the "
                             "labelled submission; writes monitor_scores.<model>.final_state_eval.jsonl")
    parser.add_argument("--instruction-file", type=pathlib.Path, default=None,
                        help="use this plain-text instruction (one {record} field) instead of "
                             "monitors.INSTRUCTION; needs --score-tag; history views, proposed actions only")
    parser.add_argument("--score-tag", default=None,
                        help="with --instruction-file, scores go to monitor_scores.<model>.<TAG>.jsonl "
                             "and monitor_manifest.<model>.<TAG>.json")
    parser.set_defaults(run=run)


def _read_rows(path: pathlib.Path) -> list[dict]:
    """Complete JSON rows; a torn final line (killed mid-write) is dropped and the file repaired."""
    if not path.exists():
        return []
    text = path.read_text()
    rows, good = [], []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
            good.append(line)
        except json.JSONDecodeError:
            pass
    if len(good) != len(text.splitlines()) or (text and not text.endswith("\n")):
        path.write_text("".join(line + "\n" for line in good))
    return rows


def _make_client(args, api: str, spec: monitors.MonitorSpec | None = None, engine: str = "llamacpp"):
    if api == "chat":
        if spec is not None and spec.engine == "vllm":
            return monitors.ChatMonitorClient(args.url, model=spec.weights, server="vllm",
                                              extra_body=spec.request_extra, reasoning_format=spec.reasoning_format,
                                              assistant_prefill=spec.assistant_prefill)
        return monitors.ChatMonitorClient(args.url)
    return llm.make_client(engine, args.url)


def _find_tokenizer_source(args, spec: monitors.MonitorSpec) -> str | None:
    """Explicit --tokenizer, else the downloaded snapshot (monitor_vllm.sbatch), else the pinned HF id."""
    if getattr(args, "tokenizer", None):
        return args.tokenizer
    local = pathlib.Path(os.environ.get("RH_HOME", pathlib.Path.home() / "rh")) / "models" / spec.weights
    return str(local) if (local / "tokenizer_config.json").exists() else spec.hf_id


def _chat_template_sha(source: str | None) -> str | None:
    """sha256 of the chat template shipped with the local snapshot (vLLM exposes no /props)."""
    if not source or not os.path.isdir(source):
        return None
    d = pathlib.Path(source)
    jinja = d / "chat_template.jinja"
    if jinja.exists():
        return hashlib.sha256(jinja.read_bytes()).hexdigest()
    try:
        tpl = json.loads((d / "tokenizer_config.json").read_text()).get("chat_template")
    except (OSError, ValueError):
        return None
    return hashlib.sha256(tpl.encode()).hexdigest() if isinstance(tpl, str) else None


def _identity(client, api: str, engine: str) -> dict:
    ident = client.identity() if hasattr(client, "identity") else {"engine": engine}
    if api == "completion":       # the server template is unused (prompts are rendered client-side)
        ident = {k: v for k, v in ident.items() if k != "server_template_sha256"}
    return ident


def _check_resume(man_path: pathlib.Path, model: str, api: str, ident: dict,
                  spec: monitors.MonitorSpec | None = None) -> None:
    if not man_path.exists():
        return
    old = json.loads(man_path.read_text())
    if spec is not None and "spec" in old.get("monitor", {}):
        for k in _SPEC_KEYS:
            if old["monitor"]["spec"].get(k) != getattr(spec, k):
                raise SystemExit(f"monitor registry entry changed ({k}: {old['monitor']['spec'].get(k)!r} -> "
                                 f"{getattr(spec, k)!r}) since these {model} monitor scores were started")
        for k, default in _SPEC_KEYS_LATE.items():
            if old["monitor"]["spec"].get(k, default) != getattr(spec, k):
                raise SystemExit(f"monitor registry entry changed ({k}: {old['monitor']['spec'].get(k, default)!r} -> "
                                 f"{getattr(spec, k)!r}) since these {model} monitor scores were started")
    old_id = old.get("monitor", {}).get("identity") or old.get("target_model", {})
    old_api = old.get("monitor", {}).get("api", "completion")
    if old_api != api:
        raise SystemExit(f"api changed ({old_api} -> {api}) since these {model} monitor scores were started")
    old_engine = old_id.get("engine", "llamacpp")
    if old_engine != ident.get("engine", old_engine):
        raise SystemExit(f"engine changed ({old_engine} -> {ident.get('engine')}) since these monitor scores were started")
    for k in _IDENTITY_KEYS["chat-vllm" if ident.get("engine") == "vllm" and api == "chat" else api][1:]:
        if old_id.get(k) != ident.get(k):
            raise SystemExit(f"monitor identity changed ({k}: {old_id.get(k)} -> {ident.get(k)}) "
                             f"since these {model} monitor scores were started")


def run(args: argparse.Namespace) -> int:
    model = getattr(args, "monitor_model", "qwen")
    if getattr(args, "measure_gpu_busy", False) and args.workers != 1:
        raise SystemExit("--measure-gpu-busy requires --workers 1 for per-check attribution")
    spec = monitors.get_spec(model)
    yarn = {"on": True, "off": False}.get(getattr(args, "yarn", "auto"))
    spec = spec.with_yarn(yarn)
    actions = getattr(args, "actions", "proposed")
    final_state_eval = getattr(args, "final_state_eval", False)
    final_state_only = getattr(args, "final_state_only", False) or final_state_eval
    if final_state_only and actions != "proposed":
        raise SystemExit("--final-state-only cannot be combined with --actions executed")
    tag = model if actions == "proposed" else f"{model}.executed"
    if final_state_only:
        tag = f"{model}.final_state_eval" if final_state_eval else f"{model}.final_state"
    custom = None                    # the run plan: (template, sha256) of a custom instruction file
    if getattr(args, "instruction_file", None) or getattr(args, "score_tag", None):
        if not (getattr(args, "instruction_file", None) and getattr(args, "score_tag", None)):
            raise SystemExit("--instruction-file and --score-tag go together")
        if final_state_only or actions != "proposed" or model in ("qwen", "glm"):
            raise SystemExit("--instruction-file needs a registry monitor, history views and proposed actions")
        if not all(c.isalnum() or c in "_-" for c in args.score_tag):
            raise SystemExit("--score-tag may hold only letters, digits, _ and -")
        from evals.reward_hacking import prompt_opt
        custom = prompt_opt.load_instruction_file(args.instruction_file)
        tag = f"{model}.{args.score_tag}"
    api = getattr(args, "api", None) or spec.api
    engine = spec.engine or getattr(args, "engine", "llamacpp")
    dirs = args.rollouts if isinstance(args.rollouts, list) else [args.rollouts]
    client = _make_client(args, api, spec, engine)
    ident = _identity(client, api, engine)
    config = spec.config() if model not in ("qwen", "glm") else monitors.MonitorConfig()
    counter, tok_info = None, None
    if model not in ("qwen", "glm"):
        src = _find_tokenizer_source(args, spec)
        counter = monitors.load_counter(src, revision=spec.revision, chars_per_token=spec.chars_per_token)
        ident["chat_template_sha256"] = ident.get("chat_template_sha256") or _chat_template_sha(src)
        ident["revision"] = spec.revision
        tok_info = {"source": counter.source, "method": counter.method,
                    "class": type(getattr(counter, "tok", None)).__name__ if counter.method == "tokenizer" else None}
    for d in dirs:
        _check_resume(manifest_path(d, tag), model, api, ident, spec)
    for d in dirs:
        manifest = {"monitor": {"name": model, "api": api, "identity": ident},
                    "monitor_config": config.as_dict(), "instruction_sha256": monitors.instruction_digest()}
        if final_state_only:
            lcb_ins, swe_ins = ((monitors.FINAL_STATE_EVAL_INSTRUCTION, monitors.FINAL_STATE_SWE_EVAL_INSTRUCTION)
                                if final_state_eval else
                                (monitors.FINAL_STATE_INSTRUCTION, monitors.FINAL_STATE_SWE_INSTRUCTION))
            manifest["observation"] = "final_state_eval" if final_state_eval else "final_state_only"
            manifest["instruction_sha256"] = hashlib.sha256(lcb_ins.encode()).hexdigest()   # LCB (as before)
            manifest["instruction_sha256_swe"] = hashlib.sha256(swe_ins.encode()).hexdigest()
        if model not in ("qwen", "glm"):
            manifest["monitor"]["spec"] = spec.as_dict()
            manifest["monitor"]["token_counting"] = tok_info
            if spec.prompt_style == "cotguard_native":
                manifest["native_prompt_sha256"] = monitors.cgn.digest()
                manifest["native_prompt_source"] = monitors.cgn.SOURCE
        if model == "qwen":
            manifest["target_model"] = ident          # pre-EQ2 key
        if custom is not None:
            old_man = manifest_path(d, tag)
            if old_man.exists() and json.loads(old_man.read_text()).get("instruction_sha256") != custom[1]:
                raise SystemExit(f"{old_man.name}: a different instruction file was used for these scores")
            manifest["instruction_sha256"] = custom[1]
            manifest["custom_instruction"] = {"file": str(args.instruction_file), "sha256": custom[1]}
        manifest_path(d, tag).write_text(json.dumps(manifest, indent=1) + "\n")
    stop = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    rc = 0
    try:
        for d in dirs:
            out = scores_path(d, tag)
            done = {(r["rollout_id"], r["view"]) for r in _read_rows(out)}
            rc = _run_rows(args, d, out, done, config, client, stop, model, api, actions, counter,
                           final_state_only=final_state_only, final_state_eval=final_state_eval,
                           custom=custom) or rc
            _write_counts(d, tag, out)
            if stop.is_set():
                return 75
        return rc
    finally:
        for sig, h in old_handlers.items():
            signal.signal(sig, h)


def count_rows(rows: list[dict]) -> dict:
    """Fail-closed accounting: unparseable scores and errors are counted, never dropped."""
    return {"rows": len(rows), "scored": sum(r.get("score") is not None for r in rows),
            "unparseable": sum("error" not in r and r.get("score") is None for r in rows),
            "errors": sum("error" in r for r in rows)}


def _write_counts(d: pathlib.Path, tag: str, out: pathlib.Path) -> None:
    rows = _read_rows(out)
    by_view = {v: count_rows([r for r in rows if r["view"] == v]) for v in sorted({r["view"] for r in rows})}
    mp = manifest_path(d, tag)
    man = json.loads(mp.read_text())
    man["counts"] = {"all": count_rows(rows), "by_view": by_view}
    mp.write_text(json.dumps(man, indent=1) + "\n")
    print(f"[{tag}] {d}: {man['counts']['all']}", flush=True)


def _run_rows(args, rollouts, out, done, config, client, stop, model="qwen", api="completion",
              actions="proposed", counter=None, final_state_only=False, final_state_eval=False, custom=None) -> int:
    views = ["black_box"] if final_state_only else [v.strip() for v in args.views.split(",")]
    parts = None if not getattr(args, "partitions", None) else {p.strip() for p in args.partitions.split(",")}
    paths = sorted((rollouts / "trajectories").glob("*.json"))[:args.limit]
    lock = threading.Lock()
    obs = ("final_state_eval" if final_state_eval else "final_state_only") if final_state_only else "history"

    def job(path) -> None:
        if stop.is_set():
            return
        traj = tj.Trajectory.from_json(path.read_text())
        rid = traj.meta["rollout_id"]
        lab = tj.label(traj)
        if lab is None or traj.meta["task_id"] in rollout.MALFORMED_TASKS:
            return
        if parts is not None and traj.meta.get("partition") not in parts:
            return
        for view in views:
            if stop.is_set():
                return
            if (rid, view) in done:
                continue
            sampler = None
            check_started = None
            try:
                kw = {} if counter is None else {"counter": counter}
                if custom is not None:
                    kw["instruction"] = custom[0]
                sampler = GpuBusySampler().start() if getattr(args, "measure_gpu_busy", False) else None
                check_started = time.perf_counter()
                r = monitors.run_monitor(client, traj, view, config=config, actions=actions,
                                         final_state_only=final_state_only,
                                         include_evaluation=final_state_eval, **kw)
                check_wall_seconds = round(time.perf_counter() - check_started, 6)
            except ValueError as exc:      # record does not fit the monitor's context
                gpu_seconds = sampler.stop() if 'sampler' in locals() and sampler else None
                check_wall_seconds = None if check_started is None else round(time.perf_counter() - check_started, 6)
                row = {"rollout_id": rid, "view": view, "monitor_model": model, "api": api,
                       "observation": obs,
                       "error": str(exc), "check_wall_seconds": check_wall_seconds,
                       "gpu_allocation_seconds": check_wall_seconds if sampler else None,
                       "gpu_allocation_seconds_note": ("equals check_wall_seconds for a dedicated 1-GPU serial job"
                                                       if sampler else None),
                       "gpu_busy_seconds": gpu_seconds}
            except requests.HTTPError as exc:
                gpu_seconds = sampler.stop() if 'sampler' in locals() and sampler else None
                check_wall_seconds = None if check_started is None else round(time.perf_counter() - check_started, 6)
                # The server rejected this request (HTTP 4xx, e.g. prompt over its context limit): a
                # fail-closed error row, as for an oversized record; 5xx and connection errors still raise.
                resp = getattr(exc, "response", None)
                if resp is None or not 400 <= resp.status_code < 500:
                    raise
                row = {"rollout_id": rid, "view": view, "monitor_model": model, "api": api,
                       "observation": obs,
                       "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                       "check_wall_seconds": check_wall_seconds,
                       "gpu_allocation_seconds": check_wall_seconds if sampler else None,
                       "gpu_allocation_seconds_note": ("equals check_wall_seconds for a dedicated 1-GPU serial job"
                                                       if sampler else None),
                       "gpu_busy_seconds": gpu_seconds}
            except Exception:
                if 'sampler' in locals() and sampler:
                    sampler.stop()
                raise
            else:
                gpu_seconds = sampler.stop() if 'sampler' in locals() and sampler else None
                gpu_allocation_seconds = check_wall_seconds if sampler else None
                row = {"rollout_id": rid, "task_id": traj.meta["task_id"], "view": view, "label": lab,
                       "score": r.score, "output": r.output, "info": r.info,
                       "prompt_tokens": r.prompt_tokens, "generated_tokens": r.generated_tokens,
                       "seconds": round(r.seconds, 3), "check_wall_seconds": check_wall_seconds,
                       "gpu_allocation_seconds": gpu_allocation_seconds,
                       "gpu_allocation_seconds_note": ("equals check_wall_seconds for a dedicated 1-GPU serial job"
                                                       if sampler else None),
                       "prompt_sha256": r.prompt_sha256,
                       "user_sha256": r.user_sha256, "finish_reason": r.finish_reason,
                       "gpu_busy_seconds": gpu_seconds,
                       "reasoning_chars": r.reasoning_chars, "monitor_model": model, "api": api,
                       "truncated_reasoning_chars": r.info.get("truncated_reasoning_chars", 0),
                       "observation": obs,
                       "user_tokens": r.info.get("user_tokens"),
                       "monitor_config": config.as_dict(),
                       "instruction_sha256": (r.info["instruction_sha256"]
                                               if final_state_only else
                                               (monitors.instruction_digest() if custom is None else custom[1]))}
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                # progress is counts-only: the label and the score are in the output file, never in the log
                print(f"{rid} {view}: row written {row.get('prompt_tokens', '-')} tok", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for f in [pool.submit(job, p) for p in paths]:
            f.result()
    if stop.is_set():
        print("interrupted; rerun the same command to resume", flush=True)
        return 75
    return 0

"""Retry of the failed or unsent items of an EQ1 monitor-cost panel run, as a separately identified export.

    python -m evals_runner.eq1_monitor_panel --retry --panel-dir OUT/<monitor> --url http://127.0.0.1:8100 \\
        --trajectories <WORK_DIR>/live_in/traj/trajectories --model-dir <snapshot dir> --out-dir OUT/retry1/<monitor> \\
        [--server-log ..] [--server-info ..] [--server-context N] [--prompts P --manifest M]
    python -m evals_runner.eq1_monitor_panel --retry-combine OUT/retry1      # summary.json and panel_table_completed.md/.tex

This is for the COST panel only. Nothing in --panel-dir is written (its config, rows and replies stay as they are); everything goes
to --out-dir, which has its own run_config.json written before the first request, rows.jsonl, replies.jsonl, sessions.jsonl,
retry_prompts.json.gz (the changed user messages) and summary.json. Detector accuracy results are not touched.

1. FAILED ITEMS BY RULE, from the original rows.jsonl: an item is failed when none of its rows is a returned reply with a valid score
   (the same validity test as the panel). Reasons: not sent (over the registry prompt budget); HTTP error (for example 400 when the prompt
   was longer than the server context); length stop; reply without a parsable score.
2. WHAT IS CHANGED, by kind:
   refit      the original prompt, plus the output cap the runner would send for it, did not fit the budget below. The prompt is rebuilt with the
              detector's normal fitting path (`monitors._fit` with the matched assembler, the code under `build_prompt_fitted`): only the agent's
              private reasoning is shortened, by middle-truncating the longest reasoning blocks, until the prompt fits. Nothing else is cut.
   raise_cap  the reply stopped at the output cap without a score (prompt unchanged): the cap is raised to what the server context allows for
              that prompt (context - prompt tokens - template overhead - margin).
   plain      any other failure whose prompt fits: the same request again.
   Budget rule (tokens, counted with the monitor's own tokenizer): prompt + template overhead <= min(registry max_prompt_tokens,
   server context - output allowance - 512), where the output allowance is the registry max_tokens (min_output_tokens for dynamic-output
   monitors), and the output cap sent is the runner's (`max(min_output, min(max_tokens, context - prompt - overhead - 512))` for dynamic
   monitors, else max_tokens). A prompt that cannot be fitted (for example a final-code view, which has nothing to shorten) is not sent and
   stays failed. A forced score is never made: a reply without a parsable score stays failed.
3. EVIDENCE: every refit item whose private reasoning was shortened carries `evidence_shortened: true`; the share of reasoning characters kept
   and the changed blocks are recorded. These are counted separately; such a judgment saw less of the agent's reasoning than the detector's.
4. TWO TIME QUANTITIES per item: successful-invocation time (the original reply time for an item that was valid at first, the retry request
   time for a retried item) and retry-inclusive time (all original attempt durations of that item plus all retry attempts).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import statistics
import sys
import time

from evals.reward_hacking import monitors as m
from evals.reward_hacking import trajectory as tj
from evals_runner import eq1_monitor_panel as P
from evals_runner import eq1_monitor_panel_prompts as prompts_mod

MARGIN = 512                       # tokens: the runner's own margin in `left = context - prompt - overhead - 512`
DEFAULT_RETRY_TIMEOUT = 7200.0     # a raised cap can mean a very long generation


def code_hashes() -> dict:
    here = pathlib.Path(__file__).resolve().parent
    files = [here / "eq1_monitor_panel_retry.py", here / "eq1_monitor_panel.py", here / "eq1_monitor_panel_prompts.py",
             here.parent / "evals" / "reward_hacking" / "monitors.py", here.parent / "evals" / "reward_hacking" / "trajectory.py"]
    return {f.name: P.sha256_file(f) for f in files if f.is_file()}


# ---------------------------------------------------------------------------------------------------------------------
# 1. failed items by rule

def original_attempts(rows: list[dict], item: int) -> list[dict]:
    return [r for r in rows if r["item"] == item]


def failed_items(rows: list[dict], n_items: int) -> dict[int, dict]:
    """item -> {reason, attempts}: the items with no returned reply that holds a valid score, by rule, from the original rows."""
    out = {}
    for i in range(n_items):
        rs = original_attempts(rows, i)
        if any(r.get("http_error") is None and not r.get("not_sent") and r.get("valid_score") for r in rs):
            continue
        sent = [r for r in rs if not r.get("not_sent")]
        if any(r.get("not_sent") for r in rs):
            reason = "not sent: prompt over the registry prompt budget"
        elif not rs:
            reason = "no row for this item"
        elif any(r.get("http_error") for r in sent) and not any(r.get("http_error") is None for r in sent):
            reason = f"http error: status {sent[-1].get('http_status')}"
        elif any(r.get("finish_reason") == "length" for r in sent):
            reason = "length stop with no valid score"
        else:
            reason = "reply without a parsable score" + (f" ({sent[-1].get('unparseable_reason')})" if sent and sent[-1].get("unparseable_reason") else "")
        out[i] = {"reason": reason, "attempts": [{k: r.get(k) for k in ("attempt", "seconds", "http_status", "http_error", "not_sent",
                                                                          "finish_reason", "valid_score", "prompt_tokens", "generated_tokens",
                                                                          "output_cap", "user_tokens_counted")} for r in rs]}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. budgets, caps and refitting

def budget_rule(cfg, server_ctx: int) -> dict:
    allowance = cfg.min_output_tokens if cfg.server_context else cfg.max_tokens
    ctx_limited = server_ctx - allowance - MARGIN
    fit = min(cfg.max_prompt_tokens, ctx_limited)
    return {"registry_prompt_budget": cfg.max_prompt_tokens, "server_context": server_ctx, "output_allowance": allowance,
            "output_allowance_is": "min_output_tokens (dynamic output)" if cfg.server_context else "max_tokens",
            "margin": MARGIN, "template_overhead": cfg.template_overhead_tokens, "context_limited_budget": ctx_limited,
            "fit_budget": fit, "binding_limit": "registry prompt budget" if cfg.max_prompt_tokens <= ctx_limited else "server context",
            "rule": "prompt tokens + template overhead <= min(registry prompt budget, server context - output allowance - margin)"}


def raised_cap(cfg, server_ctx: int, utokens: int) -> int:
    """Largest output cap the server context allows for this prompt (the runner's `left`, without the registry max_tokens)."""
    return server_ctx - utokens - cfg.template_overhead_tokens - MARGIN


def shortening_report(before: list[tj.Event], after: list[tj.Event]) -> dict:
    changed = []
    tot_before = tot_after = 0
    for b, a in zip(before, after):
        if b.kind == tj.REASONING:
            tot_before += len(b.data["text"])
            tot_after += len(a.data["text"])
            if a.data["text"] != b.data["text"]:
                changed.append({"seq": b.seq, "call_index": b.call_index, "chars_before": len(b.data["text"]), "chars_after": len(a.data["text"])})
    n_reasoning = sum(1 for e in before if e.kind == tj.REASONING)
    return {"how": "the middle of the longest private-reasoning blocks was replaced by a marker (the first and last parts of each block "
                   "are kept); no other event was changed", "reasoning_blocks": n_reasoning, "blocks_shortened": len(changed),
            "reasoning_chars_before": tot_before, "reasoning_chars_after": tot_after,
            "reasoning_fraction_kept": (tot_after / tot_before) if tot_before else None, "changed_blocks": changed}


def refit_view(traj: tj.Trajectory, item: dict, cfg, counter, fit_budget: int) -> dict:
    """The detector's fitting path for one history view. Returns user, tokens (incl. overhead), events report, or the reason it cannot."""
    view = item["view"]
    if item["input"] != "history":
        return {"ok": False, "reason": "the final-code view has nothing that can be shortened"}
    events0 = tj.view_events(traj, view, actions="proposed")
    try:
        user, events, cut, n = m._fit(events0, view, m._matched_assembler(view), counter, fit_budget, cfg.template_overhead_tokens)
    except ValueError as exc:
        return {"ok": False, "reason": f"cannot be fitted by shortening private reasoning: {str(exc)[:160]}"}
    return {"ok": True, "user": user, "user_tokens": n, "reasoning_chars_cut": cut, "report": shortening_report(events0, events)}


def plan_items(failed: dict, items: list[dict], cfg, counter, server_ctx: int, traj_dir: pathlib.Path | None, manifest: dict,
               orig_rows: list[dict]) -> tuple[list[dict], dict]:
    """The retry plan for each failed item, before any request; also the changed user messages {item: text}."""
    rule = budget_rule(cfg, server_ctx)
    plan, texts = [], {}
    for i in sorted(failed):
        it = items[i]
        ut0 = P.user_tokens(it["input"], it["user"], counter, cfg)
        cap0 = P.output_budget(cfg, ut0)
        orig_cap = next((r.get("output_cap") for r in orig_rows if r["item"] == i), cap0)
        fits = (ut0 <= rule["fit_budget"]) if it["input"] == "history" else (ut0 + cfg.template_overhead_tokens <= rule["fit_budget"])
        rec = {"item": i, "rollout_id": it["rollout_id"], "condition": P.condition_of(it), "failure_reason": failed[i]["reason"],
               "original_attempts": failed[i]["attempts"], "original_user_sha256": it["user_sha256"], "original_user_tokens": ut0,
               "original_output_cap": orig_cap, "budget": rule, "evidence_shortened": False, "sendable": True}
        if not fits:
            rec["retry_kind"] = "refit"
            if traj_dir is None:
                rec.update(sendable=False, not_sendable_reason="no --trajectories directory to rebuild the view from")
            else:
                src = prompts_mod.find_trajectory(traj_dir, it["rollout_id"])
                recorded = (manifest.get("trajectory_files") or {}).get(it.get("task_id") or it["rollout_id"].split("__")[0], {})
                if src is None:
                    rec.update(sendable=False, not_sendable_reason="trajectory file not found")
                elif recorded.get("sha256") and P.sha256_file(src) != recorded["sha256"]:
                    raise SystemExit(f"trajectory {src.name} differs from the one the prompts were built from (sha256 mismatch)")
                else:
                    traj = tj.Trajectory.from_json(src.read_text())
                    base_user = m.build_prompt_fitted(traj, it["view"], counter=m.CharCounter(3.0), budget_tokens=10 ** 9)[0] \
                        if it["input"] == "history" else None
                    if base_user is not None and hashlib.sha256(base_user.encode()).hexdigest() != it["user_sha256"]:
                        raise SystemExit(f"rebuilding {it['rollout_id']} {it['view']} does not give the original prompt (hash differs)")
                    res = refit_view(traj, it, cfg, counter, rule["fit_budget"])
                    if not res["ok"]:
                        rec.update(sendable=False, not_sendable_reason=res["reason"])
                    else:
                        cap = P.output_budget(cfg, res["user_tokens"])
                        sha = hashlib.sha256(res["user"].encode()).hexdigest()
                        rec.update(new_user_sha256=sha, new_user_tokens=res["user_tokens"], new_output_cap=cap,
                                   reasoning_chars_cut=res["reasoning_chars_cut"], evidence_shortened=res["reasoning_chars_cut"] > 0,
                                   shortening=res["report"], reason_for_change="the original prompt (plus its output cap) was longer than the "
                                   "server context allows, or longer than the registry budget; refitted with the detector's own shortening",
                                   total_fits_context=res["user_tokens"] + cap <= server_ctx)
                        if res["user_tokens"] + cap > server_ctx:
                            rec.update(sendable=False, not_sendable_reason="prompt plus output cap exceeds the server context after fitting")
                        else:
                            texts[i] = res["user"]
        elif any(r.get("finish_reason") == "length" for r in original_attempts(orig_rows, i)):
            new_cap = raised_cap(cfg, server_ctx, ut0)
            rec.update(retry_kind="raise_cap", new_user_sha256=it["user_sha256"], new_user_tokens=ut0, new_output_cap=new_cap,
                       reason_for_change=f"the reply stopped at the output cap ({orig_cap}) without a score; cap raised to the largest the "
                                         f"server context allows for this prompt; this is a retry configuration, not the registered cap")
            if new_cap <= orig_cap:
                rec.update(sendable=False, not_sendable_reason="the server context allows no larger cap")
        else:
            rec.update(retry_kind="plain", new_user_sha256=it["user_sha256"], new_user_tokens=ut0, new_output_cap=cap0,
                       reason_for_change="same prompt and cap")
        plan.append(rec)
    return plan, texts


def write_prompts_gz(path: pathlib.Path, texts: dict) -> str:
    blob = json.dumps({"kind": "eq1_monitor_panel_retry_prompts", "items": {str(k): v for k, v in sorted(texts.items())}}, separators=(",", ":")).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(blob)
    if path.exists() and path.read_bytes() != buf.getvalue():
        raise SystemExit("refusing to resume: retry_prompts.json.gz differs from the one written before")
    if not path.exists():
        path.write_bytes(buf.getvalue())
    return P.sha256_file(path)


# ---------------------------------------------------------------------------------------------------------------------
# 4. accounting: successful-invocation time and retry-inclusive time

def account(orig_rows: list[dict], retry_rows: list[dict], items: list[dict], plan: list[dict]) -> list[dict]:
    pl = {p["item"]: p for p in plan}
    out = []
    for i, it in enumerate(items):
        orig = original_attempts(orig_rows, i)
        orig_ok = [r for r in orig if r.get("http_error") is None and not r.get("not_sent") and r.get("valid_score")]
        rr = [r for r in retry_rows if r["item"] == i and not r.get("not_sent")]
        retry_ok = [r for r in rr if r.get("http_error") is None and r.get("valid_score")]
        orig_secs = sum(r.get("seconds") or 0.0 for r in orig if not r.get("not_sent"))
        retry_secs = sum(r.get("seconds") or 0.0 for r in rr)
        e = {"item": i, "rollout_id": it["rollout_id"], "condition": P.condition_of(it), "retry_kind": pl.get(i, {}).get("retry_kind"),
             "evidence_shortened": bool(pl.get(i, {}).get("evidence_shortened")), "original_valid": bool(orig_ok),
             "original_attempt_seconds": orig_secs, "retry_attempt_seconds": retry_secs, "retry_attempts": len(rr)}
        win = orig_ok[0] if orig_ok else (retry_ok[0] if retry_ok else None)
        e["final_valid"] = win is not None
        e["success_source"] = None if win is None else ("original" if orig_ok else "retry")
        e["success_seconds"] = None if win is None else win["seconds"]
        e["prompt_tokens"] = None if win is None else win["prompt_tokens"]
        e["generated_tokens"] = None if win is None else win["generated_tokens"]
        e["retry_inclusive_seconds"] = orig_secs + retry_secs
        out.append(e)
    return out


def _stats(xs):
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "p90": P.quantile(xs, .9), "min": min(xs), "max": max(xs),
            "sum": sum(xs)} if xs else None


def summarize_retry(acc: list[dict]) -> dict:
    out = {}
    for cond in P.CONDITIONS:
        es = [e for e in acc if e["condition"] == cond]
        if not es:
            continue
        ok = [e for e in es if e["final_valid"]]
        retried = [e for e in es if e["retry_kind"]]
        out[cond] = {
            "invocations": len(es), "final_valid": len(ok), "final_valid_coverage": len(ok) / len(es),
            "still_failed_items": [e["item"] for e in es if not e["final_valid"]],
            "retried_items": [e["item"] for e in retried], "retry_kinds": {k: sum(1 for e in retried if e["retry_kind"] == k) for k in
                                                                         sorted({e["retry_kind"] for e in retried})},
            "evidence_shortened_items": [e["item"] for e in es if e["evidence_shortened"]],
            "evidence_shortened_count": sum(1 for e in es if e["evidence_shortened"]),
            "successful_invocation_seconds": _stats([e["success_seconds"] for e in ok]),
            "retry_inclusive_seconds_of_valid_items": _stats([e["retry_inclusive_seconds"] for e in ok]),
            "retry_inclusive_seconds_all_items_total": sum(e["retry_inclusive_seconds"] for e in es),
            "original_failed_attempt_seconds_total": sum(e["original_attempt_seconds"] for e in retried),
            "retry_attempt_seconds_total": sum(e["retry_attempt_seconds"] for e in es),
            "input_tokens_mean": statistics.fmean(e["prompt_tokens"] for e in ok) if ok else None,
            "output_tokens_mean": statistics.fmean(e["generated_tokens"] for e in ok) if ok else None}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# run

def run_retry(a) -> int:
    panel_dir, out_dir = pathlib.Path(a.panel_dir), pathlib.Path(a.out_dir)
    if panel_dir.resolve() == out_dir.resolve():
        raise SystemExit("--out-dir must differ from --panel-dir: the original run is never written to")
    orig_cfg_doc = json.loads((panel_dir / "run_config.json").read_text())
    ocfg = orig_cfg_doc["config"]
    name = ocfg["monitor"]
    if a.monitor and a.monitor != name:
        raise SystemExit(f"--monitor {a.monitor} but the panel directory is {name}")
    spec = m.get_spec(name)
    cfg = spec.config()
    if P.sha_obj({"spec": spec.as_dict(), "config": cfg.as_dict()}) != ocfg["registry_config_sha256"]:
        raise SystemExit("the registry entry differs from the one the original panel run used")
    prompts = a.prompts or panel_dir.parent / "inputs" / "eq1_monitor_panel_prompts.json.gz"
    manifest = a.manifest or panel_dir.parent / "inputs" / "eq1_monitor_panel_prompts.manifest.json"
    items, man, prompts_sha, manifest_sha = P.load_prompts(pathlib.Path(prompts), pathlib.Path(manifest))
    if prompts_sha != ocfg["prompts_file_sha256"] or manifest_sha != ocfg["manifest_sha256"]:
        raise SystemExit("the prompt file or manifest differs from the one the original panel run used")
    items = items[:len(ocfg["schedule"])]
    orig_rows = P.read_rows(panel_dir / "rows.jsonl")
    orig_replies = panel_dir / "replies.jsonl"
    src = a.model_dir or spec.hf_id
    counter = m.load_counter(src, revision=spec.revision, local_only=True)
    method = getattr(counter, "method", "chars")
    if method != "tokenizer" and not a.allow_char_counter:
        raise SystemExit("the monitor's tokenizer did not load; refusing to fall back to a character estimate")
    client = m.ChatMonitorClient(a.url, model=spec.weights, server="vllm", extra_body=spec.request_extra,
                                 reasoning_format=spec.reasoning_format, assistant_prefill=spec.assistant_prefill)
    ident = client.identity()
    live_ctx = ident.get("max_model_len")
    server_ctx = a.server_context or live_ctx or (ocfg["server"] or {}).get("max_model_len")
    if not server_ctx:
        raise SystemExit("cannot determine the server context; give --server-context")
    if live_ctx and int(live_ctx) != int(server_ctx):
        raise SystemExit(f"--server-context {server_ctx} but the server reports max_model_len {live_ctx}")
    server_ctx = int(server_ctx)
    failed = failed_items(orig_rows, len(items))
    plan, texts = plan_items(failed, items, cfg, counter, server_ctx, pathlib.Path(a.trajectories) if a.trajectories else None, man, orig_rows)
    if a.plan_only:     # prints the plan and writes nothing: a check of token counts and budgets before any request
        print(json.dumps({"monitor": name, "server_context": server_ctx, "token_counting": method, "plan": [
            {k: v for k, v in p.items() if k not in ("original_attempts", "shortening")} | {"shortening": {k: v for k, v in (p.get("shortening") or {}).items() if k != "changed_blocks"}}
            for p in plan]}, indent=1))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_gz_sha = write_prompts_gz(out_dir / "retry_prompts.json.gz", texts)
    sinfo = P.read_json(a.server_info)
    flags = (sinfo or {}).get("flags")
    config = {
        "schema": 1, "kind": "retry", "monitor": name, "registry_config_sha256": ocfg["registry_config_sha256"],
        "registry_spec": spec.as_dict(), "monitor_config": cfg.as_dict(),
        "original": {"panel_dir": str(panel_dir), "run_config_sha256": orig_cfg_doc["config_sha256"],
                     "rows_sha256": P.sha256_file(panel_dir / "rows.jsonl"),
                     "replies_sha256": P.sha256_file(orig_replies) if orig_replies.exists() else None,
                     "prompts_file_sha256": prompts_sha, "manifest_sha256": manifest_sha, "original_server": ocfg["server"]},
        "scope": "cost panel only; frozen EQ2 accuracy results are not touched; original files are never written",
        "failed_item_rule": "an item with no returned reply holding a valid score in the original rows.jsonl",
        "failed_items": {str(k): v["reason"] for k, v in sorted(failed.items())},
        "plan": plan, "retry_prompts_sha256": prompts_gz_sha, "max_attempts": a.max_attempts, "concurrency": 1,
        "token_counting": {"method": method, "source": str(src)},
        "timed_interval": ocfg["timed_interval"], "server": {"engine": ident.get("engine"), "engine_version": ident.get("engine_version"),
                                                              "model_alias": ident.get("model_alias"), "max_model_len": server_ctx,
                                                              "launch_flags": flags},
        "same_launch_flags_as_original": (flags == (ocfg["server"] or {}).get("launch_flags")) if flags else None,
        "hardware": P.gpu_static(), "model": P.model_identity(a.model_dir, spec), "code_sha256": code_hashes(),
        "score_values_stored": False, "labels_read": False,
    }
    P.check_or_write_config(out_dir, config)
    rows_path, replies_path, sessions_path = out_dir / "rows.jsonl", out_dir / "replies.jsonl", out_dir / "sessions.jsonl"
    rows = P.read_rows(rows_path)
    st = P.item_state(rows, a.max_attempts)
    todo = [p for p in plan if not st.get(p["item"], {}).get("done")]
    if todo:
        session = {"session": len(P.read_rows(sessions_path)) + 1, "start_utc": P.now_utc(), "git": P.git_state(),
                   "server_info_session": (sinfo or {}).get("session"), "gpu_before_warmup": P.gpu_dynamic(a.server_pid),
                   "server_log_at_start": P.parse_server_log(a.server_log), "resumed_rows": len(rows), "retry": True}
        wu_user = m.FINAL_STATE_INSTRUCTION.format(reasoning_note="", record=P.WARMUP_RECORD)
        wu = P.post_once(client, a.url, wu_user, P.output_budget(cfg, P.user_tokens("final_state", wu_user, counter, cfg)), cfg, a.timeout)
        wu.pop("_text", None)
        session["warmup"] = {**wu, "untimed": True, "note": "outside every row; includes any first-request compilation"}
        session["gpu_after_warmup"] = P.gpu_dynamic(a.server_pid)
        P.append_jsonl(sessions_path, session)
        t_session = time.perf_counter()
        consecutive = 0
        for n_done, p in enumerate(todo):
            i = p["item"]
            it = items[i]
            user = texts.get(i, it["user"])
            base = {"item": i, "rollout_id": it["rollout_id"], "task_id": it.get("task_id"), "condition": P.condition_of(it), "view": it["view"],
                    "input": it["input"], "retry": True, "retry_kind": p["retry_kind"], "failure_reason_original": p["failure_reason"],
                    "original_user_sha256": p["original_user_sha256"], "user_sha256": p.get("new_user_sha256", p["original_user_sha256"]),
                    "original_user_tokens": p["original_user_tokens"], "user_tokens_counted": p.get("new_user_tokens"),
                    "original_output_cap": p["original_output_cap"], "output_cap": p.get("new_output_cap"),
                    "evidence_shortened": p["evidence_shortened"], "session": session["session"], "concurrency": 1}
            if not p["sendable"]:
                P.append_jsonl(rows_path, {**base, "attempt": 0, "not_sent": True, "http_error": None, "valid_score": False,
                                           "note": p.get("not_sendable_reason"), "start_utc": P.now_utc()})
                continue
            prior, tries = st.get(i, {}).get("attempts", 0), 0
            while tries < a.max_attempts:
                tries += 1
                start_utc, offset = P.now_utc(), time.perf_counter() - t_session
                res = P.post_once(client, a.url, user, p["new_output_cap"], cfg, a.timeout)
                text = res.pop("_text")
                P.append_jsonl(rows_path, {**base, "attempt": prior + tries, "start_utc": start_utc, "start_offset_s": round(offset, 3),
                                           **{k: (round(v, 6) if k == "seconds" else v) for k, v in res.items()}})
                if text is not None:
                    P.append_jsonl(replies_path, {"item": i, "attempt": prior + tries, "text": text})
                consecutive = consecutive + 1 if res["http_error"] else 0
                if not res["http_error"] or consecutive >= a.max_consecutive_errors:
                    break
            done = P.read_rows(rows_path)
            print(f"[retry {name}] {n_done + 1}/{len(todo)}: replies {len({r['item'] for r in done if r.get('http_error') is None and not r.get('not_sent')})}, "
                  f"valid {len({r['item'] for r in done if r.get('http_error') is None and r.get('valid_score')})}, "
                  f"http error attempts {sum(1 for r in done if r.get('http_error'))}", flush=True)
            if consecutive >= a.max_consecutive_errors:
                print(f"[retry {name}] stopping after {consecutive} consecutive errors; rerun to resume", flush=True)
                return 3
    write_retry_summary(out_dir, config, items, orig_rows, P.read_rows(rows_path), plan, a.server_log)
    return 0


def write_retry_summary(out_dir, config, items, orig_rows, retry_rows, plan, server_log=None) -> dict:
    acc = account(orig_rows, retry_rows, items, plan)
    doc = {"monitor": config["monitor"], "config_sha256": P.sha_obj(config), "original": config["original"], "failed_items": config["failed_items"],
           "per_condition": summarize_retry(acc), "items": acc,
           "plan_brief": [{k: p.get(k) for k in ("item", "condition", "retry_kind", "failure_reason", "original_user_tokens", "new_user_tokens",
                                                 "original_output_cap", "new_output_cap", "original_user_sha256", "new_user_sha256",
                                                 "evidence_shortened", "sendable", "not_sendable_reason")} for p in plan],
           "sessions": len(P.read_rows(out_dir / "sessions.jsonl")),
           "server_log_end_of_run": P.parse_server_log(server_log) if server_log else None,
           "definitions": {"successful_invocation_seconds": "original reply time for items valid at first; retry request time for retried items; "
                                                            "over items that end with a valid score",
                           "retry_inclusive_seconds": "all original attempt durations of the item plus all retry attempts; mean/median over valid items, "
                                                      "and the total over all items"}}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({c: {"final_valid": v["final_valid"], "of": v["invocations"], "retried": len(v["retried_items"]),
                          "evidence_shortened": v["evidence_shortened_count"]} for c, v in doc["per_condition"].items()}))
    return doc


# ---------------------------------------------------------------------------------------------------------------------
# combined export and table

COMPLETED_HEAD = ["Monitor", "View", "Config", "Invocations", "Seconds/invocation (mean, median)", "Input tok.", "Output tok."]


def completed_note(docs: dict) -> str:
    notes = ["Standalone judge measurement on recorded training-partition views, one request at a time, natural stops; it is not an agent-plus-judge "
             "run, and any sum with agent time is an accounting estimate. Seconds are successful-invocation time: the original reply time, or, for "
             "an input that first failed, the time of its successful retry alone (retry-inclusive cost is in the audit file)."]
    short = {n: d["per_condition"] for n, d in docs.items()}
    n_short = sum(v["evidence_shortened_count"] for pc in short.values() for v in pc.values())
    if n_short:
        notes.append("Inputs that were longer than the server context were shortened with the detector's fitting procedure (private reasoning "
                     "middle-truncated); their input tokens are the shortened counts. Affected items are listed in the audit file.")
    inc = [f"{n} {P.CONDITION_TEXT[c]}: {v['final_valid']} of {v['invocations']}" for n, pc in short.items() for c, v in pc.items()
           if v["final_valid"] < v["invocations"]]
    if inc:
        notes.append("Incomplete after retry: " + "; ".join(inc) + ".")
    return " ".join(notes)


def completed_lines(docs: dict) -> list[dict]:
    lines = []
    for n, d in docs.items():
        spec = m.get_spec(n)
        for c in P.CONDITIONS:
            v = d["per_condition"].get(c)
            if not v:
                continue
            s = v["successful_invocation_seconds"]
            lines.append({"monitor": n, "view": P.CONDITION_TEXT[c], "config": P.cap_text(spec), "invocations": v["final_valid"],
                          "seconds": f"{P.f2(s['mean'] if s else None)}, {P.f2(s['median'] if s else None)}",
                          "input_tokens": P.f2(v["input_tokens_mean"], 0), "output_tokens": P.f2(v["output_tokens_mean"], 0)})
    return lines


def completed_md(lines, unmeasured, note) -> str:
    out = ["| " + " | ".join(COMPLETED_HEAD) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * 4) + "|"]
    for r in lines:
        out.append(f"| {r['monitor']} | {r['view']} | {r['config']} | {r['invocations']} | {r['seconds']} | {r['input_tokens']} | {r['output_tokens']} |")
    for u in unmeasured:
        out.append(f"| {u['monitor']} | all | {u['config']} | not measured on this hardware | | | |")
    txt = "\n".join(out) + "\n\n" + note + "\n"
    for u in unmeasured:
        txt += f"\nNot measured: {u['monitor']}: {u['reason']}\n"
    return txt


def completed_tex(lines, unmeasured, note) -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    out = [r"\begin{table}[t]", r"\centering\footnotesize", r"\begin{tabular}{lllrrrr}", r"\toprule",
           " & ".join(["Monitor", "View", "Config", "Invoc.", r"Seconds/invocation (mean, median)", "Input tok.", "Output tok."]) + r" \\", r"\midrule"]
    for r in lines:
        out.append(" & ".join(esc(x) for x in (r["monitor"], r["view"], r["config"], r["invocations"], r["seconds"], r["input_tokens"], r["output_tokens"])) + r" \\")
    for u in unmeasured:
        out.append(" & ".join([esc(u["monitor"]), "all", esc(u["config"]), r"\multicolumn{4}{l}{not measured on this hardware}"]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{Monitor invocation cost. " + esc(note) + "}", r"\end{table}"]
    return "\n".join(out) + "\n"


def combine_retry(root: pathlib.Path, panel_root: pathlib.Path | None = None) -> int:
    """root = the retry output root (holds <monitor>/summary.json). Not-measured records are read from panel_root (default: root's parent)."""
    panel_root = panel_root or root.parent
    docs, unmeasured = {}, []
    names = [d.name for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    names.sort(key=lambda n: (P.MONITOR_ORDER.index(n) if n in P.MONITOR_ORDER else len(P.MONITOR_ORDER), n))
    for n in names:
        docs[n] = json.loads((root / n / "summary.json").read_text())
    for d in sorted(panel_root.iterdir()) if panel_root.exists() else []:
        if d.is_dir() and (d / "not_measured.json").exists() and d.name not in docs:
            unmeasured.append({"monitor": d.name, "config": P.cap_text(m.get_spec(d.name)), "reason": json.loads((d / "not_measured.json").read_text())["reason"]})
    lines, note = completed_lines(docs), completed_note(docs)
    (root / "panel_table_completed.md").write_text(completed_md(lines, unmeasured, note))
    (root / "panel_table_completed.tex").write_text(completed_tex(lines, unmeasured, note))
    (root / "summary.json").write_text(json.dumps({
        "kind": "eq1_monitor_panel_retry_export", "scope": "cost panel only; original rows, replies and configs unchanged",
        "per_monitor": {n: {"per_condition": d["per_condition"], "failed_items": d["failed_items"], "config_sha256": d["config_sha256"],
                            "original": d["original"], "plan_brief": d["plan_brief"]} for n, d in docs.items()},
        "not_measured": unmeasured,
        "time_quantities": {"successful_invocation_seconds": "table values: original time for first-pass valid items, retry request time for retried items",
                            "retry_inclusive_seconds": "original failed-attempt durations plus retry attempts, per item (mean/median over valid items; totals over all)"},
        "table_note": note}, indent=1) + "\n")
    print((root / "panel_table_completed.md").read_text())
    return 0

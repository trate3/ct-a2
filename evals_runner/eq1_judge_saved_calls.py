"""EQ1 shared-call cost table: the monitor's invocation time on the SAVED generated calls, one monitor, one request at a time.

    python -m evals_runner.eq1_judge_saved_calls --monitor gemma4-e4b --url http://127.0.0.1:8100 --out-dir OUT/gemma4-e4b \\
        --stage2-list stage2_list.json --histories histories.json --outputs generate_calls_stage2/outputs.jsonl \\
        --trajectories DIR --model-dir SNAPSHOT [--main-complete-call main_complete_call.json] [--server-log L --server-info I]
    python -m evals_runner.eq1_judge_saved_calls --combine OUT [OUT ...] --out TABLE_DIR [--agent-hardware NAME]

What is measured. For each of the 24 prompt representatives of the focused live-generation sample the monitor reads the FULL available trace through the new
response (the recorded trajectory up to the call, plus the saved A0 output of the main stage as the proposed action; built by
`eq1_judge_saved_calls_prompts.py`, whose docstring states the uniform policy for outputs without a parsed action) and judges it under its registered settings
(registry request body, reasoning switch, sampling, seed, output cap, natural stops; concurrency 1). Prefill and generation are inside the timer; the
request is an HTTP call to the local vLLM server; warm-up (one untimed request) and server load are recorded apart, never in a row.

Fitting. A prompt longer than the budget min(registry prompt budget, server context - output allowance - 512) is fitted with the detector's normal policy
(`monitors._fit`: only private reasoning is shortened, so a black-box view that is too long cannot be fitted and is recorded as not sent). Original and new
token counts, what was shortened and `evidence_shortened` are recorded. Retry policy, all attempts kept: an HTTP error is repeated; a reply that stopped at the
output cap without a score is repeated once with the cap raised to what the server context allows (`retry_kind`, `original_output_cap`, `original_failure`);
no score is ever forced. Two time quantities per prompt: successful-invocation time (the attempt that produced a valid score) and total effort (all attempts).

Positions. Each of the 30 positions (weight 1/30, from stage2_list.json) takes its representative's judged time. Estimates: weighted mean with a
task-clustered bootstrap (2,000 draws, seed 0; a task's positions kept together), so the reuse of a representative is counted together, not as 30 independent
prompts. Composed With tap = the weighted No-tap (A0) mean of main_complete_call.json plus the weighted monitor mean, bootstrapped jointly over the same positions.
Not included: model/tokenizer loading. The sum is of two measurements made one after the other on the same GPU, not one wall-clock run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time

from evals.reward_hacking import monitors as m
from evals_runner import eq1_judge_saved_calls_prompts as pr
from evals_runner import eq1_monitor_panel as P
from evals_runner import eq1_monitor_panel_retry as R

DEFAULT_TIMEOUT = 7200.0
AGENT_HARDWARE_DEFAULT = "agent GPU"
NOTE_UNIT = ("One saved A0 output per distinct prompt (24 representatives of the 30 positions, each position weighted 1/30); the monitor reads the whole recorded "
             "trace through that response (transcript view; private reasoning only in a view that explicitly provides it) and its time, prefill and generation, is "
             "measured one request at a time under its registered settings.")


# ---------------------------------------------------------------------------------------------------------------------
# retry policy and per-item state

def next_retry(res: dict, attempt: int, max_attempts: int, cfg, ctx: int, ut: int, cap: int):
    """(retry_kind, new_cap) when the attempt failed in a retryable way, else None. A reply without a score is never forced or re-rolled."""
    if attempt >= max_attempts:
        return None
    if res.get("http_error"):
        return "http_error", cap
    if not res.get("valid_score") and res.get("finish_reason") == "length":
        new = R.raised_cap(cfg, ctx, ut)
        if new > cap:
            return "raise_cap", new
    return None


def item_rows(rows: list[dict], i: int) -> list[dict]:
    return [r for r in rows if r["item"] == i]


def item_pending(rows_i: list[dict], max_attempts: int, cfg, ctx: int, plan_i: dict):
    """None when the item is finished, else (retry_kind, cap, attempts_so_far) for the next attempt (kind None = first attempt)."""
    if not rows_i:
        return (None, plan_i["output_cap"], 0)
    last = rows_i[-1]
    if last.get("not_sent") or (last.get("http_error") is None and last.get("valid_score")):
        return None
    sent = [r for r in rows_i if not r.get("not_sent")]
    nr = next_retry(last, len(sent), max_attempts, cfg, ctx, plan_i["user_tokens"], last.get("output_cap") or plan_i["output_cap"])
    return None if nr is None else (nr[0], nr[1], len(sent))


# ---------------------------------------------------------------------------------------------------------------------
# summaries

def weighted(items: list[dict], key: str) -> dict:
    from evals.tap_live.summarize_decode import task_bootstrap
    parts: dict[str, list[float]] = {}
    for it in items:
        p = parts.setdefault(it["task"], [0.0, 0.0])
        p[0] += it["weight"] * it[key]
        p[1] += it["weight"]
    b = task_bootstrap({k: tuple(v) for k, v in parts.items()})
    return {"estimate": b["estimate"], "ci95": b["ci95"], "n_tasks": b["n_tasks"]}


def per_prompt(rows: list[dict], plan: list[dict], items: list[dict]) -> list[dict]:
    out = []
    for i, (it, p) in enumerate(zip(items, plan)):
        rs = item_rows(rows, i)
        sent = [r for r in rs if not r.get("not_sent")]
        ok = [r for r in sent if r.get("http_error") is None and r.get("valid_score")]
        win = ok[0] if ok else None
        out.append({"item": i, "run_key": it["run_key"], "history_id": it["history_id"], "task": it["task"], "view": it["view"],
                    "representative_index": it["representative_index"], "final_valid": win is not None,
                    "success_seconds": win["seconds"] if win else None, "effort_seconds": sum(r.get("seconds") or 0.0 for r in sent),
                    "attempts": len(sent), "not_sent": any(r.get("not_sent") for r in rs), "prompt_tokens": win["prompt_tokens"] if win else None,
                    "generated_tokens": win["generated_tokens"] if win else None, "finish_reason": (sent[-1].get("finish_reason") if sent else None),
                    "retry_kinds": [r.get("retry_kind") for r in sent if r.get("retry_kind")],
                    "evidence_shortened": p["evidence_shortened"], "original_user_tokens": p["original_user_tokens"], "user_tokens": p["user_tokens"],
                    "flags": it["flags"], "user_sha256": p["user_sha256"], "original_user_sha256": it["user_sha256"]})
    return out


def read_main(path) -> dict | None:
    if not path:
        return None
    p = pathlib.Path(path)
    return {"file": str(p), "sha256": P.sha256_file(p), "doc": json.loads(p.read_text())}


def summarize(config: dict, items: list[dict], plan: list[dict], rows: list[dict], sel: dict, main: dict | None) -> dict:
    pp = per_prompt(rows, plan, items)
    out = {"monitor": config["monitor"], "config_sha256": P.sha_obj(config), "views": {}}
    pos_all = sel["positions"]
    a0_by_pos = {}
    if main:
        a0_by_pos = {q["history_id"]: q for q in main["doc"].get("positions", [])}
    for view in sorted({p["view"] for p in pp}, key=lambda v: list(pr.VIEWS).index(v)):
        vp = [p for p in pp if p["view"] == view]
        by_key = {p["run_key"]: p for p in vp}
        ok = [p for p in vp if p["final_valid"]]
        pos_items, missing = [], []
        for q in pos_all:
            p = by_key.get(q["run_key"])
            if not p or not p["final_valid"]:
                missing.append(q["history_id"])
                continue
            pos_items.append({"history_id": q["history_id"], "task": q["task"], "weight": q["weight"], "mon_s": p["success_seconds"], "effort_s": p["effort_seconds"],
                              "in_tok": p["prompt_tokens"], "out_tok": p["generated_tokens"]})
        v = {"n_prompts": len(vp), "n_valid": len(ok), "n_positions": len(pos_all), "n_valid_positions": len(pos_items), "missing_positions": missing,
             "complete": not missing, "per_prompt": vp,
             "evidence_shortened_prompts": [p["history_id"] for p in vp if p["evidence_shortened"]], "evidence_shortened_count": sum(p["evidence_shortened"] for p in vp),
             "not_sent_prompts": [p["history_id"] for p in vp if p["not_sent"]], "retried_prompts": [p["history_id"] for p in vp if p["retry_kinds"]],
             "finish_reasons": {k: sum(1 for p in vp if p["finish_reason"] == k) for k in sorted({p["finish_reason"] for p in vp if p["finish_reason"]})},
             "responses_without_parsed_action": sum(1 for p in vp if p["flags"]["no_parsed_action"]),
             "responses_visible_empty": sum(1 for p in vp if p["flags"]["visible_empty"])}
        if pos_items:
            v["tap_s_per_call"] = weighted(pos_items, "mon_s")
            v["tap_effort_s_per_call"] = weighted(pos_items, "effort_s")
            v["input_tokens_weighted_mean"] = weighted(pos_items, "in_tok")["estimate"]
            v["output_tokens_weighted_mean"] = weighted(pos_items, "out_tok")["estimate"]
            secs = [p["success_seconds"] for p in ok]
            v["per_prompt_seconds"] = {"mean": statistics.fmean(secs), "median": statistics.median(secs), "p90": P.quantile(secs, .9), "min": min(secs), "max": max(secs)}
            if main:
                comp = []
                for it in pos_items:
                    q = a0_by_pos.get(it["history_id"])
                    if q and q.get("valid") and q.get("a0_s") is not None:
                        comp.append({**it, "a0_s": q["a0_s"], "with_s": q["a0_s"] + it["mon_s"]})
                v["composed"] = {"n_positions": len(comp), "complete": len(comp) == len(pos_all), "main_complete_call_sha256": main["sha256"],
                                 "no_tap_s_per_call": weighted(comp, "a0_s") if comp else None, "with_tap_s_per_call": weighted(comp, "with_s") if comp else None,
                                 "tap_s_per_call_same_positions": weighted(comp, "mon_s") if comp else None,
                                 "main_complete_call_reported_no_tap": (main["doc"].get("primary") or {}).get("no_tap_mean_s_per_call"),
                                 "note": "With tap = weighted A0 mean (main_complete_call.json) + weighted monitor mean over the same positions, bootstrapped jointly "
                                         "(task-clustered, 2,000 draws, seed 0). Loading and any transfer between devices are not included."}
            else:
                v["composed"] = None
        out["views"][view] = v
    return out


def hw_label(gpu_field: str | None) -> str:
    name = (gpu_field or "").split(",")[0].strip()
    return name or "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# table

def fmt_ci(e: dict | None, nd=2) -> str:
    return "-" if not e else f"{e['estimate']:.{nd}f} [{e['ci95'][0]:.{nd}f}, {e['ci95'][1]:.{nd}f}]"


def table_notes(rows_summ: list[dict]) -> str:
    notes = [NOTE_UNIT + " With tap is the weighted No-tap generation time of the saved calls plus the monitor's time on them, added as two separate measurements; "
             "Tap is the monitor time. Model and tokenizer loading and any transfer of the saved output are not included. Where the agent and the monitor ran on "
             "different devices the two were run one after the other and the sum is not a same-device or end-to-end wall-clock measurement."]
    if any(r["summary_view"].get("evidence_shortened_count") for r in rows_summ):
        notes.append("Some inputs were longer than the monitor's supported context and were shortened with the detector's fitting procedure (private reasoning only); "
                     "the audit file lists them.")
    inc = [f"{r['monitor']} ({r['hardware']}): {r['summary_view']['n_valid_positions']} of {r['summary_view']['n_positions']} positions"
           for r in rows_summ if not r["summary_view"]["complete"]]
    if inc:
        notes.append("Incomplete: " + "; ".join(inc) + ".")
    return " ".join(notes)


def arrangement(hw: str, agent_hw: str) -> str:
    a = hw_label(agent_hw)
    return f"agent and monitor on {a}, run one after the other" if hw == a else f"agent on {a}, monitor on {hw}; two devices, run one after the other"


def combine(roots: list[pathlib.Path], out: pathlib.Path, agent_hw: str, view: str = "history_black_box") -> int:
    rows = []
    for root in roots:
        names = [d.name for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
        names.sort(key=lambda n: (P.MONITOR_ORDER.index(n) if n in P.MONITOR_ORDER else len(P.MONITOR_ORDER), n))
        for n in names:
            d = root / n
            s = json.loads((d / "summary.json").read_text())
            v = s["views"].get(view)
            if not v:
                continue
            hw = s.get("hardware_label") or "unknown"
            rows.append({"monitor": n, "hardware": hw, "arrangement": arrangement(hw, agent_hw), "config": P.cap_text(m.get_spec(n)),
                         "tap": v.get("tap_s_per_call"), "with_tap": ((v.get("composed") or {}).get("with_tap_s_per_call")),
                         "no_tap": ((v.get("composed") or {}).get("no_tap_s_per_call")), "summary_view": v, "root": str(root)})
    note = table_notes(rows)
    head = ["Monitor", "Monitor hardware (arrangement)", "Config", "With tap (s per call)", "Tap = monitor time (s per call)"]
    md = ["| " + " | ".join(head) + " |", "|---|---|---|---:|---:|"]
    for r in rows:
        md.append(f"| {r['monitor']} | {r['hardware']} ({r['arrangement']}) | {r['config']} | {fmt_ci(r['with_tap'])} | {fmt_ci(r['tap'])} |")
    txt = "\n".join(md) + "\n\n" + note + "\n"
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    tex = [r"\begin{table}[t]", r"\centering\footnotesize", r"\begin{tabular}{llcc}", r"\toprule",
           r"Monitor & Monitor hardware & With tap (s/call) & Tap (s/call) \\", r"\midrule"]
    for r in rows:
        tex.append(f"{esc(r['monitor'])} & {esc(r['hardware'])} & {esc(fmt_ci(r['with_tap']))} & {esc(fmt_ci(r['tap']))}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\caption{Monitor cost on the saved generated calls. " + esc(note) + "}", r"\end{table}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_table.md").write_text(txt)
    (out / "judge_table.tex").write_text("\n".join(tex) + "\n")
    (out / "judge_summary.json").write_text(json.dumps({"kind": "eq1_judge_saved_calls_table", "view": view, "agent_hardware": agent_hw, "note": note,
                                                         "rows": [{k: v for k, v in r.items() if k != "summary_view"} | {"counts": {
                                                             "n_valid_positions": r["summary_view"]["n_valid_positions"], "n_positions": r["summary_view"]["n_positions"],
                                                             "evidence_shortened": r["summary_view"]["evidence_shortened_count"],
                                                             "responses_without_parsed_action": r["summary_view"]["responses_without_parsed_action"]}} for r in rows]}, indent=1) + "\n")
    print(txt)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--monitor")
    ap.add_argument("--url")
    ap.add_argument("--out-dir", type=pathlib.Path)
    ap.add_argument("--stage2-list", type=pathlib.Path)
    ap.add_argument("--histories", type=pathlib.Path)
    ap.add_argument("--outputs", type=pathlib.Path)
    ap.add_argument("--trajectories", type=pathlib.Path)
    ap.add_argument("--manifest", type=pathlib.Path, help="optional inputs manifest from eq1_judge_saved_calls_prompts: every rebuilt prompt hash must match it")
    ap.add_argument("--main-complete-call", type=pathlib.Path, help="main_complete_call.json (not part of the run configuration; read for the composed row)")
    ap.add_argument("--views", default="history_black_box")
    ap.add_argument("--instruction", choices=tuple(pr.INSTRUCTIONS), default="early")
    ap.add_argument("--model-dir")
    ap.add_argument("--server-log")
    ap.add_argument("--server-info")
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--server-context", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N items of the schedule (smoke test; part of the configuration)")
    ap.add_argument("--max-consecutive-errors", type=int, default=5)
    ap.add_argument("--allow-char-counter", action="store_true", help="tests only")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="print the plan (token counts, fitting) and write nothing")
    ap.add_argument("--combine", type=pathlib.Path, nargs="+")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--agent-hardware", default=AGENT_HARDWARE_DEFAULT)
    a = ap.parse_args(argv)
    if a.combine:
        return combine(a.combine, a.out or a.combine[0], a.agent_hardware)
    for need in ("monitor", "out_dir", "stage2_list", "histories", "outputs", "trajectories"):
        if getattr(a, need) is None:
            raise SystemExit(f"--{need.replace('_', '-')} is required")
    if a.url is None and not a.plan_only and not a.summarize_only:
        raise SystemExit("--url is required")
    if a.summarize_only:     # recompute summary.json (for example with a main_complete_call.json that appeared later) from the stored configuration and rows
        stored = json.loads((a.out_dir / "run_config.json").read_text())["config"]
        world = pr.load_world(a.stage2_list, a.histories, a.outputs, a.trajectories)
        items = pr.build_inputs(world, stored["views"], stored["instruction"])[:stored.get("limit") or None]
        if [it["user_sha256"] for it in items] != [x["user_sha256"] for x in stored["schedule"]]:
            raise SystemExit("the inputs rebuilt now differ from the stored run configuration (hash differs)")
        return finish(a.out_dir, stored, items, stored["plan"], world["sel"], read_main(a.main_complete_call), a.server_log)
    views = [v for v in a.views.split(",") if v]
    if any(v not in pr.VIEWS for v in views):
        raise SystemExit(f"--views from {sorted(pr.VIEWS)}")
    spec = m.get_spec(a.monitor)
    cfg = spec.config()
    world = pr.load_world(a.stage2_list, a.histories, a.outputs, a.trajectories)
    items = pr.build_inputs(world, views, a.instruction)
    if a.limit:
        items = items[:a.limit]
    manifest_sha = None
    if a.manifest:
        man = json.loads(a.manifest.read_text())
        manifest_sha = P.sha256_file(a.manifest)
        have = {(x["run_key"], x["view"]): x["user_sha256"] for x in man["items"]}
        for it in items:
            if have.get((it["run_key"], it["view"])) != it["user_sha256"]:
                raise SystemExit(f"rebuilt input {it['history_id']} {it['view']} does not match the manifest (hash differs)")
    sel = world["sel"]
    out_dir = a.out_dir
    rows_path, replies_path, sessions_path = out_dir / "rows.jsonl", out_dir / "replies.jsonl", out_dir / "sessions.jsonl"

    src = a.model_dir or spec.hf_id
    counter = m.load_counter(src, revision=spec.revision, local_only=True)
    method = getattr(counter, "method", "chars")
    if method != "tokenizer" and not a.allow_char_counter:
        raise SystemExit("the monitor's tokenizer did not load; refusing to fall back to a character estimate")
    client = m.ChatMonitorClient(a.url or "http://127.0.0.1:1", model=spec.weights, server="vllm", extra_body=spec.request_extra,
                                 reasoning_format=spec.reasoning_format, assistant_prefill=spec.assistant_prefill)
    ident = client.identity() if a.url and not a.plan_only else {}
    live_ctx = ident.get("max_model_len")
    server_ctx = a.server_context or live_ctx or (131072 if a.plan_only else None)
    if not server_ctx:
        raise SystemExit("cannot determine the server context; give --server-context")
    if live_ctx and int(live_ctx) != int(server_ctx):
        raise SystemExit(f"--server-context {server_ctx} but the server reports max_model_len {live_ctx}")
    server_ctx = int(server_ctx)
    rule = R.budget_rule(cfg, server_ctx)
    instr = pr.INSTRUCTIONS[a.instruction]
    plan, texts = [], {}
    for i, it in enumerate(items):
        ut0 = P.user_tokens("history", it["user"], counter, cfg)
        rec = {"item": i, "run_key": it["run_key"], "history_id": it["history_id"], "view": it["view"], "original_user_sha256": it["user_sha256"],
               "original_user_tokens": ut0, "user_sha256": it["user_sha256"], "user_tokens": ut0, "sendable": True, "evidence_shortened": False,
               "fitted": False, "reason_for_change": None, "budget": rule}
        if ut0 > rule["fit_budget"]:
            res = pr.fit_view(it["_trace"], it["boundary_seq"], pr.VIEWS[it["view"]], instr, counter, rule["fit_budget"], cfg.template_overhead_tokens)
            rec.update(fitted=True, original_failure=f"prompt of {ut0} tokens over the budget {rule['fit_budget']}",
                       reason_for_change="prompt longer than min(registry prompt budget, server context - output allowance - margin): fitted with the detector's shortening")
            if not res["ok"]:
                rec.update(sendable=False, not_sendable_reason=res["reason"])
            else:
                rec.update(user_sha256=hashlib.sha256(res["user"].encode()).hexdigest(), user_tokens=res["user_tokens"], reasoning_chars_cut=res["reasoning_chars_cut"],
                           evidence_shortened=res["reasoning_chars_cut"] > 0, shortening=res["report"])
                texts[i] = res["user"]
        rec["output_cap"] = P.output_budget(cfg, rec["user_tokens"])
        if rec["sendable"] and rec["user_tokens"] + rec["output_cap"] > server_ctx:
            rec.update(sendable=False, not_sendable_reason="prompt plus output cap exceeds the server context")
        plan.append(rec)
    if a.plan_only:
        print(json.dumps({"monitor": a.monitor, "server_context": server_ctx, "token_counting": method, "budget": rule, "plan": [
            {k: v for k, v in p.items() if k not in ("budget", "shortening")} | {"flags": items[p["item"]]["flags"]} for p in plan]}, indent=1))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_items = [{k: v for k, v in it.items() if k not in ("_trace", "user")} for it in items]
    sinfo = P.read_json(a.server_info)
    flags = (sinfo or {}).get("flags")
    fitted_sha = R.write_prompts_gz(out_dir / "fitted_prompts.json.gz", texts)
    config = {
        "schema": 1, "kind": "judge_saved_calls", "monitor": a.monitor, "registry_spec": spec.as_dict(), "monitor_config": cfg.as_dict(),
        "registry_config_sha256": P.sha_obj({"spec": spec.as_dict(), "config": cfg.as_dict()}),
        "request_body_shape": client.request_body("<user message>", max_tokens=cfg.max_tokens, seed=cfg.seed, sampling=cfg.sampling, enable_thinking=cfg.enable_thinking),
        "concurrency": 1, "max_attempts": a.max_attempts, "limit": a.limit, "views": views, "instruction": a.instruction,
        "instruction_sha256": pr.sha256_bytes(instr.encode()), "policy_sha256": pr.POLICY_SHA256, "policy": pr.POLICY,
        "unit": NOTE_UNIT, "stage2_list_sha256": world["stage2_list_sha256"], "histories_sha256": world["histories_sha256"], "outputs_sha256": world["outputs_sha256"],
        "trajectory_sha256": {r["history"]["history_id"]: r["trajectory_sha256"] for r in world["reps"]}, "manifest_sha256": manifest_sha,
        "positions": [{"history_id": q["history_id"], "task": q["task"], "weight": q["weight"], "run_key": q["run_key"]} for q in sel["positions"]],
        "schedule": [{k: v for k, v in it.items() if k in ("order", "run_key", "history_id", "view", "user_sha256", "boundary_seq", "flags")} for it in pub_items],
        "plan": plan, "fitted_prompts_sha256": fitted_sha, "budget_rule": rule, "token_counting": {"method": method, "source": str(src)},
        "timed_interval": "HTTP request to the local vLLM server, sent to parsed JSON reply (prefill and generation inside); no retry inside a request; warm-up and server load recorded apart",
        "server": {"engine": ident.get("engine"), "engine_version": ident.get("engine_version"), "model_alias": ident.get("model_alias"),
                   "max_model_len": server_ctx, "launch_flags": flags},
        "hardware": P.gpu_static(), "model": P.model_identity(a.model_dir, spec), "code_sha256": {**R.code_hashes(), **{f.name: P.sha256_file(f) for f in (
            pathlib.Path(__file__), pathlib.Path(pr.__file__))}}, "score_values_stored": False, "labels_read": False,
    }
    P.check_or_write_config(out_dir, config)
    main_doc = read_main(a.main_complete_call)
    rows = P.read_rows(rows_path)
    pend = {i: item_pending(item_rows(rows, i), a.max_attempts, cfg, server_ctx, plan[i]) for i in range(len(items))}
    pend = {i: p for i, p in pend.items() if p is not None}
    if not pend:
        print(f"[judge {a.monitor}] nothing left to run; summary rewritten", flush=True)
        return finish(out_dir, config, items, plan, sel, main_doc, a.server_log)
    session = {"session": len(P.read_rows(sessions_path)) + 1, "start_utc": P.now_utc(), "git": P.git_state(), "server_info_session": (sinfo or {}).get("session"),
               "gpu_before_warmup": P.gpu_dynamic(a.server_pid), "server_log_at_start": P.parse_server_log(a.server_log), "resumed_rows": len(rows)}
    wu_user = m.FINAL_STATE_INSTRUCTION.format(reasoning_note="", record=P.WARMUP_RECORD)
    wu = P.post_once(client, a.url, wu_user, P.output_budget(cfg, P.user_tokens("final_state", wu_user, counter, cfg)), cfg, a.timeout)
    wu.pop("_text", None)
    session["warmup"] = {**wu, "untimed": True, "note": "outside every row; includes any first-request compilation"}
    session["gpu_after_warmup"] = P.gpu_dynamic(a.server_pid)
    P.append_jsonl(sessions_path, session)
    t_session = time.perf_counter()
    consecutive = 0
    for n_done, i in enumerate(sorted(pend)):
        it, p = items[i], plan[i]
        base = {"item": i, "run_key": it["run_key"], "history_id": it["history_id"], "task": it["task"], "representative_index": it["representative_index"],
                "view": it["view"], "original_user_sha256": p["original_user_sha256"], "user_sha256": p["user_sha256"], "original_user_tokens": p["original_user_tokens"],
                "user_tokens_counted": p["user_tokens"], "evidence_shortened": p["evidence_shortened"], "fitted": p["fitted"],
                **{f"response_{k}" if not k.startswith("response") else k: v for k, v in it["flags"].items()}, "session": session["session"], "concurrency": 1}
        if not p["sendable"]:
            P.append_jsonl(rows_path, {**base, "attempt": 0, "not_sent": True, "http_error": None, "valid_score": False, "output_cap": p["output_cap"],
                                       "note": p.get("not_sendable_reason"), "start_utc": P.now_utc()})
            print(f"[judge {a.monitor}] item {n_done + 1}/{len(pend)}: not sent ({p.get('not_sendable_reason')})", flush=True)
            continue
        kind, cap, made = pend[i]
        user = texts.get(i, it["user"])
        first_failure = None
        while True:
            made += 1
            start_utc, offset = P.now_utc(), time.perf_counter() - t_session
            res = P.post_once(client, a.url, user, cap, cfg, a.timeout)
            text = res.pop("_text")
            P.append_jsonl(rows_path, {**base, "attempt": made, "retry_kind": kind, "original_output_cap": p["output_cap"], "output_cap": cap,
                                       "original_failure": first_failure, "start_utc": start_utc, "start_offset_s": round(offset, 3),
                                       **{k: (round(v, 6) if k == "seconds" else v) for k, v in res.items()}})
            if text is not None:
                P.append_jsonl(replies_path, {"item": i, "attempt": made, "text": text})
            consecutive = consecutive + 1 if res["http_error"] else 0
            if first_failure is None and (res["http_error"] or not res["valid_score"]):
                first_failure = res["http_error"] or f"no valid score (finish {res['finish_reason']}, {res['unparseable_reason']})"
            nr = next_retry(res, made, a.max_attempts, cfg, server_ctx, p["user_tokens"], cap)
            if nr is None or consecutive >= a.max_consecutive_errors:
                break
            kind, cap = nr
        done = P.read_rows(rows_path)
        print(f"[judge {a.monitor}] item {n_done + 1}/{len(pend)}: replies {len({r['item'] for r in done if r.get('http_error') is None and not r.get('not_sent')})}, "
              f"valid {len({r['item'] for r in done if r.get('http_error') is None and r.get('valid_score')})}, http error attempts {sum(1 for r in done if r.get('http_error'))}", flush=True)
        if consecutive >= a.max_consecutive_errors:
            print(f"[judge {a.monitor}] stopping after {consecutive} consecutive errors; rerun to resume", flush=True)
            return 3
    return finish(out_dir, config, items, plan, sel, main_doc, a.server_log)


def finish(out_dir, config, items, plan, sel, main_doc, server_log) -> int:
    rows = P.read_rows(out_dir / "rows.jsonl")
    s = summarize(config, items, plan, rows, sel, main_doc)
    s["hardware_label"] = hw_label((config["hardware"] or {}).get("nvidia_smi_gpu_name_memory_total_mib_driver_power_limit_w"))
    s["server_log_end_of_run"] = P.parse_server_log(server_log) if server_log else None
    s["sessions"] = len(P.read_rows(out_dir / "sessions.jsonl"))
    s["note"] = NOTE_UNIT
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({v: {"valid": d["n_valid"], "of": d["n_prompts"], "shortened": d["evidence_shortened_count"]} for v, d in s["views"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
